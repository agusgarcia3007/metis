//! Ported nano tests: gradient check (with/without RoPE), the RNT generalization result, the dense
//! induction result, plus task-shape, serialization-roundtrip and determinism checks.
//!
//! The two model-training tests (`induction_learns`, `rnt_generalizes`) are marked `#[ignore]`
//! because they train a model and are slow — mirroring the Go `testing.Short()` skip. Run them
//! with `cargo test -- --ignored`.

use super::model::{Config, Gpt, Rng};
use super::task::{Task, TOK_ANS, TOK_PAD, VOCAB_SIZE};
use super::task_induction::InductionTask;
use super::task_retrieval::RetrievalTask;
use super::train::AdamW;
use super::serialize::load_gpt;

#[test]
fn grad_check() {
    let cfg = Config {
        vocab: 11,
        block: 6,
        layer: 2,
        head: 2,
        embd: 16,
        ..Default::default()
    };
    let g = Gpt::new(cfg, 42);
    let (b, t) = (2usize, cfg.block);
    let n = b * t;
    let mut rng = Rng::new(7);
    let mut idx = vec![0usize; n];
    let mut tgt = vec![0i32; n];
    for i in 0..n {
        idx[i] = (rng.next() % cfg.vocab as u64) as usize;
        tgt[i] = (rng.next() % cfg.vocab as u64) as i32;
    }
    let (max_rel, measured) = g.grad_check(&idx, &tgt, b, t, 60, 123);
    println!("max relative grad error = {max_rel:.2e} over {measured} measurable entries");
    assert!(measured >= 25, "too few measurable entries ({measured})");
    assert!(max_rel <= 3e-2, "gradient check FAILED: max rel error {max_rel:.2e} (>3e-2)");
}

#[test]
fn grad_check_no_rope() {
    let cfg = Config {
        vocab: 11,
        block: 6,
        layer: 2,
        head: 2,
        embd: 16,
        no_rope: true,
        ..Default::default()
    };
    let g = Gpt::new(cfg, 42);
    let (b, t) = (2usize, cfg.block);
    let n = b * t;
    let mut r = Rng::new(7);
    let mut idx = vec![0usize; n];
    let mut tg = vec![0i32; n];
    for i in 0..n {
        idx[i] = (r.next() % cfg.vocab as u64) as usize;
        tg[i] = (r.next() % cfg.vocab as u64) as i32;
    }
    let (mr, m) = g.grad_check(&idx, &tg, b, t, 60, 123);
    println!("NoRoPE maxRel={mr:.2e} over {m}");
    assert!(mr <= 3e-2 && m >= 25, "noRoPE gradcheck fail {mr:.2e} n={m}");
}

#[test]
fn retrieval_shape() {
    let task = RetrievalTask::new(1000, 4);
    let mut rng = Rng::new(1);
    for _ in 0..100 {
        let (seq, ans) = task.sample(&mut rng);
        assert_eq!(seq.len(), task.t, "seq len != T");
        assert_eq!(seq[task.ans_pos() + 1], ans, "answer token not at ansPos+1");
        assert_eq!(seq[task.ans_pos()], TOK_ANS, "ansPos is not the ANS token");
        assert!(ans < task.k, "answer {ans} out of range");
    }
}

#[test]
fn task_transform() {
    let task = Task::new(50);
    for v in 0..10 {
        assert_eq!(task.transform(v), (v + 3) % 10);
    }
}

#[test]
fn serialization_roundtrip() {
    let cfg = Config {
        vocab: VOCAB_SIZE,
        block: 12,
        layer: 2,
        head: 2,
        embd: 32,
        ..Default::default()
    };
    let g = Gpt::new(cfg, 5);
    let task = Task::new(50);
    let mut opt = AdamW::new(g.params(), 1e-3, 0.0);
    for s in 1..=50 {
        let (idx, tgt) = task.rnt_batch(16, s);
        g.loss_and_grad(&mut opt, &idx, &tgt, 16, task.t);
        opt.step();
    }

    let dir = std::env::temp_dir();
    let path = dir.join(format!("metis0_roundtrip_{}.bin", std::process::id()));
    let path = path.to_str().unwrap();
    g.save(path).unwrap();
    assert!(std::path::Path::new(path).exists());
    let loaded = load_gpt(path).unwrap();
    let _ = std::fs::remove_file(path);

    // parameters must match exactly
    let gp = g.params();
    let lp = loaded.params();
    for i in 0..gp.len() {
        let a = gp[i].borrow();
        let b = lp[i].borrow();
        for j in 0..a.data.len() {
            assert_eq!(a.data[j], b.data[j], "param {i}[{j}] mismatch after reload");
        }
    }
    // predictions must match
    for s in 0..20 {
        let seq = task.rnt_seq(s, s % 10, TOK_PAD);
        assert_eq!(
            g.predict_at(&seq, 1, task.t, task.rnt_ans_pos()),
            loaded.predict_at(&seq, 1, task.t, task.rnt_ans_pos()),
            "prediction mismatch after reload at subject {s}"
        );
    }
}

#[test]
fn determinism() {
    let run = || -> f32 {
        let cfg = Config {
            vocab: VOCAB_SIZE,
            block: 12,
            layer: 1,
            head: 2,
            embd: 16,
            ..Default::default()
        };
        let g = Gpt::new(cfg, 3);
        let mut opt = AdamW::new(g.params(), 1e-3, 0.0);
        let task = Task::new(50);
        let mut loss = 0.0f32;
        for s in 1..=30 {
            let (idx, tgt) = task.rnt_batch(16, s);
            loss = g.loss_and_grad(&mut opt, &idx, &tgt, 16, task.t);
            opt.step();
        }
        loss
    };
    let (a, b) = (run(), run());
    assert_eq!(a, b, "nondeterministic training: {a} != {b}");
}

#[test]
#[ignore = "trains a model; slow (mirrors Go testing.Short skip)"]
fn induction_learns() {
    let task = InductionTask::new(40, 8); // V=40, block M=8, L=16; chance = 1/40 = 2.5%
    let cfg = Config {
        vocab: 40,
        block: task.l,
        layer: 2,
        head: 4,
        embd: 64,
        ..Default::default()
    };
    let g = Gpt::new(cfg, 7);
    let mut opt = AdamW::new(g.params(), 0.0, 0.0);
    let peak = 3e-3f32;
    for s in 1..=1500 {
        opt.lr = if s < 200 { peak * s as f32 / 200.0 } else { peak };
        let (idx, tgt) = task.batch(32, s);
        g.loss_and_grad(&mut opt, &idx, &tgt, 32, task.l);
        opt.step();
    }
    let acc = task.accuracy(&g, 99, 500);
    println!("dense induction accuracy = {acc:.3} (chance 0.025)");
    assert!(acc >= 0.9, "dense induction should be ~solved (got {acc:.3})");
}

#[test]
#[ignore = "trains a model; slow (mirrors Go testing.Short skip)"]
fn rnt_generalizes() {
    let task = Task::new(20);
    let cfg = Config {
        vocab: VOCAB_SIZE,
        block: task.t,
        layer: 2,
        head: 2,
        embd: 48,
        ..Default::default()
    };
    let world_train = task.random_world(1);
    let world_new = task.random_world(999);

    // Vanilla: memorize worldTrain.
    let gv = Gpt::new(cfg, 7);
    let mut ov = AdamW::new(gv.params(), 2e-3, 0.0);
    for s in 1..=1500 {
        let (idx, tgt) = task.vanilla_batch(&world_train, 32, s);
        gv.loss_and_grad(&mut ov, &idx, &tgt, 32, task.t);
        ov.step();
    }
    let v_seen = task.vanilla_accuracy(&gv, &world_train);
    let v_new = task.vanilla_accuracy(&gv, &world_new);

    // RNT: fact retrieved into context.
    let gr = Gpt::new(cfg, 7);
    let mut or = AdamW::new(gr.params(), 2e-3, 0.0);
    for s in 1..=1500 {
        let (idx, tgt) = task.rnt_batch(32, s);
        gr.loss_and_grad(&mut or, &idx, &tgt, 32, task.t);
        or.step();
    }
    let r_new = task.rnt_accuracy(&gr, &world_new, 5, 500);

    println!("vanilla seen={v_seen:.2} new={v_new:.2} | RNT new={r_new:.2}");
    assert!(v_seen >= 0.85, "vanilla should memorize its trained world (got {v_seen:.2})");
    assert!(v_new <= 0.30, "vanilla should FAIL on a new world ~chance (got {v_new:.2})");
    assert!(r_new >= 0.90, "RNT should GENERALIZE to a new world (got {r_new:.2})");
}
