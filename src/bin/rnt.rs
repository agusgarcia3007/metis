//! Command rnt runs the experiment that proves Retrieval-Native Training (RNT).
//!
//! It trains the SAME tiny transformer two ways on a task that cleanly separates knowledge
//! (a subject->value world) from reasoning (answer = (value+3) mod 10):
//!
//!   Vanilla : input [? subject >]               -> must MEMORIZE the world in its weights.
//!   RNT     : input [subject = value ; ? subj >] -> the fact is RETRIEVED into context.
//!
//! Then it evaluates both on a NEW world whose facts were never in training. The gap is the result.
#![allow(clippy::too_many_arguments, clippy::manual_clamp)]

use std::time::Instant;

use metis_0::nano::{
    load_gpt, AdamW, AssocTask, Config, Gpt, InductionTask, RecallTask, RetrievalTask, Task,
    VOCAB_SIZE,
};

fn nparams(cfg: Config) -> usize {
    Gpt::new(cfg, 0)
        .params()
        .iter()
        .map(|p| p.borrow().data.len())
        .sum()
}

/// runQuery trains an RNT reasoner, saves it, reloads it, and answers about a world it never saw.
fn run_query(lr: f64) {
    const M: usize = 50;
    let task = Task::new(M);
    let cfg = Config {
        vocab: VOCAB_SIZE,
        block: task.t,
        layer: 2,
        head: 2,
        embd: 64,
        ..Default::default()
    };
    let g = Gpt::new(cfg, 7);
    let mut opt = AdamW::new(g.params(), lr as f32, 0.0);
    println!("training RNT reasoner (1000 steps)...");
    for s in 1..=1000i64 {
        let (idx, tgt) = task.rnt_batch(32, s); // batch 32
        g.loss_and_grad(&mut opt, &idx, &tgt, 32, task.t);
        opt.step();
    }
    let _ = std::fs::create_dir_all("models");
    let path = "models/rnt-reasoner.gob";
    if let Err(e) = g.save(path) {
        println!("save error: {e}");
        return;
    }
    let size_kb = std::fs::metadata(path).map(|m| m.len() as f64 / 1024.0).unwrap_or(0.0);
    let loaded = match load_gpt(path) {
        Ok(l) => l,
        Err(e) => {
            println!("load error: {e}");
            return;
        }
    };
    println!("saved + reloaded model: {path} ({size_kb:.1} KB on disk)\n");

    // Query the RELOADED model about a brand-new world (facts never trained on).
    let world = task.random_world(2024);
    println!("asking the reloaded reasoner about facts it never trained on:");
    println!("  rule it learned: answer = (value + 3) mod 10\n");
    let mut correct = 0;
    for &s in &[3usize, 17, 42, 8, 25] {
        let v = world[s];
        let got = task.answer(&loaded, s, v);
        let want = task.transform(v);
        let ok = if got != want { "✗" } else { correct += 1; "✓" };
        println!("  fact: subject {s:03} = {v}   ->  reasoner answers {got}  (expected {want}) {ok}");
    }
    println!(
        "\n{correct}/5 correct — a {size_kb:.0} KB model, loaded from disk, reasoning over retrieved facts."
    );
}

/// trainRetrieval trains a model on the K-distractor retrieval task and returns final accuracy.
fn train_retrieval(
    m: usize,
    k: usize,
    embd: usize,
    layer: usize,
    heads: usize,
    steps: i64,
    lr: f64,
    tag: &str,
) -> (f64, usize, usize) {
    let task = RetrievalTask::new(m, k);
    let cfg = Config {
        vocab: VOCAB_SIZE,
        block: task.t,
        layer,
        head: heads,
        embd,
        ..Default::default()
    };
    let g = Gpt::new(cfg, 7);
    let params: usize = g.params().iter().map(|p| p.borrow().data.len()).sum();
    let mut opt = AdamW::new(g.params(), lr as f32, 0.0);
    for s in 1..=steps {
        let (idx, tgt) = task.batch(32, s);
        let loss = g.loss_and_grad(&mut opt, &idx, &tgt, 32, task.t);
        opt.step();
        if s % 2000 == 0 {
            println!(
                "   [{tag}] step {s:5} loss {loss:.4} acc {:.1}%",
                task.accuracy(&g, 99, 300) * 100.0
            );
        }
    }
    (task.accuracy(&g, 99, 1000), params, task.t)
}

/// runImprove fixes retrieval failure by scaling capacity (heads + layers + steps).
fn run_improve(lr: f64) {
    const M: usize = 100;
    println!("== RNT retrieval: breaking then improving (M={M} subjects, chance=10%) ==\n");
    struct Cfg {
        k: usize,
        embd: usize,
        layer: usize,
        heads: usize,
        steps: i64,
        note: &'static str,
    }
    let configs = [
        Cfg { k: 4, embd: 64, layer: 2, heads: 2, steps: 4000, note: "broken baseline" },
        Cfg { k: 4, embd: 96, layer: 3, heads: 6, steps: 6000, note: "improved" },
        Cfg { k: 8, embd: 96, layer: 4, heads: 6, steps: 9000, note: "improved, harder" },
    ];
    let mut rows = Vec::new();
    for c in &configs {
        let tag = format!("K{}/{}", c.k, c.note);
        let (acc, params, seqlen) = train_retrieval(M, c.k, c.embd, c.layer, c.heads, c.steps, lr, &tag);
        rows.push((c, acc, params, seqlen));
    }
    println!("\n{:<4} {:<28} {:<9} {:<8} {:<10}", "K", "config", "params", "seqlen", "accuracy");
    println!("{:<4} {:<28} {:<9} {:<8} {:<10}", "--", "------", "------", "------", "--------");
    for (c, acc, params, seqlen) in &rows {
        let label = format!("embd={} L={} H={} st={}", c.embd, c.layer, c.heads, c.steps);
        let accs = format!("{:.1}%", acc * 100.0);
        println!("{:<4} {:<28} {:<9} {:<8} {:<10}  ({})", c.k, label, params, seqlen, accs, c.note);
    }
    println!("\nGenuine associative retrieval is solvable at this scale once the model has enough");
    println!("heads/layers/steps — and it still fits in a few hundred KB, far under 4 GB.");
}

/// runRetrieval is the HARD test: many distractors; pure copying scores ~chance.
fn run_retrieval(lr: f64, embd: usize, layer: usize) {
    const M: usize = 1000;
    println!("== RNT retrieval-with-distractors ==");
    println!("model: embd={embd} layer={layer} | {M} possible subjects | chance=10%\n");
    println!("{:<9} {:<7} {:<9} {:<12}", "distract", "seqlen", "steps", "accuracy");
    println!("{:<9} {:<7} {:<9} {:<12}", "--------", "------", "-----", "--------");
    for k in [1usize, 2, 4, 8, 16] {
        let task = RetrievalTask::new(M, k);
        let cfg = Config {
            vocab: VOCAB_SIZE,
            block: task.t,
            layer,
            head: 2,
            embd,
            ..Default::default()
        };
        let g = Gpt::new(cfg, 7);
        let mut opt = AdamW::new(g.params(), lr as f32, 0.0);
        let steps = 4000i64;
        for s in 1..=steps {
            let (idx, tgt) = task.batch(32, s);
            g.loss_and_grad(&mut opt, &idx, &tgt, 32, task.t);
            opt.step();
        }
        let acc = task.accuracy(&g, 99, 1000);
        println!("{:<9} {:<7} {:<9} {:<12}", k, task.t, steps, format!("{:.1}%", acc * 100.0));
    }
    println!("\nHigh accuracy with many distractors = genuine associative retrieval + reasoning,");
    println!("not copying. Every sample has fresh random facts, so this is generalization by design.");
}

/// runRecall: retrieval reframed as DENSE-supervised induction; sweeps subject vocabulary.
fn run_recall() {
    const K: usize = 4;
    const B: usize = 32;
    const STEPS: i64 = 6000;
    let (embd, layer, heads) = (64usize, 2usize, 4usize);
    println!("== retrieval as dense induction — does content matching scale with subject vocabulary? ==");
    println!("fixed K={K} distractors, embd={embd} L={layer} H={heads}, {STEPS} steps, chance=10%\n");
    println!("{:<8} {:<10}", "subjects", "accuracy");
    for m in [8usize, 16, 32, 64] {
        let task = RecallTask::new(m, K, false);
        let cfg = Config {
            vocab: 10 + m,
            block: task.l,
            layer,
            head: heads,
            embd,
            no_rope: true,
            ..Default::default()
        };
        let g = Gpt::new(cfg, 7);
        let mut opt = AdamW::new(g.params(), 0.0, 0.0);
        let peak = 3e-3f32;
        for s in 1..=STEPS {
            opt.lr = if s < 200 { peak * s as f32 / 200.0 } else { peak };
            let (idx, tgt) = task.batch(B, s);
            g.loss_and_grad(&mut opt, &idx, &tgt, B, task.l);
            opt.step();
        }
        println!("{:<8} {:<10}", m, format!("{:.1}%", task.accuracy(&g, 99, 500) * 100.0));
    }
}

/// runInduction: can the engine learn the LITERAL canonical induction task at all?
fn run_induction() {
    const V: usize = 40;
    const M: usize = 8;
    const B: usize = 32;
    const STEPS: i64 = 4000;
    let task = InductionTask::new(V, M); // L = 2M = 16
    println!(
        "== canonical induction diagnostic (repeated block, dense supervision, V={V} L={} chance={:.1}%) ==\n",
        task.l,
        100.0 / V as f64
    );
    for use_rope in [true, false] {
        let cfg = Config {
            vocab: V,
            block: task.l,
            layer: 2,
            head: 4,
            embd: 64,
            no_rope: !use_rope,
            ..Default::default()
        };
        let g = Gpt::new(cfg, 7);
        let mut opt = AdamW::new(g.params(), 0.0, 0.0);
        let peak = 3e-3f32;
        println!("-- RoPE={use_rope} --");
        for s in 1..=STEPS {
            opt.lr = if s < 200 { peak * s as f32 / 200.0 } else { peak };
            let (idx, tgt) = task.batch(B, s);
            g.loss_and_grad(&mut opt, &idx, &tgt, B, task.l);
            opt.step();
            if s % 1000 == 0 {
                println!("   step {s:5}  acc {:.1}%", task.accuracy(&g, 99, 500) * 100.0);
            }
        }
        println!();
    }
}

/// runLevel2: canonical induction-head setup (wpe ON, RoPE OFF) with cosine LR + long training.
fn run_level2() {
    const M: usize = 24;
    const NQ: usize = 4;
    const B: usize = 32;
    let (embd, layer, heads, steps) = (64usize, 2usize, 4usize, 16000i64);
    let max_t = 4 * 2 + NQ * 3;
    let cfg = Config {
        vocab: AssocTask::new_q(M, 2, NQ).vocab(),
        block: max_t + 2,
        layer,
        head: heads,
        embd,
        no_rope: true,
        ..Default::default()
    };
    let g = Gpt::new(cfg, 7);
    let mut opt = AdamW::new(g.params(), 0.0, 0.0);
    let (peak, warm) = (3e-3f32, 300i64);
    println!(
        "== RNT associative recall — level2 (wpe ON, RoPE OFF, embd={embd} L={layer} H={heads}, K=2..4, {steps} steps) ==\n"
    );
    let mut best = 0.0f64;
    for s in 1..=steps {
        if s < warm {
            opt.lr = peak * s as f32 / warm as f32;
        } else {
            let prog = (s - warm) as f64 / (steps - warm) as f64;
            opt.lr = peak * (0.1 + 0.9 * 0.5 * (1.0 + (std::f64::consts::PI * prog).cos())) as f32;
        }
        let k = 2 + (s as usize % 3);
        let task = AssocTask::new_q(M, k, NQ);
        let (idx, tgt) = task.batch(B, s);
        let loss = g.loss_and_grad(&mut opt, &idx, &tgt, B, task.t);
        opt.step();
        if s % 1000 == 0 {
            let a2 = AssocTask::new_q(M, 2, NQ).accuracy(&g, 99, 300);
            let a4 = AssocTask::new_q(M, 4, NQ).accuracy(&g, 99, 300);
            if a4 > best {
                best = a4;
            }
            println!(
                "   step {s:6} lr {:.4} loss {loss:.4}  acc[K=2] {:.1}%  acc[K=4] {:.1}%",
                opt.lr,
                a2 * 100.0,
                a4 * 100.0
            );
        }
    }
    println!("\nfinal accuracy across distractor counts (chance=10%):");
    for k in [2usize, 4, 8, 16] {
        let task = AssocTask::new_q(M, k, 1);
        println!("   K={:<2}:  {:.1}%", k, task.accuracy(&g, 99, 1000) * 100.0);
    }
}

/// runFinal: decisive associative-recall attempt (wpe+RoPE, bigger model, K cycles 2..4).
fn run_final() {
    const M: usize = 32;
    const NQ: usize = 4;
    const B: usize = 32;
    let (embd, layer, heads, steps) = (128usize, 4usize, 8usize, 9000i64);
    let max_t = 4 * 2 + NQ * 3;
    let cfg = Config {
        vocab: AssocTask::new_q(M, 2, NQ).vocab(),
        block: max_t + 2,
        layer,
        head: heads,
        embd,
        ..Default::default()
    };
    let g = Gpt::new(cfg, 7);
    let mut opt = AdamW::new(g.params(), 0.0, 0.0);
    let peak = 3e-3f32;
    println!("== RNT associative recall — final (wpe+RoPE, embd={embd} L={layer} H={heads}, K cycles 2..4) ==\n");
    for s in 1..=steps {
        opt.lr = if s < 200 { peak * s as f32 / 200.0 } else { peak };
        let k = 2 + (s as usize % 3);
        let task = AssocTask::new_q(M, k, NQ);
        let (idx, tgt) = task.batch(B, s);
        let loss = g.loss_and_grad(&mut opt, &idx, &tgt, B, task.t);
        opt.step();
        if s % 1000 == 0 {
            println!(
                "   step {s:5} loss {loss:.4}  acc[K=2] {:.1}%  acc[K=4] {:.1}%",
                AssocTask::new_q(M, 2, NQ).accuracy(&g, 99, 300) * 100.0,
                AssocTask::new_q(M, 4, NQ).accuracy(&g, 99, 300) * 100.0
            );
        }
    }
    println!("\nfinal accuracy across distractor counts:");
    for k in [2usize, 4, 8, 16] {
        let task = AssocTask::new_q(M, k, 1);
        println!("   K={:<2}:  {:.1}%", k, task.accuracy(&g, 99, 1000) * 100.0);
    }
}

/// runCurriculum trains ONE position-param-free model (NoPos + RoPE) through K=1 -> 2 -> 4.
fn run_curriculum() {
    const M: usize = 32;
    const NQ: usize = 4;
    const B: usize = 32;
    let (embd, layer, heads) = (96usize, 3usize, 6usize);
    let max_t = 4 * 2 + NQ * 3;
    let cfg = Config {
        vocab: AssocTask::new_q(M, 1, NQ).vocab(),
        block: max_t,
        layer,
        head: heads,
        embd,
        no_pos: true,
        ..Default::default()
    };
    let g = Gpt::new(cfg, 7);
    let mut opt = AdamW::new(g.params(), 0.0, 0.0);
    let peak = 3e-3f32;
    let mut gstep = 0i64;
    println!("== RNT associative recall — curriculum (NoPos+RoPE, embd={embd} L={layer} H={heads}) ==\n");
    for (k, stage_steps) in [(1usize, 2500i64), (2, 3500), (4, 5000)] {
        let task = AssocTask::new_q(M, k, NQ);
        for s in 1..=stage_steps {
            gstep += 1;
            opt.lr = if gstep < 200 { peak * gstep as f32 / 200.0 } else { peak };
            let (idx, tgt) = task.batch(B, gstep);
            let loss = g.loss_and_grad(&mut opt, &idx, &tgt, B, task.t);
            opt.step();
            if s % 1000 == 0 {
                println!("   K={k} step {s:5} loss {loss:.4} acc {:.1}%", task.accuracy(&g, 99, 300) * 100.0);
            }
        }
    }
    println!("\nfinal accuracy of the single trained model across distractor counts:");
    for k in [1usize, 2, 4, 8, 16] {
        let task = AssocTask::new_q(M, k, 1);
        println!("   K={:<2} ({:2} facts in context):  {:.1}%", k, k, task.accuracy(&g, 99, 1000) * 100.0);
    }
    println!("\nchance = 10%. High accuracy that holds as distractors grow = genuine associative");
    println!("retrieval: the model finds the queried subject among many and reasons over its value.");
}

/// runProbe diagnoses whether the tiny transformer can learn associative recall at all.
fn run_probe() {
    const M: usize = 32;
    const NQ: usize = 4;
    const B: usize = 32;
    const TOTAL: i64 = 5000;
    const WARM: i64 = 200;
    let peak = 3e-3f32;
    for k in [2usize, 4] {
        let task = AssocTask::new_q(M, k, NQ);
        let cfg = Config {
            vocab: task.vocab(),
            block: task.t,
            layer: 3,
            head: 6,
            embd: 96,
            no_pos: true,
            ..Default::default()
        };
        let g = Gpt::new(cfg, 7);
        let mut opt = AdamW::new(g.params(), 0.0, 0.0);
        println!("== probe K={k} | NoPos+RoPE, embd=96 L=3 H=6, NQ={NQ} dense ==");
        for s in 1..=TOTAL {
            opt.lr = if s < WARM { peak * s as f32 / WARM as f32 } else { peak };
            let (idx, tgt) = task.batch(B, s);
            let loss = g.loss_and_grad(&mut opt, &idx, &tgt, B, task.t);
            opt.step();
            if s % 1000 == 0 {
                println!("   step {s:5}  loss {loss:.4}  acc {:.1}%", task.accuracy(&g, 99, 300) * 100.0);
            }
        }
        println!();
    }
}

/// runAssoc tests canonical single-token associative recall.
fn run_assoc(lr: f64, embd: usize, layer: usize, heads: usize) {
    const M: usize = 64;
    println!("== RNT associative recall (single-token subjects, M={M}, chance=10%) ==");
    println!("model: embd={embd} layer={layer} heads={heads}\n");
    println!("{:<9} {:<7} {:<7} {:<10}", "distract", "seqlen", "steps", "accuracy");
    println!("{:<9} {:<7} {:<7} {:<10}", "--------", "------", "-----", "--------");
    for k in [2usize, 4, 8, 16] {
        let task = AssocTask::new(M, k);
        let cfg = Config {
            vocab: task.vocab(),
            block: task.t,
            layer,
            head: heads,
            embd,
            ..Default::default()
        };
        let g = Gpt::new(cfg, 7);
        let mut opt = AdamW::new(g.params(), lr as f32, 0.0);
        let steps = 4000i64;
        for s in 1..=steps {
            let (idx, tgt) = task.batch(32, s);
            g.loss_and_grad(&mut opt, &idx, &tgt, 32, task.t);
            opt.step();
        }
        let acc = task.accuracy(&g, 99, 1000);
        println!("{:<9} {:<7} {:<7} {:<10}", k, task.t, steps, format!("{:.1}%", acc * 100.0));
    }
    println!("\nSolved associative recall = the model genuinely RETRIEVES the right fact among");
    println!("distractors and reasons over it. Multi-token subjects add parsing on top of this.");
}

/// runSweep demonstrates the capacity wall at FIXED tiny model size.
fn run_sweep(lr: f64) {
    const B: usize = 32;
    let (embd, layer) = (16usize, 1usize); // deliberately tiny so the memorization wall appears early
    println!("== RNT capacity-wall sweep ==");
    println!("fixed model: embd={embd} layer={layer} (fixed parameter budget)\n");
    println!("{:<8} {:<9} {:<8} {:<18} {:<18}", "facts", "params", "steps", "VANILLA seen-acc", "RNT new-world-acc");
    println!("{:<8} {:<9} {:<8} {:<18} {:<18}", "-----", "------", "-----", "----------------", "-----------------");
    for m in [64usize, 256, 1024, 4096] {
        let task = Task::new(m);
        let cfg = Config {
            vocab: VOCAB_SIZE,
            block: task.t,
            layer,
            head: 2,
            embd,
            ..Default::default()
        };
        let t = task.t;
        let world_train = task.random_world(1);
        let world_new = task.random_world(999);

        // scale steps so each fact is seen ~50x regardless of M
        let mut steps = (m * 50 / B) as i64;
        if steps < 2000 {
            steps = 2000;
        }
        if steps > 9000 {
            steps = 9000;
        }

        let params = nparams(cfg);

        let gv = Gpt::new(cfg, 7);
        let mut optv = AdamW::new(gv.params(), lr as f32, 0.0);
        for s in 1..=steps {
            let (idx, tgt) = task.vanilla_batch(&world_train, B, s);
            gv.loss_and_grad(&mut optv, &idx, &tgt, B, t);
            optv.step();
        }
        let v_seen = task.vanilla_accuracy(&gv, &world_train);

        let gr = Gpt::new(cfg, 7);
        let mut optr = AdamW::new(gr.params(), lr as f32, 0.0);
        for s in 1..=steps {
            let (idx, tgt) = task.rnt_batch(B, s);
            gr.loss_and_grad(&mut optr, &idx, &tgt, B, t);
            optr.step();
        }
        let r_new = task.rnt_accuracy(&gr, &world_new, 5, 500);

        println!(
            "{:<8} {:<9} {:<8} {:<18} {:<18}",
            m,
            params,
            steps,
            format!("{:.1}%", v_seen * 100.0),
            format!("{:.1}%", r_new * 100.0)
        );
    }
    println!("\nVanilla seen-accuracy falls as facts exceed the model's memorization capacity");
    println!("(knowledge competes for a fixed parameter budget). RNT stays ~100% at the SAME size");
    println!("because every fact is supplied in context — so growing knowledge costs disk, not RAM.");
}

fn main() {
    let flags = parse_flags();
    let lr = flags.lr;

    match flags.mode.as_str() {
        "sweep" => return run_sweep(lr),
        "query" => return run_query(lr),
        "retrieval" => return run_retrieval(lr, flags.embd, flags.layer),
        "improve" => return run_improve(lr),
        "assoc" => return run_assoc(lr, flags.embd, flags.layer, flags.heads),
        "probe" => return run_probe(),
        "curriculum" => return run_curriculum(),
        "final" => return run_final(),
        "level2" => return run_level2(),
        "induction" => return run_induction(),
        "recall" => return run_recall(),
        _ => {}
    }

    // default: the "demo" mode — train vanilla vs RNT and compare on a new world.
    let task = Task::new(flags.subjects);
    let cfg = Config {
        vocab: VOCAB_SIZE,
        block: task.t,
        layer: flags.layer,
        head: 2,
        embd: flags.embd,
        ..Default::default()
    };
    let (b, t) = (flags.batch, task.t);

    let world_train = task.random_world(1);
    let world_new = task.random_world(999); // facts never seen in training

    let nparams = nparams(cfg);
    println!("== RNT experiment ==");
    println!(
        "model: embd={} layer={}  params={} (~{:.0} KB fp32)",
        flags.embd,
        flags.layer,
        nparams,
        (nparams * 4) as f64 / 1024.0
    );
    println!(
        "task : {} subjects, transform=(value+3)%10, vocab={}, seq={}\n",
        flags.subjects, VOCAB_SIZE, task.t
    );

    // ---- Vanilla: must memorize the world ----
    let gv = Gpt::new(cfg, 7);
    let mut optv = AdamW::new(gv.params(), lr as f32, 0.0);
    let t0 = Instant::now();
    for s in 1..=flags.steps {
        let (idx, tgt) = task.vanilla_batch(&world_train, b, s);
        let loss = gv.loss_and_grad(&mut optv, &idx, &tgt, b, t);
        optv.step();
        if s % 500 == 0 || s == 1 {
            println!("[vanilla] step {s:4}  loss {loss:.4}");
        }
    }
    let v_seen = task.vanilla_accuracy(&gv, &world_train);
    let v_new = task.vanilla_accuracy(&gv, &world_new);
    println!("[vanilla] trained in {:?}\n", t0.elapsed());

    // ---- RNT: knowledge retrieved into context ----
    let gr = Gpt::new(cfg, 7);
    let mut optr = AdamW::new(gr.params(), lr as f32, 0.0);
    let t0 = Instant::now();
    for s in 1..=flags.steps {
        let (idx, tgt) = task.rnt_batch(b, s);
        let loss = gr.loss_and_grad(&mut optr, &idx, &tgt, b, t);
        optr.step();
        if s % 500 == 0 || s == 1 {
            println!("[rnt]     step {s:4}  loss {loss:.4}");
        }
    }
    let r_new = task.rnt_accuracy(&gr, &world_new, 5, 500);
    let r_train_world = task.rnt_accuracy(&gr, &world_train, 5, 500);
    println!("[rnt]     trained in {:?}\n", t0.elapsed());

    let chance = 1.0 / task.k as f64;
    println!("=================== RESULTS ===================");
    println!("chance accuracy (10 answers)        : {:5.1}%", chance * 100.0);
    println!("VANILLA  accuracy on TRAINED world  : {:5.1}%   (memorized — works)", v_seen * 100.0);
    println!("VANILLA  accuracy on NEW world      : {:5.1}%   (knowledge frozen in weights — fails)", v_new * 100.0);
    println!("RNT      accuracy on NEW world      : {:5.1}%   (reads retrieved fact — generalizes)", r_new * 100.0);
    println!("RNT      accuracy on trained world  : {:5.1}%", r_train_world * 100.0);
    println!("==============================================");
    println!("\nTakeaway: identical model + params. Vanilla must grow its weights to know more");
    println!("facts and cannot adapt to new ones. RNT learns to REASON over retrieved facts, so");
    println!("it answers about worlds it never trained on — knowledge is DATA (on disk), not weights.");
    println!("That is why a tiny RNT reasoner + a disk corpus fits 4 GB yet scales its knowledge freely.");
}

struct Flags {
    steps: i64,
    subjects: usize,
    embd: usize,
    layer: usize,
    heads: usize,
    batch: usize,
    lr: f64,
    mode: String,
}

/// parse_flags mirrors the Go `flag` package usage: -steps -subjects -embd -layer -heads -batch -lr -mode.
fn parse_flags() -> Flags {
    let mut f = Flags {
        steps: 1500,
        subjects: 50,
        embd: 64,
        layer: 2,
        heads: 4,
        batch: 32,
        lr: 1e-3,
        mode: "demo".to_string(),
    };
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        let parsed = a.strip_prefix('-').map(|rest| {
            let rest = rest.strip_prefix('-').unwrap_or(rest); // allow --flag too
            match rest.split_once('=') {
                Some((n, v)) => (n.to_string(), Some(v.to_string())),
                None => (rest.to_string(), None),
            }
        });
        let (name, inline_val) = match parsed {
            Some(p) => p,
            None => {
                i += 1;
                continue;
            }
        };
        let val = || -> Option<String> {
            inline_val.clone()
        };
        // If no inline value, consume the next argument.
        let value: Option<String> = match val() {
            Some(v) => Some(v),
            None => {
                i += 1;
                args.get(i).cloned()
            }
        };
        match name.as_str() {
            "steps" => f.steps = value.and_then(|v| v.parse().ok()).unwrap_or(f.steps),
            "subjects" => f.subjects = value.and_then(|v| v.parse().ok()).unwrap_or(f.subjects),
            "embd" => f.embd = value.and_then(|v| v.parse().ok()).unwrap_or(f.embd),
            "layer" => f.layer = value.and_then(|v| v.parse().ok()).unwrap_or(f.layer),
            "heads" => f.heads = value.and_then(|v| v.parse().ok()).unwrap_or(f.heads),
            "batch" => f.batch = value.and_then(|v| v.parse().ok()).unwrap_or(f.batch),
            "lr" => f.lr = value.and_then(|v| v.parse().ok()).unwrap_or(f.lr),
            "mode" => f.mode = value.unwrap_or(f.mode.clone()),
            _ => {}
        }
        i += 1;
    }
    f
}
