//! The GPT model: config, embedding, transformer blocks, weight-tied output, and a tiny
//! deterministic RNG for reproducible initialization and data sampling.

use super::tensor::{new_param, Tape, T};
use serde::{Deserialize, Serialize};

/// Config defines the GPT shape.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Config {
    pub vocab: usize,
    pub block: usize, // context length T
    pub layer: usize,
    pub head: usize,
    pub embd: usize,   // C
    pub no_pos: bool,  // omit absolute position embedding
    pub no_rope: bool, // disable rotary position embedding in attention (pure content matching)
}

impl Tape {
    /// Embed: x[N,C] = wte[token] (+ wpe[position] if wpe is Some).
    pub(crate) fn embed(
        &mut self,
        idx: &[usize],
        wte: &T,
        wpe: Option<&T>,
        b: usize,
        t: usize,
    ) -> T {
        let c = wte.borrow().c;
        let n = b * t;
        let x = Tape::node(n, c);
        {
            let wb = wte.borrow();
            let pb = wpe.map(|p| p.borrow());
            let mut xb = x.borrow_mut();
            for nn in 0..n {
                let pos = nn % t;
                let tok = idx[nn];
                let xr = &mut xb.data[nn * c..nn * c + c];
                let wr = &wb.data[tok * c..tok * c + c];
                match &pb {
                    Some(pv) => {
                        let pr = &pv.data[pos * c..pos * c + c];
                        for cc in 0..c {
                            xr[cc] = wr[cc] + pr[cc];
                        }
                    }
                    None => xr.copy_from_slice(wr),
                }
            }
        }
        let idxc: Vec<usize> = idx.to_vec();
        let (wc, pc, xc) = (wte.clone(), wpe.cloned(), x.clone());
        self.push_bw(Box::new(move || {
            let mut wb = wc.borrow_mut();
            let mut pb = pc.as_ref().map(|p| p.borrow_mut());
            let xb = xc.borrow();
            for nn in 0..n {
                let pos = nn % t;
                let tok = idxc[nn];
                for cc in 0..c {
                    let xg = xb.grad[nn * c + cc];
                    wb.grad[tok * c + cc] += xg;
                    if let Some(pv) = pb.as_mut() {
                        pv.grad[pos * c + cc] += xg;
                    }
                }
            }
        }));
        x
    }
}

/// One transformer block's parameters.
pub(crate) struct Block {
    pub ln1g: T,
    pub ln1b: T,
    pub wqkv: T,
    pub bqkv: T,
    pub wproj: T,
    pub bproj: T,
    pub ln2g: T,
    pub ln2b: T,
    pub wfc: T,
    pub bfc: T,
    pub wfc2: T,
    pub bfc2: T,
}

/// GPT is a decoder-only transformer with tied input/output embeddings.
pub struct Gpt {
    pub cfg: Config,
    pub(crate) wte: T,
    pub(crate) wpe: T,
    pub(crate) blocks: Vec<Block>,
    pub(crate) lnfg: T,
    pub(crate) lnfb: T,
}

impl Gpt {
    /// NewGPT allocates and randomly initializes a model.
    pub fn new(cfg: Config, seed: i64) -> Gpt {
        let mut rng = Rng::new(seed);
        let c = cfg.embd;
        let std = 0.02f32;
        let mut mk = |r: usize, cc: usize, s: f32| -> T {
            let t = new_param(r, cc);
            {
                let mut tb = t.borrow_mut();
                for v in tb.data.iter_mut() {
                    *v = rng.normal() * s;
                }
            }
            t
        };
        let ones = |cc: usize| -> T {
            let t = new_param(1, cc);
            {
                let mut tb = t.borrow_mut();
                for v in tb.data.iter_mut() {
                    *v = 1.0;
                }
            }
            t
        };
        let zeros = |cc: usize| -> T { new_param(1, cc) };

        let wte = mk(cfg.vocab, c, std);
        let wpe = mk(cfg.block, c, std);
        // scale residual-projection inits by 1/sqrt(2*Layer) (GPT-2 init).
        let pscale = (0.02 / ((2 * cfg.layer) as f64).sqrt()) as f32;
        let mut blocks = Vec::with_capacity(cfg.layer);
        for _ in 0..cfg.layer {
            blocks.push(Block {
                ln1g: ones(c),
                ln1b: zeros(c),
                wqkv: mk(c, 3 * c, std),
                bqkv: zeros(3 * c),
                wproj: mk(c, c, pscale),
                bproj: zeros(c),
                ln2g: ones(c),
                ln2b: zeros(c),
                wfc: mk(c, 4 * c, std),
                bfc: zeros(4 * c),
                wfc2: mk(4 * c, c, pscale),
                bfc2: zeros(c),
            });
        }
        let lnfg = ones(c);
        let lnfb = zeros(c);
        Gpt {
            cfg,
            wte,
            wpe,
            blocks,
            lnfg,
            lnfb,
        }
    }

    /// Params returns every trainable tensor (for the optimizer & serialization), in a stable order.
    pub fn params(&self) -> Vec<T> {
        let mut ps = vec![
            self.wte.clone(),
            self.wpe.clone(),
            self.lnfg.clone(),
            self.lnfb.clone(),
        ];
        for b in &self.blocks {
            ps.push(b.ln1g.clone());
            ps.push(b.ln1b.clone());
            ps.push(b.wqkv.clone());
            ps.push(b.bqkv.clone());
            ps.push(b.wproj.clone());
            ps.push(b.bproj.clone());
            ps.push(b.ln2g.clone());
            ps.push(b.ln2b.clone());
            ps.push(b.wfc.clone());
            ps.push(b.bfc.clone());
            ps.push(b.wfc2.clone());
            ps.push(b.bfc2.clone());
        }
        ps
    }

    /// Forward runs the model over idx (length B*T) and returns logits[B*T, Vocab].
    pub(crate) fn forward(&self, tp: &mut Tape, idx: &[usize], b: usize, t: usize) -> T {
        let cfg = self.cfg;
        let wpe: Option<&T> = if cfg.no_pos { None } else { Some(&self.wpe) };
        let mut x = tp.embed(idx, &self.wte, wpe, b, t);
        for blk in &self.blocks {
            let a = tp.layer_norm(&x, &blk.ln1g, &blk.ln1b);
            let qkv = tp.linear(&a, &blk.wqkv, Some(&blk.bqkv));
            let parts = tp.split(&qkv, 3);
            let mut att = tp.attention(&parts[0], &parts[1], &parts[2], b, t, cfg.head, !cfg.no_rope);
            att = tp.linear(&att, &blk.wproj, Some(&blk.bproj));
            x = tp.add(&x, &att);
            let m = tp.layer_norm(&x, &blk.ln2g, &blk.ln2b);
            let fc = tp.linear(&m, &blk.wfc, Some(&blk.bfc));
            let hgelu = tp.gelu(&fc);
            let mlp = tp.linear(&hgelu, &blk.wfc2, Some(&blk.bfc2));
            x = tp.add(&x, &mlp);
        }
        x = tp.layer_norm(&x, &self.lnfg, &self.lnfb);
        tp.logits_tied(&x, &self.wte)
    }
}

// --- tiny deterministic RNG (xorshift + Box-Muller) so runs are reproducible ---

pub(crate) struct Rng {
    s: u64,
}

impl Rng {
    pub(crate) fn new(seed: i64) -> Rng {
        Rng {
            s: (seed as u64)
                .wrapping_mul(2862933555777941757)
                .wrapping_add(3037000493),
        }
    }

    pub(crate) fn next(&mut self) -> u64 {
        self.s ^= self.s << 13;
        self.s ^= self.s >> 7;
        self.s ^= self.s << 17;
        self.s
    }

    pub(crate) fn float(&mut self) -> f32 {
        (self.next() >> 11) as f32 / (1u64 << 53) as f32
    }

    pub(crate) fn normal(&mut self) -> f32 {
        let u1 = self.float() * 0.999999 + 1e-7;
        let u2 = self.float();
        ((-2.0 * (u1 as f64).ln()).sqrt() * (2.0 * std::f64::consts::PI * (u2 as f64)).cos()) as f32
    }
}
