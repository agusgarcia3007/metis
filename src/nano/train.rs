//! AdamW optimizer with decoupled weight decay, the train step, and a finite-difference
//! gradient checker that verifies the whole autograd engine.

use super::model::{Gpt, Rng};
use super::tensor::{Tape, T};

/// AdamW optimizer with decoupled weight decay.
pub struct AdamW {
    pub lr: f32,
    pub b1: f32,
    pub b2: f32,
    pub eps: f32,
    pub wd: f32,
    t: i64,
    m: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    params: Vec<T>,
}

impl AdamW {
    /// NewAdamW initializes optimizer state for the given parameters.
    pub fn new(params: Vec<T>, lr: f32, wd: f32) -> AdamW {
        let mut m = Vec::with_capacity(params.len());
        let mut v = Vec::with_capacity(params.len());
        for p in &params {
            let n = p.borrow().data.len();
            m.push(vec![0.0f32; n]);
            v.push(vec![0.0f32; n]);
        }
        AdamW {
            lr,
            b1: 0.9,
            b2: 0.95,
            eps: 1e-8,
            wd,
            t: 0,
            m,
            v,
            params,
        }
    }

    /// ZeroGrad clears all parameter gradients.
    pub fn zero_grad(&self) {
        for p in &self.params {
            p.borrow_mut().zero_grad();
        }
    }

    /// Step applies one AdamW update with uniform decoupled weight decay.
    ///
    /// (The Go original accepted an optional per-parameter `decayMask` used by the RNT
    /// anti-memorization experiment; every call site passes nil, so this port applies the
    /// nil-mask path: weight decay `wd` to every parameter.)
    pub fn step(&mut self) {
        self.t += 1;
        let bc1 = 1.0f32 - (self.b1 as f64).powf(self.t as f64) as f32;
        let bc2 = 1.0f32 - (self.b2 as f64).powf(self.t as f64) as f32;
        let (b1, b2, lr, eps, wd) = (self.b1, self.b2, self.lr, self.eps, self.wd);
        for (pi, p) in self.params.iter().enumerate() {
            let mut pb = p.borrow_mut();
            let m = &mut self.m[pi];
            let v = &mut self.v[pi];
            for i in 0..pb.data.len() {
                let g = pb.grad[i];
                m[i] = b1 * m[i] + (1.0 - b1) * g;
                v[i] = b2 * v[i] + (1.0 - b2) * g * g;
                let mh = m[i] / bc1;
                let vh = v[i] / bc2;
                pb.data[i] -= lr * (mh / ((vh as f64).sqrt() as f32 + eps) + wd * pb.data[i]);
            }
        }
    }
}

impl Gpt {
    /// LossAndGrad runs forward+backward for one batch and returns the mean loss.
    /// idx and targets have length B*T. It zeroes grads, so call Step after.
    pub fn loss_and_grad(
        &self,
        o: &mut AdamW,
        idx: &[usize],
        targets: &[i32],
        b: usize,
        t: usize,
    ) -> f32 {
        o.zero_grad();
        let mut tp = Tape::new();
        let logits = self.forward(&mut tp, idx, b, t);
        let (loss, val) = tp.cross_entropy(&logits, targets);
        tp.backward(&loss);
        val
    }

    /// forwardLoss recomputes just the scalar loss (for gradient checking).
    fn forward_loss(&self, idx: &[usize], targets: &[i32], b: usize, t: usize) -> f32 {
        let mut tp = Tape::new();
        let logits = self.forward(&mut tp, idx, b, t);
        let (_loss, val) = tp.cross_entropy(&logits, targets);
        val
    }

    /// GradCheck numerically verifies analytic gradients via central finite differences.
    /// It returns the max relative error over *measurable* entries and how many were measured.
    pub fn grad_check(
        &self,
        idx: &[usize],
        targets: &[i32],
        b: usize,
        t: usize,
        n: usize,
        seed: i64,
    ) -> (f32, usize) {
        let mut o = AdamW::new(self.params(), 0.0, 0.0);
        self.loss_and_grad(&mut o, idx, targets, b, t); // populate analytic grads
        let mut rng = Rng::new(seed);
        let params = self.params();
        const EPS: f32 = 4e-3; // balances float32 cancellation noise vs O(eps^2) truncation
        const FLOOR: f32 = 1e-2; // only test well-resolved gradients (signal >> float32 noise floor)
        let mut max_rel = 0.0f32;
        let mut measured = 0usize;
        let mut attempts = 0usize;
        while measured < n && attempts < n * 40 {
            attempts += 1;
            let p = &params[(rng.next() % params.len() as u64) as usize];
            let plen = p.borrow().data.len();
            let i = (rng.next() % plen as u64) as usize;
            let orig = p.borrow().data[i];
            let analytic = p.borrow().grad[i];

            p.borrow_mut().data[i] = orig + EPS;
            let lp = self.forward_loss(idx, targets, b, t);
            p.borrow_mut().data[i] = orig - EPS;
            let lm = self.forward_loss(idx, targets, b, t);
            p.borrow_mut().data[i] = orig;
            let numeric = (lp - lm) / (2.0 * EPS);

            if (analytic as f64).abs() < FLOOR as f64 && (numeric as f64).abs() < FLOOR as f64 {
                continue; // unmeasurable in float32
            }
            let denom = (analytic as f64).abs() as f32 + (numeric as f64).abs() as f32;
            let rel = ((analytic - numeric) as f64).abs() as f32 / denom;
            if rel > max_rel {
                max_rel = rel;
            }
            measured += 1;
        }
        (max_rel, measured)
    }
}
