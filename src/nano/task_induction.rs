//! The canonical induction-head training setup: a random block repeated twice (r ++ r) with DENSE
//! next-token supervision at every position. Induction needs a previous-token head, which only
//! forms under dense supervision — the lesson behind the retrieval plateau fix.

use super::model::{Gpt, Rng};

/// InductionTask: vocabulary V, block length M (sequence length L = 2M).
#[derive(Clone, Copy)]
pub struct InductionTask {
    pub v: usize,
    pub m: usize,
    pub l: usize,
}

impl InductionTask {
    /// NewInductionTask builds a repeated-block induction task with block length m.
    pub fn new(v: usize, m: usize) -> InductionTask {
        InductionTask { v, m, l: 2 * m }
    }

    fn sample(&self, rng: &mut Rng) -> Vec<usize> {
        let mut seq = vec![0usize; self.l];
        for i in 0..self.m {
            let tok = (rng.next() % self.v as u64) as usize;
            seq[i] = tok;
            seq[i + self.m] = tok;
        }
        seq
    }

    /// Batch builds B sequences with DENSE next-token targets (target[i] = seq[i+1]).
    pub fn batch(&self, b: usize, seed: i64) -> (Vec<usize>, Vec<i32>) {
        let mut rng = Rng::new(seed);
        let mut idx = vec![0usize; b * self.l];
        let mut targets = vec![0i32; b * self.l];
        for bb in 0..b {
            let seq = self.sample(&mut rng);
            idx[bb * self.l..bb * self.l + self.l].copy_from_slice(&seq);
            for i in 0..self.l {
                if i < self.l - 1 {
                    targets[bb * self.l + i] = seq[i + 1] as i32;
                } else {
                    targets[bb * self.l + i] = -1;
                }
            }
        }
        (idx, targets)
    }

    /// Accuracy measures next-token accuracy over the SECOND copy (induction-predictable positions).
    pub fn accuracy(&self, g: &Gpt, seed: i64, n: usize) -> f64 {
        let mut rng = Rng::new(seed);
        let mut correct = 0;
        let mut total = 0;
        for _ in 0..n {
            let seq = self.sample(&mut rng);
            for i in self.m..self.l - 1 {
                total += 1;
                if g.predict_at(&seq, 1, self.l, i) == seq[i + 1] {
                    correct += 1;
                }
            }
        }
        correct as f64 / total as f64
    }
}
