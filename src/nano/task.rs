//! The synthetic benchmark that proves the RNT training paradigm: a task that factorizes cleanly
//! into KNOWLEDGE (subject -> value) and REASONING (answer = (value + 3) mod 10), over a fixed
//! 15-token vocabulary so the parameter budget — and thus the memorization wall — is real.

use super::model::{Gpt, Rng};

// Fixed vocabulary (independent of the number of facts).
pub(crate) const TOK_EQ: usize = 10;
pub(crate) const TOK_SEP: usize = 11;
pub(crate) const TOK_Q: usize = 12;
pub(crate) const TOK_ANS: usize = 13;
pub(crate) const TOK_PAD: usize = 14;
pub const VOCAB_SIZE: usize = 15;

/// Task holds the encoding for a world with M subjects.
#[derive(Clone, Copy)]
pub struct Task {
    pub m: usize, // number of subjects (facts)
    pub k: usize, // number of values/answers = 10
    pub d: usize, // digits per subject code
    pub t: usize, // sequence length used by the model
}

impl Task {
    /// NewTask builds the encoding for M subjects.
    pub fn new(m: usize) -> Task {
        let mut d = 1;
        let mut p = 10;
        while p < m {
            d += 1;
            p *= 10;
        }
        Task {
            m,
            k: 10,
            d,
            t: 2 * d + 6,
        }
    }

    fn transform_(&self, v: usize) -> usize {
        (v + 3) % self.k
    }

    fn digits(&self, mut s: usize) -> Vec<usize> {
        let mut out = vec![0usize; self.d];
        for i in (0..self.d).rev() {
            out[i] = s % 10;
            s /= 10;
        }
        out
    }

    fn vanilla_ans_pos(&self) -> usize {
        self.d + 1
    }
    pub(crate) fn rnt_ans_pos(&self) -> usize {
        2 * self.d + 4
    }

    /// vanillaSeq: [Q d.. > ans PAD..] length T.
    fn vanilla_seq(&self, subj: usize, ans: usize) -> Vec<usize> {
        let mut seq = vec![TOK_PAD; self.t];
        seq[0] = TOK_Q;
        let dg = self.digits(subj);
        seq[1..1 + self.d].copy_from_slice(&dg);
        seq[self.d + 1] = TOK_ANS;
        seq[self.d + 2] = ans;
        seq
    }

    /// rntSeq: [d.. = val ; Q d.. > ans] length T.
    pub(crate) fn rnt_seq(&self, subj: usize, val: usize, ans: usize) -> Vec<usize> {
        let mut seq = vec![0usize; self.t];
        let dg = self.digits(subj);
        seq[0..self.d].copy_from_slice(&dg);
        seq[self.d] = TOK_EQ;
        seq[self.d + 1] = val;
        seq[self.d + 2] = TOK_SEP;
        seq[self.d + 3] = TOK_Q;
        seq[self.d + 4..self.d + 4 + self.d].copy_from_slice(&dg);
        seq[2 * self.d + 4] = TOK_ANS;
        seq[2 * self.d + 5] = ans;
        seq
    }

    /// RandomWorld returns a fresh subject->value mapping.
    pub fn random_world(&self, seed: i64) -> Vec<usize> {
        let mut rng = Rng::new(seed);
        let mut w = vec![0usize; self.m];
        for s in w.iter_mut() {
            *s = (rng.next() % self.k as u64) as usize;
        }
        w
    }

    /// VanillaBatch draws B queries from a fixed world (model must memorize subject->answer).
    pub fn vanilla_batch(&self, world: &[usize], b: usize, seed: i64) -> (Vec<usize>, Vec<i32>) {
        let mut rng = Rng::new(seed);
        let mut idx = vec![0usize; b * self.t];
        let mut targets = vec![-1i32; b * self.t];
        let ap = self.vanilla_ans_pos();
        for bb in 0..b {
            let s = (rng.next() % self.m as u64) as usize;
            let ans = self.transform_(world[s]);
            idx[bb * self.t..bb * self.t + self.t].copy_from_slice(&self.vanilla_seq(s, ans));
            targets[bb * self.t + ap] = ans as i32;
        }
        (idx, targets)
    }

    /// RNTBatch draws B queries with a random fact retrieved into context (model learns the transform).
    pub fn rnt_batch(&self, b: usize, seed: i64) -> (Vec<usize>, Vec<i32>) {
        let mut rng = Rng::new(seed);
        let mut idx = vec![0usize; b * self.t];
        let mut targets = vec![-1i32; b * self.t];
        let ap = self.rnt_ans_pos();
        for bb in 0..b {
            let s = (rng.next() % self.m as u64) as usize;
            let v = (rng.next() % self.k as u64) as usize;
            let ans = self.transform_(v);
            idx[bb * self.t..bb * self.t + self.t].copy_from_slice(&self.rnt_seq(s, v, ans));
            targets[bb * self.t + ap] = ans as i32;
        }
        (idx, targets)
    }

    /// VanillaAccuracy measures answer accuracy over all subjects of a world (no facts in context).
    pub fn vanilla_accuracy(&self, g: &Gpt, world: &[usize]) -> f64 {
        let mut correct = 0;
        let ap = self.vanilla_ans_pos();
        for s in 0..self.m {
            let ans = self.transform_(world[s]);
            let pred = g.predict_at(&self.vanilla_seq(s, ans), 1, self.t, ap);
            if pred == ans {
                correct += 1;
            }
        }
        correct as f64 / self.m as f64
    }

    /// Answer runs the model on a retrieved fact "subject = value" and returns its predicted answer.
    pub fn answer(&self, g: &Gpt, subj: usize, val: usize) -> usize {
        g.predict_at(&self.rnt_seq(subj, val, TOK_PAD), 1, self.t, self.rnt_ans_pos())
    }

    /// Transform exposes the reasoning rule for callers (the ground-truth answer).
    pub fn transform(&self, v: usize) -> usize {
        self.transform_(v)
    }

    /// RNTAccuracy measures answer accuracy over fresh (subject,value) pairs WITH the fact in context.
    pub fn rnt_accuracy(&self, g: &Gpt, world: &[usize], seed: i64, n: usize) -> f64 {
        let mut rng = Rng::new(seed);
        let mut correct = 0;
        let ap = self.rnt_ans_pos();
        for _ in 0..n {
            let s = (rng.next() % self.m as u64) as usize;
            let v = world[s];
            let ans = self.transform_(v);
            let pred = g.predict_at(&self.rnt_seq(s, v, ans), 1, self.t, ap);
            if pred == ans {
                correct += 1;
            }
        }
        correct as f64 / n as f64
    }
}
