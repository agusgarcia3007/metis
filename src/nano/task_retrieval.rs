//! The HARD version of the RNT benchmark: the context contains MANY facts (distractors), and the
//! model must find the queried subject among them, read ITS value, and apply the transform — genuine
//! associative retrieval + reasoning, not copying. Every sample has fresh random facts.

use super::model::{Gpt, Rng};
use super::task::{TOK_ANS, TOK_EQ, TOK_Q, TOK_SEP};
use std::collections::{HashMap, HashSet};

/// RetrievalTask holds the encoding for a K-fact retrieval problem over subjects in [0,M).
#[derive(Clone, Copy)]
pub struct RetrievalTask {
    pub m: usize,
    pub k: usize,
    pub d: usize,
    pub nfacts: usize,
    pub t: usize,
}

impl RetrievalTask {
    /// NewRetrievalTask builds a task with nFacts facts in context drawn from m subjects.
    pub fn new(m: usize, nfacts: usize) -> RetrievalTask {
        let mut d = 1;
        let mut p = 10;
        while p < m {
            d += 1;
            p *= 10;
        }
        let fact_len = d + 3; // digits + EQ + value + SEP
        let query_len = d + 3; // Q + digits + ANS + answer
        RetrievalTask {
            m,
            k: 10,
            d,
            nfacts,
            t: nfacts * fact_len + query_len,
        }
    }

    fn transform(&self, v: usize) -> usize {
        (v + 3) % self.k
    }
    pub(crate) fn ans_pos(&self) -> usize {
        self.t - 2
    } // the ANS token; predicts the final token

    /// sample builds one problem instance. Returns the token sequence and the correct answer.
    pub(crate) fn sample(&self, rng: &mut Rng) -> (Vec<usize>, usize) {
        let mut seq: Vec<usize> = Vec::with_capacity(self.t);
        // pick NFacts distinct subjects
        let mut subs: Vec<usize> = Vec::with_capacity(self.nfacts);
        let mut seen: HashSet<usize> = HashSet::new();
        while subs.len() < self.nfacts {
            let s = (rng.next() % self.m as u64) as usize;
            if !seen.contains(&s) {
                seen.insert(s);
                subs.push(s);
            }
        }
        let mut vals: HashMap<usize, usize> = HashMap::with_capacity(self.nfacts);
        for &s in &subs {
            let v = (rng.next() % self.k as u64) as usize;
            vals.insert(s, v);
            seq.extend_from_slice(&digits_to(s, self.d));
            seq.push(TOK_EQ);
            seq.push(v);
            seq.push(TOK_SEP);
        }
        // query a random one of the present subjects
        let q = subs[(rng.next() % subs.len() as u64) as usize];
        let ans = self.transform(vals[&q]);
        seq.push(TOK_Q);
        seq.extend_from_slice(&digits_to(q, self.d));
        seq.push(TOK_ANS);
        seq.push(ans);
        (seq, ans)
    }

    /// Batch builds B retrieval problems.
    pub fn batch(&self, b: usize, seed: i64) -> (Vec<usize>, Vec<i32>) {
        let mut rng = Rng::new(seed);
        let mut idx = vec![0usize; b * self.t];
        let mut targets = vec![-1i32; b * self.t];
        let ap = self.ans_pos();
        for bb in 0..b {
            let (seq, ans) = self.sample(&mut rng);
            idx[bb * self.t..bb * self.t + self.t].copy_from_slice(&seq);
            targets[bb * self.t + ap] = ans as i32;
        }
        (idx, targets)
    }

    /// Accuracy measures answer accuracy over n fresh random problems (always unseen by construction).
    pub fn accuracy(&self, g: &Gpt, seed: i64, n: usize) -> f64 {
        let mut rng = Rng::new(seed);
        let mut correct = 0;
        let ap = self.ans_pos();
        for _ in 0..n {
            let (seq, ans) = self.sample(&mut rng);
            if g.predict_at(&seq, 1, self.t, ap) == ans {
                correct += 1;
            }
        }
        correct as f64 / n as f64
    }
}

/// digitsTo returns the d-digit base-10 code of s (most significant first).
pub(crate) fn digits_to(mut s: usize, d: usize) -> Vec<usize> {
    let mut out = vec![0usize; d];
    for i in (0..d).rev() {
        out[i] = s % 10;
        s /= 10;
    }
    out
}
