//! RETRIEVAL reframed as dense-supervised induction. The first half lists K facts (subject value);
//! the second half repeats the SAME subjects shuffled, and at every second-half subject the model
//! must predict its value (or its transform) — K dense targets, the supervision density the recall
//! circuit needs.

use super::model::{Gpt, Rng};
use std::collections::{HashMap, HashSet};

/// RecallTask: M possible subjects, K facts/queries per sequence.
#[derive(Clone, Copy)]
pub struct RecallTask {
    pub m: usize,
    pub k: usize,
    pub l: usize,
    pub transform: bool, // if true, target = (value+3) mod 10; else target = value (pure recall)
}

impl RecallTask {
    /// NewRecallTask builds a recall task. L = 4K (K facts + K shuffled queries, each 2 tokens).
    pub fn new(m: usize, k: usize, transform: bool) -> RecallTask {
        RecallTask {
            m,
            k,
            l: 4 * k,
            transform,
        }
    }

    fn subj(&self, s: usize) -> usize {
        10 + s
    }
    fn ans(&self, v: usize) -> usize {
        if self.transform {
            (v + 3) % 10
        } else {
            v
        }
    }

    fn sample(&self, rng: &mut Rng) -> (Vec<usize>, Vec<i32>) {
        // K distinct subjects with random values
        let mut subs: Vec<usize> = Vec::with_capacity(self.k);
        let mut seen: HashSet<usize> = HashSet::new();
        while subs.len() < self.k {
            let s = (rng.next() % self.m as u64) as usize;
            if !seen.contains(&s) {
                seen.insert(s);
                subs.push(s);
            }
        }
        let mut val: HashMap<usize, usize> = HashMap::with_capacity(self.k);
        let mut seq: Vec<usize> = Vec::with_capacity(self.l);
        for &s in &subs {
            let v = (rng.next() % 10) as usize;
            val.insert(s, v);
            seq.push(self.subj(s));
            seq.push(v);
        }
        // shuffled second half
        let mut order = subs.clone();
        let mut i = order.len() - 1;
        while i > 0 {
            let j = (rng.next() % (i as u64 + 1)) as usize;
            order.swap(i, j);
            i -= 1;
        }
        let mut targets = vec![-1i32; self.l];
        for &s in &order {
            let pos = seq.len(); // position of this second-half subject token
            seq.push(self.subj(s));
            seq.push(val[&s]);
            targets[pos] = self.ans(val[&s]) as i32; // predict the value (or transform) AFTER the subject
        }
        (seq, targets)
    }

    /// Batch builds B sequences with K dense recall targets each.
    pub fn batch(&self, b: usize, seed: i64) -> (Vec<usize>, Vec<i32>) {
        let mut rng = Rng::new(seed);
        let mut idx = vec![0usize; b * self.l];
        let mut targets = vec![0i32; b * self.l];
        for bb in 0..b {
            let (seq, tg) = self.sample(&mut rng);
            idx[bb * self.l..bb * self.l + self.l].copy_from_slice(&seq);
            targets[bb * self.l..bb * self.l + self.l].copy_from_slice(&tg);
        }
        (idx, targets)
    }

    /// Accuracy measures recall accuracy over the K second-half query positions of n fresh sequences.
    pub fn accuracy(&self, g: &Gpt, seed: i64, n: usize) -> f64 {
        let mut rng = Rng::new(seed);
        let mut correct = 0;
        let mut total = 0;
        for _ in 0..n {
            let (seq, tg) = self.sample(&mut rng);
            for pos in 0..self.l {
                if tg[pos] < 0 {
                    continue;
                }
                total += 1;
                if g.predict_at(&seq, 1, self.l, pos) == tg[pos] as usize {
                    correct += 1;
                }
            }
        }
        correct as f64 / total as f64
    }
}
