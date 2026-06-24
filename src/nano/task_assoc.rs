//! The canonical associative-recall ("induction head") test with SINGLE-TOKEN subjects. It isolates
//! pure retrieval from multi-digit parsing: the model must match a query's subject token to one fact
//! among K distractors and read/transform its value. NQ>1 gives dense supervision.

use super::model::{Gpt, Rng};
use std::collections::{HashMap, HashSet};

/// AssocTask holds a single-token-subject associative-recall task with NFacts facts and NQ queries.
#[derive(Clone, Copy)]
pub struct AssocTask {
    pub m: usize,
    pub k: usize,
    pub nfacts: usize,
    pub nq: usize,
    pub t: usize,
    // eq/sep/ans/pad mirror the Go struct's reserved token ids; kept for fidelity.
    #[allow(dead_code)]
    eq: usize,
    #[allow(dead_code)]
    sep: usize,
    q: usize,
    #[allow(dead_code)]
    ans: usize,
    #[allow(dead_code)]
    pad: usize,
    voc: usize,
}

impl AssocTask {
    /// NewAssocTask builds a task with nFacts facts and 1 query (sparse supervision).
    pub fn new(m: usize, nfacts: usize) -> AssocTask {
        AssocTask::new_q(m, nfacts, 1)
    }

    /// NewAssocTaskQ builds a task with nFacts facts and nQ queries (nQ>1 = dense supervision).
    pub fn new_q(m: usize, nfacts: usize, nq: usize) -> AssocTask {
        let base = 10 + m; // values 0..9, subjects 10..10+m-1
        AssocTask {
            m,
            k: 10,
            nfacts,
            nq,
            t: nfacts * 2 + nq * 3, // fact = [subj val] (2); query = [? subj ans] (3)
            eq: base,
            sep: base + 1,
            q: base + 2,
            ans: base + 3,
            pad: base + 4,
            voc: base + 5,
        }
    }

    /// Vocab returns the task's vocabulary size (use for Config.vocab).
    pub fn vocab(&self) -> usize {
        self.voc
    }
    fn subj(&self, s: usize) -> usize {
        10 + s
    }
    fn transform(&self, v: usize) -> usize {
        (v + 3) % self.k
    }

    /// queryAnsPos returns the flat position of the query-subject token of query i.
    fn query_ans_pos(&self, i: usize) -> usize {
        self.nfacts * 2 + i * 3 + 1
    }

    fn sample(&self, rng: &mut Rng) -> (Vec<usize>, Vec<usize>) {
        let mut seq: Vec<usize> = Vec::with_capacity(self.t);
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
            seq.push(self.subj(s));
            seq.push(v); // [subject value]
        }
        let mut answers = vec![0usize; self.nq];
        for ans_slot in answers.iter_mut() {
            let qs = subs[(rng.next() % subs.len() as u64) as usize];
            let a = self.transform(vals[&qs]);
            *ans_slot = a;
            seq.push(self.q);
            seq.push(self.subj(qs));
            seq.push(a); // [? subject answer]
        }
        (seq, answers)
    }

    /// Batch builds B problems, supervising every query's answer.
    pub fn batch(&self, b: usize, seed: i64) -> (Vec<usize>, Vec<i32>) {
        let mut rng = Rng::new(seed);
        let mut idx = vec![0usize; b * self.t];
        let mut targets = vec![-1i32; b * self.t];
        for bb in 0..b {
            let (seq, answers) = self.sample(&mut rng);
            idx[bb * self.t..bb * self.t + self.t].copy_from_slice(&seq);
            for (i, a) in answers.iter().enumerate() {
                targets[bb * self.t + self.query_ans_pos(i)] = *a as i32;
            }
        }
        (idx, targets)
    }

    /// Accuracy over n fresh random problems, evaluated on the first query position.
    pub fn accuracy(&self, g: &Gpt, seed: i64, n: usize) -> f64 {
        let mut rng = Rng::new(seed);
        let mut correct = 0;
        let ap = self.query_ans_pos(0);
        for _ in 0..n {
            let (seq, answers) = self.sample(&mut rng);
            if g.predict_at(&seq, 1, self.t, ap) == answers[0] {
                correct += 1;
            }
        }
        correct as f64 / n as f64
    }
}
