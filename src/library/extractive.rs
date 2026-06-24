//! The cascade's fast path: for a factual lookup whose answer is a span already in a retrieved
//! chunk, score the chunk's sentences against the query with the tiny CPU embedder and return the
//! best one — WITHOUT running the generative LLM (the single biggest latency lever).

use super::{cosine, Embedder, Hit};
use regex::Regex;
use std::sync::OnceLock;

fn sentence_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?:[.!?:;]|\n)+\s*").unwrap())
}

/// SplitSentences breaks text into candidate answer spans (sentences/clauses), dropping trivia.
pub fn split_sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for p in sentence_re().split(text) {
        let trimmed = p
            .trim_matches(|c: char| matches!(c, '#' | '-' | '*' | '>' | ' '))
            .trim();
        let n = trimmed.split_whitespace().count();
        if (3..=60).contains(&n) {
            out.push(trimmed.to_string());
        }
    }
    out
}

/// Extraction is the best sentence-level answer found in the retrieved hits.
#[derive(Clone, Debug, Default)]
pub struct Extraction {
    pub answer: String,
    pub source: String,
    pub score: f32,
}

/// Extract returns the retrieved sentence most similar to the query (and its source/score).
pub fn extract(emb: &Embedder, hits: &[Hit], query: &str) -> Result<Extraction, String> {
    let mut cands: Vec<(String, String)> = Vec::new();
    for h in hits {
        for s in split_sentences(&h.chunk.text) {
            cands.push((s, h.chunk.source.clone()));
        }
    }
    if cands.is_empty() {
        return Ok(Extraction::default());
    }
    let mut texts: Vec<String> = Vec::with_capacity(cands.len() + 1);
    texts.push(query.to_string());
    for c in &cands {
        texts.push(c.0.clone());
    }
    let vecs = emb.embed(&texts)?;
    let q = &vecs[0];
    let mut best = 0usize;
    let mut best_score = -1.0f32;
    for i in 1..vecs.len() {
        let s = cosine(q, &vecs[i]);
        if s > best_score {
            best_score = s;
            best = i - 1;
        }
    }
    Ok(Extraction {
        answer: cands[best].0.clone(),
        source: cands[best].1.clone(),
        score: best_score,
    })
}

#[cfg(test)]
mod tests {
    use super::super::normalize;
    use super::*;

    #[test]
    fn test_split_sentences() {
        let txt = "The mascot is a heron. Memory must not exceed 1.84 GB; exactly 3 shards may be cached.";
        let s = split_sentences(txt);
        assert!(s.len() >= 3, "expected >=3 candidate spans, got {}: {s:?}", s.len());
        for x in &s {
            assert!(!x.is_empty(), "empty span");
        }
    }

    #[test]
    fn extract_ranking() {
        let mut q = vec![0.1f32, 0.95, 0.0];
        normalize(&mut q);
        let a = vec![0.0f32, 1.0, 0.0]; // "answer" — closest
        let b = vec![1.0f32, 0.0, 0.0]; // unrelated
        assert!(
            cosine(&q, &a) > cosine(&q, &b),
            "expected answer vector to win: a={} b={}",
            cosine(&q, &a),
            cosine(&q, &b)
        );
    }
}
