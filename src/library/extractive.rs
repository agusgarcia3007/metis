//! The cascade's fast path: for a factual lookup whose answer is a span already in a retrieved
//! chunk, score the chunk's sentences against the query with the tiny CPU embedder and return the
//! best one — WITHOUT running the generative LLM (the single biggest latency lever).

use super::{cosine, Embedder, Hit};
use regex::Regex;
use std::sync::OnceLock;

fn sentence_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    // Split on sentence punctuation ONLY when followed by whitespace (or a newline run). This
    // protects decimals like "1.84" — the dot there has no trailing space, so the number stays
    // intact instead of being severed into "1" and "84 GB".
    RE.get_or_init(|| Regex::new(r"(?:[.!?:;]\s+|\n+)").unwrap())
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

/// needs_reasoning returns true when a question cannot be answered by copying a single sentence —
/// it requires comparison, aggregation, a superlative, or chaining facts across chunks. For these,
/// the extractive fast-path is a trap: a sentence that merely *mentions* the entities scores high on
/// cosine similarity but does not actually *answer* the question (e.g. "which is larger, A or B?"
/// matches the sentence describing A without performing the comparison). Such questions must go
/// through Generate·Verify·Search so the model reasons and the answer is verified. This is a
/// high-precision guard: it only suppresses the fast-path when a reasoning marker is clearly present,
/// so genuine single-fact lookups keep their ~0.1s path.
pub fn needs_reasoning(query: &str) -> bool {
    // Normalize: lowercase, punctuation → spaces, collapse runs. This makes " strictest " match even
    // when the source was "strictest?" (the trailing "?" would otherwise defeat the space-bounded check).
    let lower = query.to_lowercase();
    let cleaned: String = lower
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { ' ' })
        .collect();
    let q = format!(" {} ", cleaned.split_whitespace().collect::<Vec<_>>().join(" "));

    // Comparison / superlative adjectives (chained as " word ") — these demand reasoning over ≥2 facts.
    const COMPARATIVES: &[&str] = &[
        "larger", "largest", "smaller", "smallest", "bigger", "biggest", "greater", "greatest",
        "higher", "highest", "lower", "lowest", "longer", "longest", "shorter", "shortest",
        "more", "most", "less", "least", "fewer", "fewest", "older", "oldest", "newer", "newest",
        "better", "best", "worse", "worst", "strictest", "strict", "loosest",
    ];
    for w in COMPARATIVES {
        if q.contains(&format!(" {w} ")) {
            return true;
        }
    }

    // Aggregation / arithmetic phrases.
    const AGGREGATIONS: &[&str] = &[
        "combined", "together", "in total", "total memory", "sum of", "difference between",
        "average", "percent", " than ", "compared to", "between ",
    ];
    for p in AGGREGATIONS {
        if q.contains(p) {
            return true;
        }
    }

    // Multi-hop relational structure: the question pivots on one entity to ask about another
    // ("the group that maintains X", "the component codenamed Y", "whose group is chaired by Z").
    const RELATIONAL: &[&str] = &[
        "maintained by", "that maintains", "responsible for", "whose ", " codenamed ",
        "chaired by", "that is chaired",
    ];
    for p in RELATIONAL {
        if q.contains(p) {
            return true;
        }
    }

    false
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
        // The decimal must survive intact — no span may end at a bare "1", and one span must
        // carry the full "1.84 GB" figure (regression guard for the sentence-splitter bug).
        assert!(s.iter().any(|x| x.contains("1.84 GB")), "decimal was severed: {s:?}");
        assert!(!s.iter().any(|x| x.ends_with(" 1")), "span truncated at a decimal: {s:?}");
    }

    #[test]
    fn needs_reasoning_flags_reasoning_questions() {
        // Comparison / superlative / aggregation / multi-hop → must skip the fast-path.
        assert!(needs_reasoning("Which has a larger memory budget, Aster or Quill?"));
        assert!(needs_reasoning("Which component has the smallest memory budget?"));
        assert!(needs_reasoning("Which conformance tier is the strictest?"));
        assert!(needs_reasoning("Combined, how much memory do Quill and Tideway use?"));
        assert!(needs_reasoning("Between the glass marmot and the veil moth, which lives longer?"));
        assert!(needs_reasoning("Who chairs the working group that maintains the Lumen component?"));
        assert!(needs_reasoning("What is the memory budget of the component codenamed Curlew?"));

        // Genuine single-fact lookups → keep the fast-path.
        assert!(!needs_reasoning("What is the memory budget of the Lumen component?"));
        assert!(!needs_reasoning("What is the codename of the Aster component?"));
        assert!(!needs_reasoning("How many eggs does the ashfoot heron lay each season?"));
        assert!(!needs_reasoning("In which city is the Orrery Foundation chartered?"));
        assert!(!needs_reasoning("What is the wingspan of the veil moth?"));
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
