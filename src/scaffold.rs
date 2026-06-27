//! scaffold — task-conditioned GVS configuration ("self-scaffolding at inference").
//!
//! Ornith-1.0's (DeepReinforce, Jun 2026) headline idea is that the agent scaffold — retry budget,
//! orchestration, temperatures — should *adapt to the task* instead of being one fixed harness
//! hand-tuned once. Ornith *learns* that scaffold during RL post-training. Metis can't retrain a
//! frontier model, but the same win is available cheaply: a deterministic classifier picks a
//! scaffold profile per query, and the profile tunes the GVS knobs (`GvsConfig`). No training, no
//! extra model, no extra latency — and a clean upgrade path to an LLM-proposed scaffold later,
//! because [`Scaffold::select`] is the only thing that would need to change.
//!
//! The four profiles below are the distinct GVS regimes the existing code already wanted but
//! applied uniformly:
//!
//!   Compute     arithmetic / exact-compute → the `calc` tool is the source of truth, so ONE
//!               low-temperature pass is right. Drawing diverse "search" candidates here mostly
//!               invites the model to *skip* the tool and guess, which is the opposite of helpful.
//!   OpenDomain  web-blended / noisy evidence → widen the search budget and decorrelate candidates
//!               (higher search temp) to cover the answer space, since the first shot is less reliable.
//!   Factual     grounded local lookup/synthesis → the balanced default GVS profile.
//!   Direct      no evidence to verify against (chitchat / unverifiable) → a single pass, no search.
//!
//! This is self-scaffolding *in inference*, not in training — the honest, runnable slice of the
//! Ornith idea for a tiny local Cortex.

use crate::conductor::GvsConfig;
use crate::library::Hit;

/// A GVS regime selected per query. See the module docs for the rationale behind each.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scaffold {
    Compute,
    OpenDomain,
    Factual,
    Direct,
}

impl Scaffold {
    pub fn as_str(&self) -> &'static str {
        match self {
            Scaffold::Compute => "compute",
            Scaffold::OpenDomain => "opendomain",
            Scaffold::Factual => "factual",
            Scaffold::Direct => "direct",
        }
    }

    /// parse maps an env / override string to a forced scaffold. `"auto"` (or anything unknown)
    /// returns None, meaning "classify the query". `"off"` is handled by the caller (it bypasses
    /// scaffolding entirely and uses the base config).
    pub fn parse(s: &str) -> Option<Scaffold> {
        match s.trim().to_ascii_lowercase().as_str() {
            "compute" => Some(Scaffold::Compute),
            "opendomain" | "open" | "web" => Some(Scaffold::OpenDomain),
            "factual" | "local" => Some(Scaffold::Factual),
            "direct" | "chitchat" => Some(Scaffold::Direct),
            _ => None,
        }
    }

    /// select classifies a query into a scaffold using cheap, deterministic heuristics over the
    /// query text and the retrieved evidence. The classifier is intentionally simple and
    /// swappable — the point is the *seam*, not the sophistication.
    ///
    ///   - no evidence retrieved        → Direct  (nothing to ground or verify against)
    ///   - looks like exact arithmetic  → Compute (defer to the calc tool, one shot)
    ///   - any web/open-domain evidence → OpenDomain (noisier, widen + decorrelate search)
    ///   - otherwise                    → Factual (the balanced default)
    pub fn select(query: &str, hits: &[Hit]) -> Scaffold {
        if hits.is_empty() {
            return Scaffold::Direct;
        }
        if looks_like_compute(query) {
            return Scaffold::Compute;
        }
        // A web hit is sourced from a URL (see `web_evidence`); local chunks carry a filename.
        if hits.iter().any(|h| h.chunk.source.starts_with("http")) {
            return Scaffold::OpenDomain;
        }
        Scaffold::Factual
    }

    /// apply tunes a base `GvsConfig` for this scaffold. It only adjusts the orchestration knobs
    /// (candidate budget + temperatures); the verifier and abstain message are left untouched, so
    /// an operator's `METIS_SEARCH` / `METIS_NLI_URL` choices still flow through as the baseline.
    pub fn apply(&self, base: GvsConfig) -> GvsConfig {
        let mut c = base;
        match self {
            Scaffold::Compute => {
                // The calc tool is authoritative; one deterministic pass, no diverse re-rolls.
                c.max_candidates = 1;
                c.gen_temp = 0.0;
            }
            Scaffold::OpenDomain => {
                // Noisier evidence: give the loop more room and decorrelate the search candidates.
                c.max_candidates = c.max_candidates.max(4);
                c.gen_temp = 0.3;
                c.search_temp = 0.9;
            }
            Scaffold::Factual => {
                // Balanced default — leave the base knobs as-is.
            }
            Scaffold::Direct => {
                // Nothing to verify against; a single pass is all that's meaningful.
                c.max_candidates = 1;
            }
        }
        c
    }
}

/// looks_like_compute is a deterministic heuristic for "this query wants exact arithmetic". It is
/// purposely conservative: it fires on an explicit numeric expression (two numbers joined by an
/// operator, e.g. `84937 * 2261`) OR on arithmetic verbs paired with at least one number. Prose
/// that merely contains a number ("version 4 of the standard") does not qualify.
fn looks_like_compute(query: &str) -> bool {
    let q = query.to_ascii_lowercase();
    let has_digit = q.bytes().any(|b| b.is_ascii_digit());
    if !has_digit {
        return false;
    }
    // An inline expression: a digit, then an operator, then (after spaces) another digit.
    if has_numeric_expression(&q) {
        return true;
    }
    // Arithmetic verbs/nouns alongside a number.
    const VERBS: [&str; 12] = [
        "multiply", "multiplied", "divide", "divided", "times", "plus", "minus", "subtract",
        "square root", "percent", "product of", "sum of",
    ];
    VERBS.iter().any(|v| q.contains(v))
}

/// has_numeric_expression scans for `<digit> <op> <digit>`, tolerating spaces around the operator.
fn has_numeric_expression(q: &str) -> bool {
    let bytes = q.as_bytes();
    let ops = b"+-*/x";
    for i in 0..bytes.len() {
        if !bytes[i].is_ascii_digit() {
            continue;
        }
        // walk forward over the rest of this number
        let mut j = i + 1;
        while j < bytes.len() && (bytes[j].is_ascii_digit() || bytes[j] == b'.' || bytes[j] == b',')
        {
            j += 1;
        }
        // skip spaces
        let mut k = j;
        while k < bytes.len() && bytes[k] == b' ' {
            k += 1;
        }
        if k < bytes.len() && ops.contains(&bytes[k]) {
            // skip operator + spaces, expect a digit next
            let mut m = k + 1;
            while m < bytes.len() && bytes[m] == b' ' {
                m += 1;
            }
            if m < bytes.len() && bytes[m].is_ascii_digit() {
                return true;
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::library::{Chunk, Hit};

    fn hit(source: &str) -> Hit {
        Hit {
            chunk: Chunk { text: "x".into(), source: source.into(), idx: 0, vec: vec![] },
            score: 0.5,
        }
    }

    #[test]
    fn empty_evidence_is_direct() {
        assert_eq!(Scaffold::select("hello there", &[]), Scaffold::Direct);
    }

    #[test]
    fn arithmetic_is_compute() {
        let h = [hit("a.md")];
        assert_eq!(Scaffold::select("What is 84937 * 2261?", &h), Scaffold::Compute);
        assert_eq!(Scaffold::select("divide 100 by 7", &h), Scaffold::Compute);
        assert_eq!(Scaffold::select("the product of 6 and 7", &h), Scaffold::Compute);
    }

    #[test]
    fn prose_number_is_not_compute() {
        let h = [hit("a.md")];
        // A bare number in factual prose must NOT trigger the compute scaffold.
        assert_eq!(Scaffold::select("In what year was version 4 ratified?", &h), Scaffold::Factual);
    }

    #[test]
    fn web_hit_is_opendomain() {
        let h = [hit("https://example.com/page")];
        assert_eq!(Scaffold::select("who won the 2025 election", &h), Scaffold::OpenDomain);
    }

    #[test]
    fn local_factual_is_factual() {
        let h = [hit("zephyr.md")];
        assert_eq!(Scaffold::select("What is the protocol mascot?", &h), Scaffold::Factual);
    }

    #[test]
    fn apply_tunes_knobs_per_scaffold() {
        let base = GvsConfig::default();
        assert_eq!(Scaffold::Compute.apply(GvsConfig::default()).max_candidates, 1);
        assert_eq!(Scaffold::Compute.apply(GvsConfig::default()).gen_temp, 0.0);
        assert!(Scaffold::OpenDomain.apply(GvsConfig::default()).max_candidates >= 4);
        assert_eq!(Scaffold::Direct.apply(GvsConfig::default()).max_candidates, 1);
        // Factual leaves the candidate budget at the base default.
        assert_eq!(Scaffold::Factual.apply(GvsConfig::default()).max_candidates, base.max_candidates);
    }

    #[test]
    fn opendomain_respects_higher_operator_budget() {
        // If an operator set METIS_SEARCH=6, OpenDomain must not shrink it to 4.
        let base = GvsConfig { max_candidates: 6, ..GvsConfig::default() };
        assert_eq!(Scaffold::OpenDomain.apply(base).max_candidates, 6);
    }

    #[test]
    fn parse_overrides() {
        assert_eq!(Scaffold::parse("compute"), Some(Scaffold::Compute));
        assert_eq!(Scaffold::parse("WEB"), Some(Scaffold::OpenDomain));
        assert_eq!(Scaffold::parse("auto"), None);
    }
}
