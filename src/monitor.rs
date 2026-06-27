//! monitor — the deterministic trust boundary (Layer 1 of verification).
//!
//! Ornith-1.0 (DeepReinforce, Jun 2026) hardens its RL scaffold with a three-layer anti-gaming
//! stack: a *fixed trust boundary*, a *deterministic monitor*, and a *frozen LLM judge*. The
//! ordering matters — the cheap, uncheatable deterministic check runs first, and only what
//! survives it reaches the (fallible, expensive) judge. Metis adopts the same shape for grounded
//! QA:
//!
//!   Layer 0  fixed trust boundary — the evidence set is immutable; the model may cite it, never edit it.
//!   Layer 1  deterministic monitor — THIS module: every inline `[n]` citation must reference a real
//!            retrieved source. Pure code, so it cannot be fooled.
//!   Layer 2  frozen judge — the existing `verifier` (LLM or NLI) entailment check.
//!
//! Why Layer 1 is necessary and not redundant with Layer 2: a tiny Cortex will happily write
//! "...caps memory at 1.84 GB [4]" when only 3 sources were retrieved, or cite `[2]` for a fact
//! that actually came from `[1]`. That sentence *reads* as grounded, and an entailment judge that
//! scores the prose against the pooled evidence can pass it — the claim may even be true. But the
//! *citation is fabricated*, which is precisely the failure mode that erodes trust in a grounded
//! system. Deterministic code catches it for free, before the judge spends a single token.

/// CitationReport is what the deterministic monitor found in a candidate answer.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CitationReport {
    /// Number of sources available to cite — valid indices are `1..=n_sources`.
    pub n_sources: usize,
    /// Distinct, in-range citation indices the answer actually used (sorted).
    pub used: Vec<usize>,
    /// Citation indices that point outside the evidence set — fabricated. `[0]` and any `[n]`
    /// with `n > n_sources` land here. Sorted, de-duplicated.
    pub out_of_range: Vec<usize>,
}

impl CitationReport {
    /// clean = no fabricated citations. A clean candidate may STILL fail the entailment judge;
    /// this only asserts that every `[n]` it wrote references a real retrieved source.
    pub fn clean(&self) -> bool {
        self.out_of_range.is_empty()
    }

    /// A short, human-readable reason for the GVS event log when the monitor rejects a candidate.
    pub fn reason(&self) -> String {
        let refs = self
            .out_of_range
            .iter()
            .map(|n| format!("[{n}]"))
            .collect::<Vec<_>>()
            .join(", ");
        format!("fabricated citation {refs} (only {} sources retrieved)", self.n_sources)
    }
}

/// extract_citations finds every inline `[n]` citation in `answer`, returning the raw 1-based
/// indices in document order (with repeats). It is deliberately tolerant of the shapes a small
/// model actually emits:
///
///   `[1]`, `[1][2]`, `[1, 2]`, `[1,2,3]`  → 1, 2, …
///
/// Non-numeric brackets are ignored, so Markdown links `[text](url)` and footnote markers like
/// `[note]` never count as citations. A bracket group that mixes text and numbers (`[ref 2]`) is
/// also ignored — citations in this codebase are always bare numbers (see `evidence_text`).
pub fn extract_citations(answer: &str) -> Vec<usize> {
    let mut out = Vec::new();
    let bytes = answer.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'[' {
            // Find the matching ']'.
            if let Some(rel) = answer[i + 1..].find(']') {
                let inner = &answer[i + 1..i + 1 + rel];
                parse_citation_group(inner, &mut out);
                i += 1 + rel + 1;
                continue;
            }
        }
        i += 1;
    }
    out
}

/// parse_citation_group reads the contents of one `[...]` group. It only accepts a comma-separated
/// list of bare non-negative integers (optionally space-padded). Anything else — letters, a URL,
/// an empty group — yields nothing, so prose brackets can never masquerade as citations.
fn parse_citation_group(inner: &str, out: &mut Vec<usize>) {
    let trimmed = inner.trim();
    if trimmed.is_empty() {
        return;
    }
    let mut parsed = Vec::new();
    for part in trimmed.split(',') {
        let p = part.trim();
        match p.parse::<usize>() {
            Ok(n) => parsed.push(n),
            Err(_) => return, // any non-numeric token disqualifies the whole group
        }
    }
    out.extend(parsed);
}

/// check_citations runs the deterministic monitor over a candidate answer given how many sources
/// were actually retrieved. Valid citations are `1..=n_sources`; `[0]` and anything larger is
/// fabricated.
///
/// Special case: if `n_sources == 0` there is nothing to cite, so any `[n]` is fabricated — but the
/// GVS loop never reaches Layer 1 with zero evidence (that path returns `Route::Unverifiable`
/// early), so in practice this is only hit by direct callers and tests.
pub fn check_citations(answer: &str, n_sources: usize) -> CitationReport {
    let mut used = Vec::new();
    let mut out_of_range = Vec::new();
    for n in extract_citations(answer) {
        if n >= 1 && n <= n_sources {
            if !used.contains(&n) {
                used.push(n);
            }
        } else if !out_of_range.contains(&n) {
            out_of_range.push(n);
        }
    }
    used.sort_unstable();
    out_of_range.sort_unstable();
    CitationReport { n_sources, used, out_of_range }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_simple_and_grouped_citations() {
        assert_eq!(extract_citations("the cap is 1.84 GB [1]."), vec![1]);
        assert_eq!(extract_citations("a [1] b [2] c"), vec![1, 2]);
        assert_eq!(extract_citations("stacked [1][2] refs"), vec![1, 2]);
        assert_eq!(extract_citations("grouped [1, 2, 3] refs"), vec![1, 2, 3]);
        assert_eq!(extract_citations("tight [1,2] refs"), vec![1, 2]);
    }

    #[test]
    fn ignores_non_numeric_brackets() {
        // Markdown links and footnote-style markers must never read as citations.
        assert_eq!(extract_citations("see [the docs](http://x) for more"), Vec::<usize>::new());
        assert_eq!(extract_citations("a footnote [note] here"), Vec::<usize>::new());
        assert_eq!(extract_citations("mixed [ref 2] token"), Vec::<usize>::new());
        assert_eq!(extract_citations("empty [] group"), Vec::<usize>::new());
        // An unterminated bracket is not a citation.
        assert_eq!(extract_citations("dangling [1 with no close"), Vec::<usize>::new());
    }

    #[test]
    fn flags_out_of_range_as_fabricated() {
        // 3 sources retrieved; [4] is invented, [0] is invalid.
        let r = check_citations("fact A [1], fact B [4], fact C [0]", 3);
        assert_eq!(r.used, vec![1]);
        assert_eq!(r.out_of_range, vec![0, 4]);
        assert!(!r.clean());
    }

    #[test]
    fn clean_when_all_citations_valid() {
        let r = check_citations("A [1] and B [2] and again [1]", 2);
        assert_eq!(r.used, vec![1, 2]); // de-duplicated
        assert!(r.out_of_range.is_empty());
        assert!(r.clean());
    }

    #[test]
    fn no_citations_is_clean() {
        // An answer that cites nothing is fine for the monitor (the judge still rules on content).
        let r = check_citations("just prose, no brackets", 3);
        assert!(r.clean());
        assert!(r.used.is_empty());
    }

    #[test]
    fn any_citation_is_fabricated_when_no_sources() {
        let r = check_citations("grounded-looking [1]", 0);
        assert_eq!(r.out_of_range, vec![1]);
        assert!(!r.clean());
    }

    #[test]
    fn reason_is_human_readable() {
        let r = check_citations("x [5]", 2);
        assert_eq!(r.reason(), "fabricated citation [5] (only 2 sources retrieved)");
    }
}
