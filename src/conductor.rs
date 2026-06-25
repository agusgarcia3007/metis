//! conductor — the Generate · Verify · Search loop (GVS): Metis's core quality mechanism.
//!
//! This is the mechanism by which a *tiny* Cortex reaches frontier-grade RELIABILITY on the
//! verifiable surface (grounded QA). The model is never trusted on a single shot. We:
//!
//!   1. GENERATE a grounded candidate answer (the Cortex may use tools).
//!   2. VERIFY it against the retrieved evidence — using the configured Verifier (LLM judge or
//!      dedicated NLI model). External framing ("does this CLAIM follow from this EVIDENCE?"),
//!      never "are you sure?". The research is unambiguous that external, grounded verification
//!      works where self-correction fails (docs/design/06).
//!   3. If unsupported/uncertain, SEARCH: draw a few *diverse* candidates and verify each, keeping
//!      the first one the evidence supports. If none survive, ABSTAIN — emit nothing unverified.
//!
//! Phase 2 note: `GvsConfig.verifier` controls which verifier runs. `VerifierKind::Llm` is the
//! default (same model as generator). `VerifierKind::Nli` delegates to a 22M NLI sidecar — the
//! Phase 2 cascade: smaller generator + specialized verifier.

use crate::kernel::{Message, OllamaKernel, Tool};
use crate::library::Hit;
use crate::verifier::VerifierKind;

// Re-export Verdict so callers using `conductor::Verdict` continue to compile.
pub use crate::verifier::Verdict;

/// Which path produced the final answer (for transparency / the HTTP API).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Route {
    /// No evidence retrieved — answered directly, cannot be grounded-verified (e.g. tool/chitchat).
    Unverifiable,
    /// First candidate was supported by the evidence.
    Verified,
    /// First candidate failed; a later search candidate was supported.
    Searched,
    /// No candidate was supported within the budget — abstained rather than emit an unverified claim.
    Abstained,
}

impl Route {
    pub fn as_str(&self) -> &'static str {
        match self {
            Route::Unverifiable => "unverifiable",
            Route::Verified => "verified",
            Route::Searched => "searched",
            Route::Abstained => "abstained",
        }
    }
}

/// The outcome of the GVS loop.
pub struct Answer {
    pub text: String,
    pub verdict: Option<Verdict>,
    pub route: Route,
    pub attempts: u32,
}

/// Knobs for the loop.
pub struct GvsConfig {
    /// Total candidate generations allowed (1 = verify-only, no search).
    pub max_candidates: u32,
    /// Temperature for the first candidate (low = the model's best single shot).
    pub gen_temp: f32,
    /// Temperature for search candidates (higher = decorrelated, to cover the answer space).
    pub search_temp: f32,
    /// What to say when nothing is verifiable.
    pub abstain_msg: String,
    /// Which verifier to use. Default: LLM judge (same model as generator).
    /// Phase 2: set to Nli { url } to use the dedicated NLI sidecar.
    pub verifier: VerifierKind,
}

impl Default for GvsConfig {
    fn default() -> Self {
        GvsConfig {
            max_candidates: 3,
            gen_temp: 0.4,
            search_temp: 0.7,
            abstain_msg: "I don't have support for a confident answer to that in my knowledge \
                          base, so I'd rather not guess."
                .to_string(),
            verifier: VerifierKind::default(),
        }
    }
}

/// answer runs the full Generate·Verify·Search loop over a prepared conversation.
///
/// `msgs` is the complete prompt (system + history + user). `hits` is the retrieved evidence; if
/// empty the answer cannot be grounded-verified and is returned as-is (Route::Unverifiable).
/// `on_event` receives progress strings (tool calls, verify verdicts, search steps) for display.
pub fn answer(
    k: &OllamaKernel,
    msgs: &[Message],
    hits: &[Hit],
    tools: &[Tool],
    cfg: &GvsConfig,
    mut on_event: Option<&mut (dyn FnMut(&str) + '_)>,
) -> Result<Answer, String> {
    // 1. GENERATE the first grounded candidate.
    let cand = k.chat_tools(msgs, cfg.gen_temp, tools, on_event.as_deref_mut())?;
    let mut attempts = 1;

    // No evidence → nothing to verify against. Return the answer untouched.
    if hits.is_empty() {
        return Ok(Answer { text: cand, verdict: None, route: Route::Unverifiable, attempts });
    }

    // 2. VERIFY the candidate against the evidence.
    let evidence = evidence_text(hits);
    let v = cfg.verifier.verify(k, &cand, &evidence);
    emit(&mut on_event, &format!("verify: {}", verdict_str(v)));
    if v == Verdict::Supported {
        return Ok(Answer { text: cand, verdict: Some(v), route: Route::Verified, attempts });
    }

    // 3. SEARCH: a few diverse candidates, accept the first the evidence supports.
    while attempts < cfg.max_candidates {
        emit(&mut on_event, &format!("search: candidate {}/{}", attempts + 1, cfg.max_candidates));
        let c = k.chat_tools(msgs, cfg.search_temp, tools, on_event.as_deref_mut())?;
        attempts += 1;
        let cv = cfg.verifier.verify(k, &c, &evidence);
        emit(&mut on_event, &format!("verify: {}", verdict_str(cv)));
        if cv == Verdict::Supported {
            return Ok(Answer { text: c, verdict: Some(cv), route: Route::Searched, attempts });
        }
    }

    // Nothing was supported within budget → ABSTAIN.
    emit(&mut on_event, "abstain: no candidate supported by the evidence");
    Ok(Answer {
        text: cfg.abstain_msg.clone(),
        verdict: Some(Verdict::Unsupported),
        route: Route::Abstained,
        attempts,
    })
}

/// evidence_text flattens the retrieved hits into a compact numbered block for the verifier.
pub fn evidence_text(hits: &[Hit]) -> String {
    let mut b = String::new();
    for (i, h) in hits.iter().enumerate() {
        b.push_str(&format!("[{}] {}\n", i + 1, h.chunk.text.trim()));
    }
    b
}

fn verdict_str(v: Verdict) -> &'static str {
    match v {
        Verdict::Supported => "supported",
        Verdict::Unsupported => "unsupported",
        Verdict::Uncertain => "uncertain",
    }
}

fn emit(on_event: &mut Option<&mut (dyn FnMut(&str) + '_)>, msg: &str) {
    if let Some(ev) = on_event.as_deref_mut() {
        ev(msg);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evidence_text_numbers_hits() {
        use crate::library::Chunk;
        let hits = vec![
            Hit {
                chunk: Chunk {
                    text: "  alpha  ".to_string(),
                    source: "a.md".to_string(),
                    idx: 0,
                    vec: vec![],
                },
                score: 0.9,
            },
            Hit {
                chunk: Chunk {
                    text: "beta".to_string(),
                    source: "b.md".to_string(),
                    idx: 1,
                    vec: vec![],
                },
                score: 0.8,
            },
        ];
        assert_eq!(evidence_text(&hits), "[1] alpha\n[2] beta\n");
    }
}
