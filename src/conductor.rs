//! conductor — the Generate · Verify · Search loop (GVS): Metis's "Opus loop".
//!
//! This is the mechanism by which a *tiny* Cortex reaches frontier-grade RELIABILITY on the
//! verifiable surface (grounded QA). The model is never trusted on a single shot. We:
//!
//!   1. GENERATE a grounded candidate answer (the Cortex may use tools).
//!   2. VERIFY it against the retrieved evidence — using the SAME model in *judge mode*, framed
//!      externally ("does this CLAIM follow from this EVIDENCE?"), never "are you sure?". The
//!      research is unambiguous that external, grounded verification works where self-correction
//!      fails (docs/design/06).
//!   3. If unsupported/uncertain, SEARCH: draw a few *diverse* candidates and verify each, keeping
//!      the first one the evidence supports. If none survive, ABSTAIN — emit nothing unverified.
//!
//! Why this turns "small" into "reliable": verifying an answer is far cheaper and far easier than
//! producing one (verification << generation). A 1.7B model is an unreliable generator but a
//! reliable grounded *judge*. So we spend the model's strength (recognition) to cover its weakness
//! (generation), at inference time — **no datasets, no training**. The verifier is the Cortex
//! itself; the search is just more sampling. Abstention makes the output trustworthy: a tiny local
//! model that emits only claims entailed by its evidence beats a frontier model that hallucinates.

use crate::kernel::{Message, OllamaKernel, Tool};
use crate::library::Hit;

/// The judge's decision about whether a candidate answer is entailed by the evidence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    Supported,
    Unsupported,
    Uncertain,
}

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

/// Knobs for the loop. Tuned small on purpose: a 4-vCPU CPU box can only afford a handful of
/// candidates, so search is "verify-then-maybe-a-few-more", never best-of-128.
pub struct GvsConfig {
    /// Total candidate generations allowed (1 = verify-only, no search).
    pub max_candidates: u32,
    /// Temperature for the first candidate (low = the model's best single shot).
    pub gen_temp: f32,
    /// Temperature for search candidates (higher = decorrelated, to cover the answer space).
    pub search_temp: f32,
    /// What to say when nothing is verifiable.
    pub abstain_msg: String,
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
        }
    }
}

/// answer runs the full Generate·Verify·Search loop over a prepared conversation.
///
/// `msgs` is the complete prompt (system + history + user). `hits` is the retrieved evidence; if it
/// is empty the answer cannot be grounded-verified and is returned as-is (`Route::Unverifiable`),
/// preserving today's behaviour for tool/chitchat queries. `on_event` receives progress strings
/// (tool calls, verify verdicts, search steps) for display.
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
        return Ok(Answer {
            text: cand,
            verdict: None,
            route: Route::Unverifiable,
            attempts,
        });
    }

    // 2. VERIFY the candidate against the evidence (external, grounded judge).
    let evidence = evidence_text(hits);
    let v = verify(k, &cand, &evidence);
    emit(&mut on_event, &format!("verify: {}", verdict_str(v)));
    if v == Verdict::Supported {
        return Ok(Answer {
            text: cand,
            verdict: Some(v),
            route: Route::Verified,
            attempts,
        });
    }

    // 3. SEARCH: a few diverse candidates, accept the first the evidence supports.
    while attempts < cfg.max_candidates {
        emit(
            &mut on_event,
            &format!("search: candidate {}/{}", attempts + 1, cfg.max_candidates),
        );
        let c = k.chat_tools(msgs, cfg.search_temp, tools, on_event.as_deref_mut())?;
        attempts += 1;
        let cv = verify(k, &c, &evidence);
        emit(&mut on_event, &format!("verify: {}", verdict_str(cv)));
        if cv == Verdict::Supported {
            return Ok(Answer {
                text: c,
                verdict: Some(cv),
                route: Route::Searched,
                attempts,
            });
        }
    }

    // Nothing was supported within budget → ABSTAIN (zero hallucination beats a confident wrong answer).
    emit(&mut on_event, "abstain: no candidate supported by the evidence");
    Ok(Answer {
        text: cfg.abstain_msg.clone(),
        verdict: Some(Verdict::Unsupported),
        route: Route::Abstained,
        attempts,
    })
}

/// verify asks the Cortex, in judge mode, whether `claim` is entailed by `evidence`.
/// External framing + grounded evidence is the regime the research shows works; we deliberately do
/// NOT ask the model to second-guess its own reasoning.
pub fn verify(k: &OllamaKernel, claim: &str, evidence: &str) -> Verdict {
    let sys = "You are a strict fact-checker. You are given EVIDENCE and a CLAIM. Decide whether \
               the CLAIM is fully supported by the EVIDENCE. Use ONLY the evidence — never outside \
               knowledge. If every factual statement in the claim is directly backed by the \
               evidence, answer SUPPORTED. If any part contradicts the evidence or is not found in \
               it, answer UNSUPPORTED. Reply with exactly one word: SUPPORTED or UNSUPPORTED.";
    let user = format!("EVIDENCE:\n{evidence}\n\nCLAIM:\n{claim}\n\nVerdict (one word):");
    let msgs = [
        Message {
            role: "system".to_string(),
            content: sys.to_string(),
        },
        Message {
            role: "user".to_string(),
            content: user,
        },
    ];
    match k.chat(&msgs, 0.0, &mut |_s: &str| {}) {
        Ok(r) => parse_verdict(&r),
        Err(_) => Verdict::Uncertain,
    }
}

/// evidence_text flattens the retrieved hits into a compact numbered block for the judge.
pub fn evidence_text(hits: &[Hit]) -> String {
    let mut b = String::new();
    for (i, h) in hits.iter().enumerate() {
        b.push_str(&format!("[{}] {}\n", i + 1, h.chunk.text.trim()));
    }
    b
}

/// parse_verdict maps a free-text judge reply onto a Verdict. Order matters: "UNSUPPORTED" contains
/// the substring "SUPPORTED", so we test for the negative first.
fn parse_verdict(s: &str) -> Verdict {
    let u = s.to_uppercase();
    if u.contains("UNSUPPORTED") || u.contains("NOT SUPPORTED") || u.contains("NOT FULLY") {
        Verdict::Unsupported
    } else if u.contains("SUPPORTED") {
        Verdict::Supported
    } else {
        Verdict::Uncertain
    }
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
    fn verdict_parsing_handles_substring_trap() {
        // "UNSUPPORTED" contains "SUPPORTED": the negative must win.
        assert_eq!(parse_verdict("UNSUPPORTED"), Verdict::Unsupported);
        assert_eq!(parse_verdict("The claim is unsupported."), Verdict::Unsupported);
        assert_eq!(parse_verdict("not supported by the evidence"), Verdict::Unsupported);
        assert_eq!(parse_verdict("SUPPORTED"), Verdict::Supported);
        assert_eq!(parse_verdict("Yes, this is supported."), Verdict::Supported);
        assert_eq!(parse_verdict("maybe?"), Verdict::Uncertain);
    }

    #[test]
    fn evidence_text_numbers_hits() {
        use crate::library::Chunk;
        let hits = vec![
            Hit {
                chunk: Chunk { text: "  alpha  ".to_string(), source: "a.md".to_string(), idx: 0, vec: vec![] },
                score: 0.9,
            },
            Hit {
                chunk: Chunk { text: "beta".to_string(), source: "b.md".to_string(), idx: 1, vec: vec![] },
                score: 0.8,
            },
        ];
        assert_eq!(evidence_text(&hits), "[1] alpha\n[2] beta\n");
    }
}
