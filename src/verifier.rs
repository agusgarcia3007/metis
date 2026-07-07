//! verifier — the Verifier abstraction and its implementations.
//!
//! A Verifier checks whether a candidate answer (claim) is entailed by a body of evidence.
//! Three implementations ship:
//!
//!   LlmVerifier  — the Cortex itself acts as judge (same model as the generator). Zero extra RAM.
//!                  Works because verification < generation: a weak model can recognize a correct
//!                  step even when it can't produce one reliably. Default (no sidecar needed).
//!
//!   NliVerifier  — a dedicated NLI sidecar (cross-encoder/nli-MiniLM-L-6-v2, 22M params, ~85MB).
//!                  Specialized for the exact task: given (evidence, claim), output entailment score.
//!                  A 22M model trained on NLI datasets beats a 1.7B generalist at this one job.
//!                  Enable with METIS_NLI_URL=http://<sidecar>:9090.
//!
//!   ExecVerifier — a sandboxed deterministic oracle for code candidates. It checks syntax, types,
//!                  lint, and tests. Unlike text verification, infrastructure failures are errors:
//!                  callers must abstain instead of interpreting a broken tool as a verdict.
//!
//! Phase 2 hypothesis: 0.6B generator + 22M NLI verifier ≈ 1.7B generalist doing both, at ~44% RAM.

use crate::hands::verify_exec::{ExecError, ExecReport, ExecRequest, ExecVerifier};
use crate::kernel::{Message, OllamaKernel};

/// The judge's decision: is the candidate answer entailed by the retrieved evidence?
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    Supported,
    Unsupported,
    Uncertain,
}

/// Which verifier to use. Controlled by METIS_NLI_URL at runtime.
pub enum VerifierKind {
    /// Use the Cortex (LLM) as a grounded fact-checker. No extra model needed.
    Llm,
    /// Call the NLI sidecar at the given URL. Faster, smaller, specialized.
    Nli { url: String },
    /// Run code candidates through an isolated compiler/type/lint/test oracle.
    Exec(Box<ExecVerifier>),
}

impl VerifierKind {
    pub fn verify(&self, k: &OllamaKernel, claim: &str, evidence: &str) -> Verdict {
        match self {
            Self::Llm => llm_verify(k, claim, evidence),
            Self::Nli { url } => nli_verify(url, claim, evidence),
            // Text claims cannot be passed to an execution verifier. Returning Uncertain preserves
            // the GVS fail-closed behavior if a caller wires the wrong verifier to grounded QA.
            Self::Exec(_) => Verdict::Uncertain,
        }
    }

    pub fn verify_code(&self, request: &ExecRequest) -> Result<ExecReport, ExecError> {
        match self {
            Self::Exec(verifier) => verifier.verify(request),
            Self::Llm | Self::Nli { .. } => Err(ExecError::WrongVerifierKind),
        }
    }
}

impl Default for VerifierKind {
    fn default() -> Self {
        Self::Llm
    }
}

/// llm_verify asks the Cortex, in judge mode, whether `claim` is entailed by `evidence`.
/// External framing ("does this CLAIM follow from this EVIDENCE?") is the key — never
/// "are you sure?". The research shows this external form works where self-correction fails.
fn llm_verify(k: &OllamaKernel, claim: &str, evidence: &str) -> Verdict {
    let sys = "You are a strict fact-checker. You are given EVIDENCE and a CLAIM. Decide whether \
               the CLAIM is fully supported by the EVIDENCE. Use ONLY the evidence — never outside \
               knowledge. If every factual statement in the claim is directly backed by the \
               evidence, answer SUPPORTED. If any part contradicts the evidence or is not found in \
               it, answer UNSUPPORTED. Reply with exactly one word: SUPPORTED or UNSUPPORTED.";
    let user = format!("EVIDENCE:\n{evidence}\n\nCLAIM:\n{claim}\n\nVerdict (one word):");
    let msgs = [
        Message { role: "system".to_string(), content: sys.to_string() },
        Message { role: "user".to_string(), content: user },
    ];
    match k.chat(&msgs, 0.0, &mut |_| {}) {
        Ok(r) => parse_llm_verdict(&r),
        Err(_) => Verdict::Uncertain,
    }
}

/// nli_verify calls the dedicated NLI sidecar (cross-encoder, 22M params).
/// The sidecar outputs {verdict: "SUPPORTED"|"UNSUPPORTED"|"UNCERTAIN", scores: {...}}.
fn nli_verify(url: &str, claim: &str, evidence: &str) -> Verdict {
    let body = serde_json::json!({ "claim": claim, "evidence": evidence });
    match ureq::post(&format!("{url}/verify"))
        .set("Content-Type", "application/json")
        .send_json(body)
    {
        Ok(resp) => {
            let parsed: serde_json::Value = resp.into_json().unwrap_or(serde_json::Value::Null);
            match parsed["verdict"].as_str() {
                Some("SUPPORTED") => Verdict::Supported,
                Some("UNSUPPORTED") => Verdict::Unsupported,
                _ => Verdict::Uncertain,
            }
        }
        Err(_) => Verdict::Uncertain,
    }
}

/// parse_llm_verdict maps a free-text LLM reply to a Verdict.
/// Order matters: "UNSUPPORTED" contains "SUPPORTED", so we test the negative first.
fn parse_llm_verdict(s: &str) -> Verdict {
    // Reasoning models (Qwen3 et al.) may prepend a <think>…</think> block that *echoes the prompt*,
    // including the literal words "SUPPORTED or UNSUPPORTED". Substring-scanning that block reads the
    // echoed "UNSUPPORTED" as the verdict and flips every answer to Unsupported (measured: it made a
    // correctly-reasoning 4B look like it rejected everything). The real verdict is what follows the
    // closing tag, so strip the think block before reading.
    let tail = match s.rfind("</think>") {
        Some(i) => &s[i + "</think>".len()..],
        None => s,
    };
    let u = tail.to_uppercase();
    if u.contains("UNSUPPORTED") || u.contains("NOT SUPPORTED") || u.contains("NOT FULLY") {
        Verdict::Unsupported
    } else if u.contains("SUPPORTED") {
        Verdict::Supported
    } else {
        Verdict::Uncertain
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verdict_parsing_handles_substring_trap() {
        assert_eq!(parse_llm_verdict("UNSUPPORTED"), Verdict::Unsupported);
        assert_eq!(parse_llm_verdict("The claim is unsupported."), Verdict::Unsupported);
        assert_eq!(parse_llm_verdict("not supported by the evidence"), Verdict::Unsupported);
        assert_eq!(parse_llm_verdict("SUPPORTED"), Verdict::Supported);
        assert_eq!(parse_llm_verdict("Yes, this is supported."), Verdict::Supported);
        assert_eq!(parse_llm_verdict("maybe?"), Verdict::Uncertain);
    }

    #[test]
    fn verdict_ignores_thinking_block_echo() {
        // A reasoning model echoes "SUPPORTED or UNSUPPORTED" inside its think block, then answers.
        // The verdict must come from AFTER </think>, not from the echoed instruction.
        let reply = "Reply with one word: SUPPORTED or UNSUPPORTED.\nThe evidence states it directly.\n</think>\n\nSUPPORTED";
        assert_eq!(parse_llm_verdict(reply), Verdict::Supported);
        let reply_neg = "The claim says 192 but evidence says 512.\n</think>\n\nUNSUPPORTED";
        assert_eq!(parse_llm_verdict(reply_neg), Verdict::Unsupported);
    }
}
