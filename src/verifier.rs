//! verifier — the Verifier abstraction and its two implementations.
//!
//! A Verifier checks whether a candidate answer (claim) is entailed by a body of evidence.
//! Two implementations ship:
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
//! Phase 2 hypothesis: 0.6B generator + 22M NLI verifier ≈ 1.7B generalist doing both, at ~44% RAM.

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
}

impl VerifierKind {
    pub fn verify(&self, k: &OllamaKernel, claim: &str, evidence: &str) -> Verdict {
        match self {
            Self::Llm => llm_verify(k, claim, evidence),
            Self::Nli { url } => nli_verify(url, claim, evidence),
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
    let u = s.to_uppercase();
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
}
