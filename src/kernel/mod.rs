//! kernel defines Metis's inference seam. The system layer — Conductor, Library, Hands — never
//! depends on a concrete inference backend; everything flows through the [`Kernel`] trait, so a stub
//! (mock) can be swapped for a real backend (ollama) without touching orchestration code.

mod mock;
mod ollama;

pub use mock::MockKernel;
pub use ollama::{Message, OllamaKernel, Tool, ToolFn};

use serde::{Deserialize, Serialize};

/// GenerateRequest configures a single generation call.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GenerateRequest {
    pub prompt: String,
    pub max_tokens: i32,
    pub temperature: f32,
    pub stop: Vec<String>,
}

/// Info is human-readable backend/model metadata.
#[derive(Clone, Debug, Default)]
pub struct Info {
    pub backend: String,
    pub model: String,
    pub ctx_len: i32,
}

/// Kernel is the one interface the rest of Metis talks to.
pub trait Kernel {
    /// Generate runs autoregressive decoding, invoking on_token for each emitted text chunk,
    /// and returns the full completion.
    fn generate(
        &self,
        req: GenerateRequest,
        on_token: &mut dyn FnMut(&str),
    ) -> Result<String, String>;

    /// Info reports the active backend and model.
    fn info(&self) -> Info;

    /// Close releases backend resources.
    fn close(&self) -> Result<(), String>;
}
