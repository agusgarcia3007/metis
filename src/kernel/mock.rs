//! A stand-in kernel used to build and test the Conductor/Library/Hands without a real backend.
//! It performs NO inference — it echoes deterministic text so the orchestration layer can be
//! developed and unit-tested independently.

use super::{GenerateRequest, Info, Kernel};

/// MockKernel echoes a deterministic placeholder response.
pub struct MockKernel {
    model: String,
}

impl MockKernel {
    /// NewMock returns a MockKernel labeled with the given model name.
    pub fn new(model: &str) -> MockKernel {
        MockKernel {
            model: model.to_string(),
        }
    }
}

impl Kernel for MockKernel {
    fn generate(
        &self,
        req: GenerateRequest,
        on_token: &mut dyn FnMut(&str),
    ) -> Result<String, String> {
        let reply = format!(
            "[mock:{}] received {} chars; real reasoning arrives with the ggml kernel (Phase 0).",
            self.model,
            req.prompt.len()
        );
        for w in reply.split_whitespace() {
            on_token(&format!("{w} "));
        }
        Ok(reply)
    }

    fn info(&self) -> Info {
        Info {
            backend: "mock".to_string(),
            model: self.model.clone(),
            ctx_len: 4096,
        }
    }

    fn close(&self) -> Result<(), String> {
        Ok(())
    }
}
