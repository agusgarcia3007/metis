//! nano is a from-scratch, dependency-free transformer language model in pure Rust:
//! a real autograd engine + GPT + trainer. It is "the smallest model we can actually build
//! and verify in-session" — trained from zero on CPU, no GPU, no downloads at runtime.
//!
//! This is a faithful port of the Go `internal/nano` package. The pointer-graph autograd is
//! modeled with `Rc<RefCell<Tensor>>` handles and boxed backward closures recorded on a Tape.

mod model;
mod ops;
mod serialize;
mod task;
mod task_assoc;
mod task_induction;
mod task_recall;
mod task_retrieval;
mod tensor;
mod train;

#[cfg(test)]
mod tests;

pub use model::{Config, Gpt};
pub use serialize::load_gpt;
pub use task::{Task, VOCAB_SIZE};
pub use task_assoc::AssocTask;
pub use task_induction::InductionTask;
pub use task_recall::RecallTask;
pub use task_retrieval::RetrievalTask;
pub use train::AdamW;
