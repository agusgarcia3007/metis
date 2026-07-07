//! hands implements Metis's tools — the capabilities the small Cortex offloads to exact,
//! deterministic code (the "Hands" of the CLH-C architecture): a calculator, a clock, web
//! retrieval, and the isolated execution surface used by the code verifier.

mod calc;
mod clock;
pub mod sandbox;
pub mod verify_exec;
pub mod web;

pub use calc::calc;
pub use clock::now;
pub use web::{search as web_search, WebResult};
