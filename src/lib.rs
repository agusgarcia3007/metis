//! Metis — frontier-grade intelligence that fits where frontier models can't.
//!
//! This is a Rust port of the Go `tiny-llm` project. The CLH-C architecture is split into:
//!   - `nano`    — a from-scratch, dependency-free transformer + autograd engine + trainer
//!   - `kernel`  — the inference seam (mock + ollama-backed Cortex)
//!   - `library` — the knowledge plane: retrieval over a disk-resident, swappable corpus
//!   - `hands`   — tools (calculator, clock) the small Cortex offloads exact work to
//!
//! The two binaries (`metis`, `rnt`) live in `src/bin/`.
//!
//! Several clippy lints are allowed crate-wide: the nano engine uses explicit index-based loops
//! that mirror the tensor math (clearer than iterator adaptors for this code), and a few float
//! constants intentionally match the Go originals.
#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::excessive_precision
)]

pub mod conductor;
pub mod hands;
pub mod kernel;
pub mod library;
pub mod nano;
