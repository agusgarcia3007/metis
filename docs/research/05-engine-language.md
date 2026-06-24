# Research: Inference Engine & Language Choice (Go vs Rust vs C++)

## TL;DR for tiny-llm

- **Pure-Go matmul is 5–16× slower than C/C++.** Go's SIMD (`simd/archsimd`, Go 1.26) is
  experimental, AMD64-only, API-unstable, and has a known inlining bug capping throughput at ~6 GB/s.
  → **Do NOT write the inference kernel in pure Go.**
- **Ollama is NOT a Go inference engine** — it's a Go control plane wrapping llama.cpp via cgo. The
  tensor math is all C++. This is the proven production pattern.
- **The resolution for our "build it in Go" goal:** Go owns the **system layer** (router, retrieval,
  tool-calling, agent loop, KV/session mgmt, HTTP/streaming API) — exactly where the novelty of
  tiny-llm lives. The hot matmul kernel is **ggml via cgo** (or a small hand-tuned BitNet kernel).
- **GC danger on 4GB:** default GOGC=100 doubles heap before collecting. Must set
  `debug.SetMemoryLimit()` + GOGC≈25–50. Keep big tensors in C/mmap memory (off the Go heap) so GC
  never scans them — another reason to delegate weights to ggml.
- **Rust + candle** is the alternative for a from-scratch engine (no GC, true SIMD via `std::arch`,
  static musl binaries) but candle trails llama.cpp ~27–40% on CPU Q4 and costs 3–6 months.

## llama.cpp / ggml — why it's the standard
- C/C++, zero-dep, mem-bandwidth-bound design. Hand-tuned SIMD: AVX/AVX2/AVX-512/AMX (x86),
  NEON/i8MM/SVE/SME (ARM). Custom threadpool, hybrid-core aware, mmap loading. GGUF k-quants.
- CPU bench (Q4_K_M, Llama 3.1 8B): Ryzen 9 7950X ~18 tok/s. 3–8× faster than Python on CPU.
- Now under Hugging Face (ggml.ai joined Feb 2026). 109K+ stars.

## Go specifics
- Pure-Go: 5–16× slower matmul (gonum ~5× slower than numpy/BLAS). Hand-written Plan 9 asm can hit
  ~9.3× over naive Go but = writing C with worse ergonomics, one ISA only.
- Go 1.26: Green Tea GC (−10–40% GC overhead), −30% cgo call overhead. Helps the cgo pattern.
- cgo breaks pure-static unless built with musl/zig cc; cross-compile needs a real toolchain.

## Rust
- candle (HF, 20.5k★): LLaMA/Mistral/Gemma/Phi, multi-backend, delegates to MKL/Accelerate (must
  dequantize for Q4 → loses to llama.cpp's native Q4 kernels). mistral.rs builds on candle.
- Bench M1 Mistral-7B Q4: llama.cpp 11 tok/s vs candle 7–8 tok/s. Phi-2 Q4: 25 vs 8.6 tok/s.
- True zero-cost SIMD, no GC, 2–5MB static musl binary. Steeper, slower to build.

## Decision for tiny-llm
**Go control plane + cgo→ggml kernel.** Rationale:
1. tiny-llm's value is the *system* (routing/retrieval/tools/agent) — Go excels there.
2. Inference perf = native llama.cpp (delegated), so we pay no Go penalty on the hot path.
3. Single-artifact deploy via musl/zig static build; weights live in mmap'd C memory, off Go heap.
4. Keep a clean kernel interface so we can later swap in a custom BitNet-1.58 ternary kernel
   (bitnet.cpp-style) for the resident reasoner without touching the system layer.
- Fallback/option: if we want zero-C, Rust+candle; accept ~30% slower + longer build.

### Sources
- llama.cpp https://github.com/ggml-org/llama.cpp · Ollama arch https://eli.thegreenplace.net/2024/the-life-of-an-ollama-prompt/
- Go 1.26 https://go.dev/doc/go1.26 · SIMD bug https://github.com/golang/go/issues/77647
- Sourcegraph Go SIMD https://sourcegraph.com/blog/slow-to-simd · candle https://github.com/huggingface/candle
- mistral.rs https://github.com/EricLBuehler/mistral.rs · bench https://medium.com/@zaiinn440/apple-mlx-vs-llama-cpp-vs-hugging-face-candle-rust-for-lightning-fast-llms-locally-5447f6e9255a
- Go GC guide https://go.dev/doc/gc-guide
