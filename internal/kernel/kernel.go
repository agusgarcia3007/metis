// Package kernel defines tiny-llm's inference seam.
//
// The whole point of the CLH-C architecture (see docs/design/01-architecture-the-grail.md)
// is that the system layer — Conductor, Library, Hands — never depends on a concrete
// inference backend. That lets us start on a stub, swap in a cgo->ggml backend (Phase 0),
// and later drop in a custom BitNet-1.58 ternary kernel (Phase 6) without touching any of
// the orchestration code. Everything flows through the Kernel interface below.
package kernel

import "context"

// GenerateRequest configures a single generation call.
type GenerateRequest struct {
	Prompt      string
	MaxTokens   int
	Temperature float32
	Stop        []string
}

// Info is human-readable backend/model metadata.
type Info struct {
	Backend string // e.g. "mock", "ggml", "bitnet"
	Model   string
	CtxLen  int
}

// Kernel is the one interface the rest of tiny-llm talks to. Implementations:
//   - MockKernel  (this package)            — build the system layer without ggml
//   - ggmlKernel  (cgo, Phase 0)            — native llama.cpp-class CPU inference
//   - bitnetKernel (cgo, Phase 6 moonshot)  — ternary, ~0.4 GB for a 2B Cortex
type Kernel interface {
	// Generate runs autoregressive decoding, invoking onToken for each emitted text
	// chunk, and returns the full completion. It must honor ctx cancellation so the
	// Conductor can abort long generations (e.g. budget exhausted).
	Generate(ctx context.Context, req GenerateRequest, onToken func(chunk string)) (string, error)

	// Info reports the active backend and model.
	Info() Info

	// Close releases backend resources (mmap'd weights, C allocations, etc.).
	Close() error
}
