package kernel

import (
	"context"
	"fmt"
	"strings"
)

// MockKernel is a stand-in used to build and test the Conductor/Library/Hands before
// the real cgo->ggml backend lands. It performs NO inference — it echoes deterministic
// text so the orchestration layer (routing, retrieval wiring, tool dispatch, streaming)
// can be developed and unit-tested independently. See docs/design/02-build-plan.md Phase 0.
type MockKernel struct{ model string }

// NewMock returns a MockKernel labeled with the given model name.
func NewMock(model string) *MockKernel { return &MockKernel{model: model} }

// Generate streams a deterministic placeholder response token-by-token.
func (m *MockKernel) Generate(ctx context.Context, req GenerateRequest, onToken func(string)) (string, error) {
	reply := fmt.Sprintf("[mock:%s] received %d chars; real reasoning arrives with the ggml kernel (Phase 0).",
		m.model, len(req.Prompt))
	for _, w := range strings.Fields(reply) {
		select {
		case <-ctx.Done():
			return "", ctx.Err()
		default:
			onToken(w + " ")
		}
	}
	return reply, nil
}

// Info reports the mock backend metadata.
func (m *MockKernel) Info() Info { return Info{Backend: "mock", Model: m.model, CtxLen: 4096} }

// Close is a no-op for the mock backend.
func (m *MockKernel) Close() error { return nil }

// compile-time check that MockKernel satisfies Kernel.
var _ Kernel = (*MockKernel)(nil)
