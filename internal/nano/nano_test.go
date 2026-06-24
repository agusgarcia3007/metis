package nano

import "testing"

// TestGradCheck verifies the full transformer's analytic gradients against central finite
// differences. Passing means the autograd engine (Linear, LayerNorm, GELU, Attention, tied
// logits, cross-entropy, embedding) is correct and training is trustworthy.
func TestGradCheck(t *testing.T) {
	cfg := Config{Vocab: 11, Block: 6, Layer: 2, Head: 2, Embd: 16}
	g := NewGPT(cfg, 42)
	B, T := 2, cfg.Block
	N := B * T
	rng := newRNG(7)
	idx := make([]int, N)
	tgt := make([]int, N)
	for i := range idx {
		idx[i] = int(rng.next() % uint64(cfg.Vocab))
		tgt[i] = int(rng.next() % uint64(cfg.Vocab))
	}
	maxRel, measured := g.GradCheck(idx, tgt, B, T, 60, 123)
	t.Logf("max relative grad error = %.2e over %d measurable entries", maxRel, measured)
	if measured < 25 {
		t.Fatalf("too few measurable entries (%d) — check is not meaningful", measured)
	}
	if maxRel > 3e-2 {
		t.Fatalf("gradient check FAILED: max rel error %.2e (>3e-2)", maxRel)
	}
}
