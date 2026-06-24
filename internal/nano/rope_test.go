package nano

import "testing"

func TestGradCheckNoRoPE(t *testing.T) {
	cfg := Config{Vocab: 11, Block: 6, Layer: 2, Head: 2, Embd: 16, NoRoPE: true}
	g := NewGPT(cfg, 42)
	B, T := 2, cfg.Block
	N := B * T
	r := newRNG(7)
	idx := make([]int, N)
	tg := make([]int, N)
	for i := range idx {
		idx[i] = int(r.next() % uint64(cfg.Vocab))
		tg[i] = int(r.next() % uint64(cfg.Vocab))
	}
	mr, m := g.GradCheck(idx, tg, B, T, 60, 123)
	t.Logf("NoRoPE maxRel=%.2e over %d", mr, m)
	if mr > 3e-2 || m < 25 {
		t.Fatalf("noRoPE gradcheck fail %.2e n=%d", mr, m)
	}
}
