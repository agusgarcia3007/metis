package nano

import "testing"

// TestInductionLearns locks in the root-cause finding: under DENSE next-token supervision the engine
// forms an induction circuit and solves the canonical repeated-block task ~perfectly. (The same model
// under SPARSE single-answer supervision stays near chance — that was the retrieval bottleneck.)
func TestInductionLearns(t *testing.T) {
	if testing.Short() {
		t.Skip("trains a model; skipped in -short")
	}
	task := NewInductionTask(40, 8) // V=40, block M=8, L=16; chance = 1/40 = 2.5%
	cfg := Config{Vocab: 40, Block: task.L, Layer: 2, Head: 4, Embd: 64}
	g := NewGPT(cfg, 7)
	opt := NewAdamW(g.Params(), 0, 0)
	peak := float32(3e-3)
	for s := 1; s <= 1500; s++ {
		if s < 200 {
			opt.LR = peak * float32(s) / 200
		} else {
			opt.LR = peak
		}
		idx, tgt := task.Batch(32, int64(s))
		g.LossAndGrad(opt, idx, tgt, 32, task.L)
		opt.Step(nil)
	}
	acc := task.Accuracy(g, 99, 500)
	t.Logf("dense induction accuracy = %.3f (chance 0.025)", acc)
	if acc < 0.9 {
		t.Fatalf("dense induction should be ~solved (got %.3f)", acc)
	}
}
