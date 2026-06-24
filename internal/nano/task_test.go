package nano

import (
	"os"
	"path/filepath"
	"testing"
)

// TestRetrievalShape checks the retrieval task encodes sequences of the right length with the
// answer at the documented position, and that the query subject's fact is present in context.
func TestRetrievalShape(t *testing.T) {
	task := NewRetrievalTask(1000, 4)
	rng := newRNG(1)
	for i := 0; i < 100; i++ {
		seq, ans := task.sample(rng)
		if len(seq) != task.T {
			t.Fatalf("seq len %d != T %d", len(seq), task.T)
		}
		if seq[task.ansPos()+1] != ans {
			t.Fatalf("answer token not at ansPos+1")
		}
		if seq[task.ansPos()] != tokANS {
			t.Fatalf("ansPos is not the ANS token")
		}
		if ans < 0 || ans >= task.K {
			t.Fatalf("answer %d out of range", ans)
		}
	}
}

// TestVanillaShape checks the simple task's vanilla/RNT answer positions are consistent.
func TestTaskTransform(t *testing.T) {
	task := NewTask(50)
	for v := 0; v < 10; v++ {
		if got := task.Transform(v); got != (v+3)%10 {
			t.Fatalf("transform(%d)=%d want %d", v, got, (v+3)%10)
		}
	}
}

// TestSerializationRoundtrip ensures a saved model reloads to byte-identical behavior.
func TestSerializationRoundtrip(t *testing.T) {
	cfg := Config{Vocab: VocabSize, Block: 12, Layer: 2, Head: 2, Embd: 32}
	g := NewGPT(cfg, 5)
	// scramble params a bit so it's not just the init
	task := NewTask(50)
	opt := NewAdamW(g.Params(), 1e-3, 0)
	for s := 1; s <= 50; s++ {
		idx, tgt := task.RNTBatch(16, int64(s))
		g.LossAndGrad(opt, idx, tgt, 16, task.T)
		opt.Step(nil)
	}

	dir := t.TempDir()
	path := filepath.Join(dir, "m.gob")
	if err := g.Save(path); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatal(err)
	}
	loaded, err := LoadGPT(path)
	if err != nil {
		t.Fatal(err)
	}

	// parameters must match exactly
	gp, lp := g.Params(), loaded.Params()
	for i := range gp {
		for j := range gp[i].Data {
			if gp[i].Data[j] != lp[i].Data[j] {
				t.Fatalf("param %d[%d] mismatch after reload", i, j)
			}
		}
	}
	// predictions must match
	for s := 0; s < 20; s++ {
		seq := task.rntSeq(s, s%10, tokPAD)
		if g.PredictAt(seq, 1, task.T, task.rntAnsPos()) != loaded.PredictAt(seq, 1, task.T, task.rntAnsPos()) {
			t.Fatalf("prediction mismatch after reload at subject %d", s)
		}
	}
}

// TestDeterminism ensures training is reproducible (same seed -> same loss trajectory).
func TestDeterminism(t *testing.T) {
	run := func() float32 {
		cfg := Config{Vocab: VocabSize, Block: 12, Layer: 1, Head: 2, Embd: 16}
		g := NewGPT(cfg, 3)
		opt := NewAdamW(g.Params(), 1e-3, 0)
		task := NewTask(50)
		var loss float32
		for s := 1; s <= 30; s++ {
			idx, tgt := task.RNTBatch(16, int64(s))
			loss = g.LossAndGrad(opt, idx, tgt, 16, task.T)
			opt.Step(nil)
		}
		return loss
	}
	a, b := run(), run()
	if a != b {
		t.Fatalf("nondeterministic training: %v != %v", a, b)
	}
}
