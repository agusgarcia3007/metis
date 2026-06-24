package nano

import "testing"

// TestRNTGeneralizes locks in the proven RNT mechanism: trained the RNT way (fact retrieved into
// context), a tiny model answers correctly about a world it never trained on; trained the vanilla
// way (memorize), the same model fails on a new world. This is the core, reproducible result.
func TestRNTGeneralizes(t *testing.T) {
	if testing.Short() {
		t.Skip("trains a model; skipped in -short")
	}
	task := NewTask(20)
	cfg := Config{Vocab: VocabSize, Block: task.T, Layer: 2, Head: 2, Embd: 48}
	worldTrain := task.RandomWorld(1)
	worldNew := task.RandomWorld(999)

	// Vanilla: memorize worldTrain.
	gV := NewGPT(cfg, 7)
	oV := NewAdamW(gV.Params(), 2e-3, 0)
	for s := 1; s <= 1500; s++ {
		idx, tgt := task.VanillaBatch(worldTrain, 32, int64(s))
		gV.LossAndGrad(oV, idx, tgt, 32, task.T)
		oV.Step(nil)
	}
	vSeen := task.VanillaAccuracy(gV, worldTrain)
	vNew := task.VanillaAccuracy(gV, worldNew)

	// RNT: fact retrieved into context.
	gR := NewGPT(cfg, 7)
	oR := NewAdamW(gR.Params(), 2e-3, 0)
	for s := 1; s <= 1500; s++ {
		idx, tgt := task.RNTBatch(32, int64(s))
		gR.LossAndGrad(oR, idx, tgt, 32, task.T)
		oR.Step(nil)
	}
	rNew := task.RNTAccuracy(gR, worldNew, 5, 500)

	t.Logf("vanilla seen=%.2f new=%.2f | RNT new=%.2f", vSeen, vNew, rNew)
	if vSeen < 0.85 {
		t.Fatalf("vanilla should memorize its trained world (got %.2f)", vSeen)
	}
	if vNew > 0.30 {
		t.Fatalf("vanilla should FAIL on a new world ~chance (got %.2f)", vNew)
	}
	if rNew < 0.90 {
		t.Fatalf("RNT should GENERALIZE to a new world (got %.2f)", rNew)
	}
}
