package nano

// task_induction.go is the canonical induction-head training setup: a random block repeated twice
// (r ++ r) with DENSE next-token supervision at every position. In the second copy, the next token
// is predictable by induction (copy the token that followed the same token in the first copy).
//
// The key lesson this encodes: induction needs a *previous-token head*, which only forms under DENSE
// next-token supervision — not the single-answer-token supervision used by the assoc/retrieval tasks.
// 2-layer transformers solve this reliably; if our engine does too, the recall circuit works and the
// retrieval plateau was a supervision-density problem.

// InductionTask: vocabulary V, block length m (sequence length L = 2m).
type InductionTask struct{ V, M, L int }

// NewInductionTask builds a repeated-block induction task with block length m.
func NewInductionTask(v, m int) InductionTask { return InductionTask{V: v, M: m, L: 2 * m} }

func (t InductionTask) sample(rng *rng) []int {
	seq := make([]int, t.L)
	for i := 0; i < t.M; i++ {
		tok := int(rng.next() % uint64(t.V))
		seq[i] = tok
		seq[i+t.M] = tok
	}
	return seq
}

// Batch builds B sequences with DENSE next-token targets (target[i] = seq[i+1]).
func (t InductionTask) Batch(B int, seed int64) (idx, targets []int) {
	rng := newRNG(seed)
	idx = make([]int, B*t.L)
	targets = make([]int, B*t.L)
	for b := 0; b < B; b++ {
		seq := t.sample(rng)
		copy(idx[b*t.L:], seq)
		for i := 0; i < t.L; i++ {
			if i < t.L-1 {
				targets[b*t.L+i] = seq[i+1]
			} else {
				targets[b*t.L+i] = -1
			}
		}
	}
	return
}

// Accuracy measures next-token accuracy over the SECOND copy (induction-predictable positions),
// where seq[i] first occurred at i-M and was followed by seq[i-M+1] == seq[i+1].
func (t InductionTask) Accuracy(g *GPT, seed, n int) float64 {
	rng := newRNG(int64(seed))
	correct, total := 0, 0
	for s := 0; s < n; s++ {
		seq := t.sample(rng)
		for i := t.M; i < t.L-1; i++ { // second copy, excluding the very last token
			total++
			if g.PredictAt(seq, 1, t.L, i) == seq[i+1] {
				correct++
			}
		}
	}
	return float64(correct) / float64(total)
}
