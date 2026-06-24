package nano

// task_recall.go is RETRIEVAL reframed as dense-supervised induction — the fix discovered after the
// canonical induction diagnostic showed the engine does induction perfectly, but only under DENSE
// supervision (the assoc/retrieval tasks supervised a single answer token and starved the circuit).
//
//   first half : [s1 v1 s2 v2 ... sK vK]          (K facts: distinct subjects, random values)
//   second half: [sπ1 vπ1 sπ2 vπ2 ... sπK vπK]    (the SAME subjects, SHUFFLED)
//   supervise  : at every second-half subject, predict its value (transform(value)) — K dense targets
//
// To answer at a second-half subject the model must find that subject among the K first-half facts
// (distractors) and copy/transform its value. That is genuine associative retrieval, now with the
// supervision density that lets the recall circuit form. With Transform off it is pure recall.

// RecallTask: M possible subjects, K facts/queries per sequence.
type RecallTask struct {
	M, K, L   int
	Transform bool // if true, target = (value+3) mod 10; else target = value (pure recall)
}

// NewRecallTask builds a recall task. L = 4K (K facts + K shuffled queries, each 2 tokens).
func NewRecallTask(m, k int, transform bool) RecallTask {
	return RecallTask{M: m, K: k, L: 4 * k, Transform: transform}
}

func (t RecallTask) subj(s int) int { return 10 + s }
func (t RecallTask) ans(v int) int {
	if t.Transform {
		return (v + 3) % 10
	}
	return v
}

func (t RecallTask) sample(rng *rng) (seq, targets []int) {
	// K distinct subjects with random values
	subs := make([]int, 0, t.K)
	seen := map[int]bool{}
	for len(subs) < t.K {
		s := int(rng.next() % uint64(t.M))
		if !seen[s] {
			seen[s] = true
			subs = append(subs, s)
		}
	}
	val := make(map[int]int, t.K)
	seq = make([]int, 0, t.L)
	for _, s := range subs {
		v := int(rng.next() % 10)
		val[s] = v
		seq = append(seq, t.subj(s), v)
	}
	// shuffled second half
	order := make([]int, len(subs))
	copy(order, subs)
	for i := len(order) - 1; i > 0; i-- {
		j := int(rng.next() % uint64(i+1))
		order[i], order[j] = order[j], order[i]
	}
	targets = make([]int, t.L)
	for i := range targets {
		targets[i] = -1
	}
	for _, s := range order {
		pos := len(seq) // position of this second-half subject token
		seq = append(seq, t.subj(s), val[s])
		targets[pos] = t.ans(val[s]) // predict the value (or its transform) AFTER the subject
	}
	return seq, targets
}

// Batch builds B sequences with K dense recall targets each.
func (t RecallTask) Batch(B int, seed int64) (idx, targets []int) {
	rng := newRNG(seed)
	idx = make([]int, B*t.L)
	targets = make([]int, B*t.L)
	for b := 0; b < B; b++ {
		seq, tg := t.sample(rng)
		copy(idx[b*t.L:], seq)
		copy(targets[b*t.L:], tg)
	}
	return
}

// Accuracy measures recall accuracy over the K second-half query positions of n fresh sequences.
func (t RecallTask) Accuracy(g *GPT, seed, n int) float64 {
	rng := newRNG(int64(seed))
	correct, total := 0, 0
	for s := 0; s < n; s++ {
		seq, tg := t.sample(rng)
		for pos := 0; pos < t.L; pos++ {
			if tg[pos] < 0 {
				continue
			}
			total++
			if g.PredictAt(seq, 1, t.L, pos) == tg[pos] {
				correct++
			}
		}
	}
	return float64(correct) / float64(total)
}
