package nano

// task_retrieval.go is the HARD version of the RNT benchmark: the context contains MANY facts
// (distractors), and the model must find the queried subject among them, read ITS value, and apply
// the transform. This kills the "it just copies the only number it sees" criticism of the simple
// task — here the model must perform genuine associative retrieval + reasoning over the context.
//
// Context: [s1 = v1 ;] [s2 = v2 ;] ... [sK = vK ;]   (K facts, random subjects/values, shuffled)
// Query:   [? sq > ans]                              (sq is one of the K subjects)
// Answer:  transform(value of sq) = (v_sq + 3) mod 10
//
// Because every sample has fresh random facts, the model can NEVER memorize — high accuracy is, by
// construction, generalization to unseen fact-sets. Chance is 10% (one of ten possible answers).

// RetrievalTask holds the encoding for a K-fact retrieval problem over subjects in [0,M).
type RetrievalTask struct {
	M, K, D, NFacts, T int
}

// NewRetrievalTask builds a task with nFacts facts in context drawn from m subjects.
func NewRetrievalTask(m, nFacts int) RetrievalTask {
	d := 1
	for p := 10; p < m; p *= 10 {
		d++
	}
	factLen := d + 3  // digits + EQ + value + SEP
	queryLen := d + 3 // Q + digits + ANS + answer
	return RetrievalTask{M: m, K: 10, D: d, NFacts: nFacts, T: nFacts*factLen + queryLen}
}

func (t RetrievalTask) transform(v int) int { return (v + 3) % t.K }
func (t RetrievalTask) ansPos() int         { return t.T - 2 } // the ANS token; predicts the final token

// sample builds one problem instance. Returns the token sequence and the correct answer.
func (t RetrievalTask) sample(rng *rng) (seq []int, ans int) {
	seq = make([]int, 0, t.T)
	// pick NFacts distinct subjects
	subs := make([]int, 0, t.NFacts)
	seen := map[int]bool{}
	for len(subs) < t.NFacts {
		s := int(rng.next() % uint64(t.M))
		if !seen[s] {
			seen[s] = true
			subs = append(subs, s)
		}
	}
	vals := make(map[int]int, t.NFacts)
	for _, s := range subs {
		v := int(rng.next() % uint64(t.K))
		vals[s] = v
		seq = append(seq, digitsTo(s, t.D)...)
		seq = append(seq, tokEQ, v, tokSEP)
	}
	// query a random one of the present subjects
	q := subs[int(rng.next()%uint64(len(subs)))]
	ans = t.transform(vals[q])
	seq = append(seq, tokQ)
	seq = append(seq, digitsTo(q, t.D)...)
	seq = append(seq, tokANS, ans)
	return seq, ans
}

// Batch builds B retrieval problems.
func (t RetrievalTask) Batch(B int, seed int64) (idx, targets []int) {
	rng := newRNG(seed)
	idx = make([]int, B*t.T)
	targets = make([]int, B*t.T)
	for i := range targets {
		targets[i] = -1
	}
	ap := t.ansPos()
	for b := 0; b < B; b++ {
		seq, ans := t.sample(rng)
		copy(idx[b*t.T:], seq)
		targets[b*t.T+ap] = ans
	}
	return
}

// Accuracy measures answer accuracy over n fresh random problems (always unseen by construction).
func (t RetrievalTask) Accuracy(g *GPT, seed int64, n int) float64 {
	rng := newRNG(seed)
	correct := 0
	ap := t.ansPos()
	for i := 0; i < n; i++ {
		seq, ans := t.sample(rng)
		if g.PredictAt(seq, 1, t.T, ap) == ans {
			correct++
		}
	}
	return float64(correct) / float64(n)
}

// digitsTo returns the d-digit base-10 code of s (most significant first).
func digitsTo(s, d int) []int {
	out := make([]int, d)
	for i := d - 1; i >= 0; i-- {
		out[i] = s % 10
		s /= 10
	}
	return out
}
