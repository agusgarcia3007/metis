package nano

// task_assoc.go is the canonical associative-recall ("induction head") test with SINGLE-TOKEN
// subjects. It isolates pure retrieval from the multi-digit parsing difficulty of the
// RetrievalTask: the model must match a query's subject token to one fact among K distractors and
// read/transform its value. This is the cleanest proof that the model can RETRIEVE, not copy.
//
// Context: [s1 = v1 ;] ... [sK = vK ;]              (single-token subjects, shuffled, random values)
// Queries: [? sq1 > a1] [? sq2 > a2] ...            (NQ queries, each about a present subject)
// Answer:  a_i = (value of sq_i + 3) mod 10
//
// NQ>1 gives DENSE supervision (many supervised tokens per sequence) — critical for the recall
// circuit to form quickly. Facts are random per sample, so the model can never memorize.

// AssocTask holds a single-token-subject associative-recall task with NFacts facts and NQ queries.
type AssocTask struct {
	M, K, NFacts, NQ, T       int
	eq, sep, q, ans, pad, voc int
}

// NewAssocTask builds a task with nFacts facts and 1 query (sparse supervision).
func NewAssocTask(m, nFacts int) AssocTask { return NewAssocTaskQ(m, nFacts, 1) }

// NewAssocTaskQ builds a task with nFacts facts and nQ queries (nQ>1 = dense supervision).
// Fact format is [subject value] (value immediately follows subject) — the easiest copy offset for
// the recall circuit. Subjects and values occupy disjoint token ranges, so the model can tell them
// apart by identity (no separators needed).
func NewAssocTaskQ(m, nFacts, nQ int) AssocTask {
	t := AssocTask{M: m, K: 10, NFacts: nFacts, NQ: nQ}
	base := 10 + m // values 0..9, subjects 10..10+m-1
	t.eq, t.sep, t.q, t.ans, t.pad = base, base+1, base+2, base+3, base+4
	t.voc = base + 5
	t.T = nFacts*2 + nQ*3 // fact = [subj val] (2); query = [? subj ans] (3)
	return t
}

// Vocab returns the task's vocabulary size (use for Config.Vocab).
func (t AssocTask) Vocab() int          { return t.voc }
func (t AssocTask) subj(s int) int      { return 10 + s }
func (t AssocTask) transform(v int) int { return (v + 3) % t.K }

// queryAnsPos returns the flat position of the query-subject token of query i, whose next-token
// prediction is that query's answer.
func (t AssocTask) queryAnsPos(i int) int { return t.NFacts*2 + i*3 + 1 }

func (t AssocTask) sample(rng *rng) (seq []int, answers []int) {
	seq = make([]int, 0, t.T)
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
		seq = append(seq, t.subj(s), v) // [subject value]
	}
	answers = make([]int, t.NQ)
	for i := 0; i < t.NQ; i++ {
		qs := subs[int(rng.next()%uint64(len(subs)))]
		a := t.transform(vals[qs])
		answers[i] = a
		seq = append(seq, t.q, t.subj(qs), a) // [? subject answer]
	}
	return seq, answers
}

// Batch builds B problems, supervising every query's answer.
func (t AssocTask) Batch(B int, seed int64) (idx, targets []int) {
	rng := newRNG(seed)
	idx = make([]int, B*t.T)
	targets = make([]int, B*t.T)
	for i := range targets {
		targets[i] = -1
	}
	for b := 0; b < B; b++ {
		seq, answers := t.sample(rng)
		copy(idx[b*t.T:], seq)
		for i, a := range answers {
			targets[b*t.T+t.queryAnsPos(i)] = a
		}
	}
	return
}

// Accuracy over n fresh random problems, evaluated on the first query position.
func (t AssocTask) Accuracy(g *GPT, seed int64, n int) float64 {
	rng := newRNG(seed)
	correct := 0
	ap := t.queryAnsPos(0)
	for i := 0; i < n; i++ {
		seq, answers := t.sample(rng)
		if g.PredictAt(seq, 1, t.T, ap) == answers[0] {
			correct++
		}
	}
	return float64(correct) / float64(n)
}
