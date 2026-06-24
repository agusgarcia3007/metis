package nano

// task.go implements the synthetic benchmark that proves the RNT training paradigm.
//
// The task factorizes cleanly into KNOWLEDGE and REASONING:
//   - KNOWLEDGE: a world maps each subject -> a value (a fact, e.g. "subject 047 = 4").
//   - REASONING: the answer is a fixed transform of the value:  answer = (value + 3) mod 10.
//
// Subjects are encoded as fixed-width DIGIT CODES over a fixed 15-token vocabulary
// {0..9, =, ;, ?, >, PAD}. Crucially the vocab does NOT grow with the number of facts, so a
// fixed-size model has a fixed parameter budget — memorizing more facts must compete for the
// same weights. This is what makes the capacity wall real (knowledge costs ~2 bits/param).
//
// Two training regimes share the SAME model:
//   - Vanilla: input [? d d d >]            -> must MEMORIZE subject->answer in its weights.
//   - RNT:     input [d d d = value ; ? d d d >] -> the fact is RETRIEVED into context.
//
// Decisive test = a NEW world whose facts were never trained on:
//   - Vanilla can't adapt (knowledge frozen in weights) -> accuracy collapses toward chance.
//   - RNT reads the fact from context and applies the transform -> accuracy stays ~perfect.

// Fixed vocabulary (independent of the number of facts).
const (
	tokEQ     = 10
	tokSEP    = 11
	tokQ      = 12
	tokANS    = 13
	tokPAD    = 14
	VocabSize = 15
)

// Task holds the encoding for a world with M subjects.
type Task struct {
	M int // number of subjects (facts)
	K int // number of values/answers = 10
	D int // digits per subject code
	T int // sequence length used by the model
}

// NewTask builds the encoding for M subjects.
func NewTask(m int) Task {
	d := 1
	for p := 10; p < m; p *= 10 {
		d++
	}
	return Task{M: m, K: 10, D: d, T: 2*d + 6}
}

func (t Task) transform(v int) int { return (v + 3) % t.K }

func (t Task) digits(s int) []int {
	out := make([]int, t.D)
	for i := t.D - 1; i >= 0; i-- {
		out[i] = s % 10
		s /= 10
	}
	return out
}

func (t Task) vanillaAnsPos() int { return t.D + 1 }
func (t Task) rntAnsPos() int     { return 2*t.D + 4 }

// vanillaSeq: [Q d.. > ans PAD..] length T.
func (t Task) vanillaSeq(subj, ans int) []int {
	seq := make([]int, t.T)
	for i := range seq {
		seq[i] = tokPAD
	}
	seq[0] = tokQ
	copy(seq[1:], t.digits(subj))
	seq[t.D+1] = tokANS
	seq[t.D+2] = ans
	return seq
}

// rntSeq: [d.. = val ; Q d.. > ans] length T.
func (t Task) rntSeq(subj, val, ans int) []int {
	seq := make([]int, t.T)
	dg := t.digits(subj)
	copy(seq[0:], dg)
	seq[t.D] = tokEQ
	seq[t.D+1] = val
	seq[t.D+2] = tokSEP
	seq[t.D+3] = tokQ
	copy(seq[t.D+4:], dg)
	seq[2*t.D+4] = tokANS
	seq[2*t.D+5] = ans
	return seq
}

// RandomWorld returns a fresh subject->value mapping.
func (t Task) RandomWorld(seed int64) []int {
	rng := newRNG(seed)
	w := make([]int, t.M)
	for s := range w {
		w[s] = int(rng.next() % uint64(t.K))
	}
	return w
}

// VanillaBatch draws B queries from a fixed world (model must memorize subject->answer).
func (t Task) VanillaBatch(world []int, B int, seed int64) (idx, targets []int) {
	rng := newRNG(seed)
	idx = make([]int, B*t.T)
	targets = make([]int, B*t.T)
	for i := range targets {
		targets[i] = -1
	}
	ap := t.vanillaAnsPos()
	for b := 0; b < B; b++ {
		s := int(rng.next() % uint64(t.M))
		ans := t.transform(world[s])
		copy(idx[b*t.T:], t.vanillaSeq(s, ans))
		targets[b*t.T+ap] = ans
	}
	return
}

// RNTBatch draws B queries with a random fact retrieved into context (model learns the transform).
func (t Task) RNTBatch(B int, seed int64) (idx, targets []int) {
	rng := newRNG(seed)
	idx = make([]int, B*t.T)
	targets = make([]int, B*t.T)
	for i := range targets {
		targets[i] = -1
	}
	ap := t.rntAnsPos()
	for b := 0; b < B; b++ {
		s := int(rng.next() % uint64(t.M))
		v := int(rng.next() % uint64(t.K))
		ans := t.transform(v)
		copy(idx[b*t.T:], t.rntSeq(s, v, ans))
		targets[b*t.T+ap] = ans
	}
	return
}

// VanillaAccuracy measures answer accuracy over all subjects of a world (no facts in context).
func (t Task) VanillaAccuracy(g *GPT, world []int) float64 {
	correct := 0
	ap := t.vanillaAnsPos()
	for s := 0; s < t.M; s++ {
		ans := t.transform(world[s])
		pred := g.PredictAt(t.vanillaSeq(s, ans), 1, t.T, ap)
		if pred == ans {
			correct++
		}
	}
	return float64(correct) / float64(t.M)
}

// Answer runs the model on a retrieved fact "subject = value" and returns its predicted answer.
// The true answer is never shown to the model (prediction is causal, before the answer position).
func (t Task) Answer(g *GPT, subj, val int) int {
	return g.PredictAt(t.rntSeq(subj, val, tokPAD), 1, t.T, t.rntAnsPos())
}

// Transform exposes the reasoning rule for callers (the ground-truth answer).
func (t Task) Transform(v int) int { return t.transform(v) }

// RNTAccuracy measures answer accuracy over fresh (subject,value) pairs WITH the fact in context.
func (t Task) RNTAccuracy(g *GPT, world []int, seed int64, n int) float64 {
	rng := newRNG(seed)
	correct := 0
	ap := t.rntAnsPos()
	for i := 0; i < n; i++ {
		s := int(rng.next() % uint64(t.M))
		v := world[s]
		ans := t.transform(v)
		pred := g.PredictAt(t.rntSeq(s, v, ans), 1, t.T, ap)
		if pred == ans {
			correct++
		}
	}
	return float64(correct) / float64(n)
}
