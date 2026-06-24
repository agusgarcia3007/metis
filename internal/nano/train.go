package nano

import (
	"math"
)

// AdamW optimizer with decoupled weight decay.
type AdamW struct {
	LR, B1, B2, Eps, WD float32
	t                   int
	m, v                [][]float32
	params              []*Tensor
}

// NewAdamW initializes optimizer state for the given parameters.
func NewAdamW(params []*Tensor, lr, wd float32) *AdamW {
	o := &AdamW{LR: lr, B1: 0.9, B2: 0.95, Eps: 1e-8, WD: wd, params: params}
	for _, p := range params {
		o.m = append(o.m, make([]float32, len(p.Data)))
		o.v = append(o.v, make([]float32, len(p.Data)))
	}
	return o
}

// ZeroGrad clears all parameter gradients.
func (o *AdamW) ZeroGrad() {
	for _, p := range o.params {
		p.ZeroGrad()
	}
}

// Step applies one AdamW update. decayMask, if non-nil, scales per-parameter weight
// decay (used by RNT to apply the anti-memorization penalty selectively to MLP weights).
func (o *AdamW) Step(decayMask map[*Tensor]float32) {
	o.t++
	bc1 := 1 - float32(math.Pow(float64(o.B1), float64(o.t)))
	bc2 := 1 - float32(math.Pow(float64(o.B2), float64(o.t)))
	for pi, p := range o.params {
		m, v := o.m[pi], o.v[pi]
		wd := o.WD
		if decayMask != nil {
			if s, ok := decayMask[p]; ok {
				wd = o.WD * s
			} else {
				wd = 0
			}
		}
		for i := range p.Data {
			g := p.Grad[i]
			m[i] = o.B1*m[i] + (1-o.B1)*g
			v[i] = o.B2*v[i] + (1-o.B2)*g*g
			mh := m[i] / bc1
			vh := v[i] / bc2
			p.Data[i] -= o.LR * (mh/(float32(math.Sqrt(float64(vh)))+o.Eps) + wd*p.Data[i])
		}
	}
}

// LossAndGrad runs forward+backward for one batch and returns the mean loss.
// idx and targets have length B*T. It zeroes grads, so call Step after.
func (g *GPT) LossAndGrad(o *AdamW, idx, targets []int, B, T int) float32 {
	o.ZeroGrad()
	tp := NewTape()
	logits := g.Forward(tp, idx, B, T)
	loss, val := tp.CrossEntropy(logits, targets)
	tp.Backward(loss)
	return val
}

// forwardLoss recomputes just the scalar loss (for gradient checking).
func (g *GPT) forwardLoss(idx, targets []int, B, T int) float32 {
	tp := NewTape()
	logits := g.Forward(tp, idx, B, T)
	_, val := tp.CrossEntropy(logits, targets)
	return val
}

// GradCheck numerically verifies analytic gradients via central finite differences.
// It returns the max relative error over *measurable* entries and how many were measured.
//
// Entries whose gradient is too small to resolve in float32 are skipped: with a loss ~O(1),
// float32 resolves ~1e-7, so a finite-difference signal (2·eps·grad) below that floor is pure
// rounding noise and untestable — this is standard practice for finite-difference grad checks.
func (g *GPT) GradCheck(idx, targets []int, B, T int, n int, seed int64) (maxRel float32, measured int) {
	o := NewAdamW(g.Params(), 0, 0)
	g.LossAndGrad(o, idx, targets, B, T) // populate analytic grads
	rng := newRNG(seed)
	params := g.Params()
	const eps = 4e-3   // step tuned to balance float32 cancellation noise vs O(eps^2) truncation
	const floor = 1e-2 // only test well-resolved gradients (signal >> float32 noise floor)
	attempts := 0
	for measured < n && attempts < n*40 {
		attempts++
		p := params[int(rng.next()%uint64(len(params)))]
		i := int(rng.next() % uint64(len(p.Data)))
		orig := p.Data[i]
		analytic := p.Grad[i]

		p.Data[i] = orig + eps
		lp := g.forwardLoss(idx, targets, B, T)
		p.Data[i] = orig - eps
		lm := g.forwardLoss(idx, targets, B, T)
		p.Data[i] = orig
		numeric := (lp - lm) / (2 * eps)

		if math.Abs(float64(analytic)) < floor && math.Abs(float64(numeric)) < floor {
			continue // unmeasurable in float32
		}
		denom := float32(math.Abs(float64(analytic)) + math.Abs(float64(numeric)))
		rel := float32(math.Abs(float64(analytic-numeric))) / denom
		if rel > maxRel {
			maxRel = rel
		}
		measured++
	}
	return maxRel, measured
}
