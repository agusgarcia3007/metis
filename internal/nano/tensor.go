// Package nano is a from-scratch, dependency-free transformer language model in pure Go:
// a real autograd engine + GPT + trainer. It is "the smallest model we can actually build
// and verify in-session" — trained from zero on CPU, no GPU, no cgo, no downloads at runtime.
//
// Design alignment: nano implements the kernel.Kernel seam (see internal/kernel/nano.go), so the
// same tiny-llm system that will later host a Qwen3 Cortex via ggml can host this native Cortex today.
package nano

import (
	"math"
	"runtime"
	"sync"
)

// Tensor is a 2D [R,C] float32 array with a gradient and a backward closure.
type Tensor struct {
	Data []float32
	Grad []float32
	R, C int
	bw   func() // accumulates this node's Grad into its parents' Grads
}

// Tape records intermediate tensors in creation (topological) order so Backward can
// replay their backward closures in reverse. Parameters are leaves and live off-tape.
type Tape struct{ nodes []*Tensor }

// NewTape returns an empty tape for one forward/backward pass.
func NewTape() *Tape { return &Tape{} }

func (tp *Tape) node(r, c int) *Tensor {
	t := &Tensor{Data: make([]float32, r*c), Grad: make([]float32, r*c), R: r, C: c}
	tp.nodes = append(tp.nodes, t)
	return t
}

// Backward seeds d(loss)/d(loss)=1 and replays every node's backward in reverse order.
func (tp *Tape) Backward(loss *Tensor) {
	loss.Grad[0] = 1
	for i := len(tp.nodes) - 1; i >= 0; i-- {
		if tp.nodes[i].bw != nil {
			tp.nodes[i].bw()
		}
	}
}

// NewParam creates a persistent leaf parameter tensor (kept across steps).
func NewParam(r, c int) *Tensor {
	return &Tensor{Data: make([]float32, r*c), Grad: make([]float32, r*c), R: r, C: c}
}

// Leaf wraps existing data as a non-tracked input (e.g. token indices, masks).
func Leaf(data []float32, r, c int) *Tensor {
	return &Tensor{Data: data, Grad: make([]float32, r*c), R: r, C: c}
}

// ZeroGrad clears a parameter's gradient buffer between steps.
func (t *Tensor) ZeroGrad() {
	for i := range t.Grad {
		t.Grad[i] = 0
	}
}

// nWorkers bounds matmul parallelism; leave 2 cores for the OS/runtime (research 00/05).
var nWorkers = max(1, runtime.NumCPU()-2)

// parFor runs fn over [0,n) split across up to nWorkers goroutines.
func parFor(n int, fn func(i int)) {
	if n == 0 {
		return
	}
	w := nWorkers
	if w > n {
		w = n
	}
	if w == 1 {
		for i := 0; i < n; i++ {
			fn(i)
		}
		return
	}
	var wg sync.WaitGroup
	chunk := (n + w - 1) / w
	for g := 0; g < w; g++ {
		lo := g * chunk
		hi := lo + chunk
		if hi > n {
			hi = n
		}
		if lo >= hi {
			break
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			for i := lo; i < hi; i++ {
				fn(i)
			}
		}(lo, hi)
	}
	wg.Wait()
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func geluTanh(x float32) float32 {
	// 0.5x(1+tanh(√(2/π)(x+0.044715x³)))
	x3 := x * x * x
	inner := 0.7978845608 * (x + 0.044715*x3)
	return 0.5 * x * (1 + float32(math.Tanh(float64(inner))))
}

func geluGrad(x float32) float32 {
	x3 := x * x * x
	inner := 0.7978845608 * (x + 0.044715*x3)
	t := float32(math.Tanh(float64(inner)))
	sech2 := 1 - t*t
	dinner := 0.7978845608 * (1 + 3*0.044715*x*x)
	return 0.5*(1+t) + 0.5*x*sech2*dinner
}
