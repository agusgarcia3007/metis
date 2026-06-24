package nano

import "math"

// Linear: y[N,O] = x[N,I] @ W[I,O] (+ b[1,O] if non-nil).
func (tp *Tape) Linear(x, W, b *Tensor) *Tensor {
	N, I, O := x.R, x.C, W.C
	y := tp.node(N, O)
	parFor(N, func(n int) {
		xr := x.Data[n*I : n*I+I]
		yr := y.Data[n*O : n*O+O]
		if b != nil {
			copy(yr, b.Data)
		}
		for i := 0; i < I; i++ {
			xv := xr[i]
			if xv == 0 {
				continue
			}
			Wr := W.Data[i*O : i*O+O]
			for o := 0; o < O; o++ {
				yr[o] += xv * Wr[o]
			}
		}
	})
	y.bw = func() {
		// dx
		parFor(N, func(n int) {
			gr := y.Grad[n*O : n*O+O]
			xg := x.Grad[n*I : n*I+I]
			for i := 0; i < I; i++ {
				Wr := W.Data[i*O : i*O+O]
				var s float32
				for o := 0; o < O; o++ {
					s += gr[o] * Wr[o]
				}
				xg[i] += s
			}
		})
		// dW (parallel over input dim i → disjoint rows of W.Grad)
		parFor(I, func(i int) {
			Wg := W.Grad[i*O : i*O+O]
			for n := 0; n < N; n++ {
				xv := x.Data[n*I+i]
				if xv == 0 {
					continue
				}
				gr := y.Grad[n*O : n*O+O]
				for o := 0; o < O; o++ {
					Wg[o] += xv * gr[o]
				}
			}
		})
		// db
		if b != nil {
			for n := 0; n < N; n++ {
				gr := y.Grad[n*O : n*O+O]
				for o := 0; o < O; o++ {
					b.Grad[o] += gr[o]
				}
			}
		}
	}
	return y
}

// Add: elementwise residual z = x + y (same shape).
func (tp *Tape) Add(x, y *Tensor) *Tensor {
	z := tp.node(x.R, x.C)
	for i := range z.Data {
		z.Data[i] = x.Data[i] + y.Data[i]
	}
	z.bw = func() {
		for i := range z.Grad {
			x.Grad[i] += z.Grad[i]
			y.Grad[i] += z.Grad[i]
		}
	}
	return z
}

// LayerNorm over the last dim C, with affine gamma,beta [1,C].
func (tp *Tape) LayerNorm(x, gamma, beta *Tensor) *Tensor {
	N, C := x.R, x.C
	const eps = 1e-5
	y := tp.node(N, C)
	mean := make([]float32, N)
	istd := make([]float32, N)
	for n := 0; n < N; n++ {
		xr := x.Data[n*C : n*C+C]
		var m float32
		for _, v := range xr {
			m += v
		}
		m /= float32(C)
		var vsum float32
		for _, v := range xr {
			d := v - m
			vsum += d * d
		}
		is := float32(1.0 / math.Sqrt(float64(vsum/float32(C)+eps)))
		mean[n], istd[n] = m, is
		yr := y.Data[n*C : n*C+C]
		for c := 0; c < C; c++ {
			yr[c] = (xr[c]-m)*is*gamma.Data[c] + beta.Data[c]
		}
	}
	y.bw = func() {
		for n := 0; n < N; n++ {
			xr := x.Data[n*C : n*C+C]
			gr := y.Grad[n*C : n*C+C]
			xg := x.Grad[n*C : n*C+C]
			m, is := mean[n], istd[n]
			var dxhatMean, dxhatXhat float32
			for c := 0; c < C; c++ {
				xhat := (xr[c] - m) * is
				dxhat := gr[c] * gamma.Data[c]
				dxhatMean += dxhat
				dxhatXhat += dxhat * xhat
				gamma.Grad[c] += gr[c] * xhat
				beta.Grad[c] += gr[c]
			}
			dxhatMean /= float32(C)
			dxhatXhat /= float32(C)
			for c := 0; c < C; c++ {
				xhat := (xr[c] - m) * is
				dxhat := gr[c] * gamma.Data[c]
				xg[c] += is * (dxhat - dxhatMean - xhat*dxhatXhat)
			}
		}
	}
	return y
}

// GELU activation (tanh approximation), elementwise.
func (tp *Tape) GELU(x *Tensor) *Tensor {
	y := tp.node(x.R, x.C)
	parFor(x.R, func(r int) {
		for c := 0; c < x.C; c++ {
			i := r*x.C + c
			y.Data[i] = geluTanh(x.Data[i])
		}
	})
	y.bw = func() {
		parFor(x.R, func(r int) {
			for c := 0; c < x.C; c++ {
				i := r*x.C + c
				x.Grad[i] += y.Grad[i] * geluGrad(x.Data[i])
			}
		})
	}
	return y
}

// Split cuts x[N, k*P] into k tensors of [N,P] (column blocks). Used for QKV.
func (tp *Tape) Split(x *Tensor, k int) []*Tensor {
	N := x.R
	P := x.C / k
	outs := make([]*Tensor, k)
	for j := 0; j < k; j++ {
		outs[j] = tp.node(N, P)
		j := j
		o := outs[j]
		for n := 0; n < N; n++ {
			copy(o.Data[n*P:n*P+P], x.Data[n*x.C+j*P:n*x.C+j*P+P])
		}
		o.bw = func() {
			for n := 0; n < N; n++ {
				src := o.Grad[n*P : n*P+P]
				dst := x.Grad[n*x.C+j*P : n*x.C+j*P+P]
				for p := 0; p < P; p++ {
					dst[p] += src[p]
				}
			}
		}
	}
	return outs
}

// ropeTables precomputes cos/sin for rotary position embeddings: [T][hd/2].
func ropeTables(T, hd int) (cos, sin []float32) {
	half := hd / 2
	cos = make([]float32, T*half)
	sin = make([]float32, T*half)
	for pos := 0; pos < T; pos++ {
		for j := 0; j < half; j++ {
			freq := math.Pow(10000, -2*float64(j)/float64(hd))
			ang := float64(pos) * freq
			cos[pos*half+j] = float32(math.Cos(ang))
			sin[pos*half+j] = float32(math.Sin(ang))
		}
	}
	return
}

// rope rotates the hd-vector src into dst using cos/sin at a position (sign=+1 forward, -1 inverse).
// RoPE rotates dimension pairs (j, j+half); the inverse (transpose) recovers gradients.
func rope(dst, src, cosRow, sinRow []float32, half int, sign float32) {
	for j := 0; j < half; j++ {
		c, s := cosRow[j], sinRow[j]*sign
		a, b := src[j], src[j+half]
		dst[j] = a*c - b*s
		dst[j+half] = a*s + b*c
	}
}

// Attention runs causal multi-head self-attention. q,k,v are [B*T,C]. With useRoPE the q/k vectors
// are rotated by position (relative positions, good for offset-copy); without it, attention is pure
// content matching plus whatever absolute position the embedding provides (the GPT-2 induction setup).
func (tp *Tape) Attention(q, k, v *Tensor, B, T, h int, useRoPE bool) *Tensor {
	C := q.C
	hd := C / h
	half := hd / 2
	scale := float32(1.0 / math.Sqrt(float64(hd)))
	N := B * T
	out := tp.node(N, C)
	perBH := T * (T + 1) / 2
	probs := make([]float32, B*h*perBH)
	cosT, sinT := ropeTables(T, hd)
	// rot applies RoPE (or copies, if disabled) so forward/backward share one path.
	rot := func(dst, src []float32, pos int, sign float32) {
		if useRoPE {
			rope(dst, src, cosT[pos*half:], sinT[pos*half:], half, sign)
		} else {
			copy(dst, src[:hd])
		}
	}

	bhFor := func(do func(b, head int)) {
		parFor(B*h, func(idx int) { do(idx/h, idx%h) })
	}
	off := func(b, head, ti int) int {
		return (b*h+head)*perBH + ti*(ti+1)/2
	}

	bhFor(func(b, head int) {
		ch := head * hd
		qrot := make([]float32, hd)
		krot := make([]float32, hd)
		for ti := 0; ti < T; ti++ {
			qi := (b*T + ti) * C
			rot(qrot, q.Data[qi+ch:qi+ch+hd], ti, 1)
			p := probs[off(b, head, ti) : off(b, head, ti)+ti+1]
			var maxs float32 = -1e30
			for kj := 0; kj <= ti; kj++ {
				kb := (b*T + kj) * C
				rot(krot, k.Data[kb+ch:kb+ch+hd], kj, 1)
				var s float32
				for d := 0; d < hd; d++ {
					s += qrot[d] * krot[d]
				}
				s *= scale
				p[kj] = s
				if s > maxs {
					maxs = s
				}
			}
			var sum float32
			for kj := 0; kj <= ti; kj++ {
				e := float32(math.Exp(float64(p[kj] - maxs)))
				p[kj] = e
				sum += e
			}
			inv := 1 / sum
			orow := out.Data[qi+ch : qi+ch+hd]
			for kj := 0; kj <= ti; kj++ {
				p[kj] *= inv
				pj := p[kj]
				vjr := v.Data[(b*T+kj)*C+ch : (b*T+kj)*C+ch+hd]
				for d := 0; d < hd; d++ {
					orow[d] += pj * vjr[d]
				}
			}
		}
	})

	out.bw = func() {
		bhFor(func(b, head int) {
			ch := head * hd
			qrot := make([]float32, hd)
			krot := make([]float32, hd)
			qgrot := make([]float32, hd)
			kgrot := make([]float32, hd)
			tmp := make([]float32, hd)
			for ti := 0; ti < T; ti++ {
				qi := (b*T + ti) * C
				go_ := out.Grad[qi+ch : qi+ch+hd]
				rot(qrot, q.Data[qi+ch:qi+ch+hd], ti, 1)
				p := probs[off(b, head, ti) : off(b, head, ti)+ti+1]
				dp := make([]float32, ti+1)
				var dot float32
				for kj := 0; kj <= ti; kj++ {
					base := (b*T+kj)*C + ch
					vjr := v.Data[base : base+hd]
					vjg := v.Grad[base : base+hd]
					var d float32
					for dd := 0; dd < hd; dd++ {
						d += go_[dd] * vjr[dd]
						vjg[dd] += p[kj] * go_[dd] // dv
					}
					dp[kj] = d
					dot += p[kj] * d
				}
				for dd := range qgrot {
					qgrot[dd] = 0
				}
				for kj := 0; kj <= ti; kj++ {
					ds := p[kj] * (dp[kj] - dot) * scale
					kb := (b*T + kj) * C
					rot(krot, k.Data[kb+ch:kb+ch+hd], kj, 1)
					// grads in ROTATED space
					for dd := 0; dd < hd; dd++ {
						qgrot[dd] += ds * krot[dd]
						kgrot[dd] = ds * qrot[dd]
					}
					// rotate k-grad back to unrotated space and accumulate
					rot(tmp, kgrot, kj, -1)
					kjg := k.Grad[kb+ch : kb+ch+hd]
					for dd := 0; dd < hd; dd++ {
						kjg[dd] += tmp[dd]
					}
				}
				// rotate q-grad back to unrotated space and accumulate
				rot(tmp, qgrot, ti, -1)
				qg := q.Grad[qi+ch : qi+ch+hd]
				for dd := 0; dd < hd; dd++ {
					qg[dd] += tmp[dd]
				}
			}
		})
	}
	return out
}

// LogitsTied computes logits[N,V] = x[N,C] @ wte[V,C]^T (weight tying with the embedding).
func (tp *Tape) LogitsTied(x, wte *Tensor) *Tensor {
	N, C, V := x.R, x.C, wte.R
	y := tp.node(N, V)
	parFor(N, func(n int) {
		xr := x.Data[n*C : n*C+C]
		yr := y.Data[n*V : n*V+V]
		for vv := 0; vv < V; vv++ {
			wr := wte.Data[vv*C : vv*C+C]
			var s float32
			for c := 0; c < C; c++ {
				s += xr[c] * wr[c]
			}
			yr[vv] = s
		}
	})
	y.bw = func() {
		parFor(N, func(n int) {
			gr := y.Grad[n*V : n*V+V]
			xg := x.Grad[n*C : n*C+C]
			for vv := 0; vv < V; vv++ {
				g := gr[vv]
				if g == 0 {
					continue
				}
				wr := wte.Data[vv*C : vv*C+C]
				for c := 0; c < C; c++ {
					xg[c] += g * wr[c]
				}
			}
		})
		parFor(V, func(vv int) {
			wg := wte.Grad[vv*C : vv*C+C]
			for n := 0; n < N; n++ {
				g := y.Grad[n*V+vv]
				if g == 0 {
					continue
				}
				xr := x.Data[n*C : n*C+C]
				for c := 0; c < C; c++ {
					wg[c] += g * xr[c]
				}
			}
		})
	}
	return y
}

// CrossEntropy returns mean token NLL over logits[N,V] given integer targets[N].
// Targets < 0 are ignored (no loss, no gradient, excluded from the mean) — this lets the
// RNT task score only the answer token and mask the rest.
func (tp *Tape) CrossEntropy(logits *Tensor, targets []int) (*Tensor, float32) {
	N, V := logits.R, logits.C
	loss := tp.node(1, 1)
	soft := make([]float32, N*V)
	var total float32
	var cnt int
	for n := 0; n < N; n++ {
		if targets[n] < 0 {
			continue
		}
		cnt++
		lr := logits.Data[n*V : n*V+V]
		var maxs float32 = -1e30
		for _, v := range lr {
			if v > maxs {
				maxs = v
			}
		}
		var sum float32
		sr := soft[n*V : n*V+V]
		for v := 0; v < V; v++ {
			e := float32(math.Exp(float64(lr[v] - maxs)))
			sr[v] = e
			sum += e
		}
		inv := 1 / sum
		for v := 0; v < V; v++ {
			sr[v] *= inv
		}
		total += -float32(math.Log(float64(sr[targets[n]]) + 1e-12))
	}
	if cnt == 0 {
		cnt = 1
	}
	avg := total / float32(cnt)
	loss.Data[0] = avg
	loss.bw = func() {
		scale := loss.Grad[0] / float32(cnt)
		for n := 0; n < N; n++ {
			if targets[n] < 0 {
				continue
			}
			lg := logits.Grad[n*V : n*V+V]
			sr := soft[n*V : n*V+V]
			for v := 0; v < V; v++ {
				g := sr[v]
				if v == targets[n] {
					g -= 1
				}
				lg[v] += scale * g
			}
		}
	}
	return loss, avg
}

// PredictAt runs a forward pass and returns the argmax token at the given flat position.
func (g *GPT) PredictAt(idx []int, B, T, pos int) int {
	tp := NewTape()
	logits := g.Forward(tp, idx, B, T)
	V := logits.C
	lr := logits.Data[pos*V : pos*V+V]
	best, bv := 0, float32(-1e30)
	for v := 0; v < V; v++ {
		if lr[v] > bv {
			bv, best = lr[v], v
		}
	}
	return best
}
