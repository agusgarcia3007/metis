package nano

import "math"

// Config defines the GPT shape. Defaults are tuned to train to legible char-level
// English on a CPU in minutes while staying tiny (research 01: small + efficient).
type Config struct {
	Vocab  int
	Block  int // context length T
	Layer  int
	Head   int
	Embd   int  // C
	NoPos  bool // omit absolute position embedding
	NoRoPE bool // disable rotary position embedding in attention (pure content matching)
}

// Embed: x[N,C] = wte[token] (+ wpe[position] if wpe != nil).
// With RoPE in attention, absolute position (wpe) is optional and is best omitted for
// associative-recall tasks so the model cannot exploit absolute-position shortcuts.
func (tp *Tape) Embed(idx []int, wte, wpe *Tensor, B, T int) *Tensor {
	C := wte.C
	N := B * T
	x := tp.node(N, C)
	for n := 0; n < N; n++ {
		pos := n % T
		tok := idx[n]
		xr := x.Data[n*C : n*C+C]
		wr := wte.Data[tok*C : tok*C+C]
		if wpe != nil {
			pr := wpe.Data[pos*C : pos*C+C]
			for c := 0; c < C; c++ {
				xr[c] = wr[c] + pr[c]
			}
		} else {
			copy(xr, wr)
		}
	}
	x.bw = func() {
		for n := 0; n < N; n++ {
			pos := n % T
			tok := idx[n]
			xg := x.Grad[n*C : n*C+C]
			wg := wte.Grad[tok*C : tok*C+C]
			for c := 0; c < C; c++ {
				wg[c] += xg[c]
			}
			if wpe != nil {
				pg := wpe.Grad[pos*C : pos*C+C]
				for c := 0; c < C; c++ {
					pg[c] += xg[c]
				}
			}
		}
	}
	return x
}

type block struct {
	ln1g, ln1b   *Tensor
	wqkv, bqkv   *Tensor
	wproj, bproj *Tensor
	ln2g, ln2b   *Tensor
	wfc, bfc     *Tensor
	wfc2, bfc2   *Tensor
}

// GPT is a decoder-only transformer with tied input/output embeddings.
type GPT struct {
	Cfg        Config
	wte, wpe   *Tensor
	blocks     []block
	lnfg, lnfb *Tensor
}

// NewGPT allocates and randomly initializes a model.
func NewGPT(cfg Config, seed int64) *GPT {
	rng := newRNG(seed)
	C := cfg.Embd
	std := float32(0.02)
	mk := func(r, c int, s float32) *Tensor {
		t := NewParam(r, c)
		for i := range t.Data {
			t.Data[i] = rng.normal() * s
		}
		return t
	}
	ones := func(c int) *Tensor {
		t := NewParam(1, c)
		for i := range t.Data {
			t.Data[i] = 1
		}
		return t
	}
	zeros := func(c int) *Tensor { return NewParam(1, c) }
	g := &GPT{Cfg: cfg}
	g.wte = mk(cfg.Vocab, C, std)
	g.wpe = mk(cfg.Block, C, std)
	// scale residual-projection inits by 1/sqrt(2*Layer) (GPT-2 init).
	pscale := float32(0.02 / math.Sqrt(float64(2*cfg.Layer)))
	for l := 0; l < cfg.Layer; l++ {
		g.blocks = append(g.blocks, block{
			ln1g: ones(C), ln1b: zeros(C),
			wqkv: mk(C, 3*C, std), bqkv: zeros(3 * C),
			wproj: mk(C, C, pscale), bproj: zeros(C),
			ln2g: ones(C), ln2b: zeros(C),
			wfc: mk(C, 4*C, std), bfc: zeros(4 * C),
			wfc2: mk(4*C, C, pscale), bfc2: zeros(C),
		})
	}
	g.lnfg, g.lnfb = ones(C), zeros(C)
	return g
}

// Params returns every trainable tensor (for the optimizer & serialization), in a stable order.
func (g *GPT) Params() []*Tensor {
	ps := []*Tensor{g.wte, g.wpe, g.lnfg, g.lnfb}
	for i := range g.blocks {
		b := &g.blocks[i]
		ps = append(ps, b.ln1g, b.ln1b, b.wqkv, b.bqkv, b.wproj, b.bproj,
			b.ln2g, b.ln2b, b.wfc, b.bfc, b.wfc2, b.bfc2)
	}
	return ps
}

// Forward runs the model over idx (length B*T) and returns logits[B*T, Vocab].
func (g *GPT) Forward(tp *Tape, idx []int, B, T int) *Tensor {
	cfg := g.Cfg
	wpe := g.wpe
	if cfg.NoPos {
		wpe = nil
	}
	x := tp.Embed(idx, g.wte, wpe, B, T)
	for i := range g.blocks {
		b := &g.blocks[i]
		a := tp.LayerNorm(x, b.ln1g, b.ln1b)
		qkv := tp.Linear(a, b.wqkv, b.bqkv)
		parts := tp.Split(qkv, 3)
		att := tp.Attention(parts[0], parts[1], parts[2], B, T, cfg.Head, !cfg.NoRoPE)
		att = tp.Linear(att, b.wproj, b.bproj)
		x = tp.Add(x, att)
		m := tp.LayerNorm(x, b.ln2g, b.ln2b)
		hgelu := tp.GELU(tp.Linear(m, b.wfc, b.bfc))
		mlp := tp.Linear(hgelu, b.wfc2, b.bfc2)
		x = tp.Add(x, mlp)
	}
	x = tp.LayerNorm(x, g.lnfg, g.lnfb)
	return tp.LogitsTied(x, g.wte)
}

// --- tiny deterministic RNG (xorshift + Box-Muller) so runs are reproducible ---

type rng struct{ s uint64 }

func newRNG(seed int64) *rng { return &rng{s: uint64(seed)*2862933555777941757 + 3037000493} }

func (r *rng) next() uint64 {
	r.s ^= r.s << 13
	r.s ^= r.s >> 7
	r.s ^= r.s << 17
	return r.s
}

func (r *rng) float() float32 { return float32(r.next()>>11) / float32(1<<53) }

func (r *rng) normal() float32 {
	u1 := r.float()*0.999999 + 1e-7
	u2 := r.float()
	return float32(math.Sqrt(-2*math.Log(float64(u1))) * math.Cos(2*math.Pi*float64(u2)))
}
