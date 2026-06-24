//! Differentiable operations on the autograd Tape: Linear, Add, LayerNorm, GELU, Split, causal
//! multi-head attention (with optional RoPE), weight-tied logits, and cross-entropy. Each op runs
//! its forward immediately and records a backward closure on the tape.

use super::model::Gpt;
use super::tensor::{gelu_grad, gelu_tanh, Tape, T};

impl Tape {
    /// Linear: y[N,O] = x[N,I] @ W[I,O] (+ b[1,O] if Some).
    pub(crate) fn linear(&mut self, x: &T, w: &T, b: Option<&T>) -> T {
        let (n, i_dim, o) = {
            let xb = x.borrow();
            (xb.r, xb.c, w.borrow().c)
        };
        let y = Tape::node(n, o);
        {
            let xb = x.borrow();
            let wb = w.borrow();
            let bb = b.map(|t| t.borrow());
            let mut yb = y.borrow_mut();
            for nn in 0..n {
                let yr = &mut yb.data[nn * o..nn * o + o];
                match &bb {
                    Some(bv) => yr.copy_from_slice(&bv.data),
                    None => yr.iter_mut().for_each(|v| *v = 0.0),
                }
                let xr = &xb.data[nn * i_dim..nn * i_dim + i_dim];
                for ii in 0..i_dim {
                    let xv = xr[ii];
                    if xv == 0.0 {
                        continue;
                    }
                    let wr = &wb.data[ii * o..ii * o + o];
                    for oo in 0..o {
                        yr[oo] += xv * wr[oo];
                    }
                }
            }
        }
        let (xc, wc, bc, yc) = (x.clone(), w.clone(), b.cloned(), y.clone());
        self.push_bw(Box::new(move || {
            let mut xb = xc.borrow_mut();
            let mut wb = wc.borrow_mut();
            let yb = yc.borrow();
            // dx
            for nn in 0..n {
                let gr = &yb.grad[nn * o..nn * o + o];
                for ii in 0..i_dim {
                    let wr = &wb.data[ii * o..ii * o + o];
                    let mut s = 0.0f32;
                    for oo in 0..o {
                        s += gr[oo] * wr[oo];
                    }
                    xb.grad[nn * i_dim + ii] += s;
                }
            }
            // dW (over input dim → disjoint rows of W.grad)
            for ii in 0..i_dim {
                for nn in 0..n {
                    let xv = xb.data[nn * i_dim + ii];
                    if xv == 0.0 {
                        continue;
                    }
                    let gr = &yb.grad[nn * o..nn * o + o];
                    let wg = &mut wb.grad[ii * o..ii * o + o];
                    for oo in 0..o {
                        wg[oo] += xv * gr[oo];
                    }
                }
            }
            // db
            if let Some(bt) = &bc {
                let mut bb = bt.borrow_mut();
                for nn in 0..n {
                    let gr = &yb.grad[nn * o..nn * o + o];
                    for oo in 0..o {
                        bb.grad[oo] += gr[oo];
                    }
                }
            }
        }));
        y
    }

    /// Add: elementwise residual z = x + y (same shape).
    pub(crate) fn add(&mut self, x: &T, y: &T) -> T {
        let (r, c) = {
            let xb = x.borrow();
            (xb.r, xb.c)
        };
        let z = Tape::node(r, c);
        {
            let xb = x.borrow();
            let yb = y.borrow();
            let mut zb = z.borrow_mut();
            for i in 0..zb.data.len() {
                zb.data[i] = xb.data[i] + yb.data[i];
            }
        }
        let (xc, yc, zc) = (x.clone(), y.clone(), z.clone());
        self.push_bw(Box::new(move || {
            let mut xb = xc.borrow_mut();
            let mut yb = yc.borrow_mut();
            let zb = zc.borrow();
            for i in 0..zb.grad.len() {
                xb.grad[i] += zb.grad[i];
                yb.grad[i] += zb.grad[i];
            }
        }));
        z
    }

    /// LayerNorm over the last dim C, with affine gamma,beta [1,C].
    pub(crate) fn layer_norm(&mut self, x: &T, gamma: &T, beta: &T) -> T {
        const EPS: f32 = 1e-5;
        let (n, c) = {
            let xb = x.borrow();
            (xb.r, xb.c)
        };
        let y = Tape::node(n, c);
        let mut mean = vec![0.0f32; n];
        let mut istd = vec![0.0f32; n];
        {
            let xb = x.borrow();
            let gb = gamma.borrow();
            let bb = beta.borrow();
            let mut yb = y.borrow_mut();
            for nn in 0..n {
                let xr = &xb.data[nn * c..nn * c + c];
                let mut m = 0.0f32;
                for &v in xr.iter() {
                    m += v;
                }
                m /= c as f32;
                let mut vsum = 0.0f32;
                for &v in xr.iter() {
                    let d = v - m;
                    vsum += d * d;
                }
                let is = (1.0 / ((vsum / c as f32 + EPS) as f64).sqrt()) as f32;
                mean[nn] = m;
                istd[nn] = is;
                let yr = &mut yb.data[nn * c..nn * c + c];
                for cc in 0..c {
                    yr[cc] = (xr[cc] - m) * is * gb.data[cc] + bb.data[cc];
                }
            }
        }
        let (xc, gc, bc, yc) = (x.clone(), gamma.clone(), beta.clone(), y.clone());
        self.push_bw(Box::new(move || {
            let mut xb = xc.borrow_mut();
            let mut gb = gc.borrow_mut();
            let mut bb = bc.borrow_mut();
            let yb = yc.borrow();
            for nn in 0..n {
                let m = mean[nn];
                let is = istd[nn];
                let mut dxhat_mean = 0.0f32;
                let mut dxhat_xhat = 0.0f32;
                for cc in 0..c {
                    let xhat = (xb.data[nn * c + cc] - m) * is;
                    let dxhat = yb.grad[nn * c + cc] * gb.data[cc];
                    dxhat_mean += dxhat;
                    dxhat_xhat += dxhat * xhat;
                    gb.grad[cc] += yb.grad[nn * c + cc] * xhat;
                    bb.grad[cc] += yb.grad[nn * c + cc];
                }
                dxhat_mean /= c as f32;
                dxhat_xhat /= c as f32;
                for cc in 0..c {
                    let xhat = (xb.data[nn * c + cc] - m) * is;
                    let dxhat = yb.grad[nn * c + cc] * gb.data[cc];
                    xb.grad[nn * c + cc] += is * (dxhat - dxhat_mean - xhat * dxhat_xhat);
                }
            }
        }));
        y
    }

    /// GELU activation (tanh approximation), elementwise.
    pub(crate) fn gelu(&mut self, x: &T) -> T {
        let (r, c) = {
            let xb = x.borrow();
            (xb.r, xb.c)
        };
        let y = Tape::node(r, c);
        {
            let xb = x.borrow();
            let mut yb = y.borrow_mut();
            for i in 0..r * c {
                yb.data[i] = gelu_tanh(xb.data[i]);
            }
        }
        let (xc, yc) = (x.clone(), y.clone());
        self.push_bw(Box::new(move || {
            let mut xb = xc.borrow_mut();
            let yb = yc.borrow();
            for i in 0..r * c {
                xb.grad[i] += yb.grad[i] * gelu_grad(xb.data[i]);
            }
        }));
        y
    }

    /// Split cuts x[N, k*P] into k tensors of [N,P] (column blocks). Used for QKV.
    pub(crate) fn split(&mut self, x: &T, k: usize) -> Vec<T> {
        let (n, xc_dim) = {
            let xb = x.borrow();
            (xb.r, xb.c)
        };
        let p = xc_dim / k;
        let mut outs = Vec::with_capacity(k);
        for j in 0..k {
            let o = Tape::node(n, p);
            {
                let xb = x.borrow();
                let mut ob = o.borrow_mut();
                for nn in 0..n {
                    ob.data[nn * p..nn * p + p]
                        .copy_from_slice(&xb.data[nn * xc_dim + j * p..nn * xc_dim + j * p + p]);
                }
            }
            let (xcl, ocl) = (x.clone(), o.clone());
            self.push_bw(Box::new(move || {
                let mut xb = xcl.borrow_mut();
                let ob = ocl.borrow();
                for nn in 0..n {
                    for pp in 0..p {
                        xb.grad[nn * xc_dim + j * p + pp] += ob.grad[nn * p + pp];
                    }
                }
            }));
            outs.push(o);
        }
        outs
    }

    /// Attention runs causal multi-head self-attention. q,k,v are [B*T,C].
    pub(crate) fn attention(
        &mut self,
        q: &T,
        k: &T,
        v: &T,
        b_size: usize,
        t_size: usize,
        h: usize,
        use_rope: bool,
    ) -> T {
        let c_dim = q.borrow().c;
        let hd = c_dim / h;
        let half = hd / 2;
        let scale = (1.0 / (hd as f64).sqrt()) as f32;
        let n = b_size * t_size;
        let out = Tape::node(n, c_dim);
        let per_bh = t_size * (t_size + 1) / 2;
        let mut probs = vec![0.0f32; b_size * h * per_bh];
        let (cos_t, sin_t) = rope_tables(t_size, hd);
        let off = move |bb: usize, head: usize, ti: usize| (bb * h + head) * per_bh + ti * (ti + 1) / 2;

        {
            let qb = q.borrow();
            let kb = k.borrow();
            let vb = v.borrow();
            let mut ob = out.borrow_mut();
            let mut qrot = vec![0.0f32; hd];
            let mut krot = vec![0.0f32; hd];
            for bb in 0..b_size {
                for head in 0..h {
                    let ch = head * hd;
                    for ti in 0..t_size {
                        let qi = (bb * t_size + ti) * c_dim;
                        rot(&mut qrot, &qb.data[qi + ch..qi + ch + hd], ti, 1.0, use_rope, &cos_t, &sin_t, half, hd);
                        let base = off(bb, head, ti);
                        let mut maxs = -1e30f32;
                        for kj in 0..=ti {
                            let kbase = (bb * t_size + kj) * c_dim;
                            rot(&mut krot, &kb.data[kbase + ch..kbase + ch + hd], kj, 1.0, use_rope, &cos_t, &sin_t, half, hd);
                            let mut s = 0.0f32;
                            for d in 0..hd {
                                s += qrot[d] * krot[d];
                            }
                            s *= scale;
                            probs[base + kj] = s;
                            if s > maxs {
                                maxs = s;
                            }
                        }
                        let mut sum = 0.0f32;
                        for kj in 0..=ti {
                            let e = ((probs[base + kj] - maxs) as f64).exp() as f32;
                            probs[base + kj] = e;
                            sum += e;
                        }
                        let inv = 1.0 / sum;
                        let orow = &mut ob.data[qi + ch..qi + ch + hd];
                        for kj in 0..=ti {
                            probs[base + kj] *= inv;
                            let pj = probs[base + kj];
                            let vbase = (bb * t_size + kj) * c_dim + ch;
                            let vjr = &vb.data[vbase..vbase + hd];
                            for d in 0..hd {
                                orow[d] += pj * vjr[d];
                            }
                        }
                    }
                }
            }
        }

        let (qc, kc, vc, oc) = (q.clone(), k.clone(), v.clone(), out.clone());
        self.push_bw(Box::new(move || {
            let mut qb = qc.borrow_mut();
            let mut kb = kc.borrow_mut();
            let mut vb = vc.borrow_mut();
            let ob = oc.borrow();
            let mut qrot = vec![0.0f32; hd];
            let mut krot = vec![0.0f32; hd];
            let mut qgrot = vec![0.0f32; hd];
            let mut kgrot = vec![0.0f32; hd];
            let mut tmp = vec![0.0f32; hd];
            for bb in 0..b_size {
                for head in 0..h {
                    let ch = head * hd;
                    for ti in 0..t_size {
                        let qi = (bb * t_size + ti) * c_dim;
                        rot(&mut qrot, &qb.data[qi + ch..qi + ch + hd], ti, 1.0, use_rope, &cos_t, &sin_t, half, hd);
                        let base = off(bb, head, ti);
                        let mut dp = vec![0.0f32; ti + 1];
                        let mut dot = 0.0f32;
                        for kj in 0..=ti {
                            let vbase = (bb * t_size + kj) * c_dim + ch;
                            let pk = probs[base + kj];
                            let mut d = 0.0f32;
                            for dd in 0..hd {
                                d += ob.grad[qi + ch + dd] * vb.data[vbase + dd];
                                vb.grad[vbase + dd] += pk * ob.grad[qi + ch + dd]; // dv
                            }
                            dp[kj] = d;
                            dot += pk * d;
                        }
                        for v in qgrot.iter_mut() {
                            *v = 0.0;
                        }
                        for kj in 0..=ti {
                            let ds = probs[base + kj] * (dp[kj] - dot) * scale;
                            let kbase = (bb * t_size + kj) * c_dim;
                            rot(&mut krot, &kb.data[kbase + ch..kbase + ch + hd], kj, 1.0, use_rope, &cos_t, &sin_t, half, hd);
                            // grads in ROTATED space
                            for dd in 0..hd {
                                qgrot[dd] += ds * krot[dd];
                                kgrot[dd] = ds * qrot[dd];
                            }
                            // rotate k-grad back to unrotated space and accumulate
                            rot(&mut tmp, &kgrot, kj, -1.0, use_rope, &cos_t, &sin_t, half, hd);
                            for dd in 0..hd {
                                kb.grad[kbase + ch + dd] += tmp[dd];
                            }
                        }
                        // rotate q-grad back to unrotated space and accumulate
                        rot(&mut tmp, &qgrot, ti, -1.0, use_rope, &cos_t, &sin_t, half, hd);
                        for dd in 0..hd {
                            qb.grad[qi + ch + dd] += tmp[dd];
                        }
                    }
                }
            }
        }));
        out
    }

    /// LogitsTied computes logits[N,V] = x[N,C] @ wte[V,C]^T (weight tying with the embedding).
    pub(crate) fn logits_tied(&mut self, x: &T, wte: &T) -> T {
        let (n, c) = {
            let xb = x.borrow();
            (xb.r, xb.c)
        };
        let vocab = wte.borrow().r;
        let y = Tape::node(n, vocab);
        {
            let xb = x.borrow();
            let wb = wte.borrow();
            let mut yb = y.borrow_mut();
            for nn in 0..n {
                let xr = &xb.data[nn * c..nn * c + c];
                let yr = &mut yb.data[nn * vocab..nn * vocab + vocab];
                for vv in 0..vocab {
                    let wr = &wb.data[vv * c..vv * c + c];
                    let mut s = 0.0f32;
                    for cc in 0..c {
                        s += xr[cc] * wr[cc];
                    }
                    yr[vv] = s;
                }
            }
        }
        let (xc, wc, yc) = (x.clone(), wte.clone(), y.clone());
        self.push_bw(Box::new(move || {
            let mut xb = xc.borrow_mut();
            let mut wb = wc.borrow_mut();
            let yb = yc.borrow();
            // dx
            for nn in 0..n {
                for vv in 0..vocab {
                    let g = yb.grad[nn * vocab + vv];
                    if g == 0.0 {
                        continue;
                    }
                    let wr = &wb.data[vv * c..vv * c + c];
                    for cc in 0..c {
                        xb.grad[nn * c + cc] += g * wr[cc];
                    }
                }
            }
            // dwte
            for vv in 0..vocab {
                for nn in 0..n {
                    let g = yb.grad[nn * vocab + vv];
                    if g == 0.0 {
                        continue;
                    }
                    let wg = &mut wb.grad[vv * c..vv * c + c];
                    for cc in 0..c {
                        wg[cc] += g * xb.data[nn * c + cc];
                    }
                }
            }
        }));
        y
    }

    /// CrossEntropy returns mean token NLL over logits[N,V] given integer targets[N].
    /// Targets < 0 are ignored (no loss, no gradient, excluded from the mean).
    pub(crate) fn cross_entropy(&mut self, logits: &T, targets: &[i32]) -> (T, f32) {
        let (n, vocab) = {
            let lb = logits.borrow();
            (lb.r, lb.c)
        };
        let loss = Tape::node(1, 1);
        let mut soft = vec![0.0f32; n * vocab];
        let mut total = 0.0f32;
        let mut cnt = 0usize;
        {
            let lb = logits.borrow();
            for nn in 0..n {
                if targets[nn] < 0 {
                    continue;
                }
                cnt += 1;
                let lr = &lb.data[nn * vocab..nn * vocab + vocab];
                let mut maxs = -1e30f32;
                for &v in lr.iter() {
                    if v > maxs {
                        maxs = v;
                    }
                }
                let mut sum = 0.0f32;
                let sr = &mut soft[nn * vocab..nn * vocab + vocab];
                for v in 0..vocab {
                    let e = ((lr[v] - maxs) as f64).exp() as f32;
                    sr[v] = e;
                    sum += e;
                }
                let inv = 1.0 / sum;
                for v in 0..vocab {
                    sr[v] *= inv;
                }
                total += -(((sr[targets[nn] as usize] as f64) + 1e-12).ln()) as f32;
            }
        }
        if cnt == 0 {
            cnt = 1;
        }
        let avg = total / cnt as f32;
        loss.borrow_mut().data[0] = avg;
        let (lc, losc) = (logits.clone(), loss.clone());
        let tgt: Vec<i32> = targets.to_vec();
        self.push_bw(Box::new(move || {
            let mut lb = lc.borrow_mut();
            let scale = losc.borrow().grad[0] / cnt as f32;
            for nn in 0..n {
                if tgt[nn] < 0 {
                    continue;
                }
                let sr = &soft[nn * vocab..nn * vocab + vocab];
                for v in 0..vocab {
                    let mut g = sr[v];
                    if v == tgt[nn] as usize {
                        g -= 1.0;
                    }
                    lb.grad[nn * vocab + v] += scale * g;
                }
            }
        }));
        (loss, avg)
    }
}

impl Gpt {
    /// PredictAt runs a forward pass and returns the argmax token at the given flat position.
    pub fn predict_at(&self, idx: &[usize], b: usize, t: usize, pos: usize) -> usize {
        let mut tp = Tape::new();
        let logits = self.forward(&mut tp, idx, b, t);
        let lb = logits.borrow();
        let vocab = lb.c;
        let lr = &lb.data[pos * vocab..pos * vocab + vocab];
        let mut best = 0usize;
        let mut bv = -1e30f32;
        for (v, &val) in lr.iter().enumerate() {
            if val > bv {
                bv = val;
                best = v;
            }
        }
        best
    }
}

/// ropeTables precomputes cos/sin for rotary position embeddings: [T][hd/2].
pub(crate) fn rope_tables(t: usize, hd: usize) -> (Vec<f32>, Vec<f32>) {
    let half = hd / 2;
    let mut cos = vec![0.0f32; t * half];
    let mut sin = vec![0.0f32; t * half];
    for pos in 0..t {
        for j in 0..half {
            let freq = 10000f64.powf(-2.0 * j as f64 / hd as f64);
            let ang = pos as f64 * freq;
            cos[pos * half + j] = ang.cos() as f32;
            sin[pos * half + j] = ang.sin() as f32;
        }
    }
    (cos, sin)
}

/// rope rotates the hd-vector src into dst using cos/sin at a position (sign=+1 forward, -1 inverse).
fn rope(dst: &mut [f32], src: &[f32], cos_row: &[f32], sin_row: &[f32], half: usize, sign: f32) {
    for j in 0..half {
        let c = cos_row[j];
        let s = sin_row[j] * sign;
        let a = src[j];
        let b = src[j + half];
        dst[j] = a * c - b * s;
        dst[j + half] = a * s + b * c;
    }
}

/// rot applies RoPE (or a plain copy, if disabled) so forward/backward share one path.
#[allow(clippy::too_many_arguments)]
fn rot(
    dst: &mut [f32],
    src: &[f32],
    pos: usize,
    sign: f32,
    use_rope: bool,
    cos_t: &[f32],
    sin_t: &[f32],
    half: usize,
    hd: usize,
) {
    if use_rope {
        rope(
            dst,
            src,
            &cos_t[pos * half..pos * half + half],
            &sin_t[pos * half..pos * half + half],
            half,
            sign,
        );
    } else {
        dst[..hd].copy_from_slice(&src[..hd]);
    }
}
