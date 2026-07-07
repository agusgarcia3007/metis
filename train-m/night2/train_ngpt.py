"""metis-1m — Night 2: nGPT (hypersphere) on top of Night 1's Muon+speedrun stack.

nGPT (Loshchilov et al., NVIDIA) reports 4-20x fewer tokens to a target loss by
putting every representation on the unit hypersphere and replacing residual adds
with a normalized LERP steered by a learned per-dim "eigen learning rate". It is
memory-neutral — no new params of note, no bigger activations — which is exactly
what a RAM-bound MacBook needs.

Open question this script answers: does nGPT COMPOSE with Muon (our 12x champion)
or fight it? We run nGPT + Muon and compare against night1's recorded curve on
the same data/size/steps.

Gentle by default: 14M params, seq 1024, short runs. Run under `nice` if worried.

Usage:
    python train_ngpt.py --data corpus.txt --steps 300 --optim muon
"""

import argparse, json, math, time
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten


def l2(x, axis=-1, eps=1e-6):
    return x * mx.rsqrt(mx.sum(mx.square(x), axis=axis, keepdims=True) + eps)


class Attn(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads, self.hd = n_heads, dim // n_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.rope = nn.RoPE(self.hd)
        self.s_qk = mx.ones((self.hd,))                 # learned QK scale (nGPT)

    def __call__(self, x, mask):
        B, L, D = x.shape
        q = self.q_proj(x).reshape(B, L, self.n_heads, self.hd).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.n_heads, self.hd).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.n_heads, self.hd).transpose(0, 2, 1, 3)
        # unit-norm q,k per head then scale -> cosine attention on the hypersphere
        q = l2(self.rope(q)) * self.s_qk
        k = l2(self.rope(k)) * self.s_qk
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.hd ** 0.5, mask=mask)
        return self.o_proj(o.transpose(0, 2, 1, 3).reshape(B, L, D))


class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w1 = nn.Linear(dim, 4 * dim, bias=False)
        self.w2 = nn.Linear(4 * dim, dim, bias=False)

    def __call__(self, x):
        return self.w2(mx.square(mx.maximum(self.w1(x), 0)))


class Block(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.attn = Attn(dim, n_heads)
        self.mlp = MLP(dim)
        self.a_A = mx.full((dim,), 0.05)                # eigen learning rate (attn)
        self.a_M = mx.full((dim,), 0.05)                # eigen learning rate (mlp)

    def __call__(self, x, mask):
        a = l2(self.attn(x, mask))
        x = l2(x + mx.abs(self.a_A) * (a - x))          # normalized LERP, nGPT residual
        m = l2(self.mlp(x))
        x = l2(x + mx.abs(self.a_M) * (m - x))
        return x


class NGPT(nn.Module):
    def __init__(self, vocab, dim, n_layers, n_heads):
        super().__init__()
        self.tok = nn.Embedding(vocab, dim)
        self.blocks = [Block(dim, n_heads) for _ in range(n_layers)]
        self.head = nn.Linear(dim, vocab, bias=False)
        self.s_z = mx.ones((vocab,))                    # learned logit scale

    def __call__(self, idx, mask):
        x = l2(self.tok(idx))
        for b in self.blocks:
            x = b(x, mask)
        return self.head(x) * self.s_z


# --- hypersphere weight normalization (applied after each optimizer step) ---
def normalize_weights(model):
    updates = []
    for k, v in tree_flatten(model.parameters()):
        if k.endswith("_proj.weight") or k.endswith("w1.weight") or k.endswith("w2.weight"):
            updates.append((k, l2(v, axis=-1)))         # unit-norm each row (input dim)
        elif k == "tok.weight" or k == "head.weight":
            updates.append((k, l2(v, axis=-1)))
    model.update(tree_unflatten(updates))


# ----------------------------------------------------------------- muon
@partial(mx.compile, shapeless=False)
def zeropower(G):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.astype(mx.bfloat16)
    X = X / (mx.linalg.norm(X) + 1e-7)
    t = X.shape[0] > X.shape[1]
    if t: X = X.T
    for _ in range(5):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return (X.T if t else X)


def is_muon(k):
    return k.startswith("blocks") and (k.endswith("_proj.weight") or
                                       k.endswith("w1.weight") or k.endswith("w2.weight"))


def batches(data, batch, seq, rng):
    while True:
        ix = rng.integers(0, len(data) - seq - 1, size=batch)
        x = np.stack([data[i:i+seq] for i in ix])
        y = np.stack([data[i+1:i+seq+1] for i in ix])
        yield mx.array(x), mx.array(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--optim", choices=["muon", "adam"], default="muon")
    ap.add_argument("--muon-lr", type=float, default=0.02)
    ap.add_argument("--adam-lr", type=float, default=3e-3)
    ap.add_argument("--val-every", type=int, default=25)
    ap.add_argument("--out", default="results-ngpt.json")
    args = ap.parse_args()

    raw = np.frombuffer(open(args.data, "rb").read(), dtype=np.uint8)
    n_val = len(raw) // 20
    tr, va = raw[:-n_val], raw[-n_val:]

    model = NGPT(256, args.dim, args.layers, args.heads)
    mx.eval(model.parameters())
    normalize_weights(model)
    N = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(f"nGPT {args.layers}L x {args.dim}d -> {N/1e6:.1f}M params, optim={args.optim}")

    mask = nn.MultiHeadAttention.create_additive_causal_mask(args.seq)
    adam = optim.AdamW(learning_rate=args.adam_lr, weight_decay=0.0)
    muon_mom = {k: mx.zeros_like(v) for k, v in tree_flatten(model.trainable_parameters())
                if is_muon(k)} if args.optim == "muon" else {}

    def loss_fn(model, x, y):
        return nn.losses.cross_entropy(model(x, mask), y).mean()

    lg = nn.value_and_grad(model, loss_fn)

    def step_fn(x, y, mlr):
        loss, grads = lg(model, x, y)
        flat = tree_flatten(grads)
        if args.optim == "muon":
            adam.update(model, tree_unflatten([(k, g) for k, g in flat if not is_muon(k)]))
            params = dict(tree_flatten(model.trainable_parameters()))
            upd = []
            for k, g in flat:
                if not is_muon(k):
                    continue
                buf = 0.95 * muon_mom[k] + g
                muon_mom[k] = buf
                u = zeropower((g + 0.95 * buf).astype(mx.bfloat16))
                sc = max(1.0, g.shape[0] / g.shape[1]) ** 0.5
                upd.append((k, params[k] - (mlr * sc) * u.astype(params[k].dtype)))
            model.update(tree_unflatten(upd))
        else:
            adam.update(model, grads)
        return loss

    def val():
        vg = batches(va, args.batch, args.seq, np.random.default_rng(7))
        return float(np.mean([loss_fn(model, *next(vg)).item() for _ in range(8)]))

    rng = np.random.default_rng(1337)
    gen = batches(tr, args.batch, args.seq, rng)
    tps = args.batch * args.seq

    # night1's recorded byte-level curve (Muon+RMSNorm, same 14M/seq1024) for reference
    n1 = {25: 2.651, 50: 2.412, 100: 1.707, 150: 1.345, 200: 1.195, 300: 0.950}

    t0, vsecs = time.time(), 0.0
    for step in range(1, args.steps + 1):
        mlr = args.muon_lr * min(1.0, step / 50)
        loss = step_fn(*next(gen), mx.array(mlr))
        mx.eval(model.state, adam.state, muon_mom, loss)
        normalize_weights(model)                        # project back to hypersphere
        mx.eval(model.parameters())
        if step % args.val_every == 0:
            tv = time.time(); vl = val(); vsecs += time.time() - tv
            el = time.time() - t0 - vsecs
            ref = n1.get(step)
            tag = f"  (night1 {ref:.3f} -> {'WIN' if vl < ref else 'behind'})" if ref else ""
            print(f"step {step:4d}  val {vl:.3f}  {step*tps/el/1e3:.0f}k tok/s  {el/60:.1f}m{tag}", flush=True)

    total = (time.time() - t0 - vsecs) / 60
    vl = val()
    json.dump({"params_M": round(N/1e6, 2), "optim": args.optim, "steps": args.steps,
               "final_val": round(vl, 3), "wall_min": round(total, 1)}, open(args.out, "w"), indent=2)
    print(json.dumps(json.load(open(args.out)), indent=2))


if __name__ == "__main__":
    main()
