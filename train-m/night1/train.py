"""metis-1m — Night 1 trainer: the NanoGPT-speedrun lessons, ported to MLX.

Changes vs night0 (each one sourced from the modded-nanogpt speedrun records):
  - Muon optimizer for the block matrices (orthogonalized momentum, ~2x vs AdamW),
    AdamW only for embeddings / head / norms
  - bf16 weights + compute (fp32 cross-entropy)
  - mx.compile'd training step
  - RoPE instead of learned positions, QK-norm, ReLU^2 MLP,
    zero-init output projections, untied head, logit soft-capping

Pre-registered metric: wall-clock time to reach night0's 16-minute validation
loss (2.618). That ratio is the honest speedup.

Usage:
    python train.py --data corpus.txt --steps 800 --target-val 2.618
"""

import argparse
import json
import math
import time
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

# ---------------------------------------------------------------- model

def rms(x, eps=1e-6):
    return x * mx.rsqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + eps)


class Attn(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.hd = dim // n_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.rope = nn.RoPE(self.hd)

    def __call__(self, x, mask):
        B, L, D = x.shape
        q = self.q_proj(x).reshape(B, L, self.n_heads, self.hd).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.n_heads, self.hd).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.n_heads, self.hd).transpose(0, 2, 1, 3)
        q, k = rms(self.rope(q)), rms(self.rope(k))          # QK-norm after RoPE
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.hd ** -0.5, mask=mask)
        return self.o_proj(o.transpose(0, 2, 1, 3).reshape(B, L, D))


class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, 4 * dim, bias=False)
        self.w2 = nn.Linear(4 * dim, dim, bias=False)

    def __call__(self, x):
        return self.w2(mx.square(mx.maximum(self.w1(x), 0)))  # ReLU^2


class Block(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim)
        self.attn = Attn(dim, n_heads)
        self.norm2 = nn.RMSNorm(dim)
        self.mlp = MLP(dim)

    def __call__(self, x, mask):
        x = x + self.attn(self.norm1(x), mask)
        return x + self.mlp(self.norm2(x))


class TrunkLet(nn.Module):
    def __init__(self, vocab: int, dim: int, n_layers: int, n_heads: int):
        super().__init__()
        self.tok = nn.Embedding(vocab, dim)
        self.blocks = [Block(dim, n_heads) for _ in range(n_layers)]
        self.norm = nn.RMSNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)          # untied

    def __call__(self, idx, mask):
        x = self.tok(idx)
        for b in self.blocks:
            x = b(x, mask)
        logits = self.head(self.norm(x))
        return 15.0 * mx.tanh(logits / 15.0)                   # logit soft-cap


# ---------------------------------------------------------------- muon

@partial(mx.compile, shapeless=False)
def zeropower(G):
    """Newton-Schulz orthogonalization (5 steps, bf16) — the heart of Muon."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.astype(mx.bfloat16)
    X = X / (mx.linalg.norm(X) + 1e-7)
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T
    for _ in range(5):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    if transposed:
        X = X.T
    return X


def is_muon_key(k: str) -> bool:
    return k.startswith("blocks") and (k.endswith("_proj.weight") or
                                       k.endswith("w1.weight") or k.endswith("w2.weight"))


# ---------------------------------------------------------------- data

def batches(data, batch, seq, rng):
    while True:
        ix = rng.integers(0, len(data) - seq - 1, size=batch)
        x = np.stack([data[i: i + seq] for i in ix])
        y = np.stack([data[i + 1: i + seq + 1] for i in ix])
        yield mx.array(x), mx.array(y)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--muon-lr", type=float, default=0.02)
    ap.add_argument("--adam-lr", type=float, default=3e-4)
    ap.add_argument("--target-val", type=float, default=2.618,
                    help="night0's 16-minute val loss — the speedup reference")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--val-every", type=int, default=50)
    ap.add_argument("--out", default="results-night1.json")
    ap.add_argument("--save-weights", default=None)
    args = ap.parse_args()

    raw = np.frombuffer(open(args.data, "rb").read(), dtype=np.uint8)
    n_val = len(raw) // 20
    train_data, val_data = raw[:-n_val], raw[-n_val:]

    model = TrunkLet(256, args.dim, args.layers, args.heads)
    # zero-init output projections (speedrun trick: blocks start as identity)
    zeros = [(f"blocks.{i}.attn.o_proj.weight", mx.zeros_like(b.attn.o_proj.weight))
             for i, b in enumerate(model.blocks)]
    zeros += [(f"blocks.{i}.mlp.w2.weight", mx.zeros_like(b.mlp.w2.weight))
              for i, b in enumerate(model.blocks)]
    model.update(tree_unflatten(zeros))
    model.apply(lambda p: p.astype(mx.bfloat16))
    mx.eval(model.parameters())
    N = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(f"model: {args.layers}L x {args.dim}d x {args.heads}h -> {N/1e6:.1f}M params (bf16)")

    mask = nn.MultiHeadAttention.create_additive_causal_mask(args.seq).astype(mx.bfloat16)
    adam = optim.AdamW(learning_rate=args.adam_lr, weight_decay=0.01)
    muon_mom = {k: mx.zeros_like(v) for k, v in tree_flatten(model.trainable_parameters())
                if is_muon_key(k)}

    def loss_fn(model, x, y):
        logits = model(x, mask).astype(mx.float32)
        return nn.losses.cross_entropy(logits, y).mean()

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    def train_step(x, y, muon_lr):
        loss, grads = loss_and_grad(model, x, y)
        flat = tree_flatten(grads)
        adam.update(model, tree_unflatten([(k, g) for k, g in flat if not is_muon_key(k)]))
        params = dict(tree_flatten(model.trainable_parameters()))
        upd = []
        for k, g in flat:
            if not is_muon_key(k):
                continue
            buf = 0.95 * muon_mom[k] + g
            muon_mom[k] = buf
            u = zeropower((g + 0.95 * buf).astype(mx.bfloat16))   # nesterov
            scale = max(1.0, g.shape[0] / g.shape[1]) ** 0.5
            upd.append((k, params[k] - (muon_lr * scale) * u.astype(params[k].dtype)))
        model.update(tree_unflatten(upd))
        return loss

    if not args.no_compile:
        state = [model.state, adam.state, muon_mom]
        train_step = mx.compile(train_step, inputs=state, outputs=state)

    def val_loss(k_batches=8):
        vgen = batches(val_data, args.batch, args.seq, np.random.default_rng(7))
        return float(np.mean([loss_fn(model, *next(vgen)).item() for _ in range(k_batches)]))

    rng = np.random.default_rng(1337)
    gen = batches(train_data, args.batch, args.seq, rng)
    tok_per_step = args.batch * args.seq
    warmup = 50

    losses, t0, hit_target_min, val_secs = [], time.time(), None, 0.0
    for step in range(1, args.steps + 1):
        lr = args.muon_lr * min(1.0, step / warmup)
        loss = train_step(*next(gen), mx.array(lr))
        mx.eval(model.state, adam.state, muon_mom, loss)
        losses.append(loss.item())
        if step % args.val_every == 0:
            tv = time.time()
            vl = val_loss()
            val_secs += time.time() - tv
            train_el = time.time() - t0 - val_secs
            print(f"step {step:4d}  train {losses[-1]:.3f}  val {vl:.3f}  "
                  f"{step*tok_per_step/train_el/1e3:.1f}k tok/s  {train_el/60:.1f} min")
            if hit_target_min is None and vl <= args.target_val:
                hit_target_min = train_el / 60
                print(f"*** target val {args.target_val} reached in {hit_target_min:.1f} min "
                      f"of training (night0: 16.0 min -> {16.0/hit_target_min:.1f}x) ***")

    total_min = (time.time() - t0 - val_secs) / 60
    vl = val_loss(20)
    tok_s = args.steps * tok_per_step / (total_min * 60)
    results = {
        "params_M": round(N / 1e6, 2),
        "tokens_per_s": round(tok_s),
        "mfu_at_5tflops": round(6 * N * tok_s / 5e12, 3),
        "final_val_loss": round(vl, 3),
        "final_bits_per_byte": round(vl / math.log(2), 3),
        "tokens_trained_M": round(args.steps * tok_per_step / 1e6, 1),
        "wall_clock_min": round(total_min, 1),
        "time_to_night0_target_min": round(hit_target_min, 1) if hit_target_min else None,
        "speedup_vs_night0": round(16.0 / hit_target_min, 1) if hit_target_min else None,
    }
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    if args.save_weights:
        model.save_weights(args.save_weights)
        cfg = {"vocab": 256, "dim": args.dim, "layers": args.layers,
               "heads": args.heads, "seq": args.seq, "arch": "night1"}
        with open(args.save_weights.replace(".safetensors", ".config.json"), "w") as f:
            json.dump(cfg, f)
        print(f"weights saved -> {args.save_weights}")


if __name__ == "__main__":
    main()
