"""metis-1m — Night 0 calibration run (doc 13 §6).

Trains a byte-level ~15M-param GPT trunk-let on real local code with MLX, and
measures the ONLY numbers that matter tonight: tokens/s, MFU, and loss descent.
Every budget in doc 13 gets re-derived from this file's JSON output.

Usage:
    python train.py --data corpus.txt --steps 800 --out results-night0.json
"""

import argparse
import json
import math
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
import numpy as np

# ---------------------------------------------------------------- model

class Block(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim)
        self.attn = nn.MultiHeadAttention(dim, n_heads)
        self.norm2 = nn.RMSNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)
        )

    def __call__(self, x, mask):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, mask=mask)
        x = x + self.mlp(self.norm2(x))
        return x


class TrunkLet(nn.Module):
    def __init__(self, vocab: int, dim: int, n_layers: int, n_heads: int, seq: int):
        super().__init__()
        self.tok = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(seq, dim)
        self.blocks = [Block(dim, n_heads) for _ in range(n_layers)]
        self.norm = nn.RMSNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)

    def __call__(self, idx, mask):
        x = self.tok(idx) + self.pos(mx.arange(idx.shape[1]))
        for b in self.blocks:
            x = b(x, mask)
        return self.head(self.norm(x))


def n_params(model) -> int:
    total = 0
    for _, v in tree_flatten(model.parameters()):
        total += v.size
    return total


# ---------------------------------------------------------------- data

def batches(data: np.ndarray, batch: int, seq: int, rng: np.random.Generator):
    while True:
        ix = rng.integers(0, len(data) - seq - 1, size=batch)
        x = np.stack([data[i : i + seq] for i in ix])
        y = np.stack([data[i + 1 : i + seq + 1] for i in ix])
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
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--peak-tflops", type=float, default=5.0,
                    help="assumed fp16 peak of the GPU, for the MFU estimate")
    ap.add_argument("--out", default="results-night0.json")
    ap.add_argument("--save-weights", default=None,
                    help="path (.safetensors) to save trained weights + a .json config next to it")
    args = ap.parse_args()

    raw = np.frombuffer(open(args.data, "rb").read(), dtype=np.uint8)
    n_val = len(raw) // 20
    train_data, val_data = raw[:-n_val], raw[-n_val:]
    print(f"corpus: {len(raw)/1e6:.1f}M bytes ({len(train_data)/1e6:.1f}M train / {n_val/1e6:.1f}M val)")

    model = TrunkLet(256, args.dim, args.layers, args.heads, args.seq)
    mx.eval(model.parameters())
    N = n_params(model)
    print(f"model: {args.layers}L x {args.dim}d x {args.heads}h -> {N/1e6:.1f}M params")

    mask = nn.MultiHeadAttention.create_additive_causal_mask(args.seq)
    opt = optim.AdamW(learning_rate=args.lr, weight_decay=0.01)

    def loss_fn(model, x, y):
        logits = model(x, mask)
        return nn.losses.cross_entropy(logits, y).mean()

    step_fn = nn.value_and_grad(model, loss_fn)

    rng = np.random.default_rng(1337)
    gen = batches(train_data, args.batch, args.seq, rng)
    tok_per_step = args.batch * args.seq

    warmup_steps = min(50, max(1, args.steps // 4))
    losses, t_start, t_lap, warm_tokens = [], time.time(), None, 0
    for step in range(1, args.steps + 1):
        x, y = next(gen)
        loss, grads = step_fn(model, x, y)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state, loss)
        losses.append(loss.item())
        if step == warmup_steps:             # end of warmup: start the clock
            t_lap, warm_tokens = time.time(), step * tok_per_step
        if step % 100 == 0:
            el = time.time() - t_start
            print(f"step {step:4d}  loss {losses[-1]:.3f}  "
                  f"({step * tok_per_step / el / 1e3:.1f}k tok/s raw)")

    total_s = time.time() - t_start
    steady_tokens = args.steps * tok_per_step - warm_tokens
    steady_s = time.time() - t_lap
    tok_s = steady_tokens / steady_s
    flops_s = 6 * N * tok_s
    mfu = flops_s / (args.peak_tflops * 1e12)

    # validation loss
    vgen = batches(val_data, args.batch, args.seq, np.random.default_rng(7))
    vloss = float(np.mean([loss_fn(model, *next(vgen)).item() for _ in range(20)]))

    # a taste of what it learned (greedy sample)
    prompt = np.frombuffer(b"function ", dtype=np.uint8)
    ctx = mx.array(prompt[None, :].astype(np.uint8))
    for _ in range(200):
        pad = args.seq - ctx.shape[1]
        logits = model(ctx, nn.MultiHeadAttention.create_additive_causal_mask(ctx.shape[1]))
        nxt = mx.argmax(logits[0, -1]).item()
        ctx = mx.concatenate([ctx, mx.array([[nxt]])], axis=1)
        if ctx.shape[1] >= args.seq:
            break
    sample = bytes([int(t) for t in np.array(ctx[0])]).decode("utf-8", errors="replace")

    night_tokens = tok_s * 8 * 3600
    results = {
        "params_M": round(N / 1e6, 2),
        "tokens_per_s": round(tok_s),
        "flops_per_s": flops_s,
        "mfu_at_assumed_peak": round(mfu, 3),
        "assumed_peak_tflops": args.peak_tflops,
        "loss_first": round(float(np.mean(losses[:10])), 3),
        "loss_last": round(float(np.mean(losses[-10:])), 3),
        "val_loss": round(vloss, 3),
        "val_bits_per_byte": round(vloss / math.log(2), 3),
        "tokens_trained_M": round(args.steps * tok_per_step / 1e6, 1),
        "wall_clock_min": round(total_s / 60, 1),
        "extrapolated_tokens_per_8h_night_M": round(night_tokens / 1e6),
        "sample": sample,
    }
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    if args.save_weights:
        model.save_weights(args.save_weights)
        cfg = {"vocab": 256, "dim": args.dim, "layers": args.layers,
               "heads": args.heads, "seq": args.seq}
        with open(args.save_weights.replace(".safetensors", ".config.json"), "w") as f:
            json.dump(cfg, f)
        print(f"weights saved -> {args.save_weights}")
    print(json.dumps({k: v for k, v in results.items() if k != "sample"}, indent=2))
    print("--- greedy sample ---")
    print(sample)


if __name__ == "__main__":
    main()
