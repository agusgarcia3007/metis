"""metis-1m — Night 2: Fill-in-the-Middle (FIM), the code-specific edge.

nGPT and fancier optimizers did NOT beat our Night-1 Muon champion at this scale
(measured: both trailed night1's curve). The real untapped lever for a *code*
model isn't the optimizer — it's the OBJECTIVE. Every serious code model
(StarCoder, DeepSeek-Coder, CodeLlama) trains with FIM at ~50%.

FIM reshapes a document into prefix / suffix / middle with sentinel tokens so the
model learns to fill a hole given BOTH sides — which is exactly what an editor
(OpenCode) asks of it, not left-to-right continuation. It is free (a data
transform), memory-neutral, and composes with Muon + BPE.

This trainer = Night-1's model & Muon, + a FIM batch generator (PSM mode).
Vocab is byte-level + 3 sentinels (256=<pre>, 257=<suf>, 258=<mid>).

Gentle by default (14M, seq 1024, short). Run under `nice` if worried.

    python train_fim.py --data corpus.txt --steps 400 --fim-rate 0.5
"""

import argparse, json, sys, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

sys.path.insert(0, __file__.rsplit("/", 2)[0] + "/night1")
from train import TrunkLet, zeropower, is_muon_key   # reuse the champion model + Muon

PRE, SUF, MID = 256, 257, 258
VOCAB = 259


def fim_transform(chunk, rng):
    """PSM: [<pre> prefix <suf> suffix <mid> middle] — predict the middle."""
    n = len(chunk)
    a, b = sorted(rng.integers(1, n - 1, size=2))
    prefix, middle, suffix = chunk[:a], chunk[a:b], chunk[b:]
    return np.concatenate([[PRE], prefix, [SUF], suffix, [MID], middle]).astype(np.int64)


def fim_batches(data, batch, seq, fim_rate, rng):
    """Half the sequences are FIM-reordered, half are plain next-token."""
    while True:
        xs, ys = [], []
        for _ in range(batch):
            i = rng.integers(0, len(data) - seq - 2)
            if rng.random() < fim_rate:
                # take a slightly shorter window, FIM-reorder, then pad/trim to seq+1
                raw = data[i:i + seq].astype(np.int64)
                s = fim_transform(raw, rng)
                if len(s) >= seq + 1:
                    s = s[:seq + 1]
                else:
                    pad = data[i + seq:i + seq + (seq + 1 - len(s))].astype(np.int64)
                    s = np.concatenate([s, pad])[:seq + 1]
            else:
                s = data[i:i + seq + 1].astype(np.int64)
            xs.append(s[:seq]); ys.append(s[1:seq + 1])
        yield mx.array(np.stack(xs)), mx.array(np.stack(ys))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--fim-rate", type=float, default=0.5)
    ap.add_argument("--muon-lr", type=float, default=0.02)
    ap.add_argument("--adam-lr", type=float, default=3e-4)
    ap.add_argument("--val-every", type=int, default=50)
    ap.add_argument("--out", default="results-fim.json")
    ap.add_argument("--save-weights", default=None)
    args = ap.parse_args()

    raw = np.frombuffer(open(args.data, "rb").read(), dtype=np.uint8)
    n_val = len(raw) // 20
    tr, va = raw[:-n_val], raw[-n_val:]

    model = TrunkLet(VOCAB, args.dim, args.layers, args.heads)
    model.apply(lambda p: p.astype(mx.bfloat16))
    mx.eval(model.parameters())
    N = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(f"FIM trainer: {args.layers}L x {args.dim}d -> {N/1e6:.1f}M params, fim_rate={args.fim_rate}")

    mask = nn.MultiHeadAttention.create_additive_causal_mask(args.seq).astype(mx.bfloat16)
    adam = optim.AdamW(learning_rate=args.adam_lr, weight_decay=0.01)
    mom = {k: mx.zeros_like(v) for k, v in tree_flatten(model.trainable_parameters()) if is_muon_key(k)}

    def loss_fn(model, x, y):
        return nn.losses.cross_entropy(model(x, mask).astype(mx.float32), y).mean()

    lg = nn.value_and_grad(model, loss_fn)

    def step_fn(x, y, mlr):
        loss, grads = lg(model, x, y)
        flat = tree_flatten(grads)
        adam.update(model, tree_unflatten([(k, g) for k, g in flat if not is_muon_key(k)]))
        params = dict(tree_flatten(model.trainable_parameters()))
        upd = []
        for k, g in flat:
            if not is_muon_key(k):
                continue
            buf = 0.95 * mom[k] + g
            mom[k] = buf
            u = zeropower((g + 0.95 * buf).astype(mx.bfloat16))
            sc = max(1.0, g.shape[0] / g.shape[1]) ** 0.5
            upd.append((k, params[k] - (mlr * sc) * u.astype(params[k].dtype)))
        model.update(tree_unflatten(upd))
        return loss

    def val(gen):
        return float(np.mean([loss_fn(model, *next(gen)).item() for _ in range(8)]))

    rng = np.random.default_rng(1337)
    gen = fim_batches(tr, args.batch, args.seq, args.fim_rate, rng)
    vgen = fim_batches(va, args.batch, args.seq, args.fim_rate, np.random.default_rng(7))
    tps = args.batch * args.seq

    t0, vsecs = time.time(), 0.0
    for step in range(1, args.steps + 1):
        mlr = args.muon_lr * min(1.0, step / 50)
        loss = step_fn(*next(gen), mx.array(mlr))
        mx.eval(model.state, adam.state, mom, loss)
        if step % args.val_every == 0:
            tv = time.time(); vl = val(vgen); vsecs += time.time() - tv
            el = time.time() - t0 - vsecs
            print(f"step {step:4d}  train {loss.item():.3f}  val(FIM) {vl:.3f}  "
                  f"{step*tps/el/1e3:.0f}k tok/s  {el/60:.1f}m", flush=True)

    total = (time.time() - t0 - vsecs) / 60
    # --- infill demo: give a hole, ask the model to fill it ---
    prefix = b"function add(a: number, b: number): number {\n  return "
    suffix = b";\n}\n"
    ctx = np.concatenate([[PRE], np.frombuffer(prefix, np.uint8),
                          [SUF], np.frombuffer(suffix, np.uint8), [MID]]).astype(np.int64)
    ids = list(ctx)
    for _ in range(20):
        c = mx.array(np.array(ids, np.int64)[None])
        m = nn.MultiHeadAttention.create_additive_causal_mask(len(ids)).astype(mx.bfloat16)
        nxt = int(mx.argmax(model(c, m)[0, -1]))
        if nxt >= 256:
            break
        ids.append(nxt)
    fill = bytes([t for t in ids[len(ctx):] if t < 256]).decode("utf-8", "replace")
    print(f"\nINFILL demo — prefix `{prefix.decode()}` ... suffix `;}}`")
    print(f"  model fills the middle -> `{fill}`")

    json.dump({"params_M": round(N/1e6, 2), "fim_rate": args.fim_rate, "steps": args.steps,
               "final_val_fim": round(val(vgen), 3), "wall_min": round(total, 1),
               "infill": fill}, open(args.out, "w"), indent=2)
    if args.save_weights:
        model.save_weights(args.save_weights)
        json.dump({"vocab": VOCAB, "dim": args.dim, "layers": args.layers,
                   "heads": args.heads, "seq": args.seq, "fim": True},
                  open(args.save_weights.replace(".safetensors", ".config.json"), "w"))
        print(f"saved -> {args.save_weights}")


if __name__ == "__main__":
    main()
