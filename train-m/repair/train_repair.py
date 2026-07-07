"""metis-1 — specialize the Cortex on repair transitions (warm-start from FIM).

Loads the night2 FIM checkpoint (already knows TS syntax) and continues training
on the mined `<state>broken</state><diagnostic>...</diagnostic><edit>gold</edit>`
sequences, with next-token loss weighted higher on the <edit> span (that is the
skill: read the diagnostic, emit the fix — VERA-R §2, Agents-A1 KAG).

Gentle: 14M params, Muon, short, `nice`-friendly. This is a smoke of the RIGHT
data shape, not the scale run — honest expectations in the README.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "night1"))
from train import TrunkLet, zeropower, is_muon_key

NIGHT2 = Path(__file__).resolve().parents[1] / "night2"
DATA = Path(__file__).parent / "data" / "transitions.jsonl"
EDIT_TAG = b"<edit>"


def load_sequences(seq_len):
    rows = [json.loads(l) for l in open(DATA)]
    seqs, wmasks = [], []
    for r in rows:
        b = r["sequence"].encode()[:seq_len]
        arr = np.frombuffer(b, np.uint8).astype(np.int64)
        if len(arr) < 16:
            continue
        # weight <edit> span 3x (the fix is the skill), everything else 1x
        w = np.ones(len(arr), np.float32)
        ei = r["sequence"].encode().find(EDIT_TAG)
        if 0 <= ei < len(arr):
            w[ei:] = 3.0
        seqs.append(arr)
        wmasks.append(w)
    return seqs, wmasks, rows


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    seq_len, batch = 512, 8
    import json as _j
    cfg = _j.load(open(NIGHT2 / "metis-fim.config.json"))
    model = TrunkLet(cfg["vocab"], cfg["dim"], cfg["layers"], cfg["heads"])
    model.load_weights(str(NIGHT2 / "metis-fim.safetensors"))   # warm start: knows TS syntax
    model.apply(lambda p: p.astype(mx.bfloat16))
    mx.eval(model.parameters())
    N = sum(v.size for _, v in tree_flatten(model.parameters()))

    seqs, wmasks, rows = load_sequences(seq_len)
    print(f"repair trainer: {N/1e6:.1f}M params (warm from FIM), {len(seqs)} transitions, {steps} steps")

    mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len).astype(mx.bfloat16)
    adam = optim.AdamW(learning_rate=3e-4, weight_decay=0.01)
    mom = {k: mx.zeros_like(v) for k, v in tree_flatten(model.trainable_parameters()) if is_muon_key(k)}
    rng = np.random.default_rng(0)

    def batch_iter():
        while True:
            idx = rng.integers(0, len(seqs), size=batch)
            xs, ys, ws = [], [], []
            for i in idx:
                a, w = seqs[i], wmasks[i]
                if len(a) < seq_len + 1:
                    pad = seq_len + 1 - len(a)
                    a = np.concatenate([a, np.zeros(pad, np.int64)])
                    w = np.concatenate([w, np.zeros(pad, np.float32)])
                xs.append(a[:seq_len]); ys.append(a[1:seq_len + 1]); ws.append(w[1:seq_len + 1])
            yield mx.array(np.stack(xs)), mx.array(np.stack(ys)), mx.array(np.stack(ws))

    def loss_fn(model, x, y, w):
        logits = model(x, mask).astype(mx.float32)
        ce = nn.losses.cross_entropy(logits, y, reduction="none")
        return (ce * w).sum() / (w.sum() + 1e-6)

    lg = nn.value_and_grad(model, loss_fn)

    def step_fn(x, y, w, mlr):
        loss, grads = lg(model, x, y, w)
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

    gen = batch_iter()
    t0 = time.time()
    for step in range(1, steps + 1):
        mlr = 0.01 * min(1.0, step / 40)
        loss = step_fn(*next(gen), mx.array(mlr))
        mx.eval(model.state, adam.state, mom, loss)
        if step % 50 == 0:
            print(f"step {step:4d}  loss {loss.item():.3f}  {(time.time()-t0)/60:.1f}m", flush=True)

    out = Path(__file__).parent / "metis-repair.safetensors"
    model.save_weights(str(out))
    _j.dump(cfg, open(str(out).replace(".safetensors", ".config.json"), "w"))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
