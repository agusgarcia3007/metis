"""metis-1 — plug the real FIM Cortex into the pass@k harness (the honest baseline).

This wires the 14M FIM checkpoint (night2) in as a `Generator` for passk.py. It
uses the model in fill-in-the-middle mode: prefix = the file up to the broken
region, suffix = the file after it, and the model fills the middle. The result is
a candidate the compiler then judges — no self-grading, ever (the Aletheia lesson).

Honest expectation (doc 14 §0): a 14M byte-level model trained ~6.5M tokens will
score near zero. That is the POINT of measuring: this is the floor the generator
has to climb from, and pass@k against the compiler is how we'll know when a better
trained Cortex is actually better — not by vibes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "night1"))
from train import TrunkLet  # the model architecture

NIGHT2 = Path(__file__).resolve().parents[1] / "night2"
PRE, SUF, MID = 256, 257, 258


def load_cortex(weights=NIGHT2 / "metis-fim.safetensors"):
    import json
    cfg = json.load(open(str(weights).replace(".safetensors", ".config.json")))
    m = TrunkLet(cfg["vocab"], cfg["dim"], cfg["layers"], cfg["heads"])
    m.load_weights(str(weights))
    m.apply(lambda p: p.astype(mx.bfloat16))
    m.eval()
    return m, cfg


def make_cortex_generator(temperature=0.7, max_fill=80):
    model, cfg = load_cortex()
    seq = cfg["seq"]
    rng = np.random.default_rng(0)

    def gen(task, i):
        """FIM-fill the function body that was broken, then return the whole file."""
        broken = task.broken
        # crude hole: replace the body of the first function with a FIM hole between
        # the last '{' before an edit region and the matching return line's ';'
        lines = broken.splitlines(keepends=True)
        # choose a hole around the first 'return' line (where our mutations live)
        idx = next((j for j, ln in enumerate(lines) if "return" in ln), len(lines) // 2)
        prefix = "".join(lines[:idx])
        suffix = "".join(lines[idx + 1:])
        ctx = np.concatenate([[PRE], np.frombuffer(prefix.encode(), np.uint8),
                              [SUF], np.frombuffer(suffix.encode(), np.uint8),
                              [MID]]).astype(np.int64)
        ids = list(ctx)[-(seq - max_fill - 1):]
        fill = []
        for _ in range(max_fill):
            c = mx.array(np.array(ids, np.int64)[None])
            mask = nn.MultiHeadAttention.create_additive_causal_mask(len(ids)).astype(mx.bfloat16)
            logits = model(c, mask)[0, -1].astype(mx.float32)
            if temperature <= 0:
                nxt = int(mx.argmax(logits))
            else:
                p = np.array(mx.softmax(logits / temperature))
                nxt = int(rng.choice(len(p), p=p / p.sum()))
            if nxt >= 256:  # sentinel => stop
                break
            fill.append(nxt)
            ids.append(nxt)
        mid = bytes([t for t in fill if t < 256]).decode("utf-8", "replace")
        return prefix + mid + "\n" + suffix

    return gen


if __name__ == "__main__":
    from breaker import make_transitions
    from passk import eval_pass_at_k, gold_generator

    gold = (Path(__file__).parent / "fixture/src/calc.ts").read_text()
    tasks = make_transitions(gold)
    cortex = make_cortex_generator()

    print("first REAL pass@k of the metis Cortex against the TypeScript compiler:\n")
    for k in (1, 4, 8):
        r = eval_pass_at_k(tasks, cortex, k=k)
        print(f"  metis-fim-14M  pass@{k}={r.pass_at_k}  mean_best_score={r.mean_best_score}")
    ref = eval_pass_at_k(tasks, gold_generator, k=1)
    print(f"\n  (reference) gold  pass@1={ref.pass_at_k}")
    print("\nBaseline recorded. This is the floor a better-trained Cortex must beat —")
    print("measured against the compiler, not vibes. That is the whole point of the ruler.")
