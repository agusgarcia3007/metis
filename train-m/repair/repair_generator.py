"""metis-1 — repair generator matching the training format (state+diagnostic -> edit).

Prompts the repair-specialized Cortex with the SAME sequence shape it was trained
on: <state>broken</state><diagnostic>compiler error</diagnostic><edit>\n , then
generates the fixed file until </edit>. The candidate is judged only by the
compiler (never self-graded). Plug into passk.py exactly like any generator.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "night1"))
from train import TrunkLet

HERE = Path(__file__).parent
END = b"</edit>"


def make_repair_generator(weights=HERE / "metis-repair.safetensors", temperature=0.6, max_new=200):
    import json
    cfg = json.load(open(str(weights).replace(".safetensors", ".config.json")))
    model = TrunkLet(cfg["vocab"], cfg["dim"], cfg["layers"], cfg["heads"])
    model.load_weights(str(weights))
    model.apply(lambda p: p.astype(mx.bfloat16))
    model.eval()
    seq = cfg["seq"]
    rng = np.random.default_rng(0)

    def gen(task, i):
        prompt = (f"<state>\nfile: {task.file}\n{task.broken}\n</state>\n"
                  f"<diagnostic>\n{task.diagnostic.strip()[:400]}\n</diagnostic>\n<edit>\n")
        ids = list(np.frombuffer(prompt.encode(), np.uint8).astype(np.int64))[-(seq - max_new - 1):]
        out = []
        for _ in range(max_new):
            c = mx.array(np.array(ids, np.int64)[None])
            m = nn.MultiHeadAttention.create_additive_causal_mask(len(ids)).astype(mx.bfloat16)
            logits = model(c, m)[0, -1].astype(mx.float32)
            if temperature <= 0:
                nxt = int(mx.argmax(logits))
            else:
                p = np.array(mx.softmax(logits / temperature))
                nxt = int(rng.choice(len(p), p=p / p.sum()))
            if nxt >= 256:
                continue
            out.append(nxt)
            ids.append(nxt)
            if bytes(out[-len(END):]) == END:
                out = out[:-len(END)]
                break
        return bytes(out).decode("utf-8", "replace").strip()

    return gen


if __name__ == "__main__":
    from breaker import make_transitions
    from passk import eval_pass_at_k, gold_generator

    gold = (HERE / "fixture/src/calc.ts").read_text()
    tasks = make_transitions(gold)   # HELD-OUT: calc.ts, never in training
    g = make_repair_generator()
    print("repair-specialized Cortex, pass@k on HELD-OUT calc.ts (vs pre-training 0.0):\n")
    for k in (1, 4, 8):
        r = eval_pass_at_k(tasks, g, k=k)
        print(f"  metis-repair-14M  pass@{k}={r.pass_at_k}  mean_best_score={r.mean_best_score}")
    print(f"\n  (reference) gold  pass@1={eval_pass_at_k(tasks, gold_generator, 1).pass_at_k}")
