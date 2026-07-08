"""metis-1 — Sakana §11.3: does SUPPORT exist? Large-k pass@k, inference only.

Runs the already-trained checkpoints at big k on the harness. No training, no
seq2048, no sustained thermal load (tsc verify gives the GPU cold gaps). The one
question: within a large sample pool, does ANY candidate reach GREEN?

  pass@big > 0  -> support exists; the bottleneck is ranking/localization, not
                  capacity. Search rescues. Cheap, huge news.
  pass@big = 0  -> no support at 14M byte-level; capacity/tokenizer is the wall,
                  now with strong evidence (not N=3, k=8).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from breaker import make_transitions
from passk import eval_pass_at_k

GOLD = (Path(__file__).parent / "fixture/src/calc.ts").read_text()


def run(name, gen_factory, ks=(16, 64, 256)):
    tasks = make_transitions(GOLD)
    gen = gen_factory()
    print(f"\n=== {name} (held-out N={len(tasks)}) ===")
    for k in ks:
        r = eval_pass_at_k(tasks, gen, k=k)
        marker = "  <-- SUPPORT" if r.solved > 0 else ""
        print(f"  pass@{k:<4} = {r.pass_at_k}  (solved {r.solved}/{r.total})  "
              f"best_score={r.mean_best_score}{marker}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    maxk = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    ks = tuple(k for k in (16, 64, 256, 512) if k <= maxk)
    if which in ("both", "repair"):
        from repair_generator import make_repair_generator
        run("metis-repair-14M (whole-file)", lambda: make_repair_generator(temperature=0.9), ks)
    if which in ("both", "edit"):
        from edit_repair import make_edit_generator
        run("metis-edit-14M (one-line)", lambda: make_edit_generator(temperature=0.9), ks)
