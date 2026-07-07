"""metis-1 — pass@k against the compiler (doc 14 §8 step 1, the generator's gate).

The single most important measurement in the project's first phase: given a broken
repo state, can a generator produce a fix that the compiler certifies GREEN within
k samples? If pass@k rises with k, intelligence is being bought with search (VERA-R
§6). If pass@1 is near zero AND pass@k stays flat, the generator is too weak and no
loop rescues it — the exact, now-falsifiable Aletheia failure (doc 14 §0).

A "generator" is any callable: (Transition) -> candidate file content (str). This
harness is model-agnostic: today we test it with a gold generator (proves the
harness detects success) and a noise generator (proves it detects failure); later
the FIM/Muon Cortex plugs in unchanged.
"""

from __future__ import annotations

import dataclasses
from typing import Callable

from breaker import Transition
from verifier import verify_patch

Generator = Callable[[Transition, int], str]  # (task, sample_index) -> new file content


@dataclasses.dataclass
class PassK:
    k: int
    solved: int                   # tasks with >=1 green candidate within k
    total: int
    best_scores: list[float]      # best dense score per task (progress even when not green)

    @property
    def pass_at_k(self) -> float:
        return round(self.solved / self.total, 3) if self.total else 0.0

    @property
    def mean_best_score(self) -> float:
        return round(sum(self.best_scores) / len(self.best_scores), 3) if self.best_scores else 0.0


def eval_pass_at_k(tasks: list[Transition], gen: Generator, k: int) -> PassK:
    solved, best = 0, []
    for task in tasks:
        task_best = 0.0
        hit = False
        for i in range(k):
            candidate = gen(task, i)
            r = verify_patch({task.file: candidate})
            task_best = max(task_best, r.score)
            if r.green:
                hit = True
                break  # first green wins (search short-circuits, VERA-R beam)
        solved += 1 if hit else 0
        best.append(task_best)
    return PassK(k=k, solved=solved, total=len(tasks), best_scores=best)


# --- reference generators (to validate the harness end-to-end) ---
def gold_generator(task: Transition, i: int) -> str:
    """Always emits the correct fix — pass@1 must be 1.0. Proves the oracle accepts truth."""
    return task.gold


def noise_generator(task: Transition, i: int) -> str:
    """Emits junk — pass@k must stay 0.0. Proves the oracle rejects garbage (fab=0)."""
    return f"export const broken_{i} = ;\n"


def stuck_generator(task: Transition, i: int) -> str:
    """Returns the broken code unchanged — pass@k = 0, but score = the broken baseline.

    Models the honest floor: a generator that does nothing makes no progress.
    """
    return task.broken


if __name__ == "__main__":
    from pathlib import Path
    from breaker import make_transitions

    gold = (Path(__file__).parent / "fixture/src/calc.ts").read_text()
    tasks = make_transitions(gold)
    print(f"tasks: {len(tasks)}\n")
    for name, g in [("gold", gold_generator), ("noise", noise_generator), ("stuck", stuck_generator)]:
        r = eval_pass_at_k(tasks, g, k=4)
        print(f"{name:6s} generator  pass@4={r.pass_at_k}  mean_best_score={r.mean_best_score}")
