"""metis-1 — Experiment A: the repair-action ranker (Sakana §2.1, §5, §7 step 3).

"Train a small action ranker/value model FIRST. If ranking does not work,
generation will not magically work." — Sakana

A tiny logistic ranker scores each typed edit candidate's P(green) from STATIC
features (no compiler). We then verify candidates in ranked order and measure how
fast we reach green — vs random order and vs a hand-heuristic. This is the cheapest
possible test of the whole repair-lattice thesis, and it runs in seconds on CPU.

Pre-registered metrics (on held-out calc.ts):
  - Top-1 / Top-8 green rate  (does the top-ranked action pass?)
  - MRR of the first green action
  - verifier_calls_to_first_green  (rank position of the first green)
Baselines: random order; heuristic = op-matches-diagnostic-family, then minimal edit.
Pass criterion (Sakana): learned ranker beats heuristic by >=25% relative, OR cuts
verifier_calls_to_first_green by >=30% at equal solved rate.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from breaker import make_transitions
from lattice import build_lattice, features

HERE = Path(__file__).parent
FEATS = ["op_match_diag", "type_from_diag", "is_strip_typo", "is_null_guard",
         "edit_minimality", "name_sim", "bias"]


def to_vec(feat: dict) -> np.ndarray:
    return np.array([feat[k] for k in FEATS], np.float32)


def train_logistic(lattices, epochs=300, lr=0.5):
    X, y = [], []
    for lat in lattices:
        for a in lat["actions"]:
            if "success" not in a:
                continue
            X.append(to_vec(a["features"])); y.append(1.0 if a["success"] else 0.0)
    X, y = np.array(X), np.array(y)
    w = np.zeros(len(FEATS), np.float32)
    for _ in range(epochs):
        p = 1.0 / (1.0 + np.exp(-(X @ w)))
        w -= lr * (X.T @ (p - y)) / len(y)
    return w


def score(w, feat):
    return float(1.0 / (1.0 + math.exp(-(to_vec(feat) @ w))))


# --- rankers to compare ---
def rank_learned(w, actions):
    return sorted(actions, key=lambda a: -score(w, a["features"]))


def rank_random(actions, rng):
    a = list(actions); rng.shuffle(a); return a


def rank_heuristic(actions):
    # op matches diagnostic family first, then most minimal edit
    return sorted(actions, key=lambda a: (-a["features"]["op_match_diag"]
                                          - 0.5 * a["features"]["type_from_diag"],
                                          -a["features"]["edit_minimality"]))


def eval_order(order):
    """Return (solved, calls_to_first_green, reciprocal_rank, top1, top8)."""
    for i, a in enumerate(order):
        if a.get("success"):
            return 1, i + 1, 1.0 / (i + 1), 1 if i == 0 else 0, 1 if i < 8 else 0
    return 0, len(order) + 1, 0.0, 0, 0


def main():
    lat_path = HERE / "data" / "lattices.jsonl"
    import json as _j
    train = _j.load(open(HERE/'data'/'lattices_fixed.json'))
    w = train_logistic(train)
    print("learned feature weights:")
    for k, v in zip(FEATS, w):
        print(f"  {k:18s} {v:+.3f}")

    # held-out: calc.ts, never in training lattices
    gold = (HERE / "fixture/src/calc.ts").read_text()
    held = [build_lattice(t.broken, t.diagnostic) for t in make_transitions(gold)]
    rng = np.random.default_rng(0)

    def summarize(name, orders):
        s = np.array([eval_order(o) for o in orders], float)
        solved, calls, rr, t1, t8 = s.sum(0)
        n = len(orders)
        print(f"  {name:10s} top1={t1/n:.2f}  top8={t8/n:.2f}  MRR={rr/n:.3f}  "
              f"calls_to_green={calls/n:.1f}  solved={int(solved)}/{n}")

    print("\nExperiment A — ranking the typed edit lattice on HELD-OUT calc.ts:")
    summarize("random", [rank_random(l["actions"], rng) for l in held])
    summarize("heuristic", [rank_heuristic(l["actions"]) for l in held])
    summarize("learned", [rank_learned(w, l["actions"]) for l in held])
    print("\n(a perfect ranker = top1 1.00, calls_to_green 1.0; the generator scored pass@1=0.0)")


if __name__ == "__main__":
    main()
