"""metis-1 — the >=100-task frozen ranker eval (Sakana §11.1, §5 Experiment A).

N=3 (calc.ts) can't separate a learned ranker from a heuristic — both ceiling. This
builds a real held-out eval of many broken states whose FUNCTIONS are disjoint from
training, then measures random vs heuristic vs learned ranking over the typed-edit
lattice: coverage (does the action space contain a fix?), top-1 / top-8 success,
MRR, and verifier_calls_to_first_green — aggregate and per diagnostic family.

    python eval_ranker.py build 140     # build held-out lattices (slow: tsc per candidate)
    python eval_ranker.py run           # train on lattices_fixed.json, eval on held-out
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

from lattice import build_lattice
from ranker import train_logistic, rank_learned, rank_heuristic, rank_random, eval_order

HERE = Path(__file__).parent
TRAIN = HERE / "data" / "lattices_fixed.json"
HELD = HERE / "data" / "heldout_lattices.json"
TRANS = HERE / "data" / "transitions.jsonl"
FNAME = re.compile(r"export function (\w+)")
N_TRAIN = 120   # lattices_fixed.json was built from transitions.jsonl[:120]


def fname(src: str) -> str:
    m = FNAME.search(src or "")
    return m.group(1) if m else "?"


def train_function_names():
    rows = [json.loads(l) for l in open(TRANS)][:N_TRAIN]
    return {fname(r["gold"]) for r in rows}


def build_heldout(n=140):
    from lattice import diag_family
    rows = [json.loads(l) for l in open(TRANS)]
    exclude = train_function_names()          # real function names used in training
    per_fam = max(1, n // 2)                   # balance TS2322 (easy) vs TS2304 (hard)
    counts, seen, held = {}, set(), []
    for r in rows[N_TRAIN:]:                   # disjoint index slice
        fn = fname(r["gold"])
        if fn in exclude or fn in seen:
            continue
        fam = diag_family(r["diagnostic"])
        if counts.get(fam, 0) >= per_fam:
            continue                            # this family is full — keep balance
        seen.add(fn)
        lat = build_lattice(r["broken"], r["diagnostic"])
        if not lat["actions"]:
            continue
        lat["function"] = fn
        held.append(lat)
        counts[fam] = counts.get(fam, 0) + 1
        if len(held) % 25 == 0:
            print(f"  built {len(held)} held-out lattices {counts}", flush=True)
        if len(held) >= n:
            break
    json.dump(held, open(HELD, "w"))
    cov = sum(1 for l in held if any(a.get("success") for a in l["actions"]))
    print(f"built {len(held)} held-out lattices {counts} ({len(exclude)} train fns excluded); "
          f"action-space coverage (has a fix): {cov}/{len(held)}")


def summarize(name, lattices, order_fn):
    rows = np.array([eval_order(order_fn(l["actions"])) for l in lattices], float)
    # only score tasks whose action space actually contains a fix (coverage-conditioned)
    solvable = [l for l in lattices if any(a.get("success") for a in l["actions"])]
    rows_s = np.array([eval_order(order_fn(l["actions"])) for l in solvable], float)
    n, ns = len(lattices), len(solvable)
    solved, calls, rr, t1, t8 = rows.sum(0)
    _, _, rr_s, t1_s, t8_s = rows_s.sum(0)
    print(f"  {name:10s} solved={int(solved)}/{n} ({solved/n:.2f})  "
          f"| on solvable(n={ns}): top1={t1_s/ns:.2f} top8={t8_s/ns:.2f} "
          f"MRR={rr_s/ns:.3f} calls={rows_s[:,1].mean():.2f}")


def run():
    train = json.load(open(TRAIN))
    held = json.load(open(HELD))
    w = train_logistic(train)
    rng = np.random.default_rng(0)
    from collections import Counter
    fams = Counter(l["diag_family"] for l in held)
    print(f"held-out: {len(held)} tasks, families {dict(fams)}\n")
    print("Ranker comparison (Sakana Experiment A, coverage-conditioned):")
    summarize("random", held, lambda a: rank_random(a, rng))
    summarize("heuristic", held, rank_heuristic)
    summarize("learned", held, lambda a: rank_learned(w, a))
    # per-family for the learned ranker
    print("\nlearned ranker, per family:")
    for fam in fams:
        sub = [l for l in held if l["diag_family"] == fam and any(a.get("success") for a in l["actions"])]
        if not sub:
            continue
        r = np.array([eval_order(rank_learned(w, l["actions"])) for l in sub], float)
        print(f"  {fam:8s} n={len(sub):3d}  top1={r[:,3].mean():.2f}  "
              f"calls_to_green={r[:,1].mean():.2f}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "build":
        build_heldout(int(sys.argv[2]) if len(sys.argv) > 2 else 140)
    else:
        run()
