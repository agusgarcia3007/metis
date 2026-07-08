"""metis-1 — repair-lattice generator (Sakana §1, §7 step 2).

For each broken state: enumerate typed edit candidates (editops.py), verify each
with the compiler, and record the full delta. One seed becomes a labeled lattice
of (state, action) -> verifier-outcome — the dense training substrate that replaces
the one-shot (broken -> gold) pair.

Also extracts STATIC features per candidate (no compiler needed at inference), so a
ranker can order candidates and we verify the promising ones first — minimizing
verifier_calls_to_first_green (Experiment A).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from editops import candidates, TS2322, TS2304, DIAG_LINE
from verifier import verify_patch

OPS = ["set_return_type", "set_return_type_from_diag", "replace_identifier",
       "strip_typo", "swap_binary_op", "add_null_guard"]


def diag_family(diag: str) -> str:
    if TS2322.search(diag):
        return "TS2322"
    if TS2304.search(diag):
        return "TS2304"
    if "TS" not in diag or "expect" in diag or "toBe" in diag:
        return "TEST"
    return "OTHER"


def features(broken: str, diagnostic: str, op: str, span: str, cand: str) -> dict:
    """Static features (computable without running the compiler)."""
    fam = diag_family(diagnostic)
    # does this op family match this diagnostic family? (the load-bearing signal)
    match = ((op.startswith("set_return_type") and fam == "TS2322") or
             (op in ("replace_identifier", "strip_typo") and fam == "TS2304") or
             (op == "swap_binary_op" and fam == "TEST"))
    # for return-type ops: does the chosen type equal the type named in the diagnostic?
    type_from_diag = 0.0
    if op == "set_return_type_from_diag":
        type_from_diag = 1.0
    elif op == "set_return_type" and (m := TS2322.search(diagnostic)):
        chosen = span.split(":")[-1]
        type_from_diag = 1.0 if chosen.replace(" ", "") == m.group(1).replace(" ", "") else 0.0
    edit_size = sum(1 for a, b in zip(broken, cand) if a != b) + abs(len(broken) - len(cand))
    return {
        "op_match_diag": 1.0 if match else 0.0,
        "type_from_diag": type_from_diag,
        "is_strip_typo": 1.0 if op == "strip_typo" else 0.0,
        "is_null_guard": 1.0 if op == "add_null_guard" else 0.0,
        "edit_minimality": 1.0 / (1.0 + edit_size / 10.0),
        "bias": 1.0,
    }


def build_lattice(broken: str, diagnostic: str, file: str = "src/calc.ts", verify=True):
    fam = diag_family(diagnostic)
    lat = []
    for op, span, cand in candidates(broken, diagnostic):
        feat = features(broken, diagnostic, op, span, cand)
        rec = {"op": op, "span": span, "candidate": cand, "features": feat}
        if verify:
            r = verify_patch({file: cand})
            rec["green"] = r.green
            rec["score"] = r.score
            # family-aware success: type errors are judged by the COMPILER (typecheck),
            # test/logic failures need the TESTS to pass. Single-function training files
            # have no matching test file, so typecheck is the right oracle there.
            rec["success"] = r.green if fam == "TEST" else r.typechecks
        lat.append(rec)
    return {"diagnostic": diagnostic, "diag_family": fam, "actions": lat}


def build_from_jsonl(src_jsonl: Path, out_jsonl: Path, limit=120):
    rows = [json.loads(l) for l in open(src_jsonl)]
    n = 0
    with open(out_jsonl, "w") as f:
        for r in rows[:limit]:
            lat = build_lattice(r["broken"], r["diagnostic"])
            greens = sum(1 for a in lat["actions"] if a.get("green"))
            if not lat["actions"]:
                continue
            lat["state_id"] = f"{r.get('function','?')}:{n}"
            f.write(json.dumps(lat) + "\n")
            n += 1
            if n % 25 == 0:
                print(f"  {n} lattices ({greens} greens in last)", flush=True)
    print(f"built {n} lattices -> {out_jsonl}")
    return n


if __name__ == "__main__":
    import sys
    HERE = Path(__file__).parent
    src = HERE / "data" / (sys.argv[1] if len(sys.argv) > 1 else "transitions.jsonl")
    out = HERE / "data" / "lattices.jsonl"
    build_from_jsonl(src, out, limit=int(sys.argv[2]) if len(sys.argv) > 2 else 120)
