"""metis-1 — high-diversity procedural function generator (defeats memorization).

Only ~19 real functions are self-contained enough to typecheck in isolation, too
few — the model memorized. This generates HUNDREDS of structurally distinct, typed
functions procedurally so no body repeats. Each is constructed to typecheck by
construction (return type matches the expression's type), then confirmed by tsc.

Diversity axes: arity (1-3), param types (number/number[]/string), and a typed
expression grammar (arithmetic, Math.*, comparisons, ternaries, array ops). The
point is that "reproduce the input and fix the flagged token" becomes the only
strategy that generalizes — copy-with-edit, not memorize-a-body.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from verifier import verify_patch

OUT = Path(__file__).parent / "data" / "functions.jsonl"

VERBS = ["compute", "derive", "resolve", "calc", "eval", "make", "find", "get", "to",
         "apply", "adjust", "scaleBy", "blendOf", "normOf", "weight", "rank", "score",
         "bound", "spread", "shiftBy", "ratioOf", "gainOf", "costOf", "sizeOf", "pickFrom"]
NOUNS = ["Value", "Total", "Factor", "Amount", "Score", "Ratio", "Level", "Rate", "Delta",
         "Index", "Weight", "Bound", "Span", "Gap", "Step", "Slice", "Norm", "Yield", "Cap"]


def num_expr(ps, rng, depth=0):
    a = rng.choice(ps)
    if depth > 1 or rng.random() < 0.4:
        op = rng.choice(["+", "-", "*"])
        return f"{a} {op} {rng.choice(ps)}"
    if rng.random() < 0.5:  # two-arg
        fn = rng.choice(["Math.max", "Math.min"])
        return f"{fn}({a}, {rng.choice(ps)})"
    fn = rng.choice(["Math.abs", "Math.round", "Math.floor", "Math.sign"])  # one-arg
    return f"{fn}({num_expr(ps, rng, depth+1)})"


def gen_function(i, rng):
    name = rng.choice(VERBS) + rng.choice(NOUNS) + str(i)
    arity = rng.randint(1, 3)
    ptype = rng.choice(["number", "number", "number[]"])
    pnames = ["a", "b", "c"][:arity]
    if ptype == "number[]":
        params = f"xs: number[], k: number"
        rtype = rng.choice(["number", "number[]"])
        if rtype == "number[]":
            body = f"return xs.map((v) => v {rng.choice(['+','-','*'])} k);"
        else:
            body = f"return xs.length {rng.choice(['+','-','*'])} k;"
    else:
        params = ", ".join(f"{p}: number" for p in pnames)
        kind = rng.random()
        if kind < 0.55:
            rtype = "number"
            body = f"return {num_expr(pnames, rng)};"
        elif kind < 0.8:
            rtype = "boolean"
            body = f"return {pnames[0]} {rng.choice(['>','<','>=','==='])} {rng.choice(pnames)};"
        else:
            rtype = "number"
            body = (f"return {pnames[0]} {rng.choice(['>','<'])} {rng.choice(pnames)} "
                    f"? {num_expr(pnames, rng)} : {num_expr(pnames, rng)};")
    return name, f"export function {name}({params}): {rtype} {{\n  {body}\n}}"


def mine(target=280, seed=1):
    rng = random.Random(seed)
    existing = []
    if OUT.exists():
        existing = [l for l in open(OUT)]
    kept, checked, seen = 0, 0, set()
    with open(OUT, "a") as f:
        i = 0
        while kept < target and checked < target * 3:
            i += 1
            name, src = gen_function(i, rng)
            if src in seen:
                continue
            seen.add(src)
            checked += 1
            r = verify_patch({"src/calc.ts": src})
            if not (r.typechecks and r.parses):
                continue
            f.write(json.dumps({"name": name, "src": src}) + "\n")
            kept += 1
            if kept % 40 == 0:
                print(f"  synth kept {kept} (checked {checked})", flush=True)
    total = len(existing) + kept
    print(f"synth added {kept} functions (checked {checked}); functions.jsonl now {total} total")
    return kept


if __name__ == "__main__":
    import sys
    mine(target=int(sys.argv[1]) if len(sys.argv) > 1 else 280)
