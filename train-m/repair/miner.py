"""metis-1 — repair-transition miner (scales breaker.py into a training set).

Generates many DIVERSE synthetic-but-real TypeScript functions, breaks each with
type-level mutations, and keeps only the ones the compiler confirms RED — writing
verified `(broken, diagnostic) -> gold` transitions to JSONL in VERA-R sequence
shape. Type-error mutations are used (caught by `tsc` alone, one call each, fast).

Held-out cleanliness: the pass@k test fixture (calc.ts: add/subtract/scale) is
NEVER emitted here — training functions are drawn from disjoint templates, so a
pass@k gain is real generalization, not memorization.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from verifier import verify_patch

OUT = Path(__file__).parent / "data" / "transitions.jsonl"

# diverse typed function templates (disjoint from the calc.ts pass@k fixture)
NAMES = ["total", "clamp", "norm", "merge", "count", "pick", "shift", "ratio", "bound",
         "sumOf", "maxOf", "minOf", "avg", "delta", "scaleUp", "offset", "combine", "reduceSum",
         "acc", "weigh", "blend", "trim", "grow", "step", "mix", "fold", "span", "gap", "rate",
         "cap", "floorTo", "ceilTo", "wrap", "join2", "dup", "halve", "twice", "incr", "decr"]
BODIES = [
    ("num2", "export function {n}(a: number, b: number): number {{\n  return a {op} b;\n}}",
     {"op": ["+", "-", "*"]}),
    ("arr", "export function {n}(xs: number[], k: number): number[] {{\n  return xs.map((x) => x {op} k);\n}}",
     {"op": ["+", "-", "*"]}),
    ("str", "export function {n}(s: string, t: string): string {{\n  return s {op} t;\n}}",
     {"op": ["+"]}),
    ("cnt", "export function {n}(xs: number[]): number {{\n  return xs.length {op} 1;\n}}",
     {"op": ["+", "-"]}),
]

# type-level mutations: (name, find, replace) — each yields a tsc-caught error
def mutations_for(src: str):
    muts = [
        ("wrong_return_type_str", "): number {", "): string {"),
        ("wrong_return_type_arr", "): number[] {", "): number {"),
        ("wrong_return_type_num", "): string {", "): number {"),
        ("undef_a", "return a ", "return aa "),
        ("undef_x", "(x) => x ", "(x) => xx "),
        ("undef_s", "return s ", "return ss "),
        ("undef_len", "xs.length ", "ys.length "),
    ]
    return [(m, f, r) for (m, f, r) in muts if f in src]


def build_functions(seed=0):
    rng = random.Random(seed)
    fns = []
    used = set()
    for name in NAMES:
        for kind, tmpl, choices in BODIES:
            op = rng.choice(choices["op"])
            src = tmpl.format(n=name, op=op)
            key = (kind, name)
            if key in used:
                continue
            used.add(key)
            fns.append(src)
    rng.shuffle(fns)
    return fns


def mine(limit=200, seed=0):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fns = build_functions(seed)
    n, kept = 0, 0
    with open(OUT, "w") as f:
        for src in fns:
            base = verify_patch({"src/calc.ts": src})   # reuse fixture workspace/tsconfig
            if not base.typechecks:
                continue                                  # only break things that started GREEN
            for mname, find, repl in mutations_for(src):
                if find not in src:
                    continue
                broken = src.replace(find, repl, 1)
                if broken == src:
                    continue
                r = verify_patch({"src/calc.ts": broken})
                n += 1
                if r.typechecks:
                    continue                              # mutation didn't break -> discard
                diag = r.diagnostic.strip()
                if not diag:
                    continue
                seq = (f"<state>\nfile: src/calc.ts\n{broken}\n</state>\n"
                       f"<diagnostic>\n{diag[:400]}\n</diagnostic>\n"
                       f"<edit>\n{src}\n</edit>\n")
                f.write(json.dumps({"mutation": mname, "broken": broken, "gold": src,
                                    "diagnostic": diag, "sequence": seq}) + "\n")
                kept += 1
                if kept >= limit:
                    print(f"mined {kept} verified transitions (checked {n}) -> {OUT}")
                    return kept
    print(f"mined {kept} verified transitions (checked {n}) -> {OUT}")
    return kept


if __name__ == "__main__":
    import sys
    mine(limit=int(sys.argv[1]) if len(sys.argv) > 1 else 200)
