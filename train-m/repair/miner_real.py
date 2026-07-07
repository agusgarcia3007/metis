"""metis-1 — repair-transition miner over REAL extracted functions.

Same verified-transition output as miner.py, but the source functions are the
hundreds of distinct real functions from extract.py (functions.jsonl) instead of
4 synthetic templates. Diversity is the whole point: the model must learn the
repair MAP (restore the type / restore the identifier), not memorize a few bodies.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from verifier import verify_patch

FUNCS = Path(__file__).parent / "data" / "functions.jsonl"
OUT = Path(__file__).parent / "data" / "transitions.jsonl"


def mutations_for(src: str):
    muts = []
    # type-error: flip the declared return type to a wrong-but-parseable one
    for good, bad in [("): number {", "): string {"), ("): string {", "): number {"),
                       ("): boolean {", "): number {"), ("): number[] {", "): number {"),
                       ("): number {", "): boolean {")]:
        if good in src:
            muts.append((f"rettype{good.strip('):{ ')}", good, bad))
            break
    # undefined-symbol: rename the first parameter at a single use site
    m = re.search(r"\(([A-Za-z_]\w*)\s*:", src)
    if m:
        p = m.group(1)
        use = re.search(rf"\b{re.escape(p)}\b", src[src.index("{"):])
        if use:
            body = src[src.index("{"):]
            broken_body = body.replace(p, p + "X", 1)
            muts.append((f"undef_{p}", body, broken_body))
    return muts


def mine(target=600):
    rows = [json.loads(l) for l in open(FUNCS)]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    kept, checked = 0, 0
    with open(OUT, "w") as f:
        for r in rows:
            src = r["src"]
            for mname, find, repl in mutations_for(src):
                if find not in src:
                    continue
                broken = src.replace(find, repl, 1)
                if broken == src:
                    continue
                checked += 1
                res = verify_patch({"src/calc.ts": broken})
                if res.typechecks:
                    continue  # didn't break -> discard
                diag = res.diagnostic.strip()
                if not diag:
                    continue
                seq = (f"<state>\nfile: src/calc.ts\n{broken}\n</state>\n"
                       f"<diagnostic>\n{diag[:400]}\n</diagnostic>\n"
                       f"<edit>\n{src}\n</edit>\n")
                f.write(json.dumps({"mutation": mname, "broken": broken, "gold": src,
                                    "diagnostic": diag, "sequence": seq}) + "\n")
                kept += 1
                if kept % 50 == 0:
                    print(f"  mined {kept} (checked {checked})", flush=True)
                if kept >= target:
                    print(f"mined {kept} real-function transitions (checked {checked}) -> {OUT}")
                    return kept
    print(f"mined {kept} real-function transitions (checked {checked}) -> {OUT}")
    return kept


if __name__ == "__main__":
    import sys
    mine(target=int(sys.argv[1]) if len(sys.argv) > 1 else 600)
