"""metis-1 — in-repo mutation miner (Sakana §4/§6): REAL code, REAL deps, REAL tsc.

Isolated verification found 0 real transitions because real code is never self-
contained. This breaks real files *inside a repo that already builds* — so the
diagnostics are real (real types, real imported symbols) — enumerates typed
repairs, and verifies each with the repo's OWN `tsc`. The output is a repair
lattice from the real distribution, the honest test of whether the ranker trained
on synthetic mutations transfers.

SAFETY: each file is snapshotted and restored in a finally block; only one file is
mutated at a time, and the repo is git-tracked, so a crash is recoverable with
`git checkout`. Never leaves a repo dirty on normal exit.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

from editops import candidates, RET_TYPE
from lattice import features, diag_family, name_similarity

FUNC_RET = re.compile(r"(export\s+)?function\s+\w+\s*\([^)]*\)\s*:\s*(number|string|boolean)\s*\{")
ERR = re.compile(r"error (TS\d+):")


def repo_tsc(repo: Path, rel: str) -> str:
    """Run the repo's own tsc; return diagnostics for the target file only."""
    try:
        p = subprocess.run(["bunx", "tsc", "--noEmit"], cwd=repo,
                           capture_output=True, text=True, timeout=90)
    except (subprocess.TimeoutExpired, OSError):
        return "TIMEOUT"
    out = p.stdout + p.stderr
    base = rel.split("/")[-1]
    return "\n".join(l for l in out.splitlines() if base in l)


def mutations_for_file(text: str):
    """Yield (name, broken_text) — typed breaks that should produce a real diagnostic."""
    muts = []
    # return-type break: flip a number return type to string (real TS2322)
    m = FUNC_RET.search(text)
    if m:
        cur = m.group(2)
        alt = "string" if cur != "string" else "number"
        s = text[:m.start(2)] + alt + text[m.end(2):]
        muts.append(("break_return_type", s))
    # identifier typo break: rename a used local identifier (real TS2304/TS2552)
    m2 = re.search(r"\bconst (\w{3,})\b", text)
    if m2:
        name = m2.group(1)
        uses = list(re.finditer(rf"\b{re.escape(name)}\b", text))
        if len(uses) >= 2:  # decl + >=1 use; rename only a use site
            u = uses[-1]
            s = text[:u.start()] + name + "X" + text[u.end():]
            muts.append(("break_identifier", s))
    return muts


def mine_repo(repo: Path, max_files=12):
    src = repo / "src"
    files = [f for f in src.rglob("*.ts") if "test" not in f.name.lower()][:120]
    lattices, tried = [], 0
    for f in files:
        if len(lattices) >= max_files:
            break
        rel = str(f.relative_to(repo))
        original = f.read_text()
        for mname, broken in mutations_for_file(original):
            tried += 1
            try:
                f.write_text(broken)
                diag = repo_tsc(repo, rel)
                if not diag or "TIMEOUT" in diag or "error" not in diag:
                    continue  # mutation didn't yield a clean single diagnostic
                # enumerate typed repairs on the mutated file, verify each with repo tsc
                cands = candidates(broken, diag)
                actions = []
                for op, span, cand in cands:
                    f.write_text(cand)
                    v = repo_tsc(repo, rel)
                    success = ("error" not in v) or (v.strip() == "")
                    feat = features(broken, diag, op, span, cand)
                    feat["name_sim"] = name_similarity(diag, op, span)
                    actions.append({"op": op, "span": span, "success": success, "features": feat})
                if actions:
                    lattices.append({"repo": repo.name, "file": rel, "mutation": mname,
                                     "diagnostic": diag, "diag_family": diag_family(diag),
                                     "actions": actions})
                    cov = "FIX" if any(a["success"] for a in actions) else "no-fix"
                    print(f"  {rel} [{mname}] {diag_family(diag)}: "
                          f"{len(actions)} actions {cov}", flush=True)
            finally:
                f.write_text(original)  # ALWAYS restore
    print(f"\nmined {len(lattices)} REAL in-repo lattices (tried {tried} mutations)")
    return lattices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repos", nargs="+")
    ap.add_argument("--max-files", type=int, default=12)
    ap.add_argument("--out", default="data/inrepo_lattices.json")
    args = ap.parse_args()
    allL = []
    for r in args.repos:
        repo = Path(r).expanduser()
        print(f"=== {repo.name} ===")
        allL += mine_repo(repo, args.max_files)
    out = Path(__file__).parent / args.out
    json.dump(allL, open(out, "w"))
    cov = sum(1 for l in allL if any(a["success"] for a in l["actions"]))
    fams = {}
    for l in allL:
        fams[l["diag_family"]] = fams.get(l["diag_family"], 0) + 1
    print(f"\ntotal: {len(allL)} lattices, coverage {cov}/{len(allL)}, families {fams} -> {out}")


if __name__ == "__main__":
    main()
