"""metis-1 — git-history repair miner (Sakana §11.2): REAL verified transitions.

Synthetic mutations teach the breaker's distribution, not the repair map. This
mines actual bug-fix commits: for every commit touching a `.ts` file, it compares
the self-contained functions before vs after. When a function CHANGED, the `after`
version typechecks GREEN in isolation, and the `before` version is RED, that pair
is a real, compiler-verified repair transition — the true training distribution.

Held-out by repo. Deduped by function. The yield is whatever it honestly is; a
small yield is itself a finding (real self-contained repair data is scarce without
a full build harness), which informs whether we need one.

    python git_miner.py ~/projects/foo ~/projects/bar --max-commits 400 --out data/git_transitions.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

from verifier import verify_patch
from extract import FUNC, is_self_contained

FIXME = re.compile(r"\b(fix|bug|repair|correct|resolve|patch|typeerror|incorrect|wrong)\b", re.I)


def sh(args, cwd, timeout=30):
    try:
        return subprocess.run(args, cwd=cwd, capture_output=True, text=True,
                              timeout=timeout, check=False).stdout
    except (subprocess.TimeoutExpired, OSError):
        return ""


def self_contained_funcs(text: str) -> dict[str, str]:
    """name -> normalized source, for functions that MIGHT be self-contained."""
    out = {}
    for m in FUNC.finditer(text or ""):
        name, params, ret, body = m.groups()
        src = f"export function {name}({params}): {ret.strip()} {{{body}}}"
        if len(src) > 400:
            continue
        if is_self_contained(name, params, ret, body):
            out[name] = src.strip()
    return out


def mine_repo(repo: Path, max_commits: int):
    if not (repo / ".git").exists():
        return []
    # candidate commits: fix-shaped OR small, touching .ts
    log = sh(["git", "log", "--no-merges", f"-n{max_commits}", "--format=%H%x00%s"], repo)
    transitions = []
    for line in log.splitlines():
        if "\x00" not in line:
            continue
        h, subject = line.split("\x00", 1)
        # files changed in this commit
        files = sh(["git", "show", "--name-only", "--format=", h], repo).split()
        ts_files = [f for f in files if f.endswith((".ts", ".tsx")) and "test" not in f.lower()]
        if not ts_files or len(ts_files) > 3:
            continue
        for f in ts_files:
            before = sh(["git", "show", f"{h}~1:{f}"], repo)
            after = sh(["git", "show", f"{h}:{f}"], repo)
            if not before or not after or before == after:
                continue
            fb, fa = self_contained_funcs(before), self_contained_funcs(after)
            for name in set(fb) & set(fa):
                if fb[name] == fa[name]:
                    continue                      # function unchanged
                gold, broken = fa[name], fb[name]
                rg = verify_patch({"src/calc.ts": gold})
                if not (rg.typechecks and rg.parses):
                    continue                      # after isn't self-contained-green
                rb = verify_patch({"src/calc.ts": broken})
                if rb.typechecks:
                    continue                      # before wasn't actually broken (in isolation)
                diag = rb.diagnostic.strip()
                if not diag:
                    continue
                transitions.append({
                    "repo": repo.name, "commit": h[:10], "subject": subject[:80],
                    "function": name, "broken": broken, "gold": gold,
                    "diagnostic": diag,
                    "sequence": (f"<state>\nfile: src/calc.ts\n{broken}\n</state>\n"
                                 f"<diagnostic>\n{diag[:400]}\n</diagnostic>\n"
                                 f"<edit>\n{gold}\n</edit>\n"),
                })
    return transitions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repos", nargs="+")
    ap.add_argument("--max-commits", type=int, default=300)
    ap.add_argument("--out", default="data/git_transitions.jsonl")
    args = ap.parse_args()

    out = Path(__file__).parent / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    seen, total = set(), 0
    with open(out, "w") as fh:
        for r in args.repos:
            repo = Path(r).expanduser()
            ts = mine_repo(repo, args.max_commits)
            kept = 0
            for t in ts:
                key = (t["broken"], t["gold"])
                if key in seen:
                    continue
                seen.add(key)
                fh.write(json.dumps(t) + "\n")
                kept += 1
            total += kept
            print(f"  {repo.name}: {kept} real transitions")
    print(f"\nmined {total} REAL verified repair transitions -> {out}")


if __name__ == "__main__":
    main()
