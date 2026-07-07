"""metis-1 — the deterministic TypeScript verifier (the oracle, doc 14/15).

This is the compiler-as-infinite-teacher, in its cheapest local form: no Docker,
just `tsc --noEmit` (typecheck) and `bun test` (tests) run over a workspace copy
in a temp dir. It returns a structured, dense reward and — crucially — the raw
compiler DIAGNOSTIC, which is the teaching signal a repair-transition model
trains on (VERA-R §2, Agents-A1 KAG).

Verification ladder (cheap -> expensive, VERA-R §8): parse -> typecheck -> tests.
Each rung is a gate; a lower rung failing short-circuits the higher ones.

This is intentionally swappable with the shipped Phase-5 Docker sandbox
(sandbox/code) later; the interface is the same Reward shape.
"""

from __future__ import annotations

import dataclasses
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

FIXTURE = Path(__file__).parent / "fixture"


@dataclasses.dataclass
class Reward:
    """Dense, ordered reward. Higher rungs only count if lower ones pass."""
    parses: bool = False
    typechecks: bool = False
    tests_passed: int = 0
    tests_total: int = 0
    diagnostic: str = ""          # raw compiler/test output — the teaching signal

    @property
    def green(self) -> bool:
        return self.typechecks and self.tests_total > 0 and self.tests_passed == self.tests_total

    @property
    def score(self) -> float:
        """Dense progress in [0,1]: parse .2, typecheck .3, tests proportional .5."""
        s = 0.2 if self.parses else 0.0
        s += 0.3 if self.typechecks else 0.0
        if self.tests_total:
            s += 0.5 * (self.tests_passed / self.tests_total)
        return round(s, 3)


def _run(cmd: list[str], cwd: Path, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                          timeout=timeout, check=False)


_TEST_LINE = re.compile(r"(\d+)\s+pass", re.I)
_FAIL_LINE = re.compile(r"(\d+)\s+fail", re.I)


def verify_workspace(ws: Path) -> Reward:
    """Run the ladder over an existing workspace dir (must contain tsconfig.json)."""
    r = Reward()

    # rung 1+2: tsc --noEmit  (parse errors and type errors both surface here)
    tsc = _run(["bunx", "tsc", "--noEmit"], ws)
    out = (tsc.stdout + tsc.stderr).strip()
    if tsc.returncode == 0:
        r.parses = True
        r.typechecks = True
    else:
        r.diagnostic = out
        # distinguish a hard parse error (TS1xxx) from a type error (TS2xxx)
        r.parses = "error TS1" not in out
        r.typechecks = False
        return r  # ladder short-circuits: don't run tests on code that won't compile

    # rung 3: bun test
    bt = _run(["bun", "test"], ws)
    bout = (bt.stdout + bt.stderr).strip()
    npass = int(m.group(1)) if (m := _TEST_LINE.search(bout)) else 0
    nfail = int(m.group(1)) if (m := _FAIL_LINE.search(bout)) else 0
    r.tests_passed, r.tests_total = npass, npass + nfail
    if nfail:
        r.diagnostic = bout
    return r


def verify_patch(patched_files: dict[str, str], base: Path = FIXTURE) -> Reward:
    """Copy `base`, overwrite `patched_files` (relpath -> content), run the ladder.

    A candidate edit is expressed as the full new content of the files it touches.
    Nothing is written to `base`; verification happens on a throwaway copy.
    """
    with tempfile.TemporaryDirectory(prefix="metis-verify-") as td:
        ws = Path(td) / "ws"
        shutil.copytree(base, ws, ignore=shutil.ignore_patterns("node_modules"))
        # reuse the base's installed typescript rather than reinstalling
        nm = base / "node_modules"
        if nm.exists():
            (ws / "node_modules").symlink_to(nm)
        for rel, content in patched_files.items():
            fp = ws / rel
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text(content)
        return verify_workspace(ws)


if __name__ == "__main__":
    # smoke: verify the untouched fixture is green
    r = verify_patch({})
    print(f"fixture: parses={r.parses} typechecks={r.typechecks} "
          f"tests={r.tests_passed}/{r.tests_total} green={r.green} score={r.score}")
