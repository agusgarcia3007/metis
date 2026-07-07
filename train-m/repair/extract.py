"""metis-1 — extract real, self-contained TS functions from a codebase.

The first repair experiment memorized because it saw only 4 function templates.
This mines HUNDREDS of distinct real functions from ~/projects, keeping only the
self-contained ones (typecheck GREEN in isolation — no imports/externals needed),
so the repair trainer must learn the repair MAP, not a few bodies.

A function is a candidate if it is a single `export function name(params): Type {
...body... }` whose body references only its own parameters and a small allowlist
of globals (Math, Number, String, Array, Object, JSON, etc.). Batch-verified with
tsc via the existing fixture workspace.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from verifier import verify_patch

OUT = Path(__file__).parent / "data" / "functions.jsonl"
PROJECTS = Path.home() / "projects"

# one self-contained exported function with an explicit return type and a body
FUNC = re.compile(
    r"export function (\w+)\(([^)]*)\)\s*:\s*([\w\[\]<>| ]+?)\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}",
    re.S,
)
ALLOWED_GLOBALS = {"Math", "Number", "String", "Array", "Object", "JSON", "Boolean",
                   "parseInt", "parseFloat", "isNaN", "isFinite", "undefined", "null",
                   "true", "false", "return", "const", "let", "if", "else", "for",
                   "of", "in", "new", "typeof", "length", "map", "filter", "reduce",
                   "push", "slice", "join", "split", "toFixed", "toString", "abs",
                   "floor", "ceil", "round", "min", "max", "pow", "sqrt", "sign"}
IDENT = re.compile(r"\b([A-Za-z_]\w*)\b")
TYPES = re.compile(r"\b(number|string|boolean|void|unknown|any|number\[\]|string\[\])\b")


def params_names(params: str) -> set[str]:
    names = set()
    for part in params.split(","):
        part = part.strip()
        if not part:
            continue
        names.add(part.split(":")[0].strip().lstrip("."))
    return names


def is_self_contained(name, params, ret, body) -> bool:
    bound = params_names(params) | ALLOWED_GLOBALS | {name}
    # every identifier in the body must be a param, an allowed global, a type, or a number/keyword
    for ident in IDENT.findall(body):
        if ident in bound or TYPES.match(ident) or ident.isdigit():
            continue
        return False
    return len(body.strip()) > 0


def candidates(limit_files=4000):
    skip = ("node_modules", ".git", "dist", "build", ".next", "venv", ".venv", "target")
    seen = set()
    files = []
    for fp in PROJECTS.rglob("*.ts"):
        if any(s in fp.parts for s in skip):
            continue
        files.append(fp)
        if len(files) >= limit_files:
            break
    for fp in files:
        try:
            text = fp.read_text(errors="replace")
        except OSError:
            continue
        for m in FUNC.finditer(text):
            name, params, ret, body = m.groups()
            src = f"export function {name}({params}): {ret.strip()} {{{body}}}"
            if len(src) > 400 or "=>" in body and "function" in body:
                continue
            if not is_self_contained(name, params, ret, body):
                continue
            key = src.strip()
            if key in seen:
                continue
            seen.add(key)
            yield name, src.strip()


def mine(target=300):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    kept, checked = 0, 0
    with open(OUT, "w") as f:
        for name, src in candidates():
            checked += 1
            r = verify_patch({"src/calc.ts": src})
            if not (r.typechecks and r.parses):
                continue
            f.write(json.dumps({"name": name, "src": src}) + "\n")
            kept += 1
            if kept % 25 == 0:
                print(f"  kept {kept} (checked {checked})", flush=True)
            if kept >= target:
                break
    print(f"extracted {kept} self-contained real functions (checked {checked}) -> {OUT}")
    return kept


if __name__ == "__main__":
    import sys
    mine(target=int(sys.argv[1]) if len(sys.argv) > 1 else 300)
