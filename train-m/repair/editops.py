"""metis-1 — typed edit operators (Sakana §1, §2.3, §7 step 1).

Given a broken TS source + the compiler diagnostic, enumerate a small set of
TYPED, localized candidate edits. This is the discrete action space of the repair
lattice: the correct fix is one action among plausible distractors, and the
compiler labels which actions are green. We do NOT enumerate arbitrary text —
only typed operators keyed by the diagnostic family, so the lattice is meaningful
(Sakana: "otherwise the model learns mutation artifacts").

Each operator returns candidates as `(op_name, span_label, new_source)`.
For the single-function fixtures a candidate is the full new source; the same
operators generalize to span-replacements inside larger files later.
"""

from __future__ import annotations

import re

TS2322 = re.compile(r"TS2322: Type '([^']+)' is not assignable to type '([^']+)'")
TS2304 = re.compile(r"TS2304: Cannot find name '([^']+)'")
TS2345 = re.compile(r"TS2345: Argument of type")
# TS2551/TS2552 carry the fix inline: "Cannot find name 'X'. Did you mean 'Y'?"
TS_SUGGEST = re.compile(r"TS255[12]: Cannot find name '([^']+)'\.\s*Did you mean '([^']+)'")
RET_TYPE = re.compile(r"\)\s*:\s*([\w\[\]]+)\s*\{")
PARAM = re.compile(r"[(,]\s*([A-Za-z_]\w*)\s*:")
DIAG_LINE = re.compile(r"\((\d+),\d+\)")


def enclosing_signature_span(src: str, diag_line: int):
    """Given the 1-based line tsc flags, find the return-type span of the enclosing
    function's signature (walk up to the nearest `): TYPE {` at/above that line)."""
    lines = src.splitlines(keepends=True)
    offsets, pos = [], 0
    for ln in lines:
        offsets.append(pos); pos += len(ln)
    for i in range(min(diag_line, len(lines)) - 1, -1, -1):
        m = RET_TYPE.search(lines[i])
        if m:
            start = offsets[i] + m.start(1)
            return start, offsets[i] + m.end(1), m.group(1)
    return None

# a small type universe for return-type repair candidates (includes the truth + distractors)
TYPE_SET = ["number", "string", "boolean", "number[]", "void", "unknown"]


def in_scope_idents(src: str) -> list[str]:
    """Parameter names + short local identifiers, as replacement candidates."""
    names = PARAM.findall(src)
    # also any 1-2 char identifiers used in the body (likely the intended symbol)
    body = src[src.index("{"):] if "{" in src else src
    names += [w for w in re.findall(r"\b([a-z]{1,3})\b", body) if w not in ("if", "of", "in")]
    seen, out = set(), []
    for n in names:
        if n not in seen:
            seen.add(n); out.append(n)
    return out


def candidates(broken: str, diagnostic: str) -> list[tuple[str, str, str]]:
    """Enumerate typed edit candidates for `broken` given its `diagnostic`."""
    out: list[tuple[str, str, str]] = []

    dline = int(m.group(1)) if (m := DIAG_LINE.search(diagnostic)) else 1

    # --- family: compiler-suggested name fix (TS2551/TS2552) ---
    # the diagnostic literally tells us the fix; apply it, plus a couple of scope
    # distractors so ranking still has to prefer the suggestion.
    if m := TS_SUGGEST.search(diagnostic):
        bad, suggestion = m.group(1), m.group(2)
        out.append(("apply_ts_suggestion", f"use:{suggestion}",
                    re.sub(rf"\b{re.escape(bad)}\b", suggestion, broken)))
        for ident in in_scope_idents(broken)[:3]:
            if ident not in (bad, suggestion):
                out.append(("replace_identifier", f"use:{ident}",
                            re.sub(rf"\b{re.escape(bad)}\b", ident, broken)))

    # --- family: return-type mismatch (TS2322) ---
    if m := TS2322.search(diagnostic):
        actual = m.group(1)                            # Type 'actual' not assignable to 'declared'
        span = enclosing_signature_span(broken, dline)  # localize to the RIGHT function
        if span:
            s, e, cur = span
            for t in TYPE_SET:
                if t == cur:
                    continue
                cand = broken[:s] + t + broken[e:]
                out.append(("set_return_type", f"signature:{t}", cand))
            # the diagnostic literally names the actual type — a high-prior candidate
            if actual.replace(" ", "") not in [t.replace(" ", "") for t in TYPE_SET]:
                cand = broken[:s] + actual + broken[e:]
                out.append(("set_return_type_from_diag", f"signature:{actual}", cand))

    # --- family: undefined symbol (TS2304) ---
    if m := TS2304.search(diagnostic):
        bad = m.group(1)
        for ident in in_scope_idents(broken):
            if ident == bad:
                continue
            # replace the undefined identifier at its use site(s)
            cand = re.sub(rf"\b{re.escape(bad)}\b", ident, broken)
            if cand != broken:
                out.append(("replace_identifier", f"use:{ident}", cand))
        # also: strip a likely typo suffix (vv -> v, xx -> x)
        if len(bad) >= 2 and bad[0] == bad[1]:
            cand = re.sub(rf"\b{re.escape(bad)}\b", bad[0], broken)
            out.append(("strip_typo", f"use:{bad[0]}", cand))

    # --- family: possibly-undefined / null (TS18048, TS2532, TS2531) ---
    if re.search(r"TS(18048|2532|2531)", diagnostic):
        span = enclosing_signature_span(broken, dline)
        # optional-chain the flagged access, and add a nullish default
        m2 = re.search(r"'([\w.]+)' is possibly", diagnostic)
        if m2:
            sym = m2.group(1)
            out.append(("optional_chain", f"use:{sym}?",
                        broken.replace(sym + ".", sym + "?.", 1)))
            out.append(("nullish_default", f"use:{sym}??",
                        re.sub(rf"\breturn ({re.escape(sym)}\b[^;]*)", r"return (\1) ?? 0", broken, count=1)))

    # --- family: expected N arguments (TS2554) ---
    if m2 := re.search(r"TS2554: Expected (\d+) arguments, but got (\d+)", diagnostic):
        want, got = int(m2.group(1)), int(m2.group(2))
        if got < want:  # add a trailing argument
            out.append(("add_argument", "call:+arg",
                        re.sub(r"\(([^()]*)\)(\s*[;,\n])", r"(\1, 0)\2", broken, count=1)))

    # --- family: test/logic failure (no TS code, or a failing assertion) ---
    # swap binary operators in the flagged line — covers wrong-operator arithmetic bugs
    is_test_fail = ("TS" not in diagnostic) or ("expect" in diagnostic) or ("toBe" in diagnostic)
    if is_test_fail:
        lines = broken.splitlines(keepends=True)
        # target the return line inside the function (best-effort without a line number)
        for i, ln in enumerate(lines):
            for a, b in [("+", "-"), ("-", "+"), ("*", "/"), ("-", "*"), ("+", "*"), ("*", "+")]:
                if f" {a} " in ln:
                    cand = "".join(lines[:i]) + ln.replace(f" {a} ", f" {b} ", 1) + "".join(lines[i+1:])
                    out.append(("swap_binary_op", f"line{i+1}:{a}->{b}", cand))

    # --- generic distractor op (usually wrong; makes ranking non-trivial) ---
    if "return " in broken:
        out.append(("add_null_guard", "body",
                    broken.replace("return ", "return 0 || ", 1)))

    # de-dup by candidate source
    seen, uniq = set(), []
    for op, span, cand in out:
        if cand in seen:
            continue
        seen.add(cand); uniq.append((op, span, cand))
    return uniq


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from breaker import make_transitions
    from pathlib import Path
    gold = (Path(__file__).parent / "fixture/src/calc.ts").read_text()
    for t in make_transitions(gold):
        cands = candidates(t.broken, t.diagnostic)
        print(f"\n{t.name}: {len(cands)} typed candidates")
        for op, span, _ in cands[:8]:
            print(f"  - {op:26s} {span}")
