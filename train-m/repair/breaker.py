"""metis-1 — the repair-transition factory (doc 15 §2, doc 16 §5).

Takes working TypeScript and applies a KNOWN mutation to break it. Because we
know the exact reverse, we get a labeled transition for free:

    (broken code, compiler diagnostic) -> gold fix

This is the training substrate the whole thesis rests on: the model learns
`diagnostic -> next fix`, and the compiler labels every example at zero cost.
Here the breakage is synthetic (deterministic mutations); the same shape is what
a git-history miner produces from real (pre-fix, diagnostic, patch) triples.

Each mutation is reversible and self-verifying: apply it, confirm the workspace
went RED, and record the transition only if the original was GREEN and the
mutation actually broke it (so we never emit a "fix" that doesn't fix anything).
"""

from __future__ import annotations

import dataclasses

from verifier import Reward, verify_patch


@dataclasses.dataclass
class Transition:
    name: str                     # which mutation
    file: str                     # relative path of the touched file
    broken: str                   # full broken file content
    gold: str                     # full correct file content (the fix)
    diagnostic: str               # compiler/test output on the broken state
    broken_reward: Reward         # proof it was actually broken

    def as_sequence(self) -> str:
        """VERA-R/RNT training-sequence shape: state+diagnostic -> edit."""
        return (
            f"<state>\nfile: {self.file}\n{self.broken}\n</state>\n"
            f"<diagnostic>\n{self.diagnostic.strip()[:600]}\n</diagnostic>\n"
            f"<edit>\n{self.gold}\n</edit>\n"
        )


# Each mutation: (name, find, replace). Applied to the GOLD source to break it.
MUTATIONS = [
    ("wrong_arith_op", "return a + b;", "return a - b;"),      # test failure
    ("wrong_return_type", "): number {\n  return a - b;", "): string {\n  return a - b;"),  # type error
    ("undefined_symbol", "values.map((v) => v * factor)", "values.map((v) => vv * factor)"),  # type/parse
    ("missing_paren", "return a + b;", "return a + b"),        # keep valid; noop-ish (control)
]


def make_transitions(gold_src: str, file: str = "src/calc.ts") -> list[Transition]:
    """Break `gold_src` each known way; keep only mutations that truly broke it."""
    base = verify_patch({file: gold_src})
    if not base.green:
        raise RuntimeError(f"gold source is not green to begin with: {base.diagnostic[:200]}")

    out: list[Transition] = []
    for name, find, repl in MUTATIONS:
        if find not in gold_src:
            continue
        broken = gold_src.replace(find, repl, 1)
        if broken == gold_src:
            continue
        r = verify_patch({file: broken})
        if r.green:
            continue  # mutation didn't actually break anything — discard (no free lunch)
        out.append(Transition(name=name, file=file, broken=broken, gold=gold_src,
                              diagnostic=r.diagnostic, broken_reward=r))
    return out


if __name__ == "__main__":
    from pathlib import Path
    gold = (Path(__file__).parent / "fixture/src/calc.ts").read_text()
    ts = make_transitions(gold)
    print(f"generated {len(ts)} verified repair transitions:")
    for t in ts:
        first = t.diagnostic.strip().splitlines()[0] if t.diagnostic.strip() else "(test failure)"
        print(f"  - {t.name:18s} broke to score={t.broken_reward.score}  |  {first[:70]}")
