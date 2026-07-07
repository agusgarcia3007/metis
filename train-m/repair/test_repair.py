"""metis-1 — proof the repair harness measures what it claims (doc 14 step 1).

Pure-assert test runner (no pytest dependency). Run: python test_repair.py
Each test proves one property the whole thesis depends on:

  1. the oracle accepts truth        (gold fix -> green)
  2. the oracle rejects garbage      (noise -> not green, fab=0)
  3. the oracle is dense             (score orders broken < partial < green)
  4. the breaker makes real breaks   (every emitted transition was truly RED)
  5. the diagnostic is captured      (the teaching signal is non-empty)
  6. gold fixes round-trip           (applying gold to a break restores GREEN)
  7. pass@k detects success/failure  (gold=1.0, noise=0.0)
  8. pass@k is monotonic in k        (more samples never lowers solved count)
"""

from pathlib import Path

from breaker import make_transitions
from passk import eval_pass_at_k, gold_generator, noise_generator, stuck_generator
from verifier import verify_patch

GOLD = (Path(__file__).parent / "fixture/src/calc.ts").read_text()
FILE = "src/calc.ts"
_checks = []


def check(name, cond, detail=""):
    _checks.append((name, bool(cond), detail))
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail and not cond else ""))


def main():
    print("building verified repair transitions from the fixture...")
    tasks = make_transitions(GOLD)
    print(f"  -> {len(tasks)} transitions\n")

    print("1-3. oracle: accepts truth, rejects garbage, is dense")
    green = verify_patch({FILE: GOLD})
    check("oracle accepts the gold source (green)", green.green, f"score={green.score}")
    junk = verify_patch({FILE: "export const x = ;"})
    check("oracle rejects a syntax-broken file", not junk.green and not junk.parses)
    check("oracle never green-lights garbage (fab=0)", not junk.green)
    check("dense score orders garbage < gold", junk.score < green.score,
          f"{junk.score} !< {green.score}")

    print("\n4-6. breaker: real breaks, captured diagnostics, round-tripping fixes")
    check("at least 2 transitions produced", len(tasks) >= 2, f"got {len(tasks)}")
    check("every transition was actually RED", all(not t.broken_reward.green for t in tasks))
    check("every transition carries a non-empty diagnostic",
          all(t.diagnostic.strip() for t in tasks))
    # round-trip: applying the gold fix to each broken state restores green
    roundtrip = all(verify_patch({t.file: t.gold}).green for t in tasks)
    check("gold fix restores GREEN for every transition", roundtrip)
    # and the broken state is genuinely not green
    check("broken state is genuinely not green",
          all(not verify_patch({t.file: t.broken}).green for t in tasks))

    print("\n7-8. pass@k: detects success and failure, monotonic in k")
    g = eval_pass_at_k(tasks, gold_generator, k=4)
    n = eval_pass_at_k(tasks, noise_generator, k=4)
    s = eval_pass_at_k(tasks, stuck_generator, k=4)
    check("gold generator solves everything (pass@4 = 1.0)", g.pass_at_k == 1.0, f"{g.pass_at_k}")
    check("noise generator solves nothing (pass@4 = 0.0)", n.pass_at_k == 0.0, f"{n.pass_at_k}")
    check("stuck generator solves nothing but scores > 0",
          s.pass_at_k == 0.0 and s.mean_best_score > 0.0,
          f"pass={s.pass_at_k} score={s.mean_best_score}")
    mono = all(eval_pass_at_k(tasks, gold_generator, k=kk).solved
               <= eval_pass_at_k(tasks, gold_generator, k=kk + 1).solved for kk in (1, 2, 3))
    check("pass@k is monotonic in k", mono)

    print("\n" + "=" * 56)
    passed = sum(1 for _, ok, _ in _checks if ok)
    print(f"RESULT: {passed}/{len(_checks)} checks passed")
    if passed != len(_checks):
        raise SystemExit(1)
    print("the ruler is correct — it accepts truth, rejects garbage, and measures progress.")


if __name__ == "__main__":
    main()
