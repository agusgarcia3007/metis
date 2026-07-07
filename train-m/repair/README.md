# metis-1 repair harness — the ruler that decides the project

This is doc 14 §8 **step 1**, built and tested: the measurement infrastructure that tells us
whether the Cortex can actually write code, **judged by the TypeScript compiler, not by vibes**.
Everything downstream (the self-improvement flywheel) is meaningless until the generator clears
this gate — that is the whole lesson of Aletheia's failure (doc 14 §0).

Mac-safe: no GPU, no Docker, no training. Just `tsc` + `bun test` over throwaway workspace copies.

## The four pieces

| file | role |
|---|---|
| `verifier.py` | the deterministic oracle — `tsc --noEmit` (typecheck) + `bun test`, a cheap→expensive ladder (parse → typecheck → tests). Returns a dense `Reward` **and the raw compiler diagnostic** (the teaching signal). Swappable with the Phase-5 Docker sandbox later. |
| `breaker.py` | the repair-transition factory — breaks working TS in known ways, keeps only mutations that truly go RED, and emits `(broken, diagnostic) → gold fix` transitions in RNT/VERA-R sequence shape. |
| `passk.py` | pass@k against the compiler — the project's headline metric. Model-agnostic: any `(task, i) → file content` callable plugs in. |
| `cortex_generator.py` | wires the real 14M FIM Cortex (night2) in as a generator — the honest baseline. |

## Run it

```sh
cd train-m/repair/fixture && bun install && cd ..     # one-time: local tsc
python test_repair.py        # 13/13 proofs the ruler is correct
python breaker.py            # see the verified repair transitions + real TS diagnostics
python passk.py              # gold=1.0, noise=0.0, stuck=0.0 (harness detects success & failure)
python cortex_generator.py   # first REAL pass@k of the Cortex vs the compiler
```

## What it proves (13/13 tests green)

The oracle **accepts truth** (gold fix → green), **rejects garbage** (noise → never green, fab=0),
and is **dense** (score orders broken < partial < green). The breaker makes **real** breaks with
**captured diagnostics** whose **gold fixes round-trip** to green. pass@k **detects success and
failure** and is **monotonic in k**.

## The first measured baseline (recorded, honest)

```
metis-fim-14M   pass@1 = 0.0   pass@4 = 0.0 (score 0.067)   pass@8 = 0.0 (score 0.133)
gold reference  pass@1 = 1.0
```

The 14M toy solves nothing — exactly as expected at ~6.5M training tokens (Aletheia's lesson made
falsifiable). But `mean_best_score` **rises with k** (0.0 → 0.067 → 0.133): more samples get
marginally further. This is the floor. A better-trained Cortex is "better" **iff this number
climbs on this harness** — the ruler, not vibes, is now the judge.

## Next

Train a stronger Cortex on repair-transition data (`breaker.py` output at scale, from a git miner),
re-run `cortex_generator.py`, and watch pass@k move off zero. Only once pass@k is non-trivial does
the doc-14 self-distillation flywheel (pass@1-vs-round) have a generator worth amplifying.
