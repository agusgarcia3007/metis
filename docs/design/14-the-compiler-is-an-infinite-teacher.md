# metis-1 — The compiler is an infinite teacher: the game-changer thesis

> This document reframes the whole project. Docs 12–13 asked "how small can a capable code model
> be?" That is the wrong question — it is a compression race against frontier labs, and you do not
> win a compression race from a MacBook. This doc states the right question and the bet that answers
> it: **a tiny local model does not need to match a frontier model at all code — it needs to beat it
> at YOUR code, which a frozen generic model structurally cannot do.**

---

## 1. The structural weakness of every frontier coding model

A frontier coding agent is a huge, frozen, generic artifact rented from a datacenter. On the day it
ships it is the best it will ever be for you, and it is identical for every user on earth. It cannot
specialize to your repo, your types, your conventions, your compiler, your test suite. That is not a
tuning gap — it is architectural. Weights trained once in a datacenter and served read-only cannot
learn from your machine.

metis inverts every one of those properties: **small, local, and alive.** The bet is not "out-
compress the frontier." It is "out-specialize it, and compound." A model that improves at your
codebase every night will cross a frozen model's quality on that codebase — while being 1000×
smaller — because the two curves move in opposite directions: theirs is flat, ours climbs.

## 2. The three facts that make this buildable (not hype)

1. **The compiler is an infinite, free, non-hallucinating teacher.** In code, `compile ∧ typecheck ∧
   tests` is a perfect oracle. It labels unlimited training signal locally, with no human annotation
   and no dataset. This is unique to code — it is why we bet the project on code (doc 11).
2. **Verifying is cheaper than generating** (docs 06, 11). A tiny model can emit many candidates; the
   verifier keeps the one that passes. Intelligence bought with search, not parameters — and search
   is cheap because the verifier is free and parallel across CPU cores.
3. **The model can retrain nightly on what it verified by day** (doc 13). Every edit that passes the
   oracle becomes training data. The system studies only its own verified successes and failures.

Each fact is proven or built in-repo: the RNT mechanism (docs 03/04), the deterministic TS verifier
(doc 11, Phase 5, shipped), Muon training on the Mac (doc bitácora, 12.3× measured), FIM as the
code-native objective (Night 2). No missing miracle — the pieces exist. What has never been run is
the loop that closes them.

## 3. The paradigm, in one sentence

> **Stop shipping intelligence. Ship a seed that grows intelligence locally — specialized to each
> codebase, taught by the compiler, compounding every night — until a 200 MB model on a laptop beats
> a datacenter model on the code that laptop actually works on.**

The retriever, the tiny reasoner, FIM, Muon — all of these are supporting actors. The headline is the
**flywheel**: search under a free verifier → distill the verified traces back into the weights →
next time, one shot does what a hundred rollouts did before → repeat, forever, on your machine.

## 4. The self-improvement loop (STaR/ReST, but the reward is a compiler, and it runs local)

```
  a coding task with tests (from your repo, or SWE-bench)
        │
        ▼
  CORTEX (tiny, FIM+Muon+BPE) ── emits K candidate edits (search, parallel on CPU cores)
        │
        ▼
  VERIFIER (compile ∧ typecheck ∧ tests) ── keeps only the edits that PASS   [free, exact]
        │
        ├── none pass → harder task, log the failure for tomorrow's curriculum
        │
        ▼
  DISTILL the passing traces back into the Cortex (a few Muon steps)         [nightly metabolism]
        │
        ▼
  pass@1 rises with FLAT parameter count ── the model absorbed its own verified search
```

The frozen frontier model has no analog of the bottom two arrows. That is the whole edge.

## 5. The one experiment that proves or kills it (pre-registered)

Everything above is rhetoric until one curve is measured. Hold **model size fixed** and run the loop:

| round | pass@1 (frozen size) | mechanism |
|---|---|---|
| 1 | baseline (low) | tiny model, one shot |
| 1 + search | higher | K rollouts under the verifier |
| 2 | ? | after distilling round-1's verified traces |
| 3 | ? | after distilling round-2's |

- **Game-changer confirmed if:** pass@1 climbs monotonically across rounds at fixed parameter count —
  the model is teaching itself with the compiler, and the gain compounds.
- **Thesis dead if:** pass@1 is flat across rounds (distillation doesn't absorb the search) — write it
  up as a first-class negative result, like H1.
- **Cheapest honest version:** a handful of TS tasks with held-out tests, the existing 14M FIM Cortex,
  the shipped Phase-5 verifier, 2–3 self-distill rounds. Runs on the Mac, gently. No GPU, no cloud.

This is the "double down" criterion of docs 11 (§5.5) and 13, finally isolated as the single headline
number of the project.

## 6. Why this changes the industry (if the curve bends up)

- **AI stops being rented.** Coding intelligence becomes a local, private, offline artifact that you
  own and that improves on your machine — no API, no per-token bill, no vendor that can deprecate you.
- **Specialization beats scale, per-codebase.** The relevant benchmark is no longer "who is best at
  all code" but "who is best at THIS repo," and a local model that trains on the repo's own compiler
  wins that by construction.
- **The moat inverts.** Frontier labs' moat is scale and frozen weights; here the moat is the user's
  own verified history, which no lab can access or replicate.

## 7. Honest limits

- Distillation of self-generated traces can collapse (mode collapse, reward hacking against the tests).
  The Phase-5 sandbox already blocks test edits / skips / network; the held-out test split is the true
  gate (doc 11 §4). If the model games it, the curve is a lie — we watch fab% every round.
- The loop needs tasks with tests. On code without tests, the verifier is weak and scale still wins
  (doc 03 §6). metis claims the verifiable core, and measures the claim.
- Compounding may saturate quickly (the model learns the easy wins, then stalls). The curve's *shape*,
  not just its first step, is the result.

## 8. Next action

Build the minimal flywheel: wire the shipped Phase-5 verifier to the FIM Cortex's search, run 2–3
self-distill rounds on a small held-out TS task set, and plot pass@1 vs round at fixed model size.
That single plot is the project. Everything else — retriever, GitHub miner, bigger Cortex, the cloud
run — is amplification of a loop that this experiment either validates or kills.
