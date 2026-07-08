# CONSULT → Sakana Fugu: maximum capability per training example, and per parameter (metis-1)

> You authored VERA-R (doc 15) and answered the generator-gate consult (doc 18), so you have the
> context. This is a deeper, more open-ended question — closer to invention than engineering. Please
> think outside the box: we want ideas that break the default assumptions, not incremental tuning.

## The setting (one paragraph)

metis-1 is a tiny (<300M target, currently piloting at ~14–40M on a MacBook), local, code-only
Cortex whose job is repair: `(broken code, compiler diagnostic, retrieved symbols) → minimal verified
edit`. Its defining asset is a **free, infinite, non-hallucinating verifier** — the TypeScript
compiler + tests — which can label unlimited data and score any candidate exactly. Its defining
constraint is that it must be **tiny and local**. We have proven the ruler (pass@k vs compiler) and
proven that a 14M byte model is too weak; the levers we know (BPE, edit-native output, capacity,
scale) are conventional. We just measured that **real isolated-verifiable repair data is nearly
nonexistent** (0 self-contained repair transitions across 194 real .ts-touching commits), so raw data
volume is scarce and expensive.

## The core question

Given a **free perfect verifier** and a **hard smallness constraint**, how do we extract the **maximum
capability per training example and per parameter** — including ideas that break the default paradigm
of *one example → one label → one gradient step*?

We are explicitly asking you to challenge assumptions like:
- one training example teaches one thing;
- the label is the human/gold answer;
- capability must be stored in parameters;
- more capability requires more data or more parameters;
- the verifier is only used at eval/reward time, not as a data engine.

## Seed directions (react to these, then go beyond them)

We already see these; tell us which are real, which are traps, and — more importantly — **what we are
NOT seeing**:

1. **Multi-task per example:** one repair transition supervises localization + generation + verifier-
   outcome-prediction + failure-class + retrieval-usefulness at once (VERA-R §5). N gradients per entry.
2. **Verifier-as-data-engine:** from one broken seed, sample K candidates, let the compiler label all
   K (green/red + dense score). The near-misses are the richest signal. One seed → K labeled examples.
3. **Soft/process labels:** distill a teacher's full top-k distribution (SVA, Agents-A1) instead of a
   1-bit hard label; and use step-level compiler reward instead of final pass/fail. More bits/token.
4. **Memory layers (PKM):** ~1M key-value slots, top-32 lookup — large capacity, tiny active compute.
   More effective parameters per FLOP (doc 13).
5. **Retrieval-native (RNT):** each example carries its own knowledge in-context, so the model learns
   to *use* information instead of memorizing it — external parameters, unbounded.

## What we want from you (please answer directly)

1. **The biggest lever we're missing.** If you had a free perfect verifier and a tiny model, what is
   the single most information-dense way to train it that we have NOT listed? Be concrete about the
   mechanism, the data shape, and why it extracts more per example than standard SFT.

2. **Break the paradigm.** Give 2–3 genuinely non-standard training formulations for this setting
   (free verifier + tiny model + scarce real data). Examples of the *kind* of answer we want:
   learning a latent edit space so one vector = one repair; the model outputting a *program of edit
   ops* so one entry teaches many actions; contrastive learning over verified-vs-refuted candidate
   pairs; energy-based / verifier-in-the-loop objectives; amortizing search into weights (distill the
   search tree, not just the winner). Invent past these if you can.

3. **Capability per parameter.** For a sub-300M code model, rank the real levers to get more capability
   per parameter (PKM/memory layers, MoE, retrieval externalization, weight tying, tokenizer choice,
   depth-vs-width, cross-layer parameter sharing, hypernetworks). What actually pays at this scale, and
   what is a distraction?

4. **Turning the verifier into a data flywheel under data scarcity.** Since real repair data barely
   exists, how do we bootstrap a large, *real-distribution* training set from a free verifier + a
   handful of working repos, without it collapsing into the synthetic-mutation distribution we already
   showed the model memorizes? (We are about to build in-repo mutation: break real files inside their
   working repo, verify with the repo's own tsc/tests. Is that the right seed, and how do we keep it
   from being just a fancier breaker?)

5. **Falsifiable + cheap-first.** For each idea, tell us the smallest experiment that would validate or
   kill it, and whether it can run on a MacBook (pilot) or genuinely needs GPU. Pre-register a metric
   on our pass@k / candidate-span harness where possible.

## One tactical rider (validate or overturn, briefly)

Our real-data miner found 0 transitions via *isolated* verification (real code is never self-contained).
We're pivoting to **in-repo mutation**: mutate real files inside a repo that already builds, verify with
that repo's own compiler + tests. Is that the right unlock, or is a full historical build-harness
(checkout → install → tsc + tests per commit) worth the extra pain for the logic-bug coverage it adds?

## Pointers (repo `metis-0`)

`docs/design/14` (compiler-as-teacher), `15` (your VERA-R), `16` (Agents-A1 + Ornith), `17`/`18` (the
generator-gate consult + your answer), `train-m/repair/` (the harness, breaker, git_miner, large_k),
and `07-implementation-and-deployment-log.md` (the measured record, incl. the 0-real-transitions finding).
