# CONSULT → Sakana Fugu: the generator gate (metis-1)

> **What this is.** A consultation brief, not a design doc. You (Sakana Fugu) authored VERA-R
> (doc 15), so you know the thesis. Since then we built the measurement infrastructure and ran the
> first real experiments on a MacBook. We hit a wall that is honest and well-characterized, and we
> want your read on the highest-leverage path forward — ideally one that squeezes a decisive signal
> out of cheap compute before committing to a paid/cloud scale run.
>
> Please answer the 5 questions in §5 directly. Be concrete and willing to say "the Mac path is
> exhausted, here is the minimal scale run" if that is the honest answer.

---

## 1. The bet (recap, so this is self-contained)

metis-1 = a tiny, local, code-only Cortex that fights frontier coding agents **not** by matching them
at all code, but by (a) externalizing knowledge to retrieval, (b) outsourcing correctness to a
deterministic compiler/test verifier, (c) buying depth with verified search, and eventually (d)
self-improving by distilling its own verified repair trajectories (the compiler as an infinite
teacher — doc 14). VERA-R (doc 15, yours) is the policy: a repair-transition model that reads
`(state, diagnostic, retrieved symbols, prior attempt) → next minimal edit`. Docs 16 stole the KAG
trajectory substrate + SVA cheap distillation from Agents-A1, and the learnable-scaffold idea from
Ornith-1.0.

**The load-bearing lesson from the sibling project Aletheia (which failed):** the loop amplifies a
capable generator; it cannot manufacture capability. Aletheia kept its model a 1-layer/d=64 toy and
died regurgitating sources. So metis's **first** job is a generator that actually writes code, gated
by the compiler — before any flywheel.

## 2. What we built (all tested, all on a MacBook M3 Pro, MLX/Python, no cloud)

- **A deterministic TS verifier** (`train-m/repair/verifier.py`): `tsc --noEmit` + `bun test`, a
  cheap→expensive ladder, returns a dense reward (parse .2 / typecheck .3 / tests .5) **and the raw
  compiler diagnostic**. Swappable with the shipped Phase-5 Docker sandbox.
- **A repair-transition factory** (`breaker.py`, `miner_real.py`, `synth.py`, `extract.py`): breaks
  working TS in known ways, keeps only mutations the compiler confirms RED, emits
  `(broken, diagnostic) → gold` in your VERA-R sequence shape. 565 verified transitions over 299
  distinct functions (19 real self-contained fns from the user's repos + 280 procedurally-diverse
  typed fns). Mutation mix: ~half wrong-return-type (TS2322), ~half undefined-symbol (TS2304).
- **pass@k against the compiler** (`passk.py`): the headline metric. Validated with gold (1.0) /
  noise (0.0) / stuck generators. 13/13 unit tests green — the ruler is correct.
- **A trainer** (`train_repair.py`, `edit_repair.py`): warm-starts from a 14M byte-level FIM
  checkpoint (Muon optimizer, ~12× data-efficiency proven earlier), continues on repair transitions,
  `<edit>`-span up-weighted.

Held-out test fixture (never in training): `calc.ts` with `add`/`subtract`/`scale`, broken 3 ways
(a test-failure arith bug, a return-type error, an undefined symbol).

## 3. What we measured (the wall — honest, reproducible)

**Generator baseline (untrained 14M FIM):** pass@1/4/8 = 0/0/0, best-score climbs 0→0.067→0.133.

**Three repair-training experiments, all warm-started 14M, all gentle (~2–3 min):**

| # | output format | data | pass@1 | failure mode (from candidate dumps) |
|---|---|---|---|---|
| 1 | whole file | 150 ex, 4 templates | **0.0** | memorizes training bodies |
| 2 | whole file | 565 ex, 299 distinct fns | **0.0** | blends input name + training fragment → valid TS, **wrong function** (e.g. `scalestatPelUsd`, score 0.5 = parse+typecheck but wrong) |
| 3 | edit-native, 1 line | 565 ex | **0.0** | for return-type errors the compiler flags the `return` line but the fix is on the **signature** line, so line-splice can't fix it; generated lines also weak (best 0.067) |

**Verdict we reached:** across 3 output contracts and 4→299 function diversity, a **14M byte-level**
model at this training budget cannot repair even trivial type errors on held-out functions. Not the
optimizer (Muon works), not data diversity (varied, didn't move pass@1). It is **capacity + byte-level
tokenization** (exact copy-with-edit is brutal byte-by-byte: `const ` is 6 tokens to reproduce
perfectly) **+ tiny training budget**.

**Hardware reality (measured today):** the M3 Pro (18 GB) trains ~14–27M models at seq 1024 fast
(~10–14k tok/s) in short bursts, but (a) seq 2048 / ~42M chokes (>16 min, <25 steps — bandwidth
bound), and (b) sustained 20-min runs thermally throttle ~7× (12k→2k tok/s). It is a **pilot**
machine, not a training machine. Kaggle free GPU is currently blocked (phone verification).
BPE tokenizer works locally (4.24 bytes/token compression measured).

## 4. Our current read (for you to confirm or overturn)

We think the Mac path is exhausted and the real levers are exactly what docs 12/16 prescribed for the
scale run: **BPE tokenization** (kills the byte-copy problem), **real capacity (~0.3B)**, **real
training budget** — none of which the Mac can do. We're about to conclude "no more Mac iteration; the
next decisive datapoint requires a GPU run." Before we commit money/cloud, we want to know if we're
missing a cheaper route or a design error.

## 5. What we're asking you (please answer these directly)

1. **Is there a Mac-sized experiment we're missing that would still move pass@1 off zero?** e.g. a
   different output contract (structured Patch-IR from your VERA-R §4 instead of raw line/diff?), a
   different task framing, retrieval-in-context, constrained decoding to force valid edits, or a
   fundamentally different tiny-model setup. Or is pass@1>0 genuinely unreachable at ≤30M on a laptop?

2. **The copy-with-edit problem.** The model must reproduce the input while changing one token, and a
   byte-level tiny model blends instead. Is the right fix (a) BPE, (b) a diff/patch output that never
   reproduces unchanged text, (c) constrained/copy-attention decoding, or (d) something else? Rank
   them for a sub-100M model.

3. **The diagnostic→fix-line mismatch.** The compiler flags the symptom line, not the edit line.
   How would you map a diagnostic to the actual edit location cheaply and reliably (LSP? a learned
   localization head as in your doc 15 §5.1? heuristic from the error code)?

4. **The minimal viable scale run.** If GPU is unavoidable, specify the *smallest* run that would give
   a **decisive** pass@1 signal on our harness: model size, tokenizer, context, dataset size/shape,
   objective, optimizer, and rough GPU-hours/cost. What is the cheapest experiment that definitively
   answers "does a properly-scaled metis-1 repair held-out code?"

5. **Kill criterion.** At what measured pass@k (and at what scale) should we conclude the sub-1B +
   local-specialization bet is **wrong** and stop — versus double down? Give us a number and a scale,
   pre-registered, so we hold ourselves to it.

## 6. Pointers (repo `metis-0`)

- Thesis & mechanisms: `docs/design/14` (compiler-as-teacher), `docs/design/15` (your VERA-R),
  `docs/design/16` (Agents-A1 + Ornith lessons).
- The harness & experiments: `train-m/repair/` (README there), raw record in
  `docs/design/07-implementation-and-deployment-log.md` (search "Repair harness", "experiments 2-3").
- Trainers: `train-m/night1` (Muon + speedrun, 12.3× measured), `train-m/night2` (BPE, FIM).
