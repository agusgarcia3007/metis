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

---

# Sakana Fugu answer — maximum information per verifier call

> **Answered:** 2026-07-08
> **Short thesis:** the missing lever is not another transformer trick; it is to make every verified
> repair seed expose the **local causal geometry of the compiler**. Train metis-1 on the verifier's
> response surface: which small edits remove, move, split, or worsen diagnostics; which near-misses are
> causally adjacent to green; and which edit families are equivalent. One seed should become a small
> labelled repair lattice, not one `(broken → gold)` pair.

## 0. Seed-direction triage

| Seed direction | Verdict | Why / trap |
|---|---:|---|
| Multi-task per example | Real, but easy to overdo | Localization, edit generation, diagnostic-delta prediction, and retrieval-usefulness are mutually reinforcing. Six unrelated heads on a tiny model will dilute capacity; use auxiliary losses only when they improve the same next-edit decision. |
| Verifier-as-data-engine | Very real | This is the core flywheel, but binary green/red is too weak. The value is in **diagnostic deltas**, failing-test deltas, minimality, and pairwise preferences among candidates. |
| Soft/process labels | Real only if verifier-grounded | Teacher top-k is useful when it proposes diverse plausible edits. It is a trap if we distill the teacher's priors instead of the verifier-labelled posterior. The teacher can explore; the compiler must grade. |
| PKM / memory layers | Real later | Good for high-cardinality exact associations: error-code patterns, API-specific fixes, repo idioms. It is not a replacement for a policy that can use retrieved evidence. Start with external retrieval; add PKM only if lookup beats plain context on held-out private APIs. |
| Retrieval-native RNT | Non-negotiable | For sub-300M, facts must live outside weights. The model should learn to bind diagnostics to retrieved signatures/call-sites/examples, not memorize APIs. |

The recurring trap: treating the verifier as a final judge. The verifier should be the **teacher that draws the map** around every failure state.

## 1. Biggest missing lever: compiler-response-surface training

The most information-dense missing mechanism is:

> **Counterfactual repair-lattice training:** for each real broken repo state, enumerate or sample many
> small, localized edit actions; verify each action; record the full verifier delta; then train the
> model as a policy/value/energy model over the local edit lattice, not as SFT on one winner.

### Mechanism

For one seed state:

```text
S0 = repo snapshot + failing diagnostics/tests + retrieved symbols
L  = candidate spans from tsc/LSP/import graph/test links
A  = small edit actions over L
     - replace type annotation
     - widen/narrow union
     - add missing property
     - change call argument order/count
     - add import
     - wrap await / remove await
     - add null guard
     - change return expression
     - replace identifier with in-scope compatible symbol
     - insert exhaustive case
```

Run the verifier for each candidate action or short action sequence:

```text
for a in A:
  S1 = apply(a, S0)
  V0 = diagnostics/tests(S0)
  V1 = diagnostics/tests(S1)
  label(a) = {
    green: bool,
    diagnostic_delta: V1 - V0,
    removed_primary_error: bool,
    introduced_errors: count/kinds,
    failing_test_delta: +/-,
    span_distance_to_root_cause,
    minimality_cost,
    touched_symbol_kind,
    verified_next_state_hash,
  }
```

Then create dense training views:

```text
(state, action) -> verifier_delta
(state, action_i, action_j) -> preference_i_over_j
(state) -> top action distribution under verifier-labelled search
(state, failed_action, new_diagnostic) -> corrective next action
(state, green_action) -> minimal Patch-IR / diff
```

This is **not** just `K candidates labelled green/red`. It is a local approximation of the compiler's
response surface: a causal map of how code edits transform verifier feedback.

### Data shape

A single training entry becomes a lattice record:

```json
{
  "state_id": "repoA:fileX:diagnosticY:seed17",
  "context": {
    "broken_span": "...",
    "diagnostics_before": ["TS2322 ..."],
    "retrieved_symbols": ["interface User ...", "call-site ..."]
  },
  "actions": [
    {
      "patch_ir": "replace_type span=signature old=string new=number",
      "diff": "@@ ...",
      "verify": {
        "green": false,
        "removed": ["TS2322"],
        "introduced": ["TS2345"],
        "score": 0.55
      }
    },
    {
      "patch_ir": "replace_return_expr ...",
      "verify": { "green": true, "score": 1.0 }
    }
  ],
  "edges": [
    { "from_action": 0, "next_diagnostic": "TS2345", "best_next_action": 7 }
  ]
}
```

### Why it extracts more per example than standard SFT

Standard SFT says: "in this state, copy this gold patch." It teaches one target string.

Repair-lattice training teaches:

1. which span is causal;
2. which edit family fits the diagnostic;
3. which almost-fix is better than which bad fix;
4. what new errors each wrong fix tends to create;
5. how to recover after a failed attempt;
6. which facts from retrieval were actually used;
7. a value estimate for search.

That is many bits of supervision from one state. It also matches inference: metis-1 will live inside a
search loop, so the model should learn the terrain that search traverses.

### The key implementation constraint

Do not enumerate arbitrary text edits. Enumerate **typed, localized edit operators**. Otherwise the
lattice is mostly nonsense and the model learns mutation artifacts. The generator can still output raw
diff as fallback, but the data engine should prefer Patch-IR operations because they produce a compact,
labelable action space.

## 2. Non-standard training formulations

### 2.1 Causal-Jacobian objective: learn diagnostic derivatives

Treat the compiler as a function:

```text
F(code) -> diagnostics/tests
```

Train on local derivatives:

```text
(code_state, edit_op) -> Δdiagnostics
```

The model does not only learn "what patch is green"; it learns the expected effect of an edit. At
inference, the Conductor asks for edits with predicted positive diagnostic gradient.

Training losses:

- action-value regression: predicted score vs verifier score;
- diagnostic-delta classification: removes TS2322, introduces TS2345, no-op, parse break, etc.;
- pairwise ranking: `edit_i > edit_j` if the verifier delta is better;
- final SFT only on green/minimal actions.

Why this is non-standard: the primary label is not the human answer; it is the **effect of an
intervention**. The model learns an inverse compiler, not a code autocomplete distribution.

Small experiment:

- Build 100–300 broken states from current repair harness.
- For each, generate `K=64` typed edits over 3–8 candidate spans.
- Verify all candidates with existing sandbox.
- Train a tiny ranker/policy head on frozen or small metis embeddings, or train a 14–40M model from scratch if convenient.
- Pre-register: candidate ranking `MRR`, top-1/top-8 action green rate, and `pass@8` when sampling by the learned value model vs random/heuristic order.
- MacBook: yes, if verification is cached and K is modest.

Kill criterion: learned ranking fails to beat simple heuristics such as "touch diagnostic span" and
"fewest introduced diagnostics" by at least 20% relative MRR on held-out files.

### 2.2 Search-tree distillation: train the tree, not the winner

Run a cheap best-first search or MCTS-like verifier loop per seed. The artifact is the tree:

```text
node: repo state / verifier output
edge: edit action
edge label: verifier delta
node value: best descendant score within budget
```

Train metis-1 to approximate:

```text
policy(state) ≈ actions that search eventually found useful
value(state, action) ≈ best reachable verifier score after taking action
repair_trace(state) ≈ compact sequence of edits to green
```

This amortizes search into the weights. The model learns not only the final patch but also which early
moves made the solution reachable.

Why it is different from ordinary SFT:

- losers are labelled, not discarded;
- dead ends become negative supervision;
- non-green but progress-making actions get credit;
- a seed that requires 2–4 edits yields a trajectory curriculum.

Small experiment:

- Use 50 seeds where a scripted/teacher/search process can find at least one green patch within budget.
- Keep a fixed verifier-call budget, e.g. 256 calls/seed.
- Compare three training targets on the same seeds:
  1. SFT on final green diff only;
  2. SFT on every green-prefix trace;
  3. policy/value distillation from the whole tree.
- Pre-register: held-out `pass@1/4/8`, mean best verifier score, and verifier calls to first green.
- MacBook: tree generation can run overnight for small repos; training small models is Mac-feasible. GPU only if moving beyond ~50M/large context.

Kill criterion: tree-distilled policy does not reduce verifier calls-to-green or improve `pass@8` over
winner-only SFT under identical search budget.

### 2.3 Equivalence-class / MDL edit grammar

Many repairs are the same latent operation with different surface names:

```text
return type mismatch       -> change annotation or returned expression
missing property           -> add field / pick correct property / widen type
possibly undefined         -> guard / optional chain / default
wrong call arity           -> insert/remove/reorder argument
async mismatch             -> add/remove await / Promise wrapper
```

Train a compact latent edit grammar:

```text
state -> edit_class -> bound_slots -> concrete patch
```

Example:

```text
edit_class: CHANGE_RETURN_TYPE_TO_MATCH_EXPR
slots:
  function_span = lines 8-8
  old_type = string
  new_type = number
patch:
  replace_lines 8 8 "export function scale(...): number {"
```

The verifier groups actions into equivalence classes by effect: two patches that remove the same
root diagnostic without regressions are equivalent even if their text differs. The model learns the
shortest description that predicts the green class.

Why this helps tiny models:

- fewer output tokens;
- more sharing across examples;
- less memorization of exact code strings;
- better compatibility with retrieval, because slots can be copied from `<lib>`.

Small experiment:

- Hand-code 8–12 Patch-IR operators covering the top TypeScript diagnostic families in the current harness.
- Auto-convert existing gold/synthetic patches into Patch-IR when possible; raw diff fallback for the rest.
- Train two same-size models:
  - compact diff target;
  - `edit_class + slots` target.
- Pre-register: parse-valid patch rate, verifier-green `pass@1/8`, output token count, and exact-slot accuracy.
- MacBook: yes for 8–12 operators and short contexts.

Kill criterion: Patch-IR improves formatting validity but not verifier score, or conversion covers too
few examples (<30%) to matter.

### 2.4 Proof-carrying patch prediction

Ask the model to emit not just a patch, but a tiny predicted certificate:

```text
<patch>...</patch>
<claim>
removes: TS2322 at src/x.ts:12
should_not_touch: tests, package config
requires: symbol User.id from <lib>
expected_remaining: none
</claim>
```

The verifier checks the claim. Wrong claims are supervision.

This adds process bits without needing chain-of-thought. It also improves rejection/ranking: a patch
with an accurate predicted verifier delta is more trustworthy than one with a hallucinated claim.

Small experiment:

- Add certificate fields to the repair-lattice dataset.
- During sampling, rank by `patch_score + claim_consistency_score`.
- Pre-register: `pass@k` and false-green/unsafe-touch rate.
- MacBook: yes.

Kill criterion: certificates are well-formed but uncorrelated with actual verifier deltas.

## 3. Capability per parameter: what pays below 300M

The most important caveat: architecture levers matter less than **problem contract**. For this task,
`retrieval + compact Patch-IR/diff + verifier-labelled search` beats almost any block-level novelty.
Within the requested levers, my ranking is:

| Rank | Lever | Verdict below 300M | Notes |
|---:|---|---|---|
| 1 | Retrieval externalization / RNT | Highest ROI | Facts, signatures, call-sites, and repo idioms should live outside weights. Test with private/mutated APIs to ensure the model uses `<lib>`. |
| 2 | Tokenizer choice | Mandatory | Code BPE/unigram with byte fallback is a direct capacity multiplier. Byte-level repair wastes parameters on spelling and whitespace. Use 8k–16k for Mac pilots, 32k-ish for larger runs. |
| 3 | Edit-native output / Patch-IR | Highest ROI, though not a parameter trick | A 50-token target is easier than a 1,000-token target. This should be treated as a capability-per-parameter lever. |
| 4 | Retrieval-conditioned small value/ranker model | High ROI | A tiny model that ranks verifier-labelled actions can guide search better than a larger model generating raw patches unaided. |
| 5 | PKM / memory layers | Promising second wave | Useful for exact high-cardinality mappings: diagnostic templates, API idioms, repo-local repair patterns. Keep active lookup small. Validate against plain retrieval first. |
| 6 | Depth over width | Usually pays | For reasoning over diagnostics, prefer moderately deeper/narrower over shallow/wide, until latency or optimization breaks. Too narrow hurts copying slots. |
| 7 | Weight tying | Free but small | Tie input/output embeddings where compatible. Helps footprint; will not rescue capability. |
| 8 | Cross-layer parameter sharing | Maybe for ultra-small pilots | Can improve memory footprint but often reduces specialization across layers. Test only if deployment memory is the blocker, not during capability discovery. |
| 9 | MoE | Mostly distraction at this scale/local target | Sparse MoE adds routing/kernels/serving complexity and needs enough data per expert. Consider only a coarse external router over tools/operators, not transformer MoE. |
| 10 | Hypernetworks | Distraction | Too much complexity and too few trustworthy labels. Repo adapters/LoRA-style deltas are more plausible than full hypernets, but not first. |

Practical architecture recommendation:

```text
50M-125M dense BPE transformer
+ compact diff/Patch-IR output
+ deterministic retrieval
+ small verifier-delta/value head
+ optional PKM only after retrieval baseline is strong
```

Do not spend the first scarce month on MoE/hypernets. Spend it on making every verifier call produce
training signal and every output token carry edit semantics.

## 4. Data flywheel under real-data scarcity

### Is in-repo mutation the right seed?

Yes. It is the right unlock for TypeScript/compiler repair because isolated verification threw away
the natural unit of correctness: repo context. Real code is rarely self-contained; the repo's own
`tsc`, tests, imports, path aliases, and config are part of the example.

But naive in-repo mutation will collapse into a fancier breaker unless the mutation distribution is
anchored to real failures and evaluated against non-mutated tasks.

### The flywheel I would build

```text
working repo snapshot
  -> extract symbols/types/call-sites/tests
  -> generate natural counterfactual breaks
  -> verify broken state actually fails for a meaningful reason
  -> enumerate typed repairs / run teacher search
  -> verify candidate repairs
  -> store repair lattice + retrieval bundle
  -> train policy/value/edit model
  -> use model to propose harder breaks and repairs
  -> periodically evaluate on historical and hand-authored real tasks
```

### How to avoid synthetic-mutation memorization

1. **Mutate from real bug priors, not random syntax.**
   Build a histogram from real TypeScript diagnostics, PR diffs, StackOverflow-like error families,
   and agent-failed patches. Sample mutation operators according to that prior. Avoid toy mutations
   such as only flipping `+/-` or simple type names.

2. **Use inverse-real commits when possible.**
   If a historical commit changes `old -> new` and the repo can build at `new`, try applying
   `new -> old` inside the modern/historical repo to recreate a plausible bug. This gives a more
   natural break than arbitrary corruption.

3. **Require semantic locality but surface diversity.**
   Keep the root cause small, but vary where the diagnostic appears: signature, call-site, inferred
   generic, config-driven import, async boundary, test assertion. The model must learn localization,
   not just edit the line `tsc` reports.

4. **Mine from agent failures.**
   Let larger models/frontier agents attempt repairs, then use the verifier to label their wrong
   patches and recovery paths. Agent mistakes are closer to the distribution Metis will see than
   random breakers.

5. **Preserve repo context in the example.**
   Store the retrieved signatures, import graph, nearby call-sites, failing test names, and the exact
   verifier command. Do not reduce examples to isolated functions.

6. **Hold out by repo, symbol family, diagnostic family, and mutation operator.**
   If train and eval share the same breaker/operator, the model can memorize the breaker. Use at least
   four eval slices:
   - seen repo / unseen file;
   - unseen repo;
   - unseen diagnostic family;
   - unseen mutation operator.

7. **Add private/mutated API canaries.**
   Invent APIs whose only definition is in `<lib>`. If the model solves public APIs but fails canaries,
   it is memorizing weights instead of using retrieval.

8. **Keep a small real historical benchmark even if expensive.**
   The training flywheel can be in-repo mutation, but claims need a real benchmark that includes
   historical logic bugs and test failures.

### Dataset acceptance filters

A generated break enters training only if:

- the clean repo verifies green before mutation;
- the mutated repo fails with a stable diagnostic/test failure;
- the root-cause patch is small and reversible;
- at least one verified repair is found;
- the repair does not touch tests/config/lockfiles unless the task class allows it;
- the example includes retrieval evidence needed to solve it;
- duplicate surface patterns are downweighted.

### Avoiding the "fancier breaker" trap

Track a **mutation-to-real transfer gap**:

```text
pass@k(mutated held-out) vs pass@k(real historical held-out)
```

If mutated `pass@8` climbs while real historical `pass@8` stays flat, the flywheel is teaching breaker
artifacts. In that case, increase inverse-commit breaks, agent-failure recovery data, and test-failure
historical tasks; do not just add more mutations.

## 5. Cheap-first experiments and pre-registered metrics

### Experiment A — Repair-lattice ranker

Purpose: validate the biggest lever before training a full generator.

Protocol:

- Select 100–300 in-repo broken states.
- Generate 64 typed candidate edits each.
- Verify all candidates.
- Train a small ranker/value head to score `(state, action)`.
- Use the ranker to order candidates under fixed verifier budget.

Metrics:

```text
MRR of first green action
Top-1 / Top-8 green action rate
mean_best_score@8
verifier_calls_to_first_green
```

Baseline:

- random candidate order;
- heuristic order: diagnostic line first, smallest patch first, no introduced diagnostics.

Pilot: MacBook yes.

Pass criterion:

```text
Top-8 green rate improves >= 25% relative over heuristic order
or verifier_calls_to_first_green drops >= 30% at same solved rate
```

### Experiment B — Patch-IR vs compact diff

Purpose: test whether latent edit classes improve capability per token.

Protocol:

- Implement 8–12 Patch-IR ops for common TS repair families.
- Convert examples where possible; keep raw diff fallback.
- Train equal-size BPE models on:
  1. compact unified diff;
  2. Patch-IR + slots;
  3. hybrid Patch-IR-or-diff.

Metrics:

```text
pass@1/4/8
parse_valid_rate
apply_valid_rate
output_tokens_per_candidate
slot_accuracy
unsafe_touch_rate
```

Pilot: MacBook yes for 14–40M and short contexts; GPU for 100M+.

Pass criterion:

```text
Patch-IR/hybrid improves pass@8 or mean_best_score@8 with lower invalid/unsafe rate,
not merely better formatting.
```

### Experiment C — Search-tree distillation

Purpose: test whether the model can learn from the whole verifier search process.

Protocol:

- For 50–100 seeds, run fixed-budget best-first search using typed edit ops plus optional teacher proposals.
- Save all nodes/edges/verifier deltas.
- Train:
  1. winner-only SFT;
  2. green-prefix SFT;
  3. policy/value tree distillation.

Metrics:

```text
pass@1/4/8 under same sampling budget
mean_best_score@k
verifier_calls_to_first_green
recovery_rate_after_first_failed_patch
```

Pilot: MacBook yes for tree generation at small K; GPU only for larger model runs.

Pass criterion:

```text
tree-distilled model beats winner-only SFT on pass@8 or calls-to-green.
```

### Experiment D — Mutation-to-real transfer benchmark

Purpose: kill or validate in-repo mutation as a real data source.

Protocol:

- Train on in-repo mutations only.
- Evaluate on four held-out slices:
  1. held-out mutations from seen repos;
  2. held-out mutation operators;
  3. unseen repos;
  4. small historical build-harness tasks.

Metrics:

```text
pass@1/4/8 per slice
mean_best_score@8 per slice
transfer_gap = pass@8(mutated_seen) - pass@8(historical)
retrieval_canary_pass@8
```

Pilot: MacBook for data generation and small model; historical harness may be slow but can run as a
nightly subset. GPU only if scaling model/corpus.

Pass criterion:

```text
historical pass@8 moves above baseline and transfer_gap does not widen as mutated score improves.
```

### Experiment E — Proof-carrying patch ranking

Purpose: determine whether predicted verifier deltas are useful for ranking.

Protocol:

- Add `<claim>` targets to lattice actions.
- Sample candidates with and without claims.
- Rank by patch probability alone vs patch probability + claim consistency.

Metrics:

```text
pass@k
claim_accuracy
claim_consistency_vs_actual_delta
unsafe_touch_rate
```

Pilot: MacBook yes.

Pass criterion:

```text
claim-aware ranking improves pass@k or reduces unsafe/irrelevant patches without lowering diversity.
```

## 6. Tactical rider: in-repo mutation vs full historical build harness

The pivot to **in-repo mutation** is the right immediate unlock. The isolated miner's `0` transitions
is not surprising: isolated verification removes the dependency graph that makes TypeScript real.
For compiler/type/lint repair, a clean repo plus controlled mutation gives stable, cheap, unlimited
labels.

But a full historical build harness is still worth building as a **small evaluation and calibration
track**, not as the first bulk data engine.

Recommended split:

```text
80-90% effort: in-repo mutation + repair lattices + verifier-labelled search
10-20% effort: historical checkout/install/tsc/test harness for benchmark slices
```

Why not make historical first:

- dependency installation and old toolchains are slow/flaky;
- many commits are not single-bug repair transitions;
- setup pain can consume weeks before the model learns anything;
- the compiler-repair flywheel needs high-throughput labels now.

Why not skip historical entirely:

- logic bugs often require tests, not `tsc`;
- real PRs include multi-file intent and non-local constraints;
- mutation success can be fake progress;
- only historical/agent-failure tasks can measure real-distribution transfer.

Concrete plan:

1. Build in-repo mutation as the main training source.
2. In parallel, create a tiny historical harness for maybe 30–100 commits across 2–4 repos.
3. Use historical tasks only as a periodic held-out gate until the harness is reliable.
4. Promote historical mining to a larger data source only if it yields stable, isolated-ish repair
   episodes at acceptable cost.

Pre-registered tactical metric:

```text
A model trained on in-repo mutation must improve pass@8 on the historical slice
within two training iterations. If not, the mutation distribution is misaligned.
```

## 7. What I would do next in metis-1

Ordered implementation path:

1. **Implement typed edit-op enumeration** for the top 8–12 TS repair families.
2. **Store repair lattices**, not only final patches: candidates, verifier deltas, pairwise prefs,
   retrieval bundle, and search edges.
3. **Train a small action ranker/value model first** before a full generator. If ranking does not work,
   generation will not magically work.
4. **Add compact Patch-IR/hybrid output** once the lattice has enough green and near-green actions.
5. **Evaluate every run on transfer slices**, especially unseen operator and historical mini-harness.
6. **Only then scale parameters** to 100M–300M or add PKM.

The bet: a tiny local model can win if it becomes an excellent amortized guide over a verifier-labelled
repair landscape. It should not try to store all programming knowledge in weights; it should learn how
compiler feedback changes when code is edited, and how to exploit retrieval plus search to move toward
green.

## 8. The five assumptions, challenged directly

The core question asked us to break specific default assumptions. Answering them one by one, mapped to
the mechanisms above:

| Assumption to break | Verdict | Replace it with |
|---|---|---|
| One training example teaches one thing | False | One broken seed becomes a **repair lattice**: many verified `(state, action) → delta` views, pairwise preferences, recovery edges, and a value target (§1, §2.1). N supervision signals per entry, not one. |
| The label is the human/gold answer | False and limiting | The label is the **verifier's response to an intervention** — diagnostic/test deltas, green/near-green ranking, equivalence class of the fix. The gold diff is just one green leaf of the lattice (§1, §2.3). |
| Capability must be stored in parameters | False below 300M | Push facts and APIs into **retrieval/RNT** and exact high-cardinality mappings into optional **PKM**; keep weights for the *policy over edits*, not for memorized code (§3). |
| More capability needs more data or more parameters | False here | The free verifier manufactures dense supervision from a handful of repos (verifier-as-data-engine), and **search distilled into weights** (§2.2) amortizes compute into capability without more real data or more parameters. |
| The verifier is only for eval/reward | The biggest miss | Promote the verifier to the **primary data engine and teacher**: it draws the local causal map around every failure, labels losers and near-misses, and grounds soft/process labels. This reframing is the throughline of the whole answer (§0, §1, §4). |

Net: the paradigm shift is from *one example → one label → one gradient step* to *one seed → a
verifier-labelled repair landscape → many gradients over policy, value, preference, and recovery*.
