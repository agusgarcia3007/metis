# Metis — Phase 4: the reasoning-compiler thesis, and the first gating experiment

> Phase 3 left Metis honest about what it is: a small model with a well-tuned RAG. Good systems
> engineering, not a new way to build intelligence. This phase asks the harder question — can the
> capability that *defines* a frontier model (reasoning depth) be moved OUT of the weights, the way
> RAG moved knowledge out of them? We turned that into one falsifiable experiment (H1) and ran it.
> The naive thesis failed its own kill criterion. The failure located the real lever precisely.

---

## 1. The thesis: reasoning as a compiler, not a model

RAG's insight was "knowledge does not have to live in the weights." The parallel bet for Metis:
**reasoning does not have to live in the weights either.** Reframe the LLM from *the reasoner* to a
cheap, replaceable micro-component with three jobs a small model already does acceptably — propose
one atomic step, verify one atomic step against evidence, emit a small program. Everything else —
planning, composition, backtracking, memory — lives *outside* the weights, in an external engine
(the Conductor) over the Library.

The load-bearing assumption, stated so it can be killed:

> The `verify < generate` asymmetry **grows as the step shrinks**. A model that cannot reliably
> *chain* a 3-hop fact can still reliably *verify* each 1-hop fact, so an external engine that only
> ever asks it to verify atomic steps can compose arbitrary depth — error stops compounding because
> every step is externally gated.

If that is true, depth becomes buyable with search + verification instead of parameters. If it is
false, the whole engine is moot. So we test it before building anything.

## 2. H1 — verifier accuracy vs reasoning depth

`src/bin/h1.rs`. We hold the **evidence constant** (the relation tables from `bench/corpus/`:
component → codename, component → working group, working group → chair, component → budget) and vary
**only the depth of the claim**: 1, 2, or 3 reasoning hops. For each depth we present both supported
claims and *plausibly* unsupported ones — the wrong answer is another real value from the same
evidence (the realistic "fabrication" negative, not obvious nonsense). We run the **exact production
verifier** (`VerifierKind::Llm`, the grounded LLM-judge) across three model sizes.

Metrics per cell: **TPR** (supported claims confirmed — recall), **TNR** (unsupported claims rejected
— precision), **fab%** (unsupported claims waved through as SUPPORTED — the dangerous error), and
**balanced accuracy** = (TPR+TNR)/2.

Pre-registered **kill criterion**: the 0.6B's balanced accuracy at g1 (atomic) must be ≥ ~0.90, or
the "atomic steps are reliably verifiable" foundation is gone.

### Results (40 items × 3 models; n=8/8 per cell at g1/g2, 4/4 at g3)

| model | g | TPR (recall) | TNR (precision) | fab% | bal-acc |
|---|---|---:|---:|---:|---:|
| qwen3:0.6b | g1 | **1.000** | 0.375 | 62 | 0.688 |
| qwen3:0.6b | g2 | 0.750 | 0.500 | 50 | 0.625 |
| qwen3:0.6b | g3 | 0.250 | 0.750 | 25 | 0.500 |
| qwen3:1.7b | g1 | **1.000** | 0.750 | 25 | 0.875 |
| qwen3:1.7b | g2 | 0.750 | 0.750 | 25 | 0.750 |
| qwen3:1.7b | g3 | 0.000 | 1.000 | 0 | 0.500 |
| qwen3:4b | g1 | **1.000** | 0.375 | 38 | 0.688 |
| qwen3:4b | g2 | **1.000** | 0.500 | 0 | 0.750 |
| qwen3:4b | g3 | **1.000** | 0.500 | 0 | 0.750 |

Raw data: `bench/results-h1.json`. The sample is small — read the *pattern*, not the third decimal.

## 3. The kill criterion failed — and that is the result

0.6B balanced accuracy at g1 = **0.688**, well under the 0.90 line. As written, the thesis is
**falsified**: atomic granularity alone does *not* make the small model a reliable verifier. We do
not move the goalpost. But the *shape* of the failure splits the foundational asymmetry cleanly in
two, and only one half holds.

**Recall (confirming a true claim) is cheap, and decomposition removes the gap entirely.**
At g1, **all three sizes have TPR = 1.000** — the 0.6B confirms every true atomic fact, same as the
4B. And recall is exactly what collapses with depth for the small models: the 0.6B's TPR falls
1.00 → 0.75 → **0.25** as the chain grows to 3 hops, while the 4B holds **1.00 at every depth**. That
4B-vs-0.6B recall gap at g3 (1.00 vs 0.25) is the reasoning-depth-scales-with-size effect, isolated.
**But it is an artifact of asking the small model to verify the whole chain at once.** Decompose the
3-hop claim into three 1-hop claims and each is verified at TPR 1.0 regardless of size. So on the
recall axis the "compose externally" move is *validated*: an engine that decomposes can confirm
correct atomic steps with a 0.6B as reliably as a 4B. This half of `verify < generate` is real.

**Precision (rejecting a plausible falsehood) is not cheap, and does not scale with size.**
This is the bottleneck, and the surprise. At atomic granularity the 0.6B rejects only 37.5% of false
claims (TNR 0.375) — and **the 4B is no better (0.375)**. Precision does not buy itself with seven
times the parameters. The small model confidently waves through 5 of 8 plausible falsehoods; the 4B
is less *confident* about it (it marks some "uncertain") but still fails to reject them. The one
bright cell — the 1.7B's 0.75 at g1 — does not survive depth.

So the fabrication problem that Phase 3 located in tier-4 (the small model fabricates more, 3 vs 0)
is **not primarily a model-size problem**. It is a verifier-*precision* problem, and precision is
flat across the sizes we can run. `verify < generate` holds for recognizing truth and fails for
rejecting plausible falsehood — and on this surface, rejecting falsehood is roughly as hard as
generating, and not bought by scale.

## 4. This unifies the E2 failure and the LLM-judge — they are mirror images

Phase 2's E2 (generic NLI verifier) failed by **over-abstaining**: it scored correct-but-paraphrased
answers as NEUTRAL → low TPR. The LLM-judge fails the opposite way: it rubber-stamps plausible
falsehoods → low TNR. **Neither is calibrated.** They sit at opposite ends of the same precision /
recall trade. That reframes "build a cheap specialized verifier" from a vague aspiration into a
sharp, two-dimensional target defined by data:

> A verifier that holds **TPR ≈ 1.0 AND lifts TNR ≈ 1.0 at atomic granularity for a sub-1B model.**

The recall half is already free (TPR 1.0 at g1, any size). The entire problem is atomic-step
precision. That is the next lever, and H1 turned it from "the model is too small" into one measurable
quantity to move.

## 5. A production bug found on the way (fixed)

The first H1 run reported the 4B rejecting *everything* (TPR 0.0 at all depths). It was a parser
artifact, not behavior: Qwen3 emits a `<think>…</think>` block that **echoes the prompt** — including
the words "SUPPORTED or UNSUPPORTED" — and `parse_llm_verdict` substring-scanned the whole reply,
catching the echoed "UNSUPPORTED" before the model's actual final verdict. Every reasoning-model
verdict was being flipped to Unsupported. Fixed in `src/verifier.rs`: strip the think block and read
the verdict from after `</think>`. Unit-tested (`verdict_ignores_thinking_block_echo`). This matters
in production whenever the verifier runs a thinking model — it was silently turning the judge into a
reject-everything oracle.

## 6. Honest status and the next lever

What we now know that we didn't this morning:
1. The reasoning-compiler thesis is **half right, and the right half is the load-bearing one for the
   engine**: decomposition to atomic steps makes recall perfect at any model size, which is exactly
   what lets external composition replace parametric chaining depth.
2. The thesis is **half wrong in the half we must now fix**: atomic-step *precision* is the
   bottleneck, it is low even at atomic granularity, and it does not scale with model size.
3. `verify < generate` is **not one asymmetry but two**, and only the recall one is favorable.

Next lever, now sharply defined: **lift atomic-step TNR for a sub-1B model while keeping TPR = 1.0.**
Candidates to test (cheap, no training): a two-sided verify (prove-supported AND prove-refuted, then
compare); a "find the exact evidence span and check the value" framing instead of a holistic
judgment; a contrastive verify that pits the claim against its sibling values from the same evidence.
The win condition is a single number moving: 0.6B g1 TNR from 0.375 toward 1.0.

The kill criterion did its job. We did not get to keep the thesis we started the day with, but we
ended it with a smaller, sharper, measurable problem — which is the whole point of writing the kill
criterion down first.
