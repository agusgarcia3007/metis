# Metis — the real lever: Generate · Verify · Search (GVS)

> The honest answer to "compete with frontier on quality with something tiny." Not a magic model — a
> different *use* of the model, grounded in the deepest pattern across the literature.

## 0. The pattern that connects every paper

Reading ~50 papers, two asymmetries explain almost every efficiency/quality result:

1. **Predictability → compute** (the *speed* lever). Speculative decoding, Mixture-of-Depths,
   early-exit, cascades, KV reuse, retrieval, extractive answering — all the same idea: *spend
   compute only where the output is genuinely novel/uncertain; skip the predictable majority.*
2. **Verification < Generation** (the *quality* lever, the important one). It is cheaper to *recognize*
   a correct answer than to *produce* one. So a **weak generator + a reliable verifier + search**
   reaches quality far above the generator's single-shot ability — and emits only verified claims, so
   hallucination → ~0.

The second asymmetry is the one that can move "tiny" toward "frontier" **on a verifiable surface**.

## 1. The evidence (research 11) — quantified, with the wall

**It works (and the size deltas are large):**
- Cobbe 2021: **6B + verifier (best-of-100) ≈ 175B** on GSM8K (20.6% → 55.2%).
- GenRM (DeepMind): GSM8K **73% → 93.4%**; 6.4× more sample-efficient than discriminative verifiers.
- Snell 2024: a small model + compute-optimal test-time search **beats a 14× larger model** on
  medium-hard MATH-500.
- **Easy-to-hard generalization**: verifiers generalize across difficulty where generators don't —
  recognizing a correct step doesn't require being able to produce one.
- **ThinkPRM-1.5B** (tiny generative verifier, 8k labels) **beats discriminative PRMs trained on 100×
  more data**. Verifiers can be small.
- **The tiny grounded verifier already EXISTS for our surface** (research 11): **MiniCheck-FT5 (770M)
  checks if a claim is supported by a document within 0.6% of GPT-4, at ~446× lower cost**;
  AlignScore (355M) beats GPT-4-based G-EVAL on factual consistency; Bespoke-MiniCheck-7B beats GPT-4o.
  So the verifier we need is not hypothetical — it's a 0.4–0.8 GB model (or our 1.7B in judge mode,
  measured at 90–100%).

**The precise operating envelope (where it beats scale):**
1. **Verifiable domain** — correctness is checkable (math, code-with-tests, **factual/grounded QA**).
2. **Competence floor** — the generator must have non-trivial pass@1; search selects, it can't conjure.
3. **EXTERNAL verifier, never self-correction** — same-model intrinsic self-correction *fails*
   (Huang 2023). The "Self-Correction Illusion": **+77 pp** when the error is framed as external vs
   the model's own reasoning. → verify a *separate claim against evidence*, don't ask "are you sure?"
4. **Grounded / rule-based verification, not neural reward** — neural reward models get *gamed* (reward
   hacking; exploit rate 0.6%→13.9% after RL). Entailment-to-a-source and exact-checks are robust.
5. **Verifier above a capability floor** (can be smaller than the generator; tolerates ~15% noise).

**The architecture is already published and beats frontier on grounded/structured tasks (research 12):**
**MCTS-RAG (Llama-3.1-8B + search) > GPT-4o on GPQA (71.3 vs 54.9)** and ComplexWebQA; Search-R1
(Qwen-7B) +238% on Musique; **MinionS (3B local) recovers 93.4% of GPT-4o at 16.6% cost**;
Speculative-RAG gets +13% accuracy AND −51% latency at once. The class works — this is not hypothesis.

**The wall (where scale still wins — stated honestly, research 11/12):**
1. **No-verifier tasks** — creative/subjective/evidence-free; BoN fails entirely, self-correction is net
   negative. 2. **Coverage floor** — search only selects what the model *can* generate; if pass@1≈0, no
   N helps (proven; RL doesn't expand coverage). 3. **Verifier ceiling/collapse** — top PRM 78→37% on the
   frontier tail. 4. **Latency** — N=128 ≈ 134 s; **interactive CPU forces tiny N** (verify-then-rarely-
   search, never big best-of-N). These bounds *define* our design, they don't refute it.

## 2. We measured the premises on OUR 1.7B model (not hand-waved)

| premise | measurement | result |
|---|---|---|
| answers are copyable (speed) | 3-gram copy-rate of grounded answers vs source | **64%** |
| the model can VERIFY groundedness (external framing) | accept-correct vs reject-wrong, easy negatives | **100%** (10/10) |
| …on HARD negatives (subtle off-by-one, plausible-unsupported) | 10 tricky claims | **90%** (1 miss = a numeric-bound edge) |
| verify-gate kills hallucination | naive vs verify-gated on unanswerable questions | naive **1/3** fabrications → verify-gated **0/3** |

The 1.7B is an unreliable *generator* but a reliable *grounded verifier* — exactly the asymmetry,
in the exact regime the research says works (external + grounded). Scripts: `/tmp/verify_hard.py`,
`/tmp/halluc.py`; copy-rate `/tmp/copyrate.py`.

## 3. The architecture (adaptive, so it stays fast on CPU)

```
query
  ├─ EXTRACTIVE fast-path (embedder picks the answer span)  ──confident?──► answer (~0.1 s) [done]
  └─ else GENERATE 1 grounded answer
            └─ VERIFY it against the retrieved evidence (external, grounded judge)
                 ├─ verified  ──► emit, with citation                 (most cases: 1× generate + cheap verify)
                 ├─ unsupported ──► ABSTAIN ("not in the knowledge")   (zero hallucination)
                 └─ wrong/uncertain ──► SEARCH: generate a few more, verify, keep the best   (rare, costs more)
```

Key design choices forced by the research + our CPU budget:
- **Verify-then-maybe-search**, not always-best-of-N: most answers pass on the first try (verify is a
  short yes/no, cheap); only uncertain ones pay the N× search cost. This is "compute ∝ uncertainty"
  applied at the query level — the speed lever and the quality lever are the *same* mechanism.
- **Grounded verifier** = entailment of the candidate against retrieved evidence (rule-ish, not a
  gameable neural reward). For non-grounded reasoning, fall back to rule checks (calc/code-run) where
  they exist.
- **Abstention is a feature**: emitting only verified claims makes a tiny local model *trustworthy*,
  which is itself a frontier-competitive property (frontier models still hallucinate).

## 4. What this is — and isn't

- **Is:** a path to **frontier-grade TRUSTWORTHINESS on the verifiable/groundable surface** (factual
  QA, code, math, anything with evidence or a checker) at tiny size, fully local. This is a real
  competitive axis: frontier models still hallucinate citations **14–95%** of the time; a verify-gated
  tiny model that emits *only* claims entailed by evidence (and abstains otherwise) can beat them on
  *reliability* where it matters. The quality comes from *recognition + search* (which a small model
  can do) and *grounding* (which removes hallucination), not from *generation alone* (which it can't).
  The unsolved sub-problem the whole field is stuck on — a small general grounded verifier — is
  **already solved for single-document entailment** (MiniCheck), which is exactly our surface.
- **Isn't:** beating frontier on open-ended creativity, taste, multi-hop novel reasoning, or
  evidence-free questions. Those have no compact verification signal (creative judge ≈73%, multi-hop
  OOD ≈0%); scale still wins. We name that wall instead of hiding it.

## 5. Build plan (turning this into the product)
1. **Verify layer** (now): a grounded-judge pass over the generated answer; abstain if unsupported.
   Wire into `ask`/`serve`/`chat`. (Verifier discrimination already measured at 90–100%.)
2. **Verify-then-search**: on a failed verify, sample a few more and keep the best verified one.
3. **Multi-vote / cross-check** for hard cases (Weaver: ensembles of weak verifiers; a 400M distilled
   verifier retains 98% at 0.03% the compute) — and a tiny distilled verifier so verification is cheap.
4. **Engine pivot → llama.cpp** so generate+verify+search fits the latency budget (research 07/09).

*Status: premises measured on our model (§2); operating envelope from research 11; two more reports
(tiny open-ended verifiers; search-vs-scale laws) will sharpen §1 and the verifier-size choice.*
