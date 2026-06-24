# Research: Verification < Generation — the quality lever (2021–2026)

> The deepest cross-paper pattern for "tiny rivals big on QUALITY": recognizing a correct answer is
> cheaper than producing one. With an external verifier + search, a weak generator punches up — and
> emits only verified claims (hallucination → ~0). Bounded by a precise envelope.

## TL;DR
- **The asymmetry is real & large.** 6B+verifier (best-of-100) ≈ 175B on GSM8K (Cobbe). GenRM 73→93%.
  Snell 2024: small model + compute-optimal search **beats a 14× larger model** (medium-hard MATH).
  Verifiers **generalize across difficulty** where generators don't (easy→hard).
- **Tiny verifiers exist for GROUNDEDNESS** (our surface): **MiniCheck-FT5 770M within 0.6% of GPT-4**
  at ~446× lower cost; **AlignScore 355M beats GPT-4-based G-EVAL**; Bespoke-MiniCheck-7B beats GPT-4o.
  **ThinkPRM-1.5B** (8k labels) beats discriminative PRMs trained on 100× more data.
- **MUST be an EXTERNAL verifier, not self-correction.** Intrinsic self-correction FAILS (Huang 2023).
  "Self-Correction Illusion": **+77 pp** when the error is framed as external vs the model's own
  reasoning. → verify a *separate claim against evidence*; never "are you sure?".
- **MUST be grounded/rule-based, not neural-reward** for robustness. Neural RMs get gamed (overopt:
  proxy↑ gold↓; RM-Bench: SOTA RMs **46.6%, below random** on style; rubric-RL hacks within 68–478 steps).
  DeepSeek-R1 hits MATH 97.3% with **rule-based** rewards only.

## The operating envelope (where verify+search beats scale)
1. **Verifiable** domain (compact checkable ground truth: entailment/math/code). ← grounded QA qualifies.
2. **Competence floor**: generator needs non-trivial pass@1 (search selects, can't conjure capability).
3. **External** verifier (different model / role / formal checker) — not self-verification.
4. **Grounded / rule-based** signal (not a gameable learned reward).
5. Verifier above a capability floor (can be smaller than the generator; tolerates ~15% noise; a 7B RM
   *degrades* with N on a strong generator → size the verifier to the generator).

## What is SOLVED vs UNSOLVED at small scale (the honest map)
| capability | best small verifier | size | vs GPT-4 |
|---|---|---|---|
| single-doc entailment / groundedness | MiniCheck-FT5 | **770M** | within 0.6% |
| factual consistency (summarization) | AlignScore | **355M** | beats G-EVAL-4 |
| RAG hallucination detection | Lynx-8B | 8B | > GPT-3.5 |
| grounded RAG generation (critique tokens) | SELF-RAG | 7B | PopQA 54.9 vs ChatGPT 29.3 |
| rubric judging (in-distribution) | Prometheus-2 | 7B | ≈ GPT-4 |

| UNSOLVED at small scale | why | best number |
|---|---|---|
| creative-writing quality | no ground truth; human–LLM ICC 0.43 | best judge 73–78% |
| multi-hop reasoning correctness | weakest-link; 0% OOD on unseen planning domains | 15% MuSiQue BoN-4 |
| open-ended subjective | humans disagree; judge biases (position 44.8% flip, length, self-pref) | 60–71% human agree |
| reward models OOD / style | gamed; below random on hard style | RM-Bench 46.6% |
| RL with open-ended verifier | reward hacking onset step 68–478, gold −18–25% | no general solution |

**Core reason:** cheap verification needs a **compact, unambiguous, checkable signal.** Grounded
entailment, math, code have it. Open-ended creativity/judgment/novel-reasoning don't — and there
scale still wins. This gap is about *verification signal*, not model scale.

## Retrieval-grounding AS verification (our path)
SELF-RAG critique tokens; RARR retrofits attribution (AIS +8–15 pp); ALCE shows citation works on
factoid (ASQA recall ~85%) but collapses on explanatory ELI5 (correctness 18%); FaithDial training
cuts hallucination 55.8→19.9%. Baseline LLM citation hallucination is **14–95%** → a verify-gate that
emits only entailed claims is a large, real trustworthiness win.

## Measured on OUR Qwen3-1.7B (premises hold)
- grounded verification: **100%** easy / **90%** hard negatives (external+grounded framing).
- verify-gate kills hallucination: naive **1/3** fabrications on unanswerable Qs → gated **0/3**.

### Sources (selected)
- Cobbe https://arxiv.org/abs/2110.14168 · Lightman(PRM) https://arxiv.org/abs/2305.20050 · GenRM https://arxiv.org/abs/2408.15240 · Snell(TTS) https://arxiv.org/abs/2408.03314
- Huang(self-correct fails) https://arxiv.org/abs/2310.01798 · Self-Correction Illusion https://arxiv.org/html/2606.05976 · Overopt https://arxiv.org/abs/2210.10760 · RM-Bench https://arxiv.org/abs/2410.16184
- MiniCheck https://arxiv.org/html/2404.10774v2 · AlignScore https://arxiv.org/abs/2305.16739 · ThinkPRM https://arxiv.org/abs/2504.16828 · SELF-RAG https://arxiv.org/html/2310.11511
- FActScore https://arxiv.org/abs/2305.14251 · Lynx https://arxiv.org/html/2407.08488v1 · ALCE https://aclanthology.org/2023.emnlp-main.398/ · LitBench https://arxiv.org/pdf/2507.00769
