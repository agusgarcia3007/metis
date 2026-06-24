# Research: Small High-Quality LLMs (0.5B–8B), 2025–2026

> Rigorous pass. **[V]** = vendor/self-reported, **[I]** = independent/third-party. Benchmark
> caveats matter here — see "reading notes" — so trust task behavior over leaderboard deltas.

## TL;DR for tiny-llm (Cortex selection)

- **Default Cortex = Qwen3-1.7B (thinking mode), Q4** (~1.1 GB) — MATH-500 93.4, AIME'24 48.3,
  MMLU-Redux 73.9 [V]. Fits the 4 GB budget comfortably alongside KV + Library.
- **Stretch Cortex = Qwen3-4B (thinking), Q4** (~2.3–2.5 GB) — MATH-500 **97.0**, AIME'24 **73.8**,
  GPQA-D 55.9 [V]. Only if the Library/runtime footprint is kept lean (tight on a true 4 GB box).
- **Moonshot = BitNet-b1.58-2B** (~0.4 GB non-embedding, native CPU speed) — train/finetune ternary.
- Alternatives: Phi-4-Mini (3.8B, best instruction-following/coding at size), Gemma-3-4B (beats
  GPT-3.5 math by 32 pts), R1-Distill-Qwen-1.5B (pure math reasoner). Qwen3 wins on
  reasoning-per-byte + Apache-2.0 license + a built-in thinking mode our Conductor can exploit.
- **7B is NOT viable on a 4 GB VPS** (needs ~6 GB). Cortex sweet spot for 4 GB = **1.7B–4B**.

## Reading notes (why the numbers wobble)
- **MMLU is contaminated** (~29% of items show signals; Mistral dropped 13 pts on clean items). Prefer
  MMLU-Pro/Redux, GPQA-Diamond, AIME, LiveCodeBench. GSM8K is saturated.
- **Qwen3 thinking vs non-thinking is a huge split** — same Qwen3-4B: AIME'24 25.0 (non-think) →
  73.8 (think). Always label the mode. Our Conductor runs thinking mode when accuracy>latency.
- Vendor↔independent gaps are mostly **CoT vs direct-answer protocol**, not fabrication.

## Qwen3 instruct — THINKING mode [V] (arXiv:2505.09388)
| Model | MMLU-Redux | GPQA-D | MATH-500 | AIME'24 | LiveCB v5 |
|---|---|---|---|---|---|
| Qwen3-0.6B | 55.6 | 27.9 | 77.6 | 10.7 | 12.3 |
| **Qwen3-1.7B** | 73.9 | 40.1 | **93.4** | 48.3 | 33.2 |
| **Qwen3-4B** | 83.7 | 55.9 | **97.0** | **73.8** | 54.2 |
| Qwen3-8B | 87.5 | 62.0 | 97.4 | 76.0 | 57.5 |

Independent MMLU [I] (arXiv:2505.02214): Qwen3-1.7B 60.0, 4B 69.7, 8B 74.7 (3–6 pts under [V]).

## Qwen2.5 instruct [V] (for reference / fallback base)
| Model | GSM8K | HumanEval | MATH | MMLU-Pro |
|---|---|---|---|---|
| Qwen2.5-1.5B-I | 73.2 | 61.6 | 55.2 | 32.4 |
| Qwen2.5-3B-I | 86.7 | 74.4 | 65.9 | 43.7 |
| Qwen2.5-7B-I | 91.6 | 84.8 | 75.5 | 56.3 |

## Other strong small models
- **Phi-4-Mini 3.8B** [V]: GSM8K 88.6, HumanEval 74.4, MATH 64.0, IFEval 70.1 (best instruction-
  following at size; +16 pts IFEval confirmed [I]). Synthetic "textbook" data recipe.
- **Gemma-3-4B** [V]: GSM8K 89.2, HumanEval 71.3, MATH 75.6 — beats GPT-3.5 on every shared metric.
- **R1-Distill-Qwen-1.5B**: MATH-500 83.9 > GPT-4o (report 06) — narrow math reasoner.
- **SmolLM3-3B**: fully-open data recipe; Global MMLU 68.9. Best for reproducibility/research.
- **Llama-3.2-1B/3B**: prune+distill from larger Llamas; weaker than purpose-trained peers at size.

## How far the frontier moved
- **3B ≈ GPT-3.5; 4B clearly beats GPT-3.5** on math/code (Gemma-3-4B MATH 75.6 vs 43.1).
- **Where small still loses to true frontier (GPT-4o/Claude/Gemini):**
  | Capability | best ~8B | frontier | gap |
  |---|---|---|---|
  | GPQA-Diamond | 62 | ~76–80 | ~15 |
  | AIME'25 | 67 | ~86–90 | ~20 |
  | LiveCodeBench | 57 | 65–75+ | ~15–20 |
  | SWE-bench agentic | ~15–25 | 40–55+ | 2–3× |
  | long-context, rare langs, long tool chains | — | — | qualitative |

## Training tricks that make small good
- **Data quality > quantity** ("textbooks": Phi). Phi-4-Mini 3.8B/5T beats many 7B on MATH.
- **Massive overtraining** beyond Chinchilla: Qwen3-0.6B ≈ **60,000 tokens/param (~3000×)**; trades
  train compute for cheap inference. (Caveat report 02: overtrained models quantize worse → QAT.)
- **Synthetic data from prior-gen models** (Qwen2.5-Math/Coder/VL feed Qwen3).
- **Strong→weak distillation**: Qwen3 distills 235B→small at **1/10 the GPU-hours** of RL, beating
  non-distilled variants (GPQA 63.3, LiveCB 60.3 for distilled 8B).

## Memory footprint (weights; KV adds ~0.3–2 GB by context)
| Params | fp16 | Q8_0 | **Q4_K_M** | Q2_K |
|---|---|---|---|---|
| 1.5–1.7B | ~3.0 | ~1.5 | **~0.9–1.1** | ~0.6 |
| 3–4B | ~6–8 | ~3–4 | **~1.7–2.5** | ~1.0–1.3 |
| 7–8B | ~14–16 | ~7–8 | **~3.9–4.5** | ~2.8 |

## CPU tokens/sec, no GPU (Q4_K_M, generation)
| Size | 4-core x86 | 4-core ARM (RPi5) | Apple M1/M2 |
|---|---|---|---|
| <1B | 25–40 | >20 | 80–120 |
| 1–1.7B | 10–20 | 5–15 | 40–70 |
| 3–4B | 5–12 | 2–5 | 20–40 |
| 7B | 5–10 (needs ≥8 GB) | n/a on 4 GB | 30–50 |
Memory-bandwidth bound (report 03). Llamafile ~3–4× faster than Ollama on low-core. Thinking mode
emits more tokens → higher quality, slower wall-clock; the Conductor budgets it.

## Implication locked in
Cortex = **Qwen3-1.7B-thinking (default) / Qwen3-4B-thinking (stretch)** at Q4, BitNet-2B as moonshot.
Reasoning is strong at this size; knowledge/MMLU is weak → offload knowledge to the Library (report 04).

### Sources
- Qwen3 https://arxiv.org/abs/2505.09388 · Qwen2.5 https://arxiv.org/pdf/2412.15115 · Phi-4-Mini https://arxiv.org/html/2503.01743v1
- Gemma-3 https://arxiv.org/html/2503.19786v1 · SmolLM2 https://arxiv.org/pdf/2502.02737 · Llama3.2 https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md
- Indep: VT/Oxford/NVIDIA https://arxiv.org/abs/2502.11569 · Qwen3-quant https://arxiv.org/html/2505.02214v1 · MMLU-CF https://arxiv.org/pdf/2412.15194
- Overtraining https://epoch.ai/data-insights/training-tokens-per-parameter · CPU/SBC https://arxiv.org/html/2511.07425v1 · llama.cpp #3167 https://github.com/ggml-org/llama.cpp/discussions/3167
