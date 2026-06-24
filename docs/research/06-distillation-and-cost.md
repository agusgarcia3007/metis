# Research: Distillation & Training Cost (2024–2026)

> The single most important finding for tiny-llm feasibility.

## TL;DR for tiny-llm

- **A small distilled REASONER beats frontier on its specialty.** DeepSeek-R1-Distill-Qwen-1.5B
  scores **MATH-500 83.9% vs GPT-4o 74.6%**; the 7B scores 92.8%. A 1.5B model beats a frontier
  model on math. This is the proof that "small reasoner" is viable.
- **Never pretrain from scratch.** Start from Qwen2.5 / Llama 3.x / Gemma 2. Continued-pretrain +
  SFT costs **100–1000× less** than from-scratch.
- **A solo dev can build a serious specialized model for $1,000–$5,000** (rent H100s, open base,
  open teacher, SFT+DPO). QLoRA experiments cost <$10.
- **Use OPEN teacher models** (Qwen2.5-72B, Llama-3.1-70B; Apache-2.0/permissive) to avoid ToS
  problems. Anthropic/OpenAI ToS forbid using Claude/GPT outputs to train competing models, and
  Anthropic now actively detects & litigates distillation attacks.
- **Distillation transfers reasoning, not broad knowledge.** Gains are domain-specific (math/code).
  General world-knowledge still needs retrieval → confirms the knowledge/reasoning split.

## DeepSeek-R1 distilled small reasoners (the headline table)
Pure SFT on 800K R1 reasoning traces (no RL on students). Base = Qwen2.5 / Llama3.

| Model | AIME'24 | MATH-500 | GPQA-D | LiveCodeBench |
|---|---|---|---|---|
| R1-Distill-Qwen-1.5B | 28.9 | **83.9** | 33.8 | 16.9 |
| R1-Distill-Qwen-7B | 55.5 | **92.8** | 49.1 | 37.6 |
| R1-Distill-Qwen-14B | 69.7 | 93.9 | 59.1 | 53.1 |
| R1-Distill-Qwen-32B | 72.6 | 94.3 | 62.1 | 57.2 |
| GPT-4o (ref) | 9.3 | 74.6 | 49.9 | 32.9 |
| Claude-3.5-Sonnet (ref) | 16.0 | 78.3 | 65.0 | 38.9 |
| o1-mini (ref) | 63.6 | 90.0 | 60.0 | 53.8 |

7B beats QwQ-32B-Preview on AIME; 32B matches o1-mini. Source: arXiv:2501.12948.

## Distillation techniques
- Logit/soft-label (cheapest, richest signal), sequence-level (R1/Llama3.2 use this), on-policy
  (MiniLLM reverse-KL, DistiLLM, GKD — fixes exposure bias).
- Structured prune + distill: 8B/4B from 15B with **1/40 the tokens**, +16% MMLU (Llama 3.2 recipe).

## Cost table (June 2026 prices; H100 fell 64–75% since 2024)
| Task | Hardware | Time | Cost |
|---|---|---|---|
| QLoRA 7B (10K samples) | 1× RTX 4090 | 20–40 min | $0–2 |
| SFT 7B on 800K samples | 8× A100 | ~22 h | ~$900 |
| Continued-pretrain 7B, 1B tok | 8× A100 | 57 h | ~$2,330 |
| R1-style distill SFT | 8× H100 | 12–48 h | $200–2,000 |
| Pretrain 1.1B from scratch (3T tok) | 16× A100, 90 d | — | $35K–140K |
| Pretrain 3.8B (Phi-3, 3.3T tok) | 512× H100, 7 d | 86K GPU-h | $172K–284K |

GPU rent: RunPod H100 ~$1.99/h, A100 ~$1.49/h; Lambda H100 $4.29/h.

## Datasets (open)
- Pretrain: FineWeb 15T (44TB), **FineWeb-Edu 1.3T** (MMLU 37 vs 33 base; 10× token efficiency),
  DCLM-Baseline 3.8T. SmolLM2 mix = 60% FineWeb-Edu + 40% DCLM.
- Synthetic: Cosmopedia 25B (Mixtral textbooks).
- Code: The Stack v2 (~900B filtered, 600+ langs).
- SFT: OpenHermes 2.5 (1M), UltraChat (774K), Magpie (300K), WizardLM Evol.
- Math: NuminaMath (860K), OpenR1-Math-220k, MetaMathQA (395K).

## Pragmatic playbook (community default)
1. Base = Qwen2.5 / Llama3.x / Gemma2.  2. Optional continued-pretrain on domain (1–50B tok).
3. SFT on 10K–1M curated/teacher traces.  4. DPO/RLHF align.  5. QLoRA to iterate.

## Legal
- Open teachers (Apache-2.0/Llama license) → outputs reusable for training. SAFE.
- Closed APIs (GPT/Claude) → ToS forbids training competitors; Anthropic detects/litigates. AVOID
  for anything resembling a general model.

### Sources
- R1 https://arxiv.org/html/2501.12948v1 · TinyLlama https://arxiv.org/html/2401.02385v2
- FineWeb https://arxiv.org/html/2406.17557v1 · FineWeb-Edu https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
- DistiLLM-2 https://arxiv.org/pdf/2503.07067 · KD survey https://arxiv.org/html/2503.12067v2
- Llama 3.2 https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/
- Anthropic distillation policy https://support.claude.com/en/articles/12326764 · cost https://www.chanl.ai/blog/fine-tuning-lora-qlora-ai-agent-builders
