# Research: MoE, Sparsity & Offloaded CPU Inference (2024–2026)

> Contains the design-critical reality check on disk streaming.

## TL;DR for tiny-llm (READ — changes the design)

- **Inference is MEMORY-BANDWIDTH bound, not compute bound.** The metric that matters is
  **bytes-read-per-token**. This is why low-bit quant buys *speed*, and why 4 vCPU is fine.
- **Streaming model weights/experts from NVMe per token is NOT viable interactively:**
  practical NVMe BW during inference is **2–5 GB/s** (not the 7–12 rated), giving **0.05–0.2 tok/s**
  for 30–70B models. Dead for chat. → **The resident model MUST fully fit in RAM.**
- **MoE doesn't rescue a 4GB box**: even though only ~5–28% of params are active per token, ALL
  experts must be resident (router is unpredictable). Qwen3-30B-A3B needs ~16GB at INT4. Doesn't fit.
- **What 4GB actually buys: a fully-resident 7B-class dense model at 4-bit (≈4–5GB, tight) or a
  ~1.5–3B model comfortably, at 5–15 tok/s on CPU.** That's our resident reasoner budget.
- **PowerInfer sparsity (11× speedup) only applies to ReLU models (95–98% sparse).** Modern SwiGLU
  models (Llama/Qwen/Mistral) are 43–53% sparse → much smaller gains. Could matter if we train our
  own ReLU/dReLU reasoner.
- **Speculative decoding = 2–6× on GPU, ~0 on CPU.** Skip it for CPU deploy.
- **Retrieval is fine** because it's a few reads per *query*, not weight streaming per *token*.

## Memory-bandwidth wall
Arithmetic intensity at batch=1 ≈ 1–2 FLOP/byte. tok/s ≈ mem_BW / bytes_per_token.
DDR5 CPU RAM ~60 GB/s practical → a 3B INT4 model (~1.8GB/token read) ≈ 30+ tok/s; a 37B-active
model ≈ 2–3 tok/s. Source: arXiv:2402.16363.

## MoE active vs total
| Model | Total | Active/tok | Ratio |
|---|---|---|---|
| Mixtral 8x7B | 46.7B | 12.9B | 28% |
| DeepSeek-V3 | 671B | 37B | 5.5% |
| Qwen3-30B-A3B | 30B | 3B | 10% |
| Qwen1.5-MoE-A2.7B | 14.3B | 2.7B | 19% |

All-experts-resident requirement: Mixtral INT4 ~23GB, DeepSeek-V3 INT4 ~336GB. None fit 4GB.

## Offloading systems
- Mixtral-offloading (RAM→GPU copy): 2.37 tok/s; expert load = 85–94% of time.
- **Fiddler** (compute experts on CPU, move only activations): >333 tok/s — but needs 90GB CPU RAM.
- HOBBIT mixed-precision edge offload: 13× vs llama.cpp on Jetson.
- **"SSD offloading for MoE considered harmful"** (arXiv:2508.06978): 3.8× energy/token; load dominates.

## Storage-tier bandwidth → tok/s bound
| Tier | Practical BW | Mixtral tok/s |
|---|---|---|
| RTX4090 GDDR6X | 700 GB/s | ~50 |
| DDR5 RAM | 60 GB/s | 2–4 |
| PCIe4 | 25 GB/s | 1–2 |
| **Gen4 NVMe** | **2–5 GB/s** | **<0.5** |

## mmap / streaming when model > RAM
- 30B (20GB) on 8GB RAM: 411K page faults, ~1.5 tok/s. Qwen2.5-14B Q4 on 8GB Mac: 0.1 tok/s.
- Apple "LLM in a flash" (windowing + row-col bundling): 4–5× over naive; runs up to 2× DRAM. Still
  fundamentally limited by flash BW.

## PowerInfer / activation sparsity
- Hot neurons (~17%) do 80% of activations → keep hot in fast mem, cold computed lazily.
- ReLU models 95–98% sparse; SwiGLU only 43–53%. PowerInfer 7–11× speedups are ReLU-only.
- PowerInfer-2 on phone (Snapdragon 8 Gen3): TurboSparse-Mixtral-47B at 11.68 tok/s, 19GB.
- **Design lever**: training our reasoner with ReLU²/dReLU activation unlocks big CPU sparsity wins.

## Honest verdict: 30–70B capability on 4–8GB
NOT achievable interactively today. 8GB buys a usable 7B dense Q4 (5–20 tok/s) or a small MoE at
~7B-equivalent quality. NVMe paging of big models = 0.05–0.2 tok/s = offline only.

### Design implications for tiny-llm
1. Resident reasoner fully in RAM (≤~2GB) — no per-token disk streaming.
2. Get "size" from RETRIEVAL (per-query reads) + TOOLS, not from paged experts.
3. Consider a ReLU²/dReLU-activated reasoner to exploit PowerInfer-style CPU sparsity.
4. Forget speculative decoding for CPU.

### Sources
- Roofline/mem-bound https://arxiv.org/html/2402.16363v4 · Mixtral https://mistral.ai/news/mixtral-of-experts/
- DeepSeek-V3 https://arxiv.org/abs/2412.19437 · Fiddler https://arxiv.org/abs/2402.07033
- Mixtral-offload https://arxiv.org/abs/2312.17238 · HOBBIT https://arxiv.org/html/2411.01433v2
- SSD-offload-harmful https://arxiv.org/html/2508.06978 · PowerInfer https://arxiv.org/abs/2312.12456
- PowerInfer-2 https://arxiv.org/html/2406.06282v2 · LLM-in-flash https://arxiv.org/abs/2312.11514
- llama.cpp mmap https://github.com/ggml-org/llama.cpp/discussions/638 · EAGLE-3 https://arxiv.org/pdf/2406.16858
