# Research: Extreme Quantization (2024–2026)

> Captured from deep-research pass. Source for the `tiny-llm` design.
> Every major claim has a source URL.

## TL;DR for tiny-llm

- **4-bit (Q4_K_M / AWQ) is the practical sweet spot**: ~1% perplexity loss, 75% memory cut, 3–4× faster. Default for anything we ship.
- **The quality cliff is at ~2.7-bit, not 4-bit.** Going 3.3→2.7 bit accelerates degradation ~5–6×.
- **BitNet b1.58 (ternary, 1.58-bit) is genuinely competitive at its size class** but MUST be trained from scratch — no conversion path. 0.4 GB for a 2B model (non-embedding). This is the single most important lever for "frontier-ish in 4GB."
- **QAT is essential below 3-bit**: PTQ at 2-bit gives 6,766 PPL on a 7B; QAT recovers it to ~30 PPL.
- **FP8 KV cache is nearly free**: 50% KV memory cut, <0.1% quality loss — use by default for long context.
- Larger models tolerate lower bits better; small (7B) models collapse earlier (IQ1_S = 28.8 PPL vs 7.49 FP16).

## BitNet b1.58 2B4T (the headline result for us)
- Ternary weights {−1,0,+1} + INT8 activations. Matmul becomes add/sub.
- 2B params, 4T tokens. **0.4 GB** non-embedding memory vs 2.0 GB (Llama 3.2 1B), 2.6 GB (Qwen2.5 1.5B).
- CPU latency 29 ms/tok (vs 48 ms Llama-1B, 65 ms Qwen-1.5B) on i7-13800H, 8 threads.
- Beats similarly-sized FP models on GSM8K (58.4), WinoGrande (71.9); trails Qwen2.5-1.5B on MMLU (53.2 vs 60.3).
- bitnet.cpp: 2.37–6.17× speedup x86, can run a **100B BitNet at 5–7 tok/s on a single CPU**.
- Catch: train-from-scratch only.
- Sources: https://arxiv.org/abs/2504.12285 · https://huggingface.co/microsoft/bitnet-b1.58-2B-4T · https://github.com/microsoft/BitNet

## Quality cliff (Llama-2-7B-Chat, WikiText-2 PPL)
| Bits | Format | PPL | Δ vs F16 | Verdict |
|---|---|---|---|---|
| 16 | F16 | 7.492 | — | baseline |
| 8 | Q8_0 | 7.493 | +0.001 | lossless |
| 5 | Q5_K_M | 7.510 | +0.017 | negligible |
| 4.89 | Q4_K_M | 7.569 | +0.077 | **sweet spot** |
| 3.3 | Q3_K_M | 7.685 | +0.193 | noticeable |
| 2.7 | IQ2_M | 8.600 | +1.108 | first cliff |
| 2.06 | IQ2_XXS | 11.03 | +3.540 | heavy |
| 1.56 | IQ1_S | 28.79 | +21.3 | near-collapse |

## PTQ methods @ 2-bit (large models only)
- QTIP 2-bit (2024, best overall): 7B PPL 5.86, 70B 3.70, 188 tok/s.
- AQLM 2-bit: best PPL (7B 5.92) but slow (~20 tok/s).
- QuIP# 2-bit: 7B 6.66, 70B 4.16, fast (106 tok/s), 1KB E8 codebook fits L1.
- HQQ 2-bit: no calibration, 5 min for 70B, 70B PPL 4.12.
- GPTQ/AWQ collapse at 2-bit on 7B (unusable).

## QAT recovery (Llama-3-8B)
- 2-bit weight-only: PTQ 6,766 PPL → QAT ~30 PPL (99% recovered).
- Overhead: ~34% slower training, +2.35 GB/GPU. <1% of pretraining compute often sufficient.

## KV cache quantization
- FP8 KV: 50% cut, <0.1% quality loss — vLLM default 2025.
- INT4 KV (KIVI, tuning-free): 73% cut; keys per-channel, values per-token.

## Outliers / activation quant
- Outliers emerge >6.7B params, 20–100× normal magnitude, <0.1% of dims.
- SmoothQuant migrates outlier difficulty activation→weight (α≈0.8–0.9), true W8A8, 1.56× speedup.
- SpQR: outlier weights in FP16, rest 3–4 bit, 3.4× compression <1% loss, 33B on 24GB.

## "Undertrained models quantize better" (arXiv:2411.17691)
- More tokens-per-param ⇒ MORE quantization damage. Future 20T-token models may be HARDER to quantize. Relevant: if we overtrain a small model heavily, expect quantization to hurt more → favor QAT/BitNet from the start.

### Primary sources
- BitNet: https://arxiv.org/abs/2504.12285 · https://github.com/microsoft/BitNet
- GPTQ https://arxiv.org/pdf/2210.17323 · AWQ https://arxiv.org/abs/2306.00978
- QuIP# https://arxiv.org/abs/2402.04396 · AQLM https://arxiv.org/pdf/2401.06118 · QTIP https://arxiv.org/html/2406.11235
- HQQ https://github.com/mobiusml/hqq · SpQR https://openreview.net/forum?id=Q1u25ahSuy · SmoothQuant https://github.com/mit-han-lab/smoothquant
- llama.cpp PPL bench https://beebopkim.github.io/2024/03/09/Benchmarks-for-lots-of-quantization-types-in-llama-cpp/
- QAT https://pytorch.org/blog/quantization-aware-training/ · Undertrained-quantize https://arxiv.org/pdf/2411.17691
