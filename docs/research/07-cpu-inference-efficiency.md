# Research: CPU-native inference efficiency (2024–2026)

> For the v2 speed architecture. What's the fastest *useful* Cortex on a 4-core CPU, and by what means.

## TL;DR for Metis v2
- **Our current path (Qwen3-1.7B via ollama) is leaving speed on the table.** Stock llama.cpp with
  the right flags runs **Qwen3-1.7B Q4_K_M at ~25–45 tok/s on a 4-core x86** (vs our slow ollama warm
  path). The engine + flags matter as much as the model.
- **BitNet b1.58 2B4T (ternary)**: **0.4 GB** weights, **15–30 tok/s** on 4-core CPU via bitnet.cpp,
  quality ≈ Qwen2.5-1.5B (beats Llama3.2-1B/Gemma3-1B). Best capability-per-byte. Catch: needs
  bitnet.cpp runtime, no ollama, 4k ctx, and you **can't** make your own (no open training recipe).
- **Activation sparsity (TurboSparse + PowerInfer)**: real but moderate **2–2.8× CPU** speedup; needs
  the PowerInfer runtime + dReLU models; 7B is borderline in 4 GB. Not worth the ecosystem pain yet.
- **The honest ceiling**: token-gen is **memory-bandwidth bound**. DDR4-3200 ≈ 50 GB/s → a 7B-Q4
  (3.8 GB) caps ~13 tok/s; a 1.7B-Q4 (~1.1 GB) caps far higher. Smaller + lower-bit = faster, period.

## Fastest useful configs on 4-core x86 (Q4_K_M, llama.cpp, measured/extrapolated)
| Model | RAM | tok/s (4-core) | MMLU | GSM8K | note |
|---|---|---|---|---|---|
| Qwen3-0.6B | 0.4 GB | 60–120 | ~42 | ~35 | ultra-fast, weak |
| **Qwen3-1.7B** | **1.5 GB** | **25–45** | **~57** | **~59** | best all-round |
| BitNet 2B4T | 0.5 GB | 15–30 | 53 | 58 | ternary, bitnet.cpp |
| Qwen3-4B | 3 GB | 10–18 | ~72 | ~79 | best quality that fits |
| Mistral-7B | 4–5 GB | 6–12 | — | — | OOM risk, slow |
> Usability threshold: <10 tok/s "feels broken"; 15–20 comfortable; >30 ~instant. On a weak/cheap
> VPS vCPU divide by ~1.5–2× (and our deployed warm latency was dominated by load + prefill, not just tg).

## bitnet.cpp (ternary) measured
- i7-13800H 8-thread, 2B4T: ~34 tok/s (29 ms/tok). Surface Laptop (i7-13800H): 5.9 tok/s 8-thread (?).
- vs llama.cpp fp16: 2.4–6.2× x86. **vs Q4 llama.cpp the real edge is ~2–3× + much less RAM/energy.**
- Jan-2026 bitnet.cpp update: +1.15–2.1× from parallel kernels.
- Quality (2B4T): MMLU 53.2, GSM8K 58.4, HumanEval+ 38.4, avg 54.2 — usable assistant, MT-Bench 5.85.
- **Must train from scratch in ternary; no open recipe → can't roll our own. Use Microsoft's artifact.**

## llama.cpp CPU speed flags (free wins)
`--threads 4` (physical cores) · `--ctx-size 2048` (less KV RAM) · `--cache-type-k q4_0 --cache-type-v q4_0`
(half the KV RAM) · Q4_0 auto-repack on AVX2 · AVX-512/VNNI auto. **ik_llama.cpp fork**: 3–7× faster
*prompt processing* (matters for RAG prefill!), 1.05–2.1× tg.

## Sparsity (for completeness)
TurboSparse-Mistral-7B: 9.94 tok/s vs 4.78 baseline (2.08×) on i7-12700K; quality ≥ original.
ProSparse-Llama2-7B 89% sparse, 16.3 tok/s CPU. All need PowerInfer (`*.powerinfer.gguf`, not llama.cpp).

## Implications for v2
1. **Switch the engine from ollama → llama.cpp** (or ik_llama.cpp) with tuned flags — likely a
   2–3× speedup on the SAME Qwen3-1.7B, for free, and it unlocks token-level control (lookup decoding,
   KV save/restore) that ollama hides. This is the single highest-ROI move.
2. Keep Qwen3-1.7B as default Cortex; offer BitNet-2B4T as a 0.4 GB ultra-light profile.
3. KV-quant + small ctx to protect the 4 GB budget.

### Sources
- BitNet CPU paper https://arxiv.org/html/2410.16144v1 · 2B4T report https://arxiv.org/abs/2504.12285 · model https://huggingface.co/microsoft/bitnet-b1.58-2B-4T
- TurboSparse https://arxiv.org/html/2406.05955v1 · ProSparse https://arxiv.org/html/2402.13516v7 · PowerInfer-2 https://arxiv.org/abs/2406.06282
- llama.cpp CPU guide https://blog.steelph0enix.dev/posts/llama-cpp-guide/ · ik_llama.cpp https://github.com/ikawrakow/ik_llama.cpp/discussions/164
- SBC bench https://arxiv.org/html/2511.07425v1 · Qwen3 speed https://qwen.readthedocs.io/en/latest/getting_started/speed_benchmark.html
