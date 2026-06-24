# Research: Decode acceleration — speculative / copy-from-context (2024–2026)

> The honest verdict on making token generation faster, with the CPU caveat front and center.

## TL;DR (read the caveat)
- **Copy-from-context speculative decoding (PLD / REST / LLMA)** is the *conceptually perfect* match
  for grounded RAG: draft tokens by copying spans from the retrieved context (no draft model, no
  training, ~0 RAM). **Acceptance scales directly with how extractive the answer is** — extractive QA
  70–90%, summarization 40–60%, creative ~0%. Our **measured 64% copy-rate** sits squarely in the
  "good acceleration" band → PLD literature predicts ~1.7–2.5×.
- **BUT — CPU benefit is NOT cleanly proven.** Speculative decoding's win comes from verifying K
  tokens in one batched pass amortizing memory reads — a GPU-centric dynamic. On CPU: one real report
  (llama-cpp-python #2110) got **zero** speedup; a 35B ngram benchmark went **net-negative**. So we
  **test it, we don't bet the architecture on it.** Treat it as a bonus lever, validate empirically.
- **llama.cpp supports it TODAY** (the engine pivot pays off): `--spec-type ngram-cache` /
  `ngram-simple`, plus **`-lcs <file>` to preload a static n-gram cache built from your corpus** =
  REST-style retrieval-draft with zero GPU. ollama exposes **none** of this (only prefix caching).

## Speculative family — speedups + CPU viability
| method | speedup | needs | CPU? |
|---|---|---|---|
| draft-model spec (Leviathan) | 2–3× (GPU, batch=1); ~0 at batch≥48 | a good small draft | poor for a 1.7B target (draft would be ~150M) |
| EAGLE-1/2/3 | 3–6.5× (GPU) | trained draft head per model | none exist for Qwen3-1.7B; GPU-only |
| Medusa / Hydra | 2–3× (GPU) | trained heads | not practical |
| **PLD / prompt-lookup** | ~2.4× (summ/QA); up to 4× extractive | nothing | CPU-plausible, unproven |
| **REST** (suffix-array datastore) | 2.36× code, 1.69× chat (GPU) | a corpus datastore | retrieval is CPU; verify was GPU |
| DReSD / RASD (dense retrieval draft) | 2.98–4.07× (GPU) | dense index | newer, GPU-benched |
| ML-SpecQD (quantized-draft, CPU) | **2.22–2.72× measured CPU** | Intel AMX / MXFP4 | yes, but high-end CPU only |

Acceptance↔copy-rate is the core law: speedup ≈ f(fraction of output copied from context). REST code
gets 2.36× because code is formulaic; chat 1.69× because language varies.

## What llama.cpp gives us (deployable now)
`--spec-type ngram-cache --spec-draft-n-max 6` · `-lcs corpus.lcs` (static lookup from the Library) ·
`-lcd dyn.lcd` (dynamic). Reported acceptance 57–70%. ollama: only automatic prefix caching.

## Honest realistic outcome for OUR setup (1.7B, CPU, grounded RAG)
The speculative agent's ranked recommendation:
1. **ollama → llama.cpp** (unlocks the flags + token control). *Reliable.*
2. enable `ngram-cache` + `-lcs` corpus cache, **measure** on our extractive answers → maybe 1.5–2.5×.
3. **quantize Q4→Q2_K/Q3_K_M** on the 1.7B → +30–50% decode (memory-bandwidth), small quality cost.
4. or drop to **Qwen3-0.6B** (40+ tok/s) since retrieval carries the facts.
Net decode: **~1.5–2.5×**, contingent + to-be-measured. Don't overclaim it.

### Sources
- PLD https://github.com/apoorvumang/prompt-lookup-decoding · REST https://arxiv.org/abs/2311.08252 · RASD https://arxiv.org/html/2503.03434v1 · DReSD https://arxiv.org/pdf/2502.15572
- ML-SpecQD (CPU) https://arxiv.org/html/2503.13565 · EAGLE-3 https://arxiv.org/html/2503.01840v1 · llama.cpp spec https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md
- CPU zero-speedup report https://github.com/abetlen/llama-cpp-python/issues/2110 · ollama spec request https://github.com/ollama/ollama/issues/5800
