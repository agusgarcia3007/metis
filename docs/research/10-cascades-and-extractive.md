# Research: Cascades, routing & extractive shortcut (2023–2026)

> The *reliable* structural win: make the common (easy/grounded) query cheap, escalate only the rare hard one.

## TL;DR for Metis v2
- **Cascades work and are deployable.** Route easy queries to the cheap path, escalate hard ones via a
  cheap confidence signal. Measured: **FrugalGPT 59–98% cost cut** at GPT-4 quality (98% only on narrow
  classification); **RouteLLM sends only 14% to the big model on MT-Bench at 95% quality** (85% cost
  cut; 45% on MMLU, 35% on GSM8K). The query-difficulty distribution is heavy-tailed → most queries
  are genuinely easy.
- **Cheap confidence gate**: **Self-REF confidence tokens** → 2× speedup, escalating only 39% of MMLU.
  **Semantic-Entropy Probes** read uncertainty from hidden states in ~one pass. Raw softmax is poorly
  calibrated; token-margin is a usable free signal.
- **Extractive shortcut is real** (skip the generative LLM for lookups): retrieval-only QA via DPR gets
  ~41 EM (NQ), DensePhrases (pure phrase index, no reader) ~41 EM; **VerbatimRAG** = dense retrieval +
  a **150M ModernBERT token-classifier** (no generative LLM) extracts verbatim spans at **53.6 F1**
  (beats an LLM extractor). So a tiny non-generative extractor can answer many grounded lookups in ~ms.

## Within-model adaptive compute (speedups, but need trained-in support)
- **CALM** (confident early-exit): up to **3×**, uses 1/3–1/2 layers on easy tokens. Needs exit heads.
- **SkipDecode**: **2–5×**, batch/KV-compatible. Needs custom inference.
- **Mixture-of-Depths**: >50% faster sampling; **train-time only** (Qwen3/Llama don't ship it).
- → These are future Cortex-training options, not retrofits for an off-the-shelf model.

## Test-time compute (buy quality cheaply on the rare hard query)
- Self-consistency: +17.9% GSM8K at N samples. s1 budget-forcing: 32B beats o1-preview on comp-math.
- A 7B + best-of-N + a verifier/PRM can match a 70B on math — but only with a verifier and N≥8 (N× latency).
- Crossover: small+test-time-compute beats bigger model when answers are **verifiable** and a verifier exists.

## The cascade for a local portable assistant (achievable operating point)
```
query → cheap pre-router (rules + embedding) 
      → EXTRACTIVE path (tiny classifier/phrase match) ──confident?──► answer (~ms, cited)
      → else GROUNDED-GENERATE (Cortex + retrieved ctx) ──confident?──► answer (cited)
      → else REASON path (more layers / test-time compute / bigger profile)   [rare]
```
Literature operating point: **60–86% of queries handled by the cheap path**, **95–97% of big-model
quality**, hard-query latency unchanged. Confidence gate <10 ms.

## Honest caveats
- "95% of GPT-4 quality" = 5% worse; high-stakes domains route worse. 98% cost-cut is a narrow
  classification result, not open chat. Best routers need preference data from the actual model pair
  (cold-start problem). Test-time compute is N× latency, not free.

### Sources
- FrugalGPT https://arxiv.org/abs/2305.05176 · RouteLLM https://arxiv.org/abs/2406.18665 · Speculative Cascades (Google) https://research.google/blog/speculative-cascades-a-hybrid-approach-for-smarter-faster-llm-inference/
- CALM https://arxiv.org/abs/2207.07061 · SkipDecode https://arxiv.org/abs/2307.02628 · MoD https://arxiv.org/abs/2404.02258
- Self-REF https://arxiv.org/html/2410.13284v3 · Semantic-Entropy Probes https://arxiv.org/abs/2406.15927
- VerbatimRAG https://arxiv.org/abs/2605.21102 · DensePhrases https://arxiv.org/abs/2012.12624 · DPR https://arxiv.org/abs/2004.04906 · s1 https://arxiv.org/abs/2501.19393
