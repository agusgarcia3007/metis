# Metis v2 — the architecture bet: speed AND quality from "something tiny"

> Working design (being validated by the 2025–2026 research sweep in progress). Honest framing:
> nobody has an architecture that beats frontier on *both* speed and quality with a tiny model —
> if they did, it would be the headline. What IS achievable, and where we can be genuinely novel,
> is an **inference architecture that makes a small local model many times faster on its supported
> surface (grounded QA + tools), at equal or better quality, on a 4 GB / 4 vCPU CPU box.**

## 0. Where the time actually goes (the bottleneck)

Measured on the live 4 GB/4 vCPU deploy (Qwen3-1.7B Q4 via ollama):
- **cold** (first call, model load): ~82 s
- **warm** (short grounded answer): ~8.5 s

For a RAG turn the warm cost decomposes as:
1. **embed the query** (load + run all-MiniLM) — small but non-zero on CPU.
2. **prefill** the prompt = system prompt + retrieved chunks (~500–800 tokens) — O(prompt) compute.
3. **decode** the answer token-by-token (~30–80 tokens) — O(tokens) × per-token cost; **memory-
   bandwidth bound** (research 03). This dominates and is the thing to kill.

The current design pays full autoregressive decode + full prefill every call. That is exactly "a
Qwen on a VPS." The architecture below attacks each term.

## 1. The thesis: a grounded **draft-verify cascade**

Most answers in a grounded assistant **copy or lightly paraphrase the retrieved text**. That single
fact is the lever. Three nested ideas, cheapest first:

### A. Extractive shortcut (skip the LLM entirely for lookups)
For a "lookup" query whose answer is a span already in a retrieved chunk, a cheap extractor (the
embedder / a tiny span scorer) can return the answer with a citation **without running generation at
all** → near-instant. A confidence gate decides extractive-vs-generative.
*Lever: removes decode AND most prefill for the common factual case.*

### B. Copy-from-context speculative decoding (when we do generate)
When generation is needed, draft the next tokens by **copying spans from the retrieved context /
prompt** (REST / Prompt-Lookup-Decoding style — no draft model needed), and let the model **verify
many tokens in one forward pass**. Grounded answers have very high copy-acceptance → multi-token-per-
step decode. *Lever: turns slow token-by-token decode into chunked verify; biggest win for grounded
text. The Library doubles as the speculative draft source — knowledge-as-data accelerates itself.*

> **MEASURED on our own deployed system (validates the bet):** over 5 real grounded answers, the
> 3-gram **copy-rate vs the source document was 64%** (range 38–89%), with **longest verbatim spans of
> 6–12 words**. So a prompt-lookup / REST decoder that drafts the longest matching span from the
> retrieved context would accept multi-token chunks most of the time. PLD/REST literature reports
> ~2–3× decode speedup on exactly this copy-heavy grounded generation — consistent with a 64%
> acceptance regime. This is the most important number we have: the *premise* holds for our workload.
>
> **Honest caveat (research 09):** speculative decoding's speedup is a GPU-centric dynamic (batched
> verify amortizes memory reads). On CPU it is **not cleanly proven** — one report got zero speedup, a
> big-MoE ngram run went net-negative. llama.cpp ships it (`--spec-type ngram-cache`, `-lcs` corpus
> cache) so it's free to try, but we **test it, we don't bet the architecture on it.** Realistic if it
> works on our extractive answers: ~1.5–2.5× decode. Treat as a *bonus* lever, not the foundation.

### C. Generative fallback with a fast Cortex
Only genuinely novel/reasoning queries fall through to full generation — and even there we use the
fastest viable Cortex (small + low-bit; ternary/BitNet or a sparse-activation model as the moonshot).
*Lever: the rare hard case is the only one that pays full price.*

This is a **cascade**: extractive (ms) → copy-spec generate (fast) → full generate (slow, rare).
Average latency collapses because the *distribution* of real queries is dominated by the cheap paths,
while quality is preserved by escalating when confidence is low.

## 2. Cutting the other terms

- **Prefill / KV reuse:** cache the KV of the **static system prompt** and, ideally, **per-chunk KV
  of retrieved documents** so a chunk is never re-prefilled. Grounded by **RAGCache** (arXiv:2404.12457,
  ACM TOCS 2025): a prefix-tree ("Knowledge Tree") of document-sequence KV caches with a prefix-aware
  PGDSF eviction policy and **speculative retrieval pipelining** (start prefill on intermediate top-k);
  reports **up to 4× lower time-to-first-token, 2.1× throughput**. Their hierarchy is GPU-L1/host-L2
  (server scale); for our CPU box the transferable parts are **per-chunk KV save/restore + prefix
  reuse of the system prompt** (llama.cpp can persist/restore KV state) and **overlapping retrieval
  with prefill**. **TurboRAG** (arXiv:2410.07590) is the most directly applicable: precompute each
  chunk's KV offline with reordered-RoPE → **up to 9.4× (avg 8.6×) lower TTFT**, accuracy ≈ normal RAG,
  no model change. For a small static knowledge base, **CAG** (arXiv:2412.15605) preloads the whole
  corpus KV once → **~40× faster generation** (2.3s vs 94s on HotPotQA). Stat that matters: naive
  whole-prefix caching hits only **~8%** of real RAG requests (Cache-Craft) — you need **chunk-level**
  KV reuse. (Full notes: `../research/08-rag-kv-cache-latency.md`.)
- **CPU-native speed (the highest-ROI move):** **switch the engine ollama → llama.cpp** with tuned
  flags. Same Qwen3-1.7B Q4 then runs **~25–45 tok/s on 4-core x86** (vs our slow ollama warm path),
  and we gain token-level control (lookup decoding, KV save/restore) that ollama hides. `ik_llama.cpp`
  fork adds **3–7× faster prompt processing** (= cheaper RAG prefill). KV q4_0 + ctx 2048 protect RAM.
  Ultra-light profile: **BitNet-b1.58-2B4T** ternary, **0.4 GB**, 15–30 tok/s, quality ≈ Qwen2.5-1.5B.
  (Full notes: `../research/07-cpu-inference-efficiency.md`.)
- **Prompt compression:** LLMLingua-family compresses retrieved chunks to cut prefill (secondary once
  chunk-KV reuse is in; complementary). > exact token-cut/quality numbers: minor, deprioritized.
- **Parallel/diffusion decode:** non-autoregressive generation as a moonshot. > CPU verdict pending
  (diffusion report); early read: GPU-bound today, not the near-term CPU lever.

## 3. What is genuinely novel here (and what is "just good engineering")

- **Novel/strong:** unifying the **Library as both the knowledge store AND the speculative-draft
  source** — retrieval-grounded answers literally accelerate themselves; plus a **confidence-gated
  extractive→speculative→generative cascade** designed for a tiny CPU box. The combination, tuned for
  portable on-device RAG, is not a thing you can `ollama run`.
- **Honest "just engineering" parts:** quantization, KV/prefix caching, prompt compression — known
  techniques, but the *integration* into one coherent portable system is the product.
- **What we are NOT claiming:** beating GPT-5 on open-domain reasoning. We claim **frontier-useful
  speed+quality on the supported surface (grounded QA, tools, summarization/extraction) at a fraction
  of the size/cost, fully local.**

## 4. The hard constraint that shapes everything

To implement A/B (extractive shortcut, copy-from-context speculative, per-chunk KV reuse) we need
**token-level control of decoding** — which `ollama` does not expose. So v2 requires going one layer
down: **llama.cpp directly**. Confirmed capabilities we'll use: built-in **prompt-lookup decoding**
(`llama-lookup` example — drafts from the prompt/context, exactly our copy-from-context mechanism),
**speculative decoding** (`--model-draft`), **KV state save/restore** (persist/reload chunk KV), and
**prefix reuse**. This is the real engineering pivot from v1 (which was an ollama wrapper).

## 6. Projected latency budget (honest estimate, to be measured)

Today: **~8.5 s** warm for a short grounded answer (ollama, Qwen3-1.7B, full prefill + full AR decode).
Stacking the validated levers (each cited above), for the **common grounded query**:

**RELIABLE levers** (well-established, deployable):
| lever | effect | source |
|---|---|---|
| engine ollama→llama.cpp + tuned flags (Q-quant, KV-quant, prefix cache, threads) | decode ~2–3× | research 07/09 |
| chunk-KV reuse for grounded prefill (TurboRAG ≤8.6× TTFT; CAG ~40× if corpus fits) | prefill ~5–8× | research 08 |
| **extractive shortcut** for pure lookups (tiny classifier / phrase match, no generation) | **skips the LLM (~ms)** | research 10 |
| cascade routing — only the rare hard query pays full price | ~60–86% of queries on the cheap path | research 10 |

**BONUS lever (promising premise, CPU-UNPROVEN — test, don't bet):**
| copy-from-context speculative (PLD/ngram), 64% measured copy-rate | maybe ~1.5–2.5× decode *if it helps on CPU* | research 09 |

Rough projection (reliable stack; speculative excluded to stay honest):
- **Pure lookup** ("what's the mascot?") → extractive path → **sub-100 ms**.
- **Grounded synthesis** → chunk-KV reuse (prefill ~6×) + faster engine (decode ~2–3×) → **~1.5–3 s**
  (down from ~8.5 s). Speculative, if it works on CPU, pushes this further.
- **Novel reasoning** (rare) → full generate → seconds, but the minority of traffic.

Net: average latency on the supported surface plausibly **~3–8×** lower from the reliable levers alone
(more if speculative pays off), at **equal or better quality** (citations preserved), same 4 GB box.
That is the gap between "a slow Qwen on a VPS" and a genuinely responsive portable assistant. **These
are projections from cited results + our measured 64% copy-rate; §5 turns them into measured numbers.**

## 5. Validation — prototyped & MEASURED (not hand-waved)

1. ✅ **Copy rate: 64%** (3-gram, grounded answers vs source) — the speculative premise holds.
2. ✅ **Extractive cascade fast-path: BUILT and MEASURED.** Implemented in `internal/library`
   (`Extract`) + a confidence gate (`extractGate=0.62`) wired into `ask`/`serve`/`chat`. Result on the
   live system:
   - lookup "how many shards?" → **104 ms** (correct span, no LLM) **vs ~8500 ms** generative = **~80×**.
   - lookup "the mascot?" → ~0.7 s incl. embedder load (correct), **score gate**: in-domain factoids
     0.7–0.9, off-topic 0.17 → off-topic/creative queries correctly fall through to the LLM.
   - This is the biggest lever (generation = ~75% of RAG latency, research 10) and it's **real now**,
     on the existing stack, no engine change. Reproduce: `metis extract` (calibration) / `metis ask`.
3. ⏭️ **Engine pivot → llama.cpp** for decode/quant/prefix-cache gains + an empirical
   prompt-lookup (ngram) test. *(llama.cpp CPU build hit a macOS OpenSSL/httplib snag; do it on Linux.)*
4. ⏭️ **Chunk-KV reuse** (TurboRAG-style) once on llama.cpp; `OLLAMA_KEEP_ALIVE=-1` already removes the
   ~82 s cold-start on the current deploy.

**Net so far:** the common lookup case went from ~8.5 s to ~0.1 s — measured, not projected.

---

*Status: design grounded in the 2025–2026 research sweep (`../research/07`, `../research/08`, +
speculative/cascade reports pending) and our own measured 64% copy-rate. The latency budget in §6 is
a projection; §5 turns it into measured numbers before any claim ships.*
