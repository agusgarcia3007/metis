# Research: RAG latency — KV-cache reuse & context (2024–2026)

> The second big latency term is PREFILL: a RAG prompt is dominated by retrieved chunks (10× the
> query), and the model recomputes their KV every call. These techniques kill that.

## TL;DR for Metis v2 (the prefill killers)
- **Precompute & cache per-chunk KV** so a retrieved chunk is never re-prefilled. This is the biggest
  single win for RAG TTFT, and it's deployable on a single box.
- **TurboRAG** (arXiv:2410.07590, EMNLP'25): offline-precomputed chunk KV + reordered-RoPE so chunks
  stay valid out of their original position → **up to 9.4× (avg 8.6×) lower TTFT**, accuracy ≈ normal
  RAG, **no model change**. *Most directly applicable to us.*
- **RAGCache** (arXiv:2404.12457, ACM TOCS'25): prefix-tree of document-sequence KV + PGDSF eviction
  + speculative retrieval pipelining → **up to 4× TTFT, 2.1× throughput**. Server-scale (GPU L1 / host
  L2) but the idea (cache chunk KV keyed by doc-sequence prefix, overlap retrieval+prefill) transfers.
- **Cache-Craft** (arXiv:2502.15734, Adobe): reuse chunk KV without exact prefix match, recompute only
  ~30% of cross-chunk tokens → 2× latency cut. Key stat: **naive prefix caching hits only ~8% of real
  RAG requests** — you need chunk-level (not whole-prefix) reuse.
- **CAG — Cache-Augmented Generation** (arXiv:2412.15605): if the whole corpus fits the context, preload
  it and cache the KV once → **generation 2.33s vs RAG 94.35s (~40×)**, quality ≥ RAG (HotPotQA). For a
  small/static knowledge base this is a stunning, simple win.
- **CacheGen** (SIGCOMM'24): compress KV bitstream **3.5–4.3× smaller, 3.2–3.7× faster fetch**, ~no
  quality loss → makes storing/loading cached chunk-KV from disk cheap (fits our knowledge-as-data).
- **Persistent Q4 KV cache on edge** (llama.cpp/MLX): reloading a cached KV is **11–136× faster** than
  recompute at 4k–32k ctx. llama.cpp supports KV state save/restore — usable today.

## The catch: positional dependency (and the fixes)
A chunk's KV encodes its absolute position / RoPE angle, so you can't blindly stitch precomputed chunk
KV at arbitrary positions. Fixes that work:
- **Reordered-RoPE** (TurboRAG) — adjust RoPE at stitch points.
- **Selective recomputation** (Cache-Craft, CacheBlend) — recompute only the ~10–30% of tokens whose
  attention actually crossed chunk boundaries.
- **Soft-token boundary adapters** (KV Packet, arXiv:2604.13226) — near-zero-FLOP wrappers, 19× TTFT.

## KV-cache compression (protect the 4 GB budget at longer ctx)
KIVI (2-bit) 2.6× mem, 2.35–3.47× tput · KVQuant (sub-4-bit, 10M ctx) · SnapKV (prefill compression) ·
H2O (29× tput OPT-30B) · CacheGen (3.5–4.3×). Architectural: GQA (standard), DeepSeek **MLA = 93% KV
reduction**. KV q4_0 in llama.cpp = ~50% KV RAM, today.

## Implications for v2 (ranked by ROI on our CPU box)
1. **Per-chunk KV precompute + reuse** (TurboRAG-style): cache each Library chunk's KV on disk (with
   reordered-RoPE), load instead of prefill → biggest TTFT cut. Knowledge-as-data now includes
   knowledge-as-*precomputed-KV*. Requires engine control (llama.cpp KV save/restore), not ollama.
2. **CAG for small corpora**: if the user's knowledge base is small/static, preload+cache once → near-
   instant grounded answers. Offer this as a mode.
3. **KV quant + small ctx** for the RAM budget.
4. **Overlap retrieval with prefill** (speculative pipelining).

### Sources
- TurboRAG https://arxiv.org/abs/2410.07590 · RAGCache https://arxiv.org/abs/2404.12457 · Cache-Craft https://arxiv.org/abs/2502.15734
- CAG https://arxiv.org/abs/2412.15605 · CacheGen https://arxiv.org/abs/2310.07240 · CacheBlend https://arxiv.org/pdf/2405.16444
- KV Packet https://arxiv.org/abs/2604.13226 · KIVI https://arxiv.org/abs/2402.02750 · MLA(DeepSeek-V2) https://arxiv.org/abs/2405.04434
- Survey https://github.com/October2001/Awesome-KV-Cache-Compression
