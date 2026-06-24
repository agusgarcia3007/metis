# tiny-llm — The Physics of the Problem (read this first)

This document fixes the hard constraints before any design. Everything downstream
must respect these numbers. The goal is a **portable, near-frontier LLM system** that
runs on a **4 GB RAM / 4 vCPU VPS** with no GPU.

## The deployment envelope (4 GB / 4 vCPU, no GPU)

| Resource | Budget | Notes |
|---|---|---|
| RAM total | 4096 MB | Hard ceiling. OS + runtime eat ~300–600 MB. |
| RAM for model+KV+runtime | ~3200 MB usable | Be conservative; OOM kills the box. |
| CPU | 4 vCPU | Memory-bandwidth bound, not FLOP bound, at inference. |
| Disk | assume 40–80 GB SSD/NVMe | This is our secret weapon: cheap, large, streamable. |
| NVMe read BW | ~1–3.5 GB/s typical VPS | Bounds any weight-streaming scheme. |

## The uncomfortable truth (so the design is honest)

A **monolithic** model "comparable to frontier" (GPT-4o / Claude / Gemini class) does
NOT fit in 4 GB. Frontier models are 100B–2T+ params. At 4-bit:

| Params | 4-bit size | Fits in 4 GB? |
|---|---|---|
| 1.5B | ~1.0 GB | yes, comfortably |
| 3B | ~2.0 GB | yes |
| 7–8B | ~4.0–4.5 GB | NO (too tight with KV+runtime) |
| 7–8B @ BitNet 1.58 | ~1.2–1.8 GB | **yes** |
| 14B @ 2-bit | ~4.5 GB | no (and 2-bit on small models collapses) |
| 70B | ~35 GB | only via disk streaming, ~1 tok/s |

So **raw parameter count for general intelligence cannot be faked** in 4 GB. Anyone who
claims a 4 GB box equals GPT-4 *as a single model* is selling something.

## The reframe that makes it possible — "different system"

The user asked for a *different system*, not a smaller copy of the same thing. The
design thesis (validated against the research notes in `../research/`):

> **A frontier model spends most of its parameters MEMORIZING facts, not reasoning.**
> Knowledge-capacity scaling work (Allen-Zhu, "Physics of LMs") estimates ~2 bits of
> stored knowledge per parameter. If we *move knowledge out of the weights* into a
> cheap, streamable external store (retrieval) and tools (calculator, code, search),
> then the resident model only has to be a strong **reasoner + router + writer** — and
> a strong reasoner can be small.

tiny-llm is therefore **not a model. It is a system**:

```
            ┌──────────────────────────────────────────────────────┐
            │  tiny-llm runtime (Go, single static binary)          │
            │                                                       │
  query ──► │  Router  ─►  Small resident REASONER (in RAM)         │
            │              │   • BitNet/4-bit, ~1.5–3B effective    │
            │              │   • strong CoT / RL-reasoning          │
            │              ▼                                        │
            │   ┌─────────────────────────────────────────────┐    │
            │   │ Knowledge plane (mostly on DISK, streamed)   │    │
            │   │  • Retrieval over a big embedded corpus      │    │
            │   │  • MoE experts paged from NVMe on demand     │    │
            │   │  • Tools: calc, code-exec, web/search        │    │
            │   └─────────────────────────────────────────────┘    │
            └──────────────────────────────────────────────────────┘
```

The resident footprint stays under ~2 GB; the *apparent* capability is far larger
because knowledge and rarely-used experts live on disk and are paged in per token/query.

## What "comparable to frontier" will and won't mean (set expectations now)

- **Will get close on**: reasoning/math/code with CoT, retrieval-grounded factual Q&A,
  tool-using agentic tasks, summarization/extraction/rewriting over provided context.
  On these, small distilled reasoners + retrieval already rival much larger models
  (see research notes 01, 04, 06).
- **Will still lose on**: broad zero-shot world knowledge without retrieval, very long
  unaided multi-step reasoning, the long tail of niche facts, raw "knows everything"
  feel. We close most of this gap with retrieval + tools, but not 100%.
- **Honest target**: match/beat a 7–14B general model and approach GPT-3.5/early-GPT-4
  quality *on the supported task surface*, at <2 GB resident + disk.

## Non-negotiable design rules (derived from physics)

1. **Resident RAM ≤ ~2 GB** for weights+activations; KV cache budgeted separately and
   quantized (FP8/INT4). Leave headroom — OOM = dead VPS.
2. **Disk is a first-class tier.** Knowledge and cold experts stream from NVMe via mmap.
   Design around NVMe bandwidth, not capacity.
3. **Memory-bandwidth bound, not compute bound.** Optimize bytes-moved-per-token. This
   is why low-bit quantization buys *speed*, not just size.
4. **Quantize for the cliff**: ship 4-bit by default; pursue BitNet-1.58 for the resident
   reasoner (best capability-per-byte; needs train/finetune-from-scratch in ternary).
5. **Pragmatism over purity**: start from the best open base (Qwen/Llama/Gemma) and
   specialize/distill; only go train-from-scratch where it pays (the BitNet reasoner).
6. **Single static binary deploy** (Go) — `scp` one file, run. No CUDA, no Python.
