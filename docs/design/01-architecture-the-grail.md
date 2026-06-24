# tiny-llm — The System Design ("El Santo Grial")

> This is the design we commit to. The build plan (`02-build-plan.md`) must respect it.
> It is grounded in `../research/` (every claim there has a source URL).

## 0. The one idea

A frontier model is **a reasoning engine welded to an encyclopedia welded to a calculator**,
all compressed into one giant blob of weights. We **unweld them**:

| Frontier (monolith) | tiny-llm (system) | Where it lives |
|---|---|---|
| Reasoning circuits | **Cortex** — small RL-reasoner | RAM (~1.3–2 GB) |
| Memorized facts (~2 bits/param) | **Library** — retrieval over disk corpus | NVMe/SSD (10s of GB) |
| Exact compute / live data | **Hands** — tools (calc, code, web) | external processes |
| Inner monologue / planning | **Conductor** — agentic loop | Go control plane |

Research basis: ~2 bits-of-knowledge/param ceiling and facts-live-in-MLPs (research 04);
RETRO 7.5B≈GPT-3-175B and Atlas-11B>PaLM-540B via retrieval (04); R1-Distill-Qwen-1.5B beats
GPT-4o on MATH-500 (06); TinyAgent-1.1B≈GPT-4-Turbo on tools (04). The pieces are each proven;
tiny-llm is the **integration** into one portable system that fits 4 GB.

We call it the **CLH-C architecture: Cortex · Library · Hands · Conductor.**

## 1. Why this is the only honest path to "portable frontier"

From `00-constraints-and-physics.md`: a monolith comparable to GPT-4 cannot fit in 4 GB, and
per-token weight streaming from NVMe is **0.05–0.2 tok/s = dead** (research 03). Two hard rules fall out:

1. **The Cortex must fit entirely in RAM** (no per-token disk paging).
2. **Bigness comes from per-*query* retrieval + tools**, never from per-*token* paging.

That is the whole trick. Retrieval touches disk a few times per request (~5 ms, research 04);
the hot autoregressive loop never leaves RAM.

## 2. The four components

### 2.1 Cortex — the resident reasoner (RAM)
- **What:** a small, strongly-distilled **reasoning** model. Target **1.7–4B effective params**.
- **Chosen base (research 01):** **Qwen3-1.7B in *thinking* mode** as the default Cortex
  (MATH-500 93.4, AIME'24 48.3 [V], ~1.1 GB at Q4, Apache-2.0). **Qwen3-4B-thinking** is the stretch
  option (MATH-500 97.0, AIME'24 73.8) when the Library/runtime footprint is kept lean. Qwen3 wins on
  reasoning-per-byte, a clean license for distillation, and a built-in thinking mode the Conductor
  exploits as a test-time-compute knob. Fallbacks: Phi-4-Mini (instruction/coding), R1-Distill-Qwen-1.5B (pure math).
- **Quantization:** ship **4-bit (Q4_K_M)** day one (~1.0–2.5 GB, ~1% quality loss, research 02).
  Moonshot track: a **BitNet b1.58 ternary** Cortex (0.4 GB for 2B non-embedding, native CPU speed,
  research 02) — must be trained/finetuned in ternary (QAT), the highest capability-per-byte option.
- **Why a reasoner, not a generalist:** reasoning distills into tiny models extraordinarily well
  (R1-Distill-Qwen-1.5B = 83.9 MATH-500 > GPT-4o); broad knowledge does NOT — so we don't store it here.
- **Activation choice (moonshot):** train with ReLU²/dReLU to unlock PowerInfer-style CPU activation
  sparsity (research 03); SwiGLU only gives 43–53% sparsity, ReLU-family 90–98%.
- **KV cache:** quantized FP8 by default (50% cut, <0.1% loss); SnapKV-style compression for long
  context (92% KV cut, research 02/04). Keeps the 4 GB budget safe at long sessions.

### 2.2 Library — the knowledge plane (disk)
- **What:** the "facts" we removed from the weights, as an external, swappable corpus.
- **Index:** **DiskANN-style SSD-resident ANN** — 1B vectors from SSD at <3 ms on low RAM (research 04).
  This is the key: the index lives on NVMe, only a small navigation cache sits in RAM.
- **Embeddings:** a tiny CPU embedder (all-MiniLM-L6-v2 22M/80MB, 14k sent/s, or nomic-embed 137M).
- **Contents:** curated corpus (e.g., FineWeb-Edu slice, Wikipedia, domain docs, the user's own data).
  Swappable per deployment → the *same* binary becomes a medical / legal / coding assistant by
  swapping the Library. This modularity is itself the product.
- **Retrieval discipline:** put retrieved chunks at prompt start/end (dodge "lost in the middle"),
  rerank, and feed only what's needed (token budget is precious on CPU).

### 2.3 Hands — tools (external processes)
- Calculator / symbolic math, sandboxed code execution, web/search, structured DB/API calls.
- **ToolRAG:** retrieve the relevant tool schemas per query so the Cortex reasons over few tools
  (research 04 — the trick that took TinyAgent-1.1B to GPT-4-Turbo level).
- Tools offload exactly what small models are worst at: exact arithmetic, fresh data, long code exec.

### 2.4 Conductor — the agentic loop (Go control plane)
- Owns: request lifecycle, planning, the retrieve→reason→act→verify loop, tool dispatch, streaming,
  sessions, KV/memory management, structured/constrained decoding, self-consistency / budget-forcing
  when accuracy matters and latency allows.
- **This is where tiny-llm's novelty and most of its code live — and Go is ideal here** (research 05).

## 3. Engine & language (from research 05)
- **Go control plane + cgo→ggml inference kernel.** Pure-Go matmul is 5–16× too slow and Go SIMD is
  experimental/buggy; ggml gives native llama.cpp speed on the hot path.
- Big tensors live in **mmap'd C memory, off the Go heap**, so the GC never scans them. Set
  `GOMEMLIMIT` + `GOGC≈30` to stay safe in 4 GB.
- Clean `Kernel` interface so we can swap ggml ↔ a custom **BitNet ternary kernel** (bitnet.cpp-style)
  without touching the Conductor/Library/Hands.
- Single static binary via musl/zig-cc. `scp` one file + a Library directory, run. No CUDA, no Python.
- Optional alt: Rust+candle (no GC, clean SIMD) — ~30% slower on CPU Q4, 3–6 mo more work. Not chosen.

## 4. Memory budget (4 GB box, concrete)
| Item | Budget |
|---|---|
| OS + Go runtime | ~0.4 GB |
| Cortex weights (3B Q4 or 2B BitNet) | 1.0–2.0 GB |
| KV cache (FP8 + SnapKV, ~8–16k ctx) | 0.3–0.6 GB |
| Embedder + ANN nav cache | 0.2–0.4 GB |
| Working/activations/headroom | 0.5–0.8 GB |
| **Library index + corpus** | **on NVMe, not counted** |
| **Total RAM** | **≈ 2.4–3.6 GB ✅** |

## 5. Capability envelope (honest, from the research)
**Will rival much larger / frontier models on:** math & logical reasoning (distilled reasoner),
retrieval-grounded factual Q&A (Library), tool/agentic tasks with schemas (Hands+ToolRAG),
summarize/extract/rewrite over provided context.
**Will still trail frontier on:** open-ended agentic coding (SWE-bench 7B ~20% vs ~77%), frontier
science (GPQA), the very hardest unaided reasoning, and broad zero-shot knowledge when the Library
lacks the domain. We narrow these with retrieval+tools+test-time compute; we do not claim to erase them.
**Target:** match/beat a 7–14B general model and approach GPT-3.5 / early-GPT-4 *on the supported
surface*, at <3.6 GB RAM + disk, on a 4-vCPU VPS.

## 6. What makes this a genuinely "different system"
1. Knowledge is **data, not weights** — hot-swappable, updatable without retraining, auditable.
2. The model is a **reasoning CPU**; disk is **RAM-of-facts**; tools are **peripherals**; Go is the **OS**.
3. Capability scales by **growing the Library and Hands**, not by growing the resident model — so it
   stays portable forever while getting smarter.
4. One static binary + a swappable corpus = a frontier-ish assistant you `scp` to a $5/mo VPS.
