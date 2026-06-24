# tiny-llm — Build Plan

> Respects `01-architecture-the-grail.md`. Phased, each phase shippable & measurable.
> Philosophy: **stand on giants** (research 06: never pretrain from scratch). Start from the best
> open base, build the *system* around it, specialize via cheap distillation last.

## Guiding constraints (non-negotiable, from design)
- Cortex fully in RAM; no per-token disk paging. · Bigness from per-query retrieval + tools.
- Go control plane + cgo→ggml kernel; tensors off the Go heap; `GOMEMLIMIT` set.
- Ship 4-bit; FP8 KV; target ≤3.6 GB RAM on 4 vCPU.

## Repository layout
```
tiny-llm/
├── cmd/tinyllm/          # main: serve | chat | index | eval
├── internal/
│   ├── kernel/           # cgo→ggml bridge; Kernel interface (swap BitNet later)
│   ├── cortex/           # model load, sampling, KV mgmt, structured decoding
│   ├── library/          # embedder + DiskANN-style ANN + chunk store + RAG
│   ├── hands/            # tool registry, ToolRAG, calc/code/web adapters
│   ├── conductor/        # agentic loop: plan→retrieve→reason→act→verify
│   └── server/           # HTTP/SSE API, sessions, auth, metrics
├── models/               # GGUF weights (gitignored)
├── library/              # corpus + on-disk index (gitignored)
├── bench/                # eval harness (MMLU/GSM8K/HumanEval subset, RAG, tools)
└── docs/                 # research/ + design/
```

## Phase 0 — Engine skeleton (proves the kernel) ⏱ ~1–2 wk
- cgo bridge to ggml; load a GGUF; greedy + sampled token generation; SSE streaming HTTP API.
- Weights via mmap (off Go heap); set `GOMEMLIMIT`, `GOGC=30`; measure RSS.
- **Done when:** a 3B Q4 model answers over HTTP on the 4 GB VPS at **≥6 tok/s**, RSS <2.5 GB.

## Phase 1 — Cortex + minimal Conductor (MVP chat) ⏱ ~1–2 wk
- Base finalized (research 01): **Qwen3-1.7B-thinking** (default) / **Qwen3-4B-thinking** (stretch),
  Q4_K_M. Wire the thinking-mode tags so the Conductor can toggle reasoning depth. Chat templating,
  stop tokens, CoT scaffolding.
- Structured/constrained decoding (JSON/grammar) for later tool calls.
- Sampling controls; basic self-consistency toggle.
- **Done when:** GSM8K/MATH subset matches the base model's published numbers (no regression) on CPU.

## Phase 2 — Library (knowledge plane) ⏱ ~2–3 wk
- CPU embedder (all-MiniLM-L6-v2 / nomic-embed). `tinyllm index <corpus>` builds an on-disk
  DiskANN-style ANN + chunk store; only a nav cache in RAM.
- RAG pipeline: embed query → ANN search → rerank → place chunks at prompt start/end → generate.
- Swappable corpora (general / domain). Citations in output.
- **Done when:** on a knowledge-QA set, Cortex+Library beats bare Cortex by a large margin; retrieval
  adds <30 ms/query; RAM stays <3 GB with a 10 GB+ on-disk index.

## Phase 3 — Hands (tools) ⏱ ~2 wk
- Tool registry + JSON-schema contracts; **ToolRAG** (retrieve relevant tool schemas per query).
- Adapters: calculator/symbolic, sandboxed code exec, web/search, generic HTTP/DB.
- **Done when:** on a tool-use benchmark, tiny-llm reliably selects+calls the right tool; exact-math
  and live-data queries that the bare Cortex fails are now solved.

## Phase 4 — Conductor agentic loop ⏱ ~2–3 wk
- plan → retrieve → reason → act(tool) → observe → verify → answer; bounded iterations.
- Test-time compute knobs: self-consistency, budget-forcing ("Wait") when accuracy>latency.
- Verification pass (self-check / tool-check) before finalizing.
- **Done when:** end-to-end agentic tasks (multi-step QA needing retrieval+tools) succeed; ablations
  show each component contributes.

## Phase 5 — Specialize the Cortex (cheap distillation) ⏱ ~1–2 wk + GPU rental
- Generate reasoning/tool-use traces from an **open teacher** (Qwen2.5-72B / Llama-3.1-70B — license-
  safe; never closed APIs, research 06). SFT + DPO on the chosen small base; optionally QLoRA first.
- Target the *system's* weak spots: tool-call formatting, retrieval-grounded answering, CoT for our
  domains. **Budget: $1–5K** (research 06). Re-quantize to Q4; re-bench.
- **Done when:** specialized Cortex beats the off-the-shelf base on our eval harness at equal size.

## Phase 6 — Moonshot: BitNet-1.58 ternary Cortex ⏱ research-grade, optional
- Train/finetune the Cortex in **ternary (QAT)** for ~0.4 GB @ 2B + native CPU speed (research 02),
  optionally with ReLU²/dReLU activations for PowerInfer-style CPU sparsity (research 03).
- Custom ggml-compatible ternary kernel behind the `Kernel` interface (bitnet.cpp-style, 2–6× CPU).
- Highest capability-per-byte; biggest training cost/risk → do only after the system proves out.
- **Done when:** ternary Cortex matches the Q4 Cortex quality at a fraction of RAM and higher tok/s.

## Cross-cutting tracks (run continuously)
- **Quantization:** Q4_K_M default; FP8 KV; SnapKV long-context; evaluate IQ-quants if we go smaller.
- **Eval harness (bench/):** MMLU/GSM8K/HumanEval subsets + RAG-QA + tool-use + latency/RSS on the
  actual 4 GB VPS. No claim ships without a number from this box.
- **Deploy:** static musl/zig build; `scp` binary + library dir; systemd unit; health/metrics.
- **Honesty ledger:** keep the capability-envelope table (design §5) updated with measured gaps.

## Sequencing & risk
- Critical path: Phase 0 → 1 → 2 → 4 (this is the demoable "different system"). Phases 3 & 5
  multiply quality. Phase 6 is the research bet — isolate it so it can't block the product.
- Biggest risks: (a) cgo static-build friction → mitigate with zig-cc early; (b) CPU tok/s too low →
  smaller/BitNet Cortex + KV compression; (c) RAG quality → reranking + corpus curation; (d) Phase 6
  training cost → keep optional, fund only after Phases 0–4 demonstrate value.

## First concrete step
Scaffold `cmd/tinyllm` + `internal/kernel` and get ggml loading a Q4 GGUF behind the `Kernel`
interface (Phase 0). Everything else builds on that bridge.
