# Metis — Phase 2: Decomposed Cascade Architecture

> Phase 1 proved the system beats the weights (same 1.7B: 0/8 → 8/8 factual, 4/4 → 0/4
> fabrications). Phase 2 asks: how small can we make the generator while preserving that quality?
> Answer: very small — by specializing the verifier.

---

## 0. The core insight

The 1.7B Cortex currently does four distinct cognitive jobs:

| Job | What it requires | Minimum viable model |
|-----|-----------------|----------------------|
| Routing (tool vs knowledge?) | intent classification | rule-based / 50M |
| Extraction (which span is the answer?) | reading comprehension | already separated (cosine fast-path) |
| **Generation** (synthesize a grounded answer) | language model | small LLM (0.6B floor) |
| **Verification** (is the claim entailed by evidence?) | NLI / entailment | specialized 22M model |

Generation and verification are the only jobs that need a neural model. But they need **different
kinds** of neural model. Mixing them in one generalist means the model is sub-optimal for both.

**The hypothesis:**
> On a grounded QA surface, a specialized 22M NLI model verifies better than a 1.7B generalist.
> A 0.6B generator + 22M verifier achieves equal or better quality than a 1.7B generalist doing both,
> at ~44% of the resident RAM.

---

## 1. The experiments

| ID | Generator | Verifier | Resident RAM | Prediction |
|----|-----------|----------|-------------|------------|
| **E0** (baseline) | qwen3:1.7B | same 1.7B (LLM judge) | ~1.1 GB | 8/8 facts, 0/4 fabrications (measured) |
| **E1** (naive down) | qwen3:0.6B | same 0.6B (LLM judge) | ~0.4 GB | degraded on both |
| **E2** (cascade) | qwen3:0.6B | NLI-MiniLM (22M) | ~0.5 GB | generation ≈ E1, verification ≈ E0 → net ≈ E0 |

E1 is expected to be bad: the 0.6B struggles as a generator *and* as a verifier. E2 compensates
for the weaker generator by pairing it with a verifier that is *better than the 1.7B at its one job*.

The key result to look for in E2: **does the specialized verifier recover the quality gap vs E1?**
If yes, the bottleneck was never the generator — it was the generalist verifier.

---

## 2. Why the 22M NLI model is a better verifier than the 1.7B generalist

`cross-encoder/nli-MiniLM-L-6-v2` (22M params, ~85MB) is a cross-encoder fine-tuned specifically
on NLI datasets (SNLI, MultiNLI). Its sole job: given (premise, hypothesis), output
{contradiction, entailment, neutral} scores. Contrast with the 1.7B asked to output "SUPPORTED
or UNSUPPORTED" via a system prompt — it's a generalist interpreting a specialized task via
instruction following, which is inherently noisier.

Verification asymmetry from the research (doc 06): recognizing a correct step is far easier than
producing one. A tiny model trained on that exact recognition task outperforms a large model that
has to interpret it as a generation task.

---

## 3. Architecture change

```
                   ┌── Phase 1 ──────────────────────────────┐
                   │  qwen3:1.7B                              │
  query + evidence │    ├── generate answer  (generation)     │
  ─────────────►   │    └── verify claim    (NLI/entailment)  │
                   └──────────────────────────────────────────┘

                   ┌── Phase 2 (cascade) ────────────────────┐
                   │  qwen3:0.6B ─── generate answer          │
  query + evidence │                                          │
  ─────────────►   │  NLI-MiniLM (22M) ─── verify claim      │
                   └──────────────────────────────────────────┘
```

The sidecar pattern already exists in Metis (SearXNG for web search). The NLI verifier is the
second sidecar: a tiny Python service that loads the cross-encoder at startup and exposes one
endpoint `POST /verify` → `{verdict, scores}`.

---

## 4. Implementation

**`src/verifier.rs`** — new module: `Verdict` enum, `VerifierKind` enum (Llm | Nli), `verify()`
dispatch. The conductor's `GvsConfig` gains a `verifier: VerifierKind` field.

**`conductor.rs`** — `answer()` calls `cfg.verifier.verify(k, &cand, &evidence)` instead of the
hardcoded LLM judge. `Verdict` moves to `verifier.rs` (re-exported from `conductor` for compat).

**`nli/`** — Python sidecar: `server.py` (22 lines, stdlib HTTP + sentence-transformers),
`requirements.txt`, `Dockerfile` (pre-bakes the model). Adds ~256 MB to the deployment image
(model cached at build time), ~85 MB resident RAM.

**`docker-compose.yml`** — adds `nli` service. `METIS_NLI_URL=http://nli:9090` wires it up.
`METIS_MODEL=qwen3:0.6b` downgrades the generator for E2.

**`bench/benchmark.py`** — adds `METIS_VARIANTS` env var for multi-config comparison in one run.

---

## 5. Wiring / env vars

| Env var | Values | Effect |
|---------|--------|--------|
| `METIS_MODEL` | `qwen3:1.7b` (E0/E1) / `qwen3:0.6b` (E2) | selects the generator |
| `METIS_NLI_URL` | unset (E0/E1) / `http://nli:9090` (E2) | selects the verifier |
| `METIS_SEARCH` | 1–N | max GVS candidates (unchanged) |

Setting `METIS_NLI_URL` automatically switches `VerifierKind::Nli`; leaving it unset keeps
`VerifierKind::Llm` (current behavior). Fully backward-compatible.

---

## 6. Honest expected results and the wall

**If E2 ≈ E0 quality at 44% RAM:** the bottleneck was the generalist verifier. Scaling the
generator below a floor does not hurt quality on grounded tasks — which means the "minimum viable
generator" for this surface is very small (or even extractive).

**If E2 ≪ E0:** the generator is the bottleneck. The 0.6B cannot produce candidates good enough
for even a specialized verifier to select from. This is also useful data: it sets the generator
floor empirically rather than by assumption.

**The wall that specialization can't fix:** if the right answer is never in the generator's
candidate set (pass@1 ≈ 0), no verifier rescues it. The 64% copy-rate (doc 06) makes this
unlikely for grounded tasks — most answers are extractive — but it's the risk.

**Next step if E2 works:** replace the extractive fast-path (cosine over sentences) with a
dedicated extractive QA model (DeBERTa-SQuAD2, 184M). The generator would only be called for
the ~36% of queries where extraction fails — dropping total LLM calls by >60%.
