# Metis — implementation & deployment log (the bitácora)

> The earlier design docs (00–06) are the *plan*. This one is the *record*: what got built, what we
> measured, and the production gotchas we hit and fixed. Every number here was measured on real
> hardware (a local Mac for iteration, a Railway shared-CPU box for production), not projected.

---

## 1. What is actually built and live

| Plane | Status | Where |
|---|---|---|
| **Cortex** — qwen3:1.7b (4-bit) via ollama | live | RAM |
| **Library** — local retrieval over a swappable index | live | disk |
| **Hands** — calc, clock, **web search (SearXNG)** | live | subprocess / sidecar |
| **Conductor** — the Generate·Verify·Search (GVS) loop | live | `src/conductor.rs` |

Live deployment: a single-container Railway service (ollama + metis) + a SearXNG sidecar.
`POST /ask {"q":"..."}` · `GET /healthz` · `GET /readyz`.

---

## 2. The Conductor — Generate · Verify · Search (`src/conductor.rs`)

The mechanism that turns an unreliable tiny *generator* into a reliable *system*, at inference time,
**with no datasets and no retraining**:

1. **Generate** a grounded candidate (the Cortex may call tools).
2. **Verify** it against the retrieved evidence using the *same* qwen3 in judge mode — external
   framing ("does this CLAIM follow from this EVIDENCE?"), never "are you sure?".
3. If unsupported/uncertain, **Search**: draw a few diverse candidates (higher temperature), verify
   each, keep the first the evidence supports.
4. If none survive, **Abstain** — emit nothing unverified.

Why it works: verifying an answer is far easier than producing one. A 1.7B is a poor generator but a
reliable grounded *judge*, so we spend its strength (recognition) to cover its weakness (generation).
Knobs: `METIS_SEARCH=N` (candidates; 1 = verify-only). The HTTP response exposes `path`
(`extractive` | `verified` | `searched` | `abstained` | `unverifiable`), `verified`, `attempts`.

---

## 3. The web as a verified Library (`src/hands/web.rs` + `research()`)

The most "Metis" way to go open-domain without bigger weights: **the web is just a Library too big to
store.** `research()` blends local hits with live web results and runs the *same*
ground→verify→cite→abstain loop over the combined evidence — the model never trusts the web raw.

- **Backend:** a self-hosted **SearXNG** metasearch instance. Metis talks **plain HTTP** to it over
  the private network; SearXNG does the HTTPS to upstream engines — so the Metis binary stays
  **TLS-free** (no `ring`/openssl on musl). Sovereign, no API key, no vendor lock-in.
- **Wiring:** web results are embedded with the same CPU embedder, ranked by cosine, and become
  `Hit`s with `source = url`. The extractive fast-path is **skipped** for web hits (a title that
  echoes the query scores high but is not the answer).
- **Gate:** web is only consulted when local grounding is thin (local top score < 0.55), so private
  docs still win outright.

Enable with `METIS_SEARCH_URL=http://<searxng>:8080`.

### Live demos (against the production URL)

- *"Who wrote Dune and when?"* → "Frank Herbert … 1965 **[1]**" citing Britannica. A fact **not** in
  the local Library, answered from the live web, verified.
- *"Benchmark Opus 4.8 vs Sakana Fugu Ultra"* → before web: abstained (`sources: []`). After web:
  synthesized a cited comparison from real sources and verified it. (With no web, the bare model
  fabricated a full fake benchmark — see §4.)

---

## 4. Benchmark — bare Cortex vs Metis (same weights)

Hold the weights fixed (`qwen3:1.7b`), change only the architecture. Surface: a private doc
(`sample-docs/zephyr.md`) the model never trained on, plus unanswerable and general questions.
Harness: `bench/benchmark.py`. Full write-up: `bench/RESULTS.md`.

| metric | BARE (ollama, no RAG/verify) | METIS — local | METIS — Railway (live) |
|---|---:|---:|---:|
| **answerable facts correct** | 0 / 8 | **8 / 8** | **8 / 8** |
| **fabrications on unanswerable** (lower=better) | 4 / 4 | **0 / 4** | **0 / 4** |
| **general** (incl. exact math) | 1 / 2 | **2 / 2** | **2 / 2** |

The bare model invented a ratification year ("2015"), a codename ("Zephyr"), a chairperson
("John W. Lott"), a HQ ("San Francisco"), and got the multiplication wrong (191,737,397 vs
192,042,557). The *same* model inside Metis got every fact right with citations, abstained on all
four unanswerable questions, and used the calc tool for exact math. **The architecture, not the
weights, is the difference.**

---

## 5. Production hardening log (symptom → cause → fix)

The valuable part of the bitácora — small things that decide "unusable" vs "responsive" on a cheap box.

1. **Index load crashes with a huge bogus allocation.**
   Cause: a stale `index.gob` from an older struct/bincode layout. Fix: rebuild the index with the
   current binary.

2. **One answerable fact missed (`1.84 GB` → `"…exceed 1"`).**
   Cause: the sentence splitter broke on the `.` inside the decimal `1.84`. Fix: only split on
   sentence punctuation **followed by whitespace** — decimals have no space after the dot
   (`src/library/extractive.rs`). Regression test added.

3. **`llama-server` segfaults on load.**
   Cause: `OLLAMA_KV_CACHE_TYPE=q8_0` — quantized KV needs flash-attention; without it it crashes.
   Fix: drop it. Default f16 KV at ctx 2048 is only ~224 MiB anyway.

4. **Decode collapses to 0.09 tok/s (a 34-token answer took 380 s).**
   Cause: llama.cpp spawns one thread per *host* core, but a shared PaaS container's CPU *quota* is a
   fraction of that. Prefill (batched) tolerated it; decode (per-token barrier sync) thrashed.
   Fix: cap threads to the real allocation via `num_thread` (env `METIS_NUM_THREAD`, plumbed into the
   ollama `options`). **Result: prefill 19.7 → 218 tok/s, decode 0.09 → 35.9 tok/s, a verified answer
   408 s → 3.4 s.** Single highest-impact fix of the whole effort.

5. **`llama-server` segfaults on load again — but not on memory.**
   Logs showed a **380 GB host with AMX** (Intel Sapphire Rapids). ollama's AMX / flash-attn codepath
   segfaults loading qwen3 on those hosts (intermittent: earlier deploys landed on hosts where it
   didn't). Fix: `OLLAMA_FLASH_ATTENTION=0` + `OLLAMA_LLM_LIBRARY=cpu_avx2` (trade a little speed for
   not crashing) + a **persistent volume** at `/root/.ollama` so the model isn't re-pulled (and
   re-risked) every deploy.

6. **Web query returned a YouTube *title* instead of an answer.**
   Cause: the extractive fast-path fired on a web snippet whose title echoed the query (score 0.88 >
   gate). Fix: never extractive-shortcut web evidence — web hits must be synthesized and verified.

---

## 6. Deployment topology (Railway)

Project `metis-0`, two services on the private network:

- **`metis-0`** — single container: `ollama serve` + `metis serve`, model on a persistent volume.
  Key env: `METIS_MODEL=qwen3:1.7b`, `METIS_SEARCH=2`, `METIS_NUM_THREAD=4`,
  `OLLAMA_KEEP_ALIVE=-1`, `OLLAMA_CONTEXT_LENGTH=2048`, `OLLAMA_FLASH_ATTENTION=0`,
  `OLLAMA_LLM_LIBRARY=cpu_avx2`, `METIS_SEARCH_URL=http://searxng.railway.internal:8080`.
  Built from `Dockerfile.railway` + `entrypoint.railway.sh`.
- **`searxng`** — `searxng/` image with JSON API enabled and `bind_address: "::"` (IPv6 — required
  for Railway's private network). Built from `searxng/Dockerfile` + `searxng/settings.yml`.

---

## 7. Honest envelope & next levers

What this is: **frontier-useful on the verifiable / researchable surface** (grounded QA, math, code,
"compare X and Y", research) at tiny size, fully local — winning on *reliability* (it abstains, it
cites, it verifies) where frontier models still hallucinate. What it is **not**: open-ended
creativity, taste, or deep novel reasoning on questions with no checkable answer.

Two known rough edges and the next levers:

- **Quality inherits source quality.** The verifier checks the answer against what was retrieved; it
  does not yet judge the *trustworthiness* of the source. Junk web results (clickbait, video titles)
  leak in. Next lever: domain ranking / source filtering to raise the floor.
- **The web-trigger threshold (0.55)** sometimes fetches web for locally-answerable questions, adding
  noise (the answer stays correct, just noisier sources). Tunable.

Bigger bets still ahead (from docs 01/06): the **verified-trace flywheel** (every verified answer
becomes a retrievable exemplar — learning without training) and **cheap latent search** (make a
reasoning step cost ms so test-time search is affordable on CPU).

---

## 8. The Ornith-1.0 study — self-scaffolding & a deterministic trust boundary (2026-06-27)

### 8.1 What prompted it

On 2026-06-25 DeepReinforce released **Ornith-1.0**, an MIT-licensed open-source family of *agentic
coding* models (9B Dense, 31B Dense, 35B MoE, 397B MoE; built on Gemma 4 + Qwen 3.5). It posted
frontier-adjacent numbers:

| benchmark | Ornith-397B | Claude Opus 4.7 | Claude Opus 4.8 |
|---|---:|---:|---:|
| SWE-Bench Verified | **82.4** | 80.8 | 87.6 |
| Terminal-Bench 2.1 | **77.5** | 70.3 | 85.0 |
| SWE-Bench Pro | 62.2 | 64.3 | 69.2 |

The headline that matters for Metis is **not** the flagship — it is the **9B**: 43.1 on Terminal-Bench
(≈ Gemma 4-31B, ~3.4× larger) and 69.4 on SWE-Bench Verified (vs 53.2 for Qwen3.5-9B). *A small model,
well-scaffolded, fights models several times its size.* That is, almost word for word, the Metis bet
(`README.md`, doc 01). Independent June-2026 confirmation that **scaffold beats parameters** is worth
banking.

Sources: DeepReinforce blog (`deep-reinforce.com/ornith_1_0.html`), MarkTechPost (2026-06-25),
explainx.ai, testingcatalog.com, HF `deepreinforce-ai/Ornith-1.0-9B`.

### 8.2 The two ideas we took

1. **Self-scaffolding.** Ornith does not hard-code the agent harness (retry budget, orchestration,
   temperatures). It *learns* the scaffold during RL: conditioned on the task and the last scaffold, the
   model proposes a refined scaffold, then a solution under it; reward flows to **both** stages, so good
   orchestration patterns survive by selection. The Conductor's GVS loop **is** exactly such a scaffold —
   but ours was *fixed* (one `GvsConfig` for every query). Ornith's lesson: make it *adapt to the task*.

2. **A deterministic trust boundary, monitor-first.** Against reward hacking, Ornith stacks three layers:
   a *fixed trust boundary* (env/tools immutable), a *deterministic monitor* (catches forbidden-path /
   unauthorised-tool gaming → zero reward), and a *frozen LLM judge* (catches intent-level gaming). The
   cheap, **uncheatable** deterministic check runs *before* the fallible judge. GVS only had the judge.

We **deliberately did not** copy Ornith's mechanism — 397B, multi-hour pipeline-RL with
staleness-weighted GRPO. That is the cloud-bound, retrain-the-weights world Metis bets against. We took
the *ideas* and implemented the honest, runnable, **no-training** slice of each.

### 8.3 What we built

- **Layer 1 — deterministic citation monitor** (`src/monitor.rs`). Pure code: every inline `[n]`
  citation must reference a real retrieved source (`1..=n_sources`). `[4]` with 3 sources, `[0]`, or a
  cite for a fact that came from a different chunk — all caught for free, *before* the judge spends a
  token. This closes a real gap: a tiny Cortex emits grounded-*looking* citations that an entailment
  judge (which scores prose against pooled evidence) can wave through even when the citation is invented.
  Markdown links `[text](url)` and footnote markers `[note]` are correctly ignored. Wired into
  `conductor::answer` as a gate: a candidate is accepted only if it clears **Layer 1 (monitor) AND
  Layer 2 (judge)**, on both the first generation and every search candidate.

- **Self-scaffolding at inference** (`src/scaffold.rs`). A deterministic classifier picks a profile per
  query and tunes the GVS knobs — the cheap version of Ornith's learned scaffold, with the *seam* in the
  right place (swap `Scaffold::select` for an LLM proposer later):
  - `Compute` (exact arithmetic) → 1 low-temp pass; the `calc` tool is authoritative, so diverse
    re-rolls only invite the model to *skip* the tool.
  - `OpenDomain` (web-blended evidence) → wider budget (≥4) + decorrelated search (temp 0.9).
  - `Factual` (local lookup/synthesis) → the balanced default.
  - `Direct` (no evidence) → single pass; nothing to verify against.
  Controlled by `METIS_SCAFFOLD=auto|off|compute|opendomain|factual|direct`; `off` restores the legacy
  fixed config exactly (backward-compatible). Operator overrides (`METIS_SEARCH`, `METIS_NLI_URL`) still
  flow through as the baseline the scaffold tunes.

### 8.4 The benchmark run — what is honest here

The user asked to run the standard benchmarks (SWE-Bench / ARC-AGI-style, bare-then-Metis). **That run
was not possible in this sandbox, and pretending otherwise would be the one unforgivable thing.** Why:

- **No live Cortex.** ollama is not installed; the network policy allows only package registries
  (crates.io, PyPI), so the ollama binary (404) and any model pull are blocked. Every inference
  benchmark — `bench/benchmark.py`, SWE-Bench, ARC-AGI — needs a model generating tokens.
- **Railway API down.** The public deploy returned 404 (nginx) on `/healthz` and `/ask`, so the live
  Metis side could not be exercised either.
- **Scope.** SWE-Bench/Terminal-Bench are *coding-agent* harnesses (patch a repo, run its tests).
  Metis is a grounded-QA / RAG system, not a coding agent — those numbers measure a different machine.
  Metis's own apples-to-apples test is the bare-vs-Metis grounded-QA harness in `bench/`.

What we **could** run, and did:

- **`cargo test` — 32 passed, 0 failed** (incl. the new `monitor` + `scaffold` + `benchlayers` suites,
  alongside the nano gradient-checks).
- **Offline layer benchmark** (`metis bench-layers`, `src/benchlayers.rs` → `bench/results-layers.json`).
  No model required — it exercises the deterministic layers directly, BASE (behaviour before this
  change) vs METIS:

  | Layer 1 — citation monitor (13 cases: 6 fabricated, 7 clean) | fabricated caught | clean preserved | false rejects |
  |---|---:|---:|---:|
  | BASE (Layer 2 / judge only) | 0 / 6 | 7 / 7 | 0 |
  | METIS (Layer 1 + 2)         | **6 / 6** | 7 / 7 | **0** |

  Self-scaffolding routing: **9 / 9** queries routed to the correct profile. The monitor catches every
  fabricated citation the judge-only path lets through, with zero false rejects on clean answers.

The live bare-vs-Metis run is **ready to execute** wherever a Cortex is reachable — unchanged command
in `bench/RESULTS.md`. It belongs on a box with ollama, not in this network-restricted container.

### 8.5 Envelope

What `bench-layers` proves: the deterministic layer does exactly what it claims, deterministically, on
a labelled adversarial suite. What it does **not** prove: any end-to-end answer-quality delta — that
still needs the live harness, and the monitor's value only shows up when a real Cortex actually
fabricates a citation. Ornith's own caveat, adopted here verbatim: *scores vary with harness,
temperature, and context window — reproduce on your stack.* Next lever in this thread: replace the
heuristic `Scaffold::select` with a one-shot LLM scaffold proposer (true self-scaffolding), and feed
monitor rejections into the verified-trace flywheel.

---

### 2026-07-01 — Phase 5.0: TypeScript sandbox + Exec verifier

- **Built:** `VerifierKind::Exec`; a typed `Reward`/`ExecReport`; syntax, `tsc`, ESLint, and Vitest
  gates; and a pinned Node 22/TypeScript toolchain in `sandbox/code`. Every gate runs in a fresh
  Docker container with no network, read-only mounts/rootfs, 1 CPU, 1 GiB RAM, 256 PIDs, bounded
  output, and a 120 s timeout. Candidate patches cannot edit tests/tooling, add common skip/type
  bypasses, or mutate the sealed workspace at runtime. Optional held-out tests are injected only
  after the patch is applied. Tool/setup failures return `UNCERTAIN`; they never masquerade as a
  rejected patch.
- **Measured:** on an Apple M3 Pro (11 logical CPUs, 18 GiB RAM), six cold verifier runs took
  3,311–4,364 ms each (mean **3,612 ms**). Mean gate times were parse **597 ms**, typecheck
  **1,371 ms**, lint **832 ms**, and tests **807 ms**. The local image was 434,157,574 bytes.
  The workspace immutability check failed closed with `Permission denied`, as intended.
- **Decisions:** TypeScript is the first and only language surface. Commands cross the boundary as
  argv, never model-controlled shell text. A successful test reward requires both exact counts and
  a successful Vitest process/report. Four isolated containers cost startup latency but keep each
  gate independent and deterministic; optimize only after retaining these semantics.
- **Surprises:** the first run over-abstained because Vitest's transitive declarations required
  Node types and a newer standard library. Pinning `@types/node`, `lib: ESNext`, and
  `skipLibCheck` in the smoke fixture fixed the environment rather than weakening the verifier.
- **Verdict:** **go** for Phase 5.0. The oracle is executable, isolated, fail-closed, and returns the
  complete structured reward required by the search phase.
- **Next:** run H2 on the preregistered ~20 real code tasks/candidates, then use measured verifier
  latency to choose the Phase 5.2 parallel rollout budget.

### 2026-07-01 — Phase 5.1 scaffold: H2 harness smoke (not the preregistered experiment)

- **Built:** `src/bin/h2.rs` consumes a fixed JSON candidate set, preserves full raw gate evidence,
  and reports TPR, TNR, fabrication rate, balanced accuracy, and uncertainty by edit depth. A
  six-candidate g1/g2/g3 fixture includes visible and held-out tests.
- **Measured:** the hand-authored smoke classified all 3 supported and all 3 unsupported patches
  correctly: TPR **1.00**, TNR **1.00**, fabrication **0%**, balanced accuracy **1.00**, uncertainty
  **0** at every depth. Raw data: `bench/results-h2-smoke.json`.
- **Decisions:** policy rejections count as deterministic `UNSUPPORTED`; runtime, timeout, setup, and
  malformed-report failures count as `UNCERTAIN`. Candidate generation remains outside the harness
  so the same patches can be replayed against different verifier implementations.
- **Surprises:** none after the Phase 5.0 toolchain correction.
- **Verdict:** **pending**. This smoke validates plumbing only and does not satisfy the H2 success
  criterion; claiming `<2%` requires the preregistered real-task set.
- **Next:** freeze task IDs, patches, expected labels, and held-out tests for the real H2 dataset
  before running it.

### 2026-07-07 — metis-1m Night 0: MLX calibration on the M3 Pro (doc 13 §6)

- **Built:** `train-m/night0/train.py` — a byte-level ~15M-param GPT trunk-let trained with MLX on
  36.7 MB of real local code (8,000 TS/Rust/Python/Go/MD files, 34.9M train / 1.8M val bytes).
- **Measured (800 steps, 13.1M tokens, 16.0 min wall-clock, fp32):** **13,677 tok/s** steady;
  **1.21e12 FLOPs/s** ≈ **MFU 0.24** at an assumed 5-TFLOPS fp16 peak; train loss 4.16 → 2.63;
  **val loss 2.618 = 3.78 bits/byte**; extrapolation: **394M tokens per 8-hour night** at 15M params.
  Raw data: `train-m/night0/results-night0.json`.
- **Decisions:** all doc-13 budgets now derive from the measured 1.2e12 FLOPs/s, not the assumed
  2e12. At fp32 the 40M trunk costs ~14 nights for 2B tokens (vs the ~5–7 projected); the gap is
  precision and kernel overhead, not thesis. bf16 + `mx.compile` + larger batch are the identified
  levers to close it before Night 1.
- **Surprises:** MFU landed almost exactly on the doc-13 assumption (0.24 vs 0.25) *without any
  optimization* — in fp32. The greedy sample degenerates into repetition, as expected at 0.4
  epochs of byte-level training; the val bits/byte curve, not the sample, is the signal tonight.
- **Verdict:** **go.** The MacBook trains real models on real code at a measured, budgetable rate;
  the Night-0 gate ("recompute every budget from a measured number") is satisfied.
- **Next:** bf16 + compiled step to target ≥2e12 FLOPs/s, then Nights 1–7: the 40M trunk with
  RNT-shaped sequences (doc 12 data factory) instead of raw concatenated bytes.

### 2026-07-07 — metis-1 MVP: our own weights, end to end in OpenCode

- **Built:** the full sovereign loop on the M3 Pro — `train-m/night0/train.py` (now saves
  checkpoints), `serve.py` (OpenAI-compatible server with SSE streaming), `run.sh` (self-bootstrapping
  launcher: corpus/train/serve), and a `metis` provider block in OpenCode's config. Journey entry 12
  ("Our Own Weights") published on metis-web.
- **Measured:** 2,500 steps, 41.0M tokens, 50.8 min wall-clock at 13,462 tok/s sustained (MFU 0.24,
  fp32). Val loss **1.092 = 1.575 bits/byte** (Night-0 16-min checkpoint: 3.776). Qualitative arc,
  same prompt through OpenCode: 30-second checkpoint → letter soup with braces; 51-minute checkpoint
  → `//` comments, object literals with keys and quoted string values, `await db.end` in greedy
  sampling. OpenCode → local server → MLX weights round-trip verified streaming and non-streaming.
- **Decisions:** MVP serves raw MLX weights via a Python endpoint rather than GGUF conversion (the
  trunk-let's GPT-2-style arch isn't worth a converter; metis-1 proper will be llama-compatible from
  the start). Provider added alongside existing ones, not replacing.
- **Surprises:** none mechanical — the pipe worked on the second try (a session restart orphaned the
  first training run; `run.sh` now makes every piece re-runnable without this session's venv).
- **Verdict:** **go.** The pipeline (train → serve → agent) is proven end to end at US$0; quality is
  now an iteration loop, not a bet.
- **Next:** bf16 + `mx.compile` for ~2× throughput; the GitHub miner (issue → merged PR → diff + CI
  verdict) as M1.0's centerpiece; first RNT-shaped nights per docs 12/13.

### 2026-07-07 — Night 1: the speedrun lessons land — 12.3× measured

- **Built:** `train-m/night1/` — the modded-nanogpt record levers ported to MLX: **Muon** for the
  block matrices (Newton-Schulz orthogonalized momentum, AdamW only for embeddings/head/norms),
  bf16 weights + compute, `mx.compile`d step, RoPE, QK-norm, ReLU² MLP, zero-init output
  projections, untied head, logit soft-capping. Plus a night1 `serve.py` (KV cache with RoPE
  offsets) and `run.sh`.
- **Measured (same budget as Night 0: 14M params, 800 steps, 13.1M tokens):** reached Night-0's
  16-minute val loss (2.618) in **1.3 min → 12.3×** (upper-bounded by the 25-step val cadence).
  Final val **0.694 = 1.004 bits/byte** vs Night-0's 2.618/3.776 — and far past the 51-minute MVP
  checkpoint (1.092/1.575), crossed at ~5 min ≈ **10× to MVP quality**. Sustained **14.3k tok/s**
  (MFU 0.25) — bf16+compile fully paid for Muon's ~48 orthogonalizations/step, so the entire win
  is data efficiency at equal wall-clock throughput. Serving: 150 tokens in 0.66 s; qualitative
  jump at equal budget: `const { t } = useTranslation();` — an idiomatic React hook line.
- **Decisions:** night1 is the default trainer going forward. Doc-13 budgets re-derive at ~12×
  data efficiency: the 40M hive trunk drops from ~14 nights to **~1–2**; the doc-12 0.3B rental
  from ~US$2–3k to **~US$300–500** at equal target quality.
- **Surprises:** Muon's data efficiency at this scale exceeded the literature's ~2× headline —
  likely because it compounds with the architecture package (QK-norm, zero-init, soft-cap) and a
  narrow, code-only byte distribution. The smoke run's apparent throughput drop was val-eval time
  polluting the counter; excluded, throughput matched fp32 baseline.
- **Verdict:** **go.** Pre-registered metric (time to Night-0 target) beaten by an order of
  magnitude on measured hardware.
- **Next:** stacked/vectorized Newton-Schulz if throughput ever binds; then M1.0 (GitHub miner,
  RNT sequences) feeding night1 as the standard trainer.

### 2026-07-07 — Off the Mac: free-GPU training path (Kaggle) after local training overheated

- **Problem:** scaling toward Night 2 (50M params, seq 2048) saturated the 18 GB M3 Pro — training
  and using the machine at once was unsustainable. Algorithmic wins (Muon 12×, BPE 3.3×) don't
  change the physics of one shared-memory laptop doing datacenter work.
- **Built:** `train-m/kaggle/` — a self-contained PyTorch port of the Night-1 architecture + Muon
  (`kaggle_train.py`, one paste-able cell), a code-BPE tokenizer trained inline, and a local
  CPU-only OpenAI-compatible server (`serve_torch.py`, KV-cached, self-throttled threads) so the
  Mac only edits and serves. Target: Kaggle free **T4 x2** (30 h/week, background exec).
- **Measured (CPU validation, no GPU touched):** model+Muon backward through Newton-Schulz learns
  on synthetic data (loss < ln(vocab)); the serving KV cache matches the full forward to **9.5e-7**
  max abs diff — generation is provably correct before the first Kaggle run.
- **Decisions:** cloud = free Kaggle T4 (chosen over paid rental/Modal) — ~9× the Mac's GPU at zero
  cost and zero local load. PyTorch on cloud (not MLX) lets us run the reference speedrun stack;
  weights come back as `.pt` and serve locally on CPU, which is light for a ≤50M model.
- **Verdict:** **go.** The train→download→serve→OpenCode loop is wired and unit-validated; the only
  remaining step is a human running one Kaggle cell.
- **Next:** run the Kaggle cell (15–25 min), pull the 3 artifacts, `serve_torch.py`, test in
  OpenCode. Then M1.0's GitHub miner feeds larger corpora to the same free-GPU trainer.

### 2026-07-07 — Night 2: what the edge is NOT (nGPT) and what it IS (FIM)

- **Investigated (Mac-safe, gentle runs):** two candidate architectural edges for cheaper results.
  1. **nGPT / hypersphere normalization** (reported 4–20× fewer tokens). Ported to MLX
     (`night2/train_ngpt.py`): unit-norm representations, cosine attention with learned QK scale,
     normalized-LERP residuals with per-dim eigen learning rates, post-step weight normalization.
     Tested nGPT+Muon and nGPT+Adam vs Night-1's recorded byte-level curve, same 14M/seq1024.
  2. **Fill-in-the-Middle (FIM)** — the code-specific objective (StarCoder/DeepSeek-Coder standard
     at rate 0.5). Implemented in `night2/train_fim.py` as a PSM data transform (byte vocab + 3
     sentinels) on top of the Night-1 Muon champion.
- **Measured:** nGPT **lost** at this scale — nGPT+Muon val 3.75 @ step50 (night1: 2.41);
  nGPT+Adam val 3.25 @ step150 (night1: 1.35). The speedrun stack (Muon+RMSNorm+RoPE+QK-norm+
  zero-init+soft-cap) is already a strong baseline; a faithful-enough nGPT-lite did not beat it and
  the missing pieces (s_u/s_v scaling, alpha init schedule) are a known-finicky rabbit hole.
  FIM trained stably at 13k tok/s with no extra memory or FLOPs.
- **Decisions:** the edge at this scale is the **objective, not the optimizer**. Keep Night-1 (Muon)
  as the trainer; adopt **FIM** as the default objective (it matches OpenCode's actual job — infill
  an edit given both sides — and is free). nGPT shelved as a negative result, not deleted; revisit
  only with the full scaling-factor recipe if we ever hit an optimizer wall.
- **Verdict:** **go** on FIM (code-native, memory-neutral, composes with Muon+BPE). **no-go** on
  nGPT as built.
- **Infill caveat (honest):** FIM loss descended stably to val 0.919 @ 400 steps, but greedy AND
  sampled infill on the 14M byte-level checkpoint collapsed to whitespace — expected at only 6.5M
  tokens (night0's MVP was still "soup" at 41M). The FIM *mechanism* is validated (loss, sentinels,
  Muon-composition); crisp infill needs the real run (BPE + far more tokens), not more architecture.
- **Next:** FIM + BPE + Muon is the metis-1 pretraining recipe. At this scale the architecture is
  near-optimal; the remaining big levers are **data quality and teacher distillation**, not the
  optimizer or the block design.

### 2026-07-07 — Repair harness: the ruler that gates the generator (doc 14 step 1)

- **Built:** `train-m/repair/` — the Mac-safe (no GPU/Docker/training) measurement infra that decides
  whether the Cortex can write code, judged by the compiler. Four pieces: `verifier.py` (deterministic
  `tsc`+`bun test` ladder returning a dense `Reward` + the raw diagnostic — the teaching signal),
  `breaker.py` (repair-transition factory: breaks working TS in known ways, keeps only truly-RED
  mutations, emits `(broken, diagnostic) → gold` in VERA-R sequence shape), `passk.py` (pass@k vs the
  compiler, model-agnostic), `cortex_generator.py` (the 14M FIM Cortex as a generator).
- **Measured:** `test_repair.py` **13/13 green** — oracle accepts truth / rejects garbage (fab=0) /
  dense score orders broken<partial<green; breaker makes real breaks with captured diagnostics
  (TS2322, TS2304) whose gold fixes round-trip; pass@k detects success (gold=1.0) and failure
  (noise=0.0) and is monotonic in k. **First real Cortex pass@k vs the compiler:** metis-fim-14M
  pass@1/4/8 = 0.0/0.0/0.0, but `mean_best_score` rises 0.0→0.067→0.133 with k. gold ref pass@1=1.0.
- **Decisions:** the compiler is the only judge — the model never self-grades (Aletheia lesson,
  doc 14 §0). Verifier is a local no-Docker `tsc`/`bun` ladder now, interface-compatible with the
  Phase-5 sandbox for later. Dense score (parse .2 / typecheck .3 / tests .5) gives search a gradient,
  not a flat pass/fail (Agents-A1 step-level reward, doc 16 §5).
- **Surprises:** the 14M floor is not flat — best-score climbing with k means the toy already emits
  marginally-closer candidates; the ruler is sensitive enough to see it.
- **Verdict:** **go.** The project now has an honest, tested ruler. "Better Cortex" is henceforth a
  number on this harness, not a vibe.
- **Next:** scale `breaker.py` into a git-history repair miner; train a stronger Cortex on the
  transitions; re-run `cortex_generator.py` and watch pass@k leave zero. Only then does the doc-14
  flywheel (pass@1-vs-round) have a generator worth amplifying.

### 2026-07-07 — First repair-training experiment: the ruler catches memorization

- **Built:** `miner.py` (scales the breaker to 150 verified type-error transitions over disjoint
  synthetic functions, real `tsc` diagnostics), `train_repair.py` (warm-start from the night2 FIM
  14M, Muon, `<edit>`-span 3× weighted, gentle: 400 steps / 2.0 min / `nice`), `repair_generator.py`
  (inference in the trained `<state><diagnostic><edit>` format). calc.ts held out (never mined).
- **Measured (pass@k on HELD-OUT calc.ts, vs untrained baseline 0.0 → best-score 0.13):**
  metis-repair-14M pass@1/4/8 = **0.0**, but **mean_best_score 0.13 → 0.333**. Dump of the actual
  candidates decomposes it exactly: `wrong_arith_op` (a test-failure mutation *never in training*) →
  0.0 garbage; `wrong_return_type` & `undefined_symbol` → **0.5** each — syntactically/type-valid TS,
  but the model **emits a memorized training function (`scaleUp`, `a+b`)** instead of repairing the
  held-out `add`/`scale`. mean = (0+0.5+0.5)/3 = 0.333.
- **Decisions/diagnosis:** training on repair-transition data *does* move the model (0.13→0.33) — the
  data shape is right (VERA-R §2 confirmed). But 14M + 150 synthetic examples **overfits to
  memorization**, not repair skill: it produces valid TS copied from training, not a fix for the
  input. pass@1=0 is honest and correct. The failure is capacity+data, not the optimizer.
- **Surprises:** the ruler is precise enough to separate "garbage" (0.0), "valid-but-wrong" (0.5),
  and would flag "green" (1.0) — it diagnosed the exact failure mode (memorization) from the score
  alone, then confirmed by candidate dump. This is the measurement discipline the project needed.
- **Verdict:** **partial go.** Direction confirmed (repair data helps, measurably); the tiny-model +
  tiny-synthetic-data combo memorizes. Need real diverse data + more capacity to cross to green.
- **Next:** replace synthetic mutations with a **git-history repair miner** (thousands of real
  (pre-fix, diagnostic, patch) triples across many repos) so the model must learn the repair MAP, not
  a handful of function bodies; only then is pass@1 off zero a real generalization result.

### 2026-07-07 — Repair experiments 2-3: a measured capacity verdict (14M is too weak, honestly)

- **Built:** `extract.py` (mine self-contained real TS functions from ~/projects — only ~19 typecheck
  in isolation), `synth.py` (280 procedurally-diverse typed functions, tsc-verified), `miner_real.py`
  (565 verified repair transitions over 299 distinct functions), `edit_repair.py` (edit-native output:
  the model emits ONE corrected line, spliced at the diagnostic's line number — VERA-R §4).
- **Measured (pass@1 on HELD-OUT calc.ts, three formats, all warm-started 14M, all gentle ~2-3 min):**
  1. whole-file / 4 templates / 150 ex → pass@1 **0.0**, best 0.333 (memorizes bodies).
  2. whole-file / 299 diverse fns / 565 ex → pass@1 **0.0**, best 0.333 (candidate dump: blends input
     name with a training fragment, e.g. `scalestatPelUsd` — valid TS, wrong function).
  3. edit-native / 1 line / 565 ex → pass@1 **0.0**, best 0.067 (candidate dump: for return-type
     errors the compiler flags the `return` line but the fix is on the *signature* line, so splicing
     the flagged line can't fix it; generated lines are also weak).
- **Verdict — capacity, measured not assumed:** across three output contracts and 4→299 function
  diversity, a 14M **byte-level** Cortex at ~2-3 min training cannot repair even trivial type errors
  on held-out functions. The ruler is sensitive (0.0 garbage / 0.333 valid-but-wrong / would show
  1.0 green) and consistently reports the same floor. This is **not** an optimizer or data-diversity
  problem — both were varied and didn't move pass@1. It is model capacity + byte-level tokenization
  (exact copy-with-edit is brutal byte-by-byte) + tiny training budget.
- **Decision:** stop Mac-gentle iteration on the 14M byte model — it has told us what it can't do,
  clearly. The levers that remain are exactly docs 12/16's real-run prescription, none of them a Mac
  tweak: **(a) BPE tokenization** (byte-level makes `const ` 6 tokens to copy perfectly vs 1),
  **(b) real capacity** (~0.3B), **(c) real training budget** (the cloud/nights run). The edit-native
  insight stands and should carry to the real run, but it needs the diagnostic→*fix-line* mapping, not
  the flagged line.
- **Next:** the real run is now well-specified and de-risked by these negatives: BPE + ~0.3B +
  repair-transition data at scale, measured on this exact harness. pass@1 leaving zero there is a real
  result; here it honestly cannot.
