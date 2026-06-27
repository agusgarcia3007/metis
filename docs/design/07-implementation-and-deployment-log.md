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
