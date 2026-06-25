# Metis benchmark — bare Cortex vs the Metis system

**Question:** does the Metis architecture (RAG + Generate·Verify·Search) actually make a *tiny* model
behave better, or is it just a wrapper? We hold the **weights fixed** (`qwen3:1.7b`, 4-bit) and change
*only* the architecture around them.

- **BARE** — `qwen3:1.7b` called directly via ollama. No retrieval, no verification. What `ollama run`
  gives you.
- **METIS** — the *same* `qwen3:1.7b` + the Library (retrieval) + the GVS loop (generate → verify the
  answer against the retrieved evidence → search a few diverse candidates if it fails → **abstain**
  instead of guessing).

The knowledge surface is a private document (`sample-docs/zephyr.md`) the model never trained on, so
this isolates *architecture*, not memorized trivia.

## Head-to-head (local, `qwen3:1.7b`, `METIS_SEARCH=2`)

| metric | BARE | METIS |
|---|---:|---:|
| **answerable facts correct** | 0 / 8 | **8 / 8** |
| **fabrications on unanswerable** (lower is better) | 4 / 4 | **0 / 4** |
| **general** (incl. exact math via the calc tool) | 1 / 2 | **2 / 2** |
| avg latency | ~0.4 s | ~0.6 s |

### What the bare model did
- **Invented facts** for every corpus question (ratified "2015", codename "Zephyr", an ARM Cortex-A55
  target, a "winged" mascot — all wrong).
- **Fabricated** confident answers to questions with no answer (a chairperson "John W. Lott", HQ in
  "San Francisco", Marlowe "written in Scala", certification "$150").
- Got the exact multiplication **wrong** (191,737,397 vs 192,042,557).

### What Metis did (same weights)
- Answered all 8 corpus facts correctly, with citations — partly via the ~100 ms extractive fast-path,
  partly via generate-then-verify.
- **Abstained on all 4 unanswerable questions** — zero hallucination — because the grounded verifier
  refused to certify a claim the evidence didn't support.
- Used the `calc` tool for exact math.

The takeaway: the architecture, not the weights, is what turns an unreliable 1.7B generator into a
trustworthy grounded system. **Same model. Zero datasets. Zero training.**

## Reproduce

```sh
# local: full bare-vs-Metis comparison (both reachable)
METIS_MODEL=qwen3:1.7b METIS_SEARCH=2 PORT=8080 ./target/release/metis serve &
METIS_URL=http://127.0.0.1:8080 OLLAMA_URL=http://127.0.0.1:11434 MODEL=qwen3:1.7b \
    python3 bench/benchmark.py

# against the live Railway deploy (Metis-only; bare ollama isn't exposed there)
METIS_URL=https://metis-0-production.up.railway.app MODEL=qwen3:1.7b python3 bench/benchmark.py
```

Raw per-question results: `bench/results-local.json`.

## Live deployment

Single-container Railway service (ollama + metis), `qwen3:1.7b`:
**https://metis-0-production.up.railway.app** — `POST /ask {"q":"..."}` · `GET /healthz` · `GET /readyz`.

Built from `Dockerfile.railway` + `entrypoint.railway.sh` (config in `railway.json`).

### Live benchmark (Metis-only, run against the public URL)

Same 14 questions, same result as local — the architecture behaves identically in production:

| metric | METIS @ Railway |
|---|---:|
| answerable facts correct | **8 / 8** |
| fabrications on unanswerable | **0 / 4** |
| general (incl. exact math) | **2 / 2** |
| avg latency | ~21 s (verified answers ~3–5 s; multi-candidate *search* answers longer) |

Raw: `bench/results-railway.json`.

### Two production gotchas found and fixed on Railway

1. **Quantized KV cache segfaults.** `OLLAMA_KV_CACHE_TYPE=q8_0` crashes llama-server without
   flash-attention. Removed it — default f16 KV at ctx 2048 is only ~224 MiB.
2. **Thread oversubscription destroys decode.** llama.cpp spawns one thread per *host* core, but a
   shared PaaS container's CPU *quota* is a fraction of that. Prefill (batched) tolerated it; decode
   (per-token barrier sync) collapsed to **0.09 tok/s** — a 34-token answer took 380 s. Pinning
   threads to the real allocation (`METIS_NUM_THREAD=4`, plumbed into the ollama `options`) restored
   **decode to ~36 tok/s and prefill to ~218 tok/s** — a single verified answer dropped from ~408 s to
   ~3.4 s. This is the difference between "unusable" and "responsive" on a small box, and it's the kind
   of control the architecture is built to exploit.
