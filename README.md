<h1 align="center">Μῆτις · Metis <sub>(Rust)</sub></h1>
<p align="center"><b>Frontier-grade intelligence that fits where frontier models can't.</b></p>
<p align="center"><i>Win by cunning, not by size.</i></p>

---

> **About this directory.** `metis-0` is a faithful, from-scratch **Rust** port of the original Go
> `tiny-llm` project — same architecture, same algorithms, same results. The autograd engine, the
> RNT experiments, the RAG Library, the tools, and the CLI are all reimplemented in safe Rust. The
> `docs/` and `data/` directories are copied verbatim from the original.

In Greek myth, **Metis** is the Titaness of *practical wisdom and cunning intelligence* — the wise
counsel even Zeus sought. Not raw power. Cleverness.

That is the whole bet of this project. Today's AI race is an arms race of size: bigger models, bigger
clusters, bigger bills, run in someone else's data center. **Metis takes the opposite bet** — a small
reasoning core that wins through *wisdom* (reasoning), *counsel* (retrieved knowledge), and *craft*
(tools), running **entirely on hardware you already own**.

## Why this is a potential game-changer

Frontier models today weld three things into one giant, cloud-bound blob of weights: a reasoning
engine, an encyclopedia, and a calculator. That design forces three costs onto the world — **money,
privacy, and access**. Metis **unwelds them**:

| What a frontier model fuses | Metis splits out | Where it lives |
|---|---|---|
| Reasoning circuits | **Cortex** — a small reasoner | RAM (~1–2 GB) |
| Memorized facts | **Library** — retrieval over a disk corpus | a file on disk |
| Exact compute / live data | **Hands** — tools (calc, clock, …) | local subprocesses |
| Planning / inner monologue | **Conductor** — the agentic loop | the program itself |

→ **Cortex · Library · Hands · Conductor.** A small brain whose *knowledge is data you can swap*.

If this thesis fully lands, the implications are global:

- **Privacy & sovereignty.** Your data, your model, your machine. Nothing leaves the device. Hospitals,
  courts, governments, and individuals get capable AI **without shipping their secrets to a cloud.**
- **Access & cost.** Useful AI on a **$5/month VPS, a laptop, or an offline edge box** — not a
  $40k GPU node. That puts frontier-*useful* assistance in reach of the 90% of the world priced out
  of cloud AI.
- **Auditable, updatable knowledge.** The "brain" is fixed and small; what it *knows* is a file you
  can read, version, swap, and trust. Update the world's knowledge **without retraining anything.**
- **Resilience.** No internet, no API key, no rate limit, no vendor that can deprecate you. It just runs.

Knowledge-as-data, run locally, is how you democratize frontier intelligence. That's the world-change.

## Why it can work (grounded in the research, not hype)

This isn't a vibe — it's built on published results (see [`docs/research/`](docs/research/), every
claim sourced):

- **Most of a big model's parameters memorize facts, not reasoning** (~2 bits/param; facts live in
  the MLP layers). Move that knowledge to disk and the model can be tiny. *(research 04)*
- **Retrieval collapses a >25× parameter gap**: RETRO-7.5B ≈ GPT-3-175B; Atlas-11B beats PaLM-540B on
  knowledge tasks. A small reasoner + retrieval rivals a giant. *(research 04)*
- **Reasoning distills into tiny models**: a 1.5B distilled reasoner beats GPT-4o on MATH-500. *(research 06)*
- **Tools let small models punch up**: a 1.1B tool-user matched GPT-4-Turbo on agentic tasks. *(research 04)*

## What works **today** (honest status)

A real, runnable V1 — 100% local, no GPU:

```sh
ollama serve &                       # local inference engine (bundles ggml)
ollama pull qwen3:4b                 # Cortex (~2.5 GB, ~GPT-4o-mini-class reasoning, fits 4 GB)
ollama pull all-minilm               # the Library's embedder (~45 MB)

cargo build --release                # build the metis + rnt binaries
./target/release/metis index ./docs  # turn your files into swappable knowledge
./target/release/metis chat          # grounded, tool-using, fully local
```

**Knowledge-as-data, demonstrated** — a fact the model cannot have trained on, answered from the index:

```
$ metis index sample-docs
Library built: 1 chunks, dim=384 -> library/index.gob

$ metis ask "What does the Zephyrian Protocol mandate about memory, and what's its mascot?"
The Zephyrian Protocol caps resident memory at 1.84 GB [1]. Its mascot is a blue heron named Pippa.
sources: [1] zephyr.md (0.32)
```

**Tools, where the weights would fail:**

```
you> What is 84937 × 2261, divided by 7?
  [tool] calc(84937*2261) = 192042557
metis> 192042557 ÷ 7 = 27434651.
```

Multi-turn memory, a `/think` toggle (model reasoning), relevance-gated citations (no spurious
sources), and native tool-calling. Swap the `library/` index → swap the assistant's entire knowledge,
no retraining. Override the Cortex with `METIS_MODEL=...`.

**Generate · Verify · Search (GVS) — reliability from a tiny model.** The Conductor
(`src/conductor.rs`) never trusts a single generation: it generates a grounded answer, **verifies** it
against the retrieved evidence with the same model in judge mode, **searches** a few diverse
candidates if it fails, and **abstains** rather than emit an unverified claim. No datasets, no
retraining — verifying is cheaper than generating, and a 1.7B is a reliable grounded *judge* even when
it's a shaky generator.

**The web as a verified Library — open-domain, still grounded.** Point Metis at a self-hosted SearXNG
(`METIS_SEARCH_URL=...`) and the live web becomes "a Library too big to store": results flow through
the *same* ground→verify→cite→abstain loop, so it answers questions far outside its local corpus
**with citations**, and abstains when even the web doesn't support it. The binary stays TLS-free —
SearXNG does the HTTPS.

### Benchmark — same weights, architecture is the only difference

`qwen3:1.7b` called bare (via ollama) vs the *same* model inside Metis, on a private corpus it never
trained on (harness: `bench/benchmark.py`, full write-up: [`bench/RESULTS.md`](bench/RESULTS.md)):

| metric | BARE | METIS |
|---|---:|---:|
| answerable facts correct | **0 / 8** | **8 / 8** |
| fabrications on unanswerable (lower=better) | **4 / 4** | **0 / 4** |
| general (incl. exact math via the calc tool) | 1 / 2 | **2 / 2** |

The bare model invents facts and confidently fabricates answers that don't exist; the same model
inside Metis answers grounded, cites, and refuses to guess. Reproduced identically on the live Railway
deploy. The full engineering record — every production fix (KV segfault, CPU thread-thrash that tanked
decode to 0.09 tok/s, the AMX segfault) and the architecture decisions — is in
[`docs/design/07-implementation-and-deployment-log.md`](docs/design/07-implementation-and-deployment-log.md).

## The proven core: Retrieval-Native Training (RNT)

The thesis mechanism is proven in miniature by a from-scratch, gradient-checked transformer
(`src/nano`): knowledge-in-context generalizes to unseen facts while knowledge-in-weights does not.

```sh
cargo run --release --bin rnt              # vanilla vs RNT on a new world (the headline result)
cargo run --release --bin rnt -- -mode query   # train, save, reload, reason over unseen facts
cargo run --release --bin rnt -- -mode sweep   # the capacity wall: memorization degrades, RNT stays flat
```

The decisive result, reproduced by this Rust port:

```
VANILLA  accuracy on TRAINED world  : 100.0%   (memorized — works)
VANILLA  accuracy on NEW world      :  10.0%   (knowledge frozen in weights — fails ~chance)
RNT      accuracy on NEW world      : 100.0%   (reads retrieved fact — generalizes)
```

Other `-mode` values mirror the Go original: `retrieval`, `improve`, `assoc`, `probe`, `curriculum`,
`final`, `level2`, `induction`, `recall`.

## Deploy on a 4 GB / 4 vCPU VPS (Docker)

One command brings up the Cortex (ollama) + Metis (HTTP API), tuned to the budget:

```sh
docker compose up -d --build                      # first run pulls ~1.5 GB of models, then serves
docker compose run --rm metis index /app/docs     # build the Library from ./sample-docs
docker compose restart metis                       # serve picks up the new knowledge

curl -s localhost:8080/ask -d '{"q":"What does the Zephyrian Protocol mandate?"}'
# {"answer":"... caps resident memory at 1.84 GB [1] ...","sources":[{"n":1,"source":"zephyr.md","score":0.32}]}
```

Endpoints: `GET /healthz`, `GET /readyz`, `POST /ask {"q":"...","think":false}`. Put your own
`.md`/`.txt` files in `./sample-docs`. `mem_limit`s keep ollama at 3 GB and Metis at ~0.4 GB. Local
chat without Docker: `./target/release/metis chat`.

## Architecture & docs

- **Design** — [`docs/design/`](docs/design/): the constraints, the architecture, the build plan, the
  GVS thesis, and the [implementation & deployment **log (the bitácora)**](docs/design/07-implementation-and-deployment-log.md) —
  what got built, what we measured, and every production fix.
- **Benchmark** — [`bench/`](bench/): the bare-vs-Metis harness (`benchmark.py`), the
  [results write-up](bench/RESULTS.md), and the raw numbers (`results-*.json`).
- **Research** — [`docs/research/`](docs/research/): twelve deep, sourced notes.
- **Code** — `src/kernel` (Cortex backend), `src/library` (knowledge plane), `src/hands` (tools:
  calc, clock, **web/SearXNG**), `src/conductor.rs` (the **GVS** loop), `src/nano` (from-scratch
  transformer + the RNT experiments), `src/bin/metis.rs` (the CLI), `src/bin/rnt.rs` (RNT runner).
- **Deploy** — single-container Railway image (`Dockerfile.railway`) + SearXNG sidecar (`searxng/`);
  see the log for the topology and the hard-won CPU-stability env.

```sh
cargo test                       # engine gradient-check, library, tools — all green
cargo test --release -- --ignored   # the slower RNT/induction training tests (~1 min)
cargo run --release --bin metis -- chat
```

## Notes on the Rust port

- The autograd engine (`src/nano`) models the Go pointer graph with `Rc<RefCell<Tensor>>` handles and
  boxed backward closures recorded on a `Tape`. It is numerically identical and fully deterministic.
- Where the Go engine parallelized matmuls across goroutines, this port computes those loops
  sequentially — each output element's reduction is independent of worker count, so results match
  exactly; only throughput differs.
- Model/index persistence uses `bincode` (the closest Rust analog to Go's `gob`); file paths are kept
  identical for CLI parity.

---

<p align="center"><i>A small mind, made vast by the knowledge you give it — and yours alone.</i></p>
