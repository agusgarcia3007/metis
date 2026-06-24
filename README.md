<h1 align="center">Μῆτις · Metis</h1>
<p align="center"><b>Frontier-grade intelligence that fits where frontier models can't.</b></p>
<p align="center"><i>Win by cunning, not by size.</i></p>

---

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

go run ./cmd/metis index ./docs      # turn your files into swappable knowledge
go run ./cmd/metis chat              # grounded, tool-using, fully local
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

## The honest frontier (mission vs. proven)

I will not pretend the war is won. Here is exactly where we stand:

- ✅ **The system is real and local.** Cortex + Library (RAG with citations) + Hands (tools), working.
- ✅ **The thesis mechanism is proven in miniature.** A from-scratch, gradient-checked transformer
  (`internal/nano`) shows knowledge-in-context generalizes to unseen facts while knowledge-in-weights
  does not, and quantifies the capacity wall (see [`docs/design/04`](docs/design/04-RNT-results-log.md)).
- ⏳ **Not yet proven:** real numbers (tokens/sec, RAM, quality) on an actual **4 GB / 4 vCPU** box.
  *That benchmark is the next milestone and the claim's real proof.*
- ⏳ **V1 limits:** the Cortex is an off-the-shelf small model (Qwen3-1.7B); retrieval is flat cosine
  (great for moderate corpora, needs a disk-ANN index to scale to web-scale knowledge); small-model
  tool-use needs careful prompting. None are dead-ends; all are on the roadmap.

The genuinely hard research bet — training a model that *refuses to memorize* so it spends all its
capacity on reasoning (`docs/design/03-training-system-RNT.md`) — hit a real boundary in-session and
needs GPU-scale training. The shipping product uses the **pragmatic, working version of the same
thesis**: retrieval. That's the honest path from "bold idea" to "useful today."

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
chat without Docker: `go run ./cmd/metis chat`.

## Architecture & docs

- **Design** — [`docs/design/`](docs/design/): the constraints, the architecture, the build plan, and
  the training-system research log.
- **Research** — [`docs/research/`](docs/research/): six deep, sourced notes (small models, quantization,
  MoE/offloading, retrieval/tools, engine/language, distillation/cost).
- **Code** — `internal/kernel` (Cortex backend), `internal/library` (knowledge plane),
  `internal/hands` (tools), `internal/nano` (from-scratch transformer + the RNT experiments),
  `cmd/metis` (the CLI).

```sh
go test ./...          # engine gradient-check, library, tools, RNT mechanism — all green
go run ./cmd/metis chat
```

---

<p align="center"><i>A small mind, made vast by the knowledge you give it — and yours alone.</i></p>
