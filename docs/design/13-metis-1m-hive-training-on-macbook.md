# metis-1m — the Hive Cortex: an architecture that trains on a MacBook, by design

> Doc 12 specified metis-1 (0.7B, ~8.4e20 FLOPs) for a rented 8×H100 node. This doc answers a harder
> question: **what architecture would let us train the Cortex on the machine we already own — an
> M3 Pro, 14 GPU cores, 18 GB unified — quickly?** Brute force is out by four orders of magnitude
> (~6e16 FLOPs per 8-hour night ≈ 40 years for doc-12 metis-1). So the answer cannot be "a smaller
> monolith." It has to be a different *kind* of model.
>
> The invention: stop training a model. **Grow an organism.** A frozen shared trunk + sparse
> product-key memory + tiny specialist modules, fed every night by an active-learning curriculum
> that the deterministic verifier writes during the day. Training stops being a datacenter event
> and becomes a **continuous nightly metabolism** on hardware you own.

---

## 1. The hardware, measured honestly

| Resource | M3 Pro (this machine) | Implication |
|---|---|---|
| GPU fp16 peak | ~7 TFLOPS (14 cores) | ~2e12 FLOPs/s effective at 25–30% training MFU on MLX |
| One 8h night | **~6e16 FLOPs** | trains ~6*N*D → a 30M model on ~330M tokens, per night |
| Unified RAM | 18 GB | model+Adam+activations for ≤150M dense params comfortably; but **RAM is 100× cheaper than FLOPs here** — the architecture should spend memory, not compute |
| Always available | every night, forever | favors *continual* training over one-shot runs |

The asymmetry in row 3 is the design key: Apple Silicon is **FLOP-poor but memory-rich and
bandwidth-decent**. The winning architecture buys capability with cheap memory lookups and sells
expensive matmuls.

## 2. Hidden assumptions of "training a model" — and their inversions

1. *One monolithic network learns everything* → **a federation of tiny modules, each learns one skill.**
2. *All parameters participate in every forward/backward* → **sparse memory layers: touch top-k slots, update top-k slots.**
3. *Capacity requires FLOPs* → **capacity in lookup tables costs RAM, not matmuls.**
4. *Training is one run, then you ship* → **training is a nightly loop that never ends; the model is always mid-metabolism.**
5. *The curriculum is fixed before training* → **today's verified failures are tonight's curriculum.**
6. *Data is tokens from a corpus* → **data is (failure, verified fix) pairs the system itself harvests in the sandbox.**
7. *You need the whole 200B-token distribution* → **you need the ~1% of it the current organism actually gets wrong.**
8. *Later capabilities require retraining earlier ones* → **frozen trunk: new skills are new modules; old skills are physically untouched (no catastrophic forgetting, by construction).**

## 3. Candidates considered (scored, then composed)

| # | Candidate | Mechanism | Why alone it's not enough |
|---|---|---|---|
| C1 | **Hive of specialists** | frozen shared trunk (~40M) + tiny per-skill modules (~10–25M), Conductor routes | composition without a cheap-capacity substrate still caps skill depth |
| C2 | **Product-key memory (PKM) layers** | replace MLPs with sparse key-value tables: forward touches top-k of ~1M slots; backward updates only those rows | capacity w/o curriculum still wastes nights on already-known tokens |
| C3 | **Verifier-driven active learning** | train ONLY on sequences the current model fails in the sandbox; compile∧typecheck∧tests label them for free | needs C1/C2 to have something cheap to update nightly |
| C4 | **Logit distillation from a teacher** | one-time cloud pass: teacher logits cached on the curated corpus; local training gets dense soft targets (~2–5× data efficiency) | a multiplier, not an architecture |
| C5 | Ternary/BitNet QAT | {-1,0,+1} weights | saves RAM we don't lack; FLOP savings need custom kernels — keep as deploy-time option |
| C6 | Hypernetwork weight genesis | generate specialist weights from descriptions | too speculative for the critical path; revisit after C1 works |

**metis-1m = C1 + C2 + C3 + C4.** They compose because they attack orthogonal walls: FLOPs (C2),
tokens (C3+C4), and forgetting/scale (C1).

## 4. The Hive Cortex architecture

```
              ┌─ TRUNK (frozen after week 1) ──────────────────────────┐
tokens ──►    │ embeddings + 8 layers, d=512  (~40M params, bf16)      │
              └──────────────┬─────────────────────────────────────────┘
                             │ shared representation
        ┌────────────────────┼──────────────────────┬───────────────────┐
        ▼                    ▼                      ▼                   ▼
  MODULE: edit-TS      MODULE: test-read      MODULE: tool-call    MODULE: (next…)
  4 layers d=512       4 layers d=512         2 layers d=512       one new module
  + PKM (1M slots,     + PKM                  + PKM                per skill,
  top-32, ~0.5GB RAM)  …                      …                    one/few nights each
        │                    │                      │
        └────────────────────┴──────────┬───────────┘
                                        ▼
                     CONDUCTOR routes by task type; every output
                     gated by the deterministic verifier (doc 11)
```

- **Trunk (~40M):** generic code reading. Trained ONCE on ~2–3B RNT-shaped, scorer-selected TS
  tokens with C4 distilled logits ≈ 1.2e18 FLOPs ≈ **5–7 nights**. Then frozen forever.
- **PKM layers:** each module's "knowledge of patterns" lives in a product-key memory — ~1M slots,
  top-32 lookup. Forward cost: two small key matmuls + 32 row reads (~1000× fewer FLOPs than a dense
  MLP of equal capacity). Backward: **only 32 rows get gradients** — sparse Adam on rows. This is
  what converts 18 GB of unified RAM into model capacity that the 14-core GPU can afford.
- **Specialist modules (~10–25M each):** small transformer heads on the frozen trunk, one skill
  each, matching the edit-native action space (emit diff · read test failure · call tool · select
  from `<lib>`). Each trains in **1–2 nights** on its active-learning slice.
- **The nightly metabolism (C3):** by day, the system runs tasks (benchmarks, your real OpenCode
  usage); the sandbox verifier labels every failure *for free*. By night, the failures — and only
  the failures — plus teacher logits for them, become the training set. Mastery-based: solved
  patterns retire from the curriculum. The organism only ever studies what it got wrong.
- **RNT invariants carried over (docs 03/04):** retrieval-native sequences, dense supervision
  (Round-4 rule), anti-memorization decay on whatever dense MLPs remain.

**What "training metis-1m" costs:** trunk ≈ one week of nights, once. Each specialist ≈ 1–2
nights. A working 4-module hive ≈ **~2 weeks of nights on this MacBook**, then it *keeps improving
nightly forever* — which a one-shot datacenter run structurally cannot do.

## 5. Prototype spec (MLX, this repo)

- `train-m/trunk/` — MLX trainer: 8L×512d GQA decoder, bf16, seq-pack 4k, sparse-Adam-ready.
- `train-m/pkm/` — product-key memory layer for MLX (two half-key matmuls + gather/scatter-add;
  the only custom op needed).
- `train-m/harvest/` — the day-side: wrap the Phase-5 sandbox so every verified failure is appended
  to `curriculum/night-N.jsonl` with its retrieved `<lib>` context.
- `train-m/distill/` — one-time teacher-logit cache over the curated corpus (top-8 logits/token,
  int8 — fits on disk).
- Benchmark: the 20-task H2 set + held-out APIs (doc 12 rules: eval sets frozen, never trained on).

## 6. Experiment plan — pre-registered numbers

- **Night 0 (calibration, ~3h):** train a 15M trunk-let on 150M tokens; measure real tok/s and MFU
  on MLX. Every budget above is recomputed from this measured number. *(Kill nothing yet — this is
  the ruler.)*
- **Nights 1–7 (trunk):** train the 40M trunk. **Gate:** beats n-gram/tiny baselines on held-out-API
  completion given `<lib>`; RNT effect (retrieval > no-retrieval arms) reproduces at 40M.
  **Kill if** RNT effect doesn't show — same gate as doc 12 M1.1, and it transfers there.
- **Nights 8–9 (first specialist + PKM):** edit-TS module. **Gate:** PKM module ≥ dense-MLP module
  of equal FLOPs by a clear margin (this is C2's whole claim). **Kill PKM if not** — fall back to
  dense 25M specialists (hive still stands).
- **Nights 10–14 (metabolism):** run the day/night loop on the H2 set. **Gate:** pass@1 rises
  week-over-week with a *flat* nightly FLOP budget — capability bought with selection, not compute.
  **Kill the active-learning arm if** the curve is flat vs. random-sample training.
- **Doubling-down criterion:** if the 4-module hive on this MacBook matches doc-12's projected
  125M-dense-nano quality on the TS surface, the hive replaces the monolith as metis-1's actual
  architecture — and the H100 budget in doc 12 gets spent on a *bigger hive*, not a bigger blob.

## 7. Research honesty

- **Known:** PKM layers work at scale (Lample et al.; Meta's memory layers, 2024); active/selected
  training beats uniform (Rho-1 et al.); distillation multiplies small-model data efficiency; the
  RNT mechanism and dense-supervision rule are proven in-repo (docs 03/04); MLX trains ≤150M models
  on this RAM comfortably.
- **Unknown:** PKM training stability on MLX (custom op, night-8 gate); whether frozen-trunk
  specialists compose without interference at this scale (the Conductor routing hides some of this
  risk, but it's the biggest one); whether the nightly failure-harvest yields enough volume early
  on (cold-start may need synthetic drills from doc 12's data factory).
- **Speculation:** that the hive matches a dense 125M monolith at equal total FLOPs — that is the
  experiment, not a promise.
- **Testable tomorrow:** Night 0 costs three hours and zero dollars, and every number in this doc
  gets re-derived from its output.
