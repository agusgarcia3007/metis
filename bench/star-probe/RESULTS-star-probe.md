# STaR coverage probe — the decisive measurement for local teacher-free self-training

**Question.** Can a model that fits a 4 GB / 4-core VPS (qwen3:1.7b) self-improve on *this* codebase
with no teacher — using `cargo test` as the only (free, local) verifier? STaR/rejection-sampling
trains on *passing* samples, so the whole approach is dead-on-arrival if the base can't generate a
single passing candidate. That's the floor we measured first, before building any LoRA pipeline.

**Method.** For 4 pure functions in the metis repo (`parse_verdict`, `evidence_text`,
`split_sentences`, `parse_unary`): blank the body, give the model the full file (tests visible),
ask it to reimplement, splice the candidate back, run the real `cargo test` for that module.
Thinking disabled, `num_predict=1200`. Repo restored via `git checkout` after every candidate.

## Result (qwen3:1.7b)

| metric | result |
|---|---:|
| pass@1 | **0 / 4** |
| pass@5 | **0 / 4** (0/20 samples passed) |
| pass with 2 rounds of execution-feedback **repair** | **0 / 4** |

**The failures are near-misses, not garbage** — every candidate compiled and was structurally right:
- `evidence_text`: failed by a single missing `.trim()` (`"[1]   alpha  \n"` vs `"[1] alpha\n"`).
- `parse_verdict`: used `strip_prefix` (exact) instead of `contains` (substring); missed `NOT SUPPORTED`.

**The killer detail.** Handed the exact failing assertion (`left` vs `right`), its own code, AND an
explicit natural-language hint ("the output has extra spaces the expected output does not, fix it"),
the model returned **byte-identical broken code** twice. It cannot use the verifier's signal even
spoon-fed. This is a deeper floor than "needs more samples."

## Verdict for the thesis

- **The competence floor (research 11/12, wall #1) is binding** for qwen3:1.7b on this Rust repo:
  pass@k ≈ 0 → **zero seed data** → STaR/rejection-sampling has nothing to bootstrap from.
  **Do not build the phase-2 LoRA loop on this base/language — it would train on an empty set.**
- This is the cheap kill-signal working as designed: a few minutes of measurement instead of building
  a whole CPU QLoRA pipeline for nothing.

## Caveats (so the negative isn't overclaimed)

1. **Rust is hard-mode** for a tiny model; qwen3:1.7b is far stronger at Python. A Python manifold
   would very likely show nonzero pass@k. "0/4 on Rust" ≠ "0 on all code".
2. **1.7b is the floor model.** qwen3:4b (what Railway runs, ~2.5 GB) and code-specialized small
   models (Qwen2.5-Coder-3B) are dramatically stronger per-param at exactly this.
3. Small sample (4 functions) — directional, not statistical.

## The corrective insight / next probe

The bottleneck is the **base model's competence floor**, not the loop. The thesis *requires* starting
above the floor (search selects, it cannot conjure). Next cheap measurement: re-run this exact probe
with **qwen3:4b** and a **code-specialized 3B** to locate where pass@k crosses zero. That point is the
real minimum viable base for the local self-training bet.

Scripts: `star_probe.py` (pass@k), `repair_probe.py` (execution-feedback repair). Run:
`PROBE_MODEL=qwen3:4b python3 star_probe.py`.
