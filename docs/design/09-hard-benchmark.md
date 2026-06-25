# Metis — Phase 3: the hard benchmark, and what broke

> Phase 2 measured "0.6B = 1.7B" on 14 mostly-extractive questions over one document. The honest
> next step was to widen and harden the benchmark until the tie either survived or broke. It broke —
> and on the way it exposed an architecture bug that mattered more than model size.

---

## 1. The benchmark

A new corpus of **4 fictional, internally-consistent documents** (`bench/corpus/`): a technical
spec, a governance doc, a changelog, and an unrelated biology doc (to force retrieval to
discriminate across domains). All facts are invented, so neither model could have memorized them.

**42 questions** (`bench/questions.json`) graded into five difficulty tiers. The key design choice:
separate the *model floor* from the *system floor*.

| tier | what it tests | floor it probes |
|---|---|---|
| 1 — extractive | answer is a span in one chunk | retrieval + copy |
| 2 — synthesis | reason over facts co-retrievable in one doc (compare, add, pick) | **model reasoning** |
| 3 — multihop | chain facts across documents (codename→component→working-group→chair) | **system: iterative retrieval + model** |
| 4 — unanswerable | plausible, on-topic, absent | abstention discipline |
| 5 — general | model-unaided + tool (math) | no regression |

Tier 2 is reasoning over evidence the system *can* gather in one shot, so it isolates the model.
Tier 3 needs the system to gather a chain it cannot get in one retrieval, so a failure there is
ambiguous (model or retrieval). Keeping them separate is what makes the result interpretable.

---

## 2. Results — four cells (E0=1.7B, E1=0.6B; fast-path on vs off)

Same weights, same corpus, same hardware. `METIS_EXTRACT_GATE=2.0` disables the extractive
fast-path (every query goes through Generate·Verify·Search).

| tier (max) | E0 fp-ON | E1 fp-ON | **E0 fp-OFF** | **E1 fp-OFF** |
|---|---:|---:|---:|---:|
| 1 extractive (12) | 10 | 10 | **12** | 10 |
| 2 synthesis (10) | 3 | 2 | **9** | 7 |
| 3 multihop (8) | 6 | 3 | **7** | 3 |
| 4 unanswerable (8) | 6 | 5 | **8** | 5 |
| 5 general (4) | 4 | 3 | **4** | 3 |
| **answerable /30** | 19 | 15 | **28** | 20 |
| **fabrications** (↓) | 2 | 3 | **0** | 3 |

Raw data: `bench/results-hard.json` (fp-on), `bench/results-hard-nofastpath.json` (fp-off).

---

## 3. Finding A — the extractive fast-path was sabotaging quality

Turning the fast-path **off** took the 1.7B from **19→28** answerable and **2→0** fabrications.
That is a larger swing than the entire model-size difference.

Why: the fast-path fires whenever a retrieved chunk's cosine similarity to the question exceeds
0.62, and returns the raw chunk **without reasoning and without the verify gate**. Two failure modes:

1. **Synthesis questions short-circuited.** "Which has a larger budget, Aster or Quill?" scores high
   against the chunk that mentions Quill, so the fast-path returned that chunk verbatim — never doing
   the comparison. Tier 2 went 3→9 for the 1.7B once the model was allowed to actually reason.
2. **Unanswerable questions fabricated.** An absent-fact question can still score >0.62 against a
   topically-near chunk; the fast-path returned it as if it were the answer, bypassing abstention.
   This directly violates the project's core promise (abstain, don't fabricate). Tier 4 went 6→8,
   fabrications 2→0.

The fast-path trades quality for ~0.5 s of latency, and the trade is bad. The fix is not to delete
it but to **gate it on question type** — only single-fact lookups should take it; comparison,
arithmetic, and multi-hop questions must go through GVS. That is the next concrete lever.

## 4. Finding B — the model floor is real, and located

With the fast-path off (the fair comparison), the 1.7B clearly beats the 0.6B: **28 vs 20**. Phase 2's
"0.6B = 1.7B" was an artifact of an easy, mostly-extractive benchmark. But the gap is not uniform —
it is concentrated in exactly two capabilities:

- **Multi-hop chaining (tier 3): 7 vs 3.** The 0.6B cannot reliably chain facts across documents
  (codename → component → working group → chair). It abstains when the chain gets long. The 1.7B
  holds it together. This is the clearest model-size effect in the data.
- **Abstention discipline (tier 4): 8 vs 5, fabrications 0 vs 3.** The smaller model is worse at
  recognizing "this is not in the evidence" and fabricates more. Knowing when to shut up is itself a
  capability that scales with size.

Where they are **close**: extraction (12 vs 10) and single-hop synthesis (9 vs 7). So the 0.6B is
viable for lookup and simple reasoning over co-retrieved facts — it falls down on chaining and on
restraint.

---

## 5. Honest status and next levers

What we now know that we didn't before:
1. The single biggest quality lever on this surface is **not model size — it's fixing the fast-path**.
2. The model floor is real and **located**: multi-hop reasoning and abstention discipline.

Next levers, in order of expected payoff:
1. **Type-gate the fast-path** (single-fact only). Cheap; recovers most of the 19→28 swing in
   production without paying full GVS latency on every query.
2. **Iterative / multi-hop retrieval** (retrieve → read → retrieve again) to lift tier 3 for *both*
   models — this targets the system floor, and may let the 0.6B chain where it currently can't.
3. **Abstention calibration** for the small model (a stricter verify pass, or a cheap "is this in
   the evidence at all" check) to close the fabrication gap.

The benchmark is still small (42 Q, 4 docs) but it now spans difficulty tiers and has already
falsified the easy-benchmark conclusion — which is exactly what a good benchmark is for.
