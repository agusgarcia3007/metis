# Metis — Phase 5: code specialization, where the verifier is a compiler

> Phase 4 (`10-reasoning-compiler-h1.md`) asked whether reasoning depth can be moved OUT of the
> weights the way RAG moved knowledge out. The naive thesis died on its kill criterion — but it died
> for a specific reason: the verifier was an **LLM-judge**, and a 0.6B judge waves through plausible
> fabrications (fab% up to 62 at g1). The failure *located the lever*: the reasoning-compiler works
> only as well as its verifier is trustworthy.
>
> This phase takes the lever to the one domain where the verifier is not a noisy LLM but a
> **deterministic oracle**: code. A compiler and a test suite are near-perfect verifiers — they do not
> hallucinate SUPPORTED. If `verify < generate` ever holds, it holds hardest here. Phase 5 turns Metis
> into a **code agent that operates OpenCode**, and uses code as the clean testbed for the whole thesis.

---

## 1. Why code is the ideal domain for the Metis thesis

Everything Metis already splits out (Cortex · Library · Hands · Conductor) maps onto software
engineering *better* than onto open-domain QA, because each externalized plane has a **deterministic,
free, ground-truth oracle** in code that QA never had:

| Metis plane | In QA (Phases 1–4) | In code (Phase 5) | Why it's stronger |
|---|---|---|---|
| **Cortex** (reasoning) | small generator, shaky judge | proposes an *edit*, not prose | the action space is tiny and checkable |
| **Library** (knowledge) | RAG over docs, LLM-judged relevance | API signatures, types, examples on disk | retrieval is grounded by the type system |
| **Hands** (tools) | calc, clock, web | **compiler · type-checker · test runner · linter** | tools return *ground truth*, not estimates |
| **Conductor** (GVS) | verify with an LLM-judge (**noisy** — see H1) | verify with `compile ∧ typecheck ∧ tests` (**exact**) | the verifier cannot fabricate SUPPORTED |

**The reframed bet (falsifiable):** the Phase-4 asymmetry `verify < generate` failed with an LLM
verifier but should *hold* with a compiler+tests verifier, because `fab% → ~0`. If a deterministic
oracle gates every step, an external engine can compose depth that a tiny generator could never chain
alone — **on code specifically, before any claim about general reasoning.**

This does not require pretraining from scratch (research 06, build-plan philosophy: *stand on giants*).
We start from the best small open coder base, build the *system*, and specialize by cheap distillation
+ RLVR last.

---

## 2. What we externalize out of the weights (the code-specific split)

The general thesis is "weights hold only reasoning." For code we can be far more aggressive, because
each of these has a machine that already knows the answer:

| Externalization | Removes from weights | Mechanism | Builds on |
|---|---|---|---|
| **X1 — Verifier oracle** | "does this work?" | `compile ∧ typecheck ∧ tests` as the GVS verifier | doc 06 (GVS), doc 11 §4 |
| **X2 — Edit-native action space** | boilerplate, spelling APIs | Cortex emits **AST edits**, not free text | new: `src/hands/edit` |
| **X3 — Type-state decoding** | syntax + type bookkeeping | constrained decoding masked by grammar + in-scope types | doc 02 (structured decoding), extend |
| **X4 — Code Library (RNT)** | API/library knowledge | retrieval-native over signatures/docs/examples | doc 03 (RNT), doc 04 (RNT results) |
| **X5 — Trajectory distillation** | agentic behavior | imitate a frontier teacher's *verified* tool-use traces | doc 06 distill, research 06 |

The load-bearing claim of the whole phase: **with X1–X4 in place, the parameter count needed to hit a
target on a code benchmark collapses.** That claim is measured directly in §6 (the ablation curve),
and it is the scientific output of the phase — positive or negative.

---

## 3. Architecture — the Conductor becomes a code agent

The existing Generate·Verify·Search loop (`src/conductor.rs`) generalizes almost unchanged; only the
*candidate* and the *verifier* change type.

```
                        ┌── OpenCode session (task: fix issue / add feature) ──┐
                        │                                                      │
  read repo context ──► retrieve (Code Library, X4) ──► propose EDIT (Cortex, X2)
        ▲                                                        │
        │                                                        ▼
   refine / backtrack ◄── VERIFY (Hands: compile ∧ type ∧ tests, X1) ◄── apply edit
        │                                                        │
        └──────────────── SEARCH: parallel rollouts over edits (32 cores) ◄────┘
                          reward = {compiles+, lint+, tests-partial, tests-full}
                          decoder masked by grammar + type-state (X3)
                          abstain if a tool errors or retrieval < threshold
```

- **Cortex** = a small coder (start: `qwen3-coder`-class 0.6–1.7B, Q4) whose output is edits, not files.
- **Library** = the Code Library: an RNT index of API signatures, type stubs, and worked examples.
- **Hands** = compiler, type-checker, test runner, linter, dataflow — all in the sandbox.
- **Conductor** = GVS, now with a *deterministic* verifier and search parallelized across 32 cores.

**Deployment target unchanged from the project's DNA:** CPU-only, fits the box (here we may stretch to
the 32 GB / 32 vCPU VPS). Q4 weights + KV + search rollouts must stay within budget; §6 measures it.

---

## 4. The verifier — a deterministic oracle (this is the whole point)

Phase 4's verifier was `VerifierKind::Llm`. Phase 5 adds `VerifierKind::Exec`:

```
verify(edit) -> Reward {
    compiles:      bool,     // parser + compiler front-end
    typechecks:    bool,     // tsc / rustc --emit=metadata
    lint_clean:    bool,     // eslint / clippy
    tests_passed:  u32,      // of tests_total
    tests_total:   u32,
}
```

Reward shaping (dense, to make search tractable — sparse pass/fail is too flat):
`compiles(+) → typechecks(+) → lint(+) → tests_partial(proportional) → tests_full(+full)`.

Anti-`reward-hacking` (research 11, verification asymmetry — but now the attacker is the policy):
the sandbox blocks the model from *editing the tests*, from `skip`/`xfail`, and from network; a held-out
test split (not shown to the policy) is the real gate. A verifier that can be gamed is worse than none.

**Pre-registered success criterion for the phase-defining claim:** with `VerifierKind::Exec`,
**fab% (false SUPPORTED) < 2%** at g1 — versus 62% for the LLM judge in H1. If the deterministic
verifier does *not* crush fabrication, the entire premise of §1 is wrong and we stop.

---

## 5. Execution plan (phased, each shippable & measurable — house style)

> Philosophy (build-plan §0): *stand on giants*, ship 4-bit, specialize by cheap distillation last.
> Each phase ends with a **bitácora entry** in `07-implementation-and-deployment-log.md` (see §8).

### Phase 5.0 — Sandbox + Exec verifier ⏱ ~few days
- `src/hands/sandbox`: ephemeral container, **no network**, cpu/mem/time limits.
- `src/hands/verify_exec.rs`: implement `VerifierKind::Exec` returning the `Reward` above.
- Pilot language: **TypeScript** (strong type-state, mature tooling: `tsc`, `vitest`, `eslint`, `ts-morph`).
- **Done when:** given a patch, the verifier returns `{compiles, typechecks, lint, tests}` deterministically.
- **Bitácora:** environment, pilot-language decision, verifier latency (ms/verify — it bounds search).

### Phase 5.1 — H2: fab% with a deterministic verifier ⏱ ~1 day (**the critical, cheapest experiment**)
> Direct successor to H1. Re-run the asymmetry test, swapping the LLM judge for `VerifierKind::Exec`.
- `src/bin/h2.rs`: hold code tasks constant, vary claim depth (1/2/3-edit chains), measure TPR/TNR/**fab%**/bal-acc.
- Use the **existing 0.6B**, no training, on ~20 SWE-bench Lite tasks with tests.
- **Success:** fab% < 2% at g1 (vs 62% in H1) → the reasoning-compiler premise revives *on code*.
- **Kill:** if fab% stays high even with a compiler oracle → the bottleneck was never the verifier; stop and rethink.
- **Bitácora:** the H1-vs-H2 table side by side; go/no-go verdict.

### Phase 5.2 — Verifier-guided search (Candidate C, no training) ⏱ ~1 week
- `src/conductor.rs`: extend GVS to draw *edit* candidates; `search/`: parallel rollouts on 32 cores.
- Measure **pass@k vs search budget** and **rollouts/min on CPU** on the 20-task set.
- **Success:** pass@k rises clearly with budget (intelligence bought with search, not params).
- **Kill:** flat pass@k → the verifier signal doesn't guide → the search arm is dead.
- **Bitácora:** pass@k curve, CPU throughput, RAM footprint under search.

### Phase 5.3 — Edit-native action space + type-state decoding (X2 + X3) ⏱ ~1–2 weeks
- `src/hands/edit`: typed AST-edit algebra (insert/replace/wrap/rename; symbols resolved by env).
- Extend structured decoding (build-plan Phase 1) to **type-state masking**: only tokens keeping the
  program well-typed in scope.
- Measure **edit-algebra coverage** (% of tasks expressible as edits) and **syntax/type error rate** (target 0).
- **Kill:** >40% of tasks not expressible as edits → narrow the domain or revisit the algebra.
- **Bitácora:** coverage, validity rate, non-expressible examples.

### Phase 5.4 — Code Library via RNT + trajectory distillation (X4 + X5) ⏱ ~1 month
- Extend the RNT corpus (doc 03) to **API signatures / type stubs / worked examples**; APIs seen *only*
  in retrieval. Re-run the RNT generalization test on **held-out APIs**.
- `train/distill`: distill a frontier teacher's **verified** OpenCode trajectories (process, not text).
- **Success:** retrieval beats the no-retrieval baseline on held-out APIs (knowledge is out of the weights).
- **Kill:** retrieval doesn't help held-out APIs → fall back to bolted-on RAG.
- **Bitácora:** held-out API accuracy with/without RNT; distilled-trajectory quality.

### Phase 5.5 — The ablation curve (the scientific output) ⏱ ongoing
- `bench/ablation`: fix a target score, measure the **minimum active params** to reach it, ablating
  each of X1–X4. Baseline: a dense ~3B coder with none of them.
- **Double down if:** a <300M Cortex with X1–X4 matches the dense ~3B on SWE-bench Lite.
- **Kill (the thesis):** removing knowledge/syntax/verify does *not* let the Cortex shrink → code
  capability is **not separable** from scale. Document as a first-class negative result (like H1).
- **Bitácora:** the full params↔capability curve + honest interpretation.

---

## 6. Benchmarks & budget

| Benchmark | Measures | Why |
|---|---|---|
| **H2 (fab% vs depth)** | verifier trustworthiness on code | the premise of the whole phase (§4) |
| **SWE-bench Lite / Verified** | agentic repo-level patch | the real goal (operate OpenCode) |
| **LiveCodeBench** | function-level, contamination-resistant | base capability without test-set memory |
| **Ablation curve** | min params per target, per externalization | **proves/refutes separability** (§5.5) |
| **CPU throughput** | tok/s and **rollouts/min** on 32 cores | bounds how much search (C) is affordable |
| **RAM footprint** | Q4 weights + KV + rollouts | must fit 32 GB with margin |

---

## 7. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Code capability **not separable** from scale | The ablation curve (§5.5) measures it; a negative result is valid and publishable (cf. H1) |
| **Reward hacking** (policy edits tests, skips, hardcodes) | Sandbox forbids test edits/`skip`/network; held-out test split is the true gate (§4) |
| Verifier latency throttles search | Measure ms/verify from 5.0; cache compiles; incremental typecheck |
| Edit algebra can't express green-field code | Start on refactor/bugfix tasks; combine edits with constrained generation |
| Type-state machine costly per language | One typed pilot (TS) first; dynamic langs (Python) later |
| Cortex ignores retrieval | RNT fusion (doc 03), not bolted-on; verify on held-out APIs (§5.4) |

---

## 8. Closing every phase — the bitácora

This document (11) is the **plan**. The **record** goes in `07-implementation-and-deployment-log.md`
(the bitácora), house-style: every number measured on real hardware, not projected. On completing each
phase above, append an entry there:

```
### [DATE] — Phase 5.N: <title>
- Built: what actually shipped (not what was attempted).
- Measured: concrete numbers (fab%, pass@k, coverage, rollouts/min, params, RAM).
- Decisions: what was decided and why.
- Surprises: what differed from the plan.
- Verdict: go / no-go / pivot — against this phase's pre-registered criterion.
- Next: the concrete next action.
```

Start the record at Phase 5.0, and — mirroring H1 — log H2 the moment it runs, win or lose. The kill
criteria in §5 are pre-registered on purpose: the bitácora is where we hold ourselves to them.

---

## References (internal)

- `03-training-system-RNT.md`, `04-RNT-results-log.md` — retrieval-native training (X4).
- `06-generate-verify-search.md` — the GVS loop the code agent extends.
- `10-reasoning-compiler-h1.md` — Phase 4, the H1 experiment this phase answers with H2.
- `research/11-verification-asymmetry.md`, `research/12-test-time-search.md` — the `verify < generate` and search foundations.
- External: Agents-A1 (arXiv 2606.30616, horizon scaling + on-policy multi-teacher distillation);
  RETRO/Atlas (retrieval-pretraining); AlphaCode (generate-and-filter); RLVR (verifiable reward).
