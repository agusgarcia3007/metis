# metis-1 VERA-R — verifier-native architecture for a tiny coding model that can fight frontier agents

> **Authored by Sakana Fugu Ultra (external review), 2026-07-07.** Renumbered from 14→15 to sit
> beside doc 14 (*The compiler is an infinite teacher*). The two are complementary and were reached
> independently: **doc 14 is the learning loop** (why a tiny local model beats a frozen frontier one
> — self-distilling verified search, compounding nightly); **doc 15 (this doc) is the policy
> architecture** (what that model consumes and emits — a repair-transition policy over diagnostics,
> retrieval, and a verification ladder). VERA-R is the *policy* inside doc 14's *loop*.
>
> **Adoption discipline (house style):** the ideas here are adopted incrementally, one measurable
> step at a time — NOT all six heads + Patch IR + proof-carrying patches at once, which would spread
> a tiny model's capacity too thin. The highest-leverage adoptions for the local flywheel, in order:
> (1) the **warm verification ladder** (§8) — makes CPU search affordable, the precondition for the
> flywheel to run on a Mac; (2) **repair-transition / diagnostic-first data** (§2, §15.1) — what the
> flywheel naturally produces; (3) the **localization head** (§5.1); (4) the **contamination
> firewall + mutated-API evals** (§12, §15.3) — required before any "beats frontier" claim.
>
> **Convergence signal:** every VERA-R bet is about the *objective, retrieval, verifier, and search*
> — none about a fancier transformer block. This independently matches the Night-2 finding (docs
> bitácora: nGPT lost to the Muon champion; the block is near-optimal at this scale). Two independent
> sources, same north.
>
> **Status:** proposal only. This document records architecture ideas for training **metis-1**, a brand-new tiny coding Cortex, without implementing any of them.
>
> **Thesis:** metis-1 should not try to be a small frontier chat model. It should be the best tiny model at one narrow job: reading a failing repo state and proposing the next verified repair step inside the Metis system.
>
> **Target claim:** `metis-1 + Library + deterministic verifier + search` can beat frontier coding agents on a constrained, verifiable TS/JS repair surface by cost-adjusted solved rate, and maybe by absolute quality once the verifier/search loop is strong enough.

---

## 0. What not to claim

The wrong claim:

> A 300M bare metis-1 beats GPT/Claude/Gemini at all coding.

That is probably false.

The credible claim:

> **metis-1 is a tiny code-repair policy inside Metis: Library + deterministic verifier + search + sandbox. It beats frontier coding agents on a constrained, verifiable TS/JS repair surface by cost-adjusted solved rate, and competes on absolute quality where tests/types make verification strong.**

The model should not win by knowing more. It should win by needing less knowledge in weights, outsourcing correctness to deterministic tools, and buying depth with verified search.

---

## 1. Core architecture: VERA-R

**VERA-R = Verifier-native, Edit-native, Retrieval-native Repair policy.**

The model's job is not:

```text
prompt -> code answer
```

The model's job is:

```text
(repo state, task, retrieved symbols, failing signal, previous attempts)
  -> next minimal edit that moves the repo closer to green verification
```

Training and inference should both treat coding as a repair/search process:

```text
state_0
  -> candidate edit
  -> deterministic verification
  -> diagnostic / reward
  -> next state
  -> next candidate edit
```

Metis-1 is therefore a **repair transition model**, not a general chat/code model.

---

## 2. Train on repair transitions, not only final patches

Most tiny-coder plans start with `instruction -> code`. For Metis, invert the distribution:

```text
broken repo state
+ compiler/test/lint diagnostic
+ retrieved relevant symbols
+ prior failed patch
-> minimal repair edit
```

Canonical sequence shape:

```text
<state>
  repo graph summary
  touched files
  failing tests
  compiler/linter diagnostics
  previous patch attempts
</state>

<lib>
  exact imported symbols
  type signatures
  local examples
  package docs
</lib>

<edit>
  minimal patch
</edit>

<verify>
  expected check commands
</verify>
```

The key skill is **diagnostic -> next fix**. A tiny model may not solve hard repo tasks in one shot, but it can become excellent at interpreting verifier feedback and making the next useful move.

### Training examples to mine

- compile error -> patch that fixes compile;
- type error -> patch that fixes typecheck;
- failing unit test -> smallest patch that makes it pass;
- lint failure -> style-safe repair;
- previous bad model patch -> corrective patch;
- issue + existing failing regression test -> final merged patch;
- test-generated reproducer -> implementation patch.

---

## 3. Retrieval is part of the model contract, not bolted-on RAG

For metis-1, knowledge should be treated as external by default.

The model is trained to assume:

```text
If a fact is not in <lib>, do not pretend to know it.
```

The Code Library should have two retrieval layers.

### 3.1 Deterministic retrieval

Use exact code tooling first:

- LSP / tsserver;
- import graph;
- export graph;
- local call graph;
- type signatures;
- interface and type definitions;
- test-to-source links;
- package entry points;
- recently changed files;
- symbol references.

For code, exact symbol retrieval is usually more valuable than fuzzy semantic search.

### 3.2 Learned example retrieval

Use embeddings for examples that exact tooling will not catch:

- similar bugfixes;
- similar compiler diagnostics;
- similar test failures;
- similar API usage examples;
- style-local examples from the same repo;
- similar previous Metis repair trajectories.

### 3.3 Retrieval-native anti-memorization eval

Create private/mutated APIs that cannot be in the weights:

```ts
import { rotateFenwickSession } from "@private/auth-v17";
```

Only expose the real signature in `<lib>`. If metis-1 solves it, it learned to read retrieval. If it solves popular public APIs but fails private/mutated APIs, it memorized instead of becoming retrieval-native.

---

## 4. Output diffs first, Patch IR second

Do not make a custom AST-edit language the only output at the start.

The largest natural supervision source is git:

```text
commit / PR / patch / diff
```

So the primary output should be:

```text
compact unified diff
```

Then add a secondary Patch IR for common operations:

```text
replace_function_body
insert_import
rename_symbol
add_type_guard
change_call_args
wrap_expression
add_case
update_type_annotation
extract_helper
```

### Why this order

- Diffs preserve real-world data scale.
- Patch IR improves validity for frequent edits.
- Raw diff fallback avoids blocking green-field or unusual changes.
- The Conductor can learn when Patch IR is safe and when raw diff is needed.

Recommended output contract:

```text
<edit_format>diff</edit_format>
<edit>
--- a/src/auth.ts
+++ b/src/auth.ts
@@ ...
</edit>

<checks>
  bun test src/auth.test.ts
  bun run typecheck
</checks>
```

Optional structured version:

```text
<edit_format>patch_ir</edit_format>
<edit>
insert_import(file="src/auth.ts", symbol="Session", from="./types")
replace_function_body(symbol="refreshSession", ...)
</edit>
```

---

## 5. Add auxiliary heads, but never replace the verifier

Use one shared tiny trunk with multiple heads:

```text
shared transformer trunk
  ├── next edit token head
  ├── localization head
  ├── verifier prediction head
  ├── failure class head
  ├── retrieval usefulness head
  └── next tool/action head
```

### 5.1 Localization head

Predict:

```text
files_to_edit
symbols_to_edit
tests_to_run
context_needed_next
```

Repo-level coding often fails because the agent edits the wrong place. Localization may be higher leverage than raw generation quality.

Training labels come from real diffs:

```text
issue + repo summary -> files/symbols touched by eventual patch
```

### 5.2 Verifier prediction head

Predict:

```text
P(patch_applies)
P(typecheck_passes)
P(lint_passes)
P(visible_tests_pass)
P(heldout_tests_pass)
likely_failure_kind
expected_tests_passed
```

But this head is a **ranker only**.

Correct use:

```text
sample 64 candidate edits
rank with verifier head
keep diverse top 12
run real sandbox verifier
train on actual verifier result
```

Incorrect use:

```text
verifier head says pass -> accept patch
```

That would recreate the learned-judge failure mode Metis is specifically avoiding.

### 5.3 Failure-class head

Predict what kind of failure a candidate or state represents:

```text
syntax_error
type_error
lint_error
test_assertion_failure
runtime_exception
missing_import
wrong_signature
wrong_file
incomplete_patch
reward_hacking_attempt
```

This head helps the Conductor choose the next action and helps the repair policy specialize.

### 5.4 Retrieval-usefulness head

Predict which retrieved chunks mattered for the patch.

Use it to:

- prune noisy context;
- train better retrieval;
- attribute edits to source facts;
- detect when the model is ignoring `<lib>`.

---

## 6. Train metis-1 as a search policy

Metis already wants Generate -> Verify -> Search. Metis-1 should be the policy/value network inside that loop.

Represent solving as a tree:

```text
state_0
  ├── edit_a -> verifier_a -> state_a
  ├── edit_b -> verifier_b -> state_b
  └── edit_c -> verifier_c -> state_c
```

Train:

```text
policy(state) -> candidate edits
value(state) -> probability branch can be solved
repair(state, verifier_failure) -> next edit
```

The compiler/tests remain the oracle. The learned value function only controls branch ordering.

### Search metrics

Measure both:

```text
pass@k       = can search find it within k samples?
pass@inf-ish = did any sample in a large pool contain a valid fix?
```

Interpretation:

- If `pass@inf-ish` is low, the model's sampling support is too weak. Search cannot rescue it.
- If `pass@inf-ish` is high but `pass@k` is low, ranking/search is the problem.
- If `pass@k` rises with budget, intelligence is being bought with search, not parameters.

---

## 7. Two-tier constrained decoding

Full type-state decoding per token is probably too slow. Use two layers instead.

### 7.1 Always-on cheap constraints

Always enforce:

```text
valid diff format
valid JSON/tool-call format
valid Patch IR syntax when using Patch IR
balanced fences / tags
no edits to forbidden files
no test skip markers
no @ts-ignore / @ts-expect-error unless explicitly allowed
```

### 7.2 Decision-point type constraints

Only query type state at key choices:

```text
choosing symbol name
choosing import
choosing call arguments
choosing property access
choosing return type
choosing object field
choosing discriminated-union case
```

Use cached scope tables and tsserver snapshots instead of live compiler calls every token.

Goal: most validity benefits without destroying CPU throughput.

---

## 8. Warm incremental verifier

Search dies if every candidate costs seconds. Cold full verification is still the final gate, but search needs cheap intermediate gates.

Architecture:

```text
persistent warm sandbox
persistent language server
incremental tsc
compile cache
dependency cache
affected-test selection
read-only test/config layer
held-out tests injected after candidate patch
cold full verification as final accept gate
```

Verification ladder:

```text
patch applies
-> syntax parses
-> affected file typechecks
-> lint on touched files
-> affected tests
-> full visible tests
-> held-out tests
```

This is where Metis can turn CPUs into intelligence.

---

## 9. Verifier-strengthening through generated tests

The deterministic verifier only helps when tests/types are strong. Real repos often have weak tests.

Add a **Verifier-Strengthener** component:

```text
current behavior -> characterization tests
issue description -> candidate regression tests
public API contract -> property checks
failing stacktrace -> targeted reproducer
```

Rules:

- generate tests before the implementation patch;
- freeze generated tests;
- the repair policy cannot edit them;
- final acceptance uses original tests + generated tests + held-out tests;
- generated tests are useful only if they encode the task contract, not the model's patch.

This avoids becoming benchmark-only and helps Metis operate on under-tested repos.

---

## 10. Executable proof-carrying patches

Ask the model to emit obligations, but only accept machine-checkable obligations.

Example:

```text
PATCH:
  ...

OBLIGATIONS:
  - preserves refreshSession public signature
  - fixes auth-refresh.spec.ts
  - does not edit token expiry logic
  - no network behavior changed

CHECKS:
  - bun test auth-refresh.spec.ts
  - bun run typecheck
  - rg "fetch\\(" src/lib/auth.ts should be unchanged
```

Reject decorative prose. If an obligation cannot become a check, it does not count.

The point is to train metis-1 to think in verifier terms.

---

## 11. Proposed model shape

Start boring and tiny.

```text
metis-1 VERA-R

Core:
  300M-500M dense decoder
  code-native tokenizer
  8k pretrain context -> 32k extension
  local/sliding attention over code
  global attention over task, diagnostics, failing tests
  retrieval cross-attention or explicit <lib> blocks

Inputs:
  task
  repo graph summary
  exact LSP-retrieved symbols
  relevant examples
  failing tests
  compiler/lint diagnostics
  previous failed attempts

Outputs:
  compact unified diff
  optional Patch IR
  verifier commands
  executable obligations

Heads:
  edit-token head
  localization head
  verifier-ranker head
  failure-class head
  retrieval-usefulness head
  tool-action head

Runtime:
  Metis GVS search
  warm incremental sandbox
  real compiler/typecheck/test oracle
  held-out tests
  test-tamper protections
```

Avoid a giant MoE at first. Metis's local-first goal favors simple dense models that are easier to train, quantize, and serve. Conditional compute may be useful later, but it should not be on the critical path.

---

## 12. Data firewall before data factory

Before mining GitHub-scale data, build contamination controls.

Required exclusions:

```text
repo-level benchmark blocklist
commit-hash exclusion
issue/PR URL exclusion
temporal cutoff
fork/mirror detection
near-duplicate diff detection
benchmark-specific held-out set
private/mutated API evals
```

Without this, any claim about beating frontier coding agents is not credible.

Benchmark-derived GitHub tasks can leak through issues, PRs, commits, forks, and copied patches. Treat contamination prevention as part of the architecture, not an afterthought.

---

## 13. Training plan

### Stage 0 — Tokenizer + data factory

Build:

- code-native tokenizer;
- TS/JS import resolver;
- `<lib>` block builder;
- sequence packer;
- diff miner;
- issue/PR/CI miner;
- benchmark contamination firewall;
- private/mutated API eval generator.

Done when:

```text
10B tokens of deterministic RNT-shaped TS/JS data can stream reproducibly.
```

Kill criterion:

```text
If import/type retrieval yields usable <lib> blocks for <60% of files, fix extraction before renting GPU.
```

### Stage 1 — metis-1-nano experiments

Train 50M-125M variants before scaling.

Arms:

```text
A. from-scratch + retrieval-native + diff output
B. same, no retrieval
C. same, Patch IR primary
D. retrieval-retrofitted open small coder
E. RNT without dense supervision on <lib>
```

Important: include arm D. Do not assume from-scratch is better. If retrofitting retrieval into a small open coder wins, use it.

Success metrics:

```text
held-out API success
localization accuracy
patch applies rate
typecheck prediction calibration
pass@inf-ish
pass@k
wall-clock per solved task
```

Kill criterion:

```text
If retrieval-native nano does not beat no-retrieval nano on held-out APIs, do not scale.
```

### Stage 2 — base pretraining

Candidate data mixture:

```text
35% TS/JS repo-level code
15% Python repair/code
20% real commits and PR diffs
10% issue -> PR -> merged patch pairs
10% compiler/test failure -> repair pairs
5% verified agent trajectories
5% synthetic/private API retrieval drills
```

Bias harder toward repair traces than generic code. The model does not need broad language coverage; it needs to move real repos toward green checks.

Training objective:

```text
next-token loss on full sequence
higher weight on edit spans
auxiliary localization loss
auxiliary verifier-outcome loss
auxiliary failure-class loss
retrieval-usefulness contrastive loss
```

### Stage 3 — SFT on verified trajectories only

Teacher agents are useful only when filtered by deterministic verification.

Keep trajectories only if:

```text
patch applies
typecheck passes
visible tests pass
held-out tests pass
no forbidden file edits
no test tampering
no network attempt
patch size reasonable
```

Do not imitate beautiful failures.

Avoid assuming dense teacher-logit distillation across hundreds of billions of tokens is affordable. Use sparse/top-k logits or trajectory-only distillation unless the budget is re-estimated.

### Stage 4 — RLVR inside the Metis sandbox

Reward must be dense but grounded:

```text
+ patch applies
+ syntax parses
+ typecheck improves
+ lint improves
+ visible tests improve
+ held-out tests pass
- edits tests/config unless explicitly allowed
- disables tests
- adds .skip/.only
- adds @ts-ignore to hide failure
- giant rewrite without need
- network attempt
- public API break
```

Curriculum:

```text
1-edit fixes
-> single-file multi-edit
-> multi-file bugfix
-> small feature with tests
-> dependency/API migration
```

Tiny models need curriculum. Do not throw full agentic repair at them from day one.

---

## 14. Benchmark plan

Be precise about the frontier fight.

Primary public/private surface:

```text
custom contamination-controlled TS/JS OpenCode suite
SWE-bench Multilingual JS/TS subset
LiveCodeBench-style TS/JS tasks where applicable
```

Secondary surface:

```text
SWE-bench Verified only if Python becomes first-class
```

Do not overclaim on a Python-heavy public benchmark while calling the model TypeScript-first.

### Metrics to report

```text
resolved rate
cost per solved task
wall-clock per solved task
pass@1
pass@k
pass@inf-ish
patch size
regression rate
held-out test pass rate
test-tampering rate
localization accuracy
verifier latency
```

Claim format:

```text
metis-1 + Metis beats frontier agents on cost-adjusted solved rate
and aims to match/exceed quality on verified TS/JS repair tasks.
```

Avoid:

```text
metis-1 bare model beats frontier models generally.
```

---

## 15. Most promising novel bets

### 15.1 Diagnostic-first pretraining

Make compiler/test output a first-class language:

```text
diagnostic tokens are not comments;
they are the control signal.
```

Train millions of:

```text
broken patch -> diagnostic -> minimal repair
```

This is likely higher leverage than generic code pretraining.

### 15.2 Localization as a supervised head

Train:

```text
issue + repo summary -> files/symbols touched by eventual patch
```

If metis-1 can localize cheaply, the edit itself gets much easier.

### 15.3 Retrieval mutation evals

Every API eval has two versions:

```text
real popular package
mutated private package with same shape but changed names/signatures
```

This tests whether metis-1 reads Library context or memorizes GitHub.

### 15.4 Warm verifier search

Build search as a verification ladder:

```text
candidate pool
-> cheap syntactic filters
-> incremental type filters
-> affected tests
-> full tests
-> held-out tests
```

This is the place where Metis can buy intelligence with CPUs instead of parameters.

### 15.5 Executable proof-carrying patches

Patch plus obligations plus checks. The model proposes its own verification plan, but the Conductor only accepts machine-checkable obligations.

---

## 16. Recommended first implementation sequence later

This section is a future implementation order, not applied now.

1. Build the contamination firewall and benchmark holdout registry.
2. Build TS/JS deterministic retrieval: imports, exports, symbols, signatures, call graph.
3. Build the RNT-shaped sequence packer with `<state>`, `<lib>`, `<edit>`, `<verify>`.
4. Mine repair transitions from existing repos and synthetic broken patches.
5. Train 50M-125M nano arms A-E.
6. Evaluate held-out API reading, localization, patch validity, and pass@k.
7. Only scale if retrieval-native nano beats no-retrieval nano.
8. Add warm verifier search and verifier-ranker integration.
9. Add verified-trajectory SFT.
10. Add RLVR against the existing Phase-5 sandbox.

---

## 17. Final architecture summary

```text
metis-1 VERA-R

A tiny repair policy trained to read:
  - repo state
  - exact retrieved symbols
  - failing diagnostics
  - tests
  - previous failed attempts

And emit:
  - minimal diffs
  - optional Patch IR
  - verifier commands
  - executable obligations

Inside:
  - Metis Library for API/repo knowledge
  - deterministic verifier for truth
  - warm sandbox for fast search
  - held-out tests for reward integrity
  - GVS Conductor for exploration/backtracking

Winning path:
  - not more parametric knowledge
  - better retrieval
  - better localization
  - better repair transitions
  - better verifier-guided search
  - lower cost per solved task
```

The essence:

> **metis-1 should be the world's best tiny model at reading a failing repo state and proposing the next verified repair step.**
>
> That is the path where a tiny model can plausibly beat frontier systems: not by having more knowledge, but by needing less knowledge in weights; not by reasoning longer internally, but by outsourcing depth to verified search; not by sounding smarter, but by making repos green.
