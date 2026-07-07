# metis-1 — the first Metis-trained model: a from-scratch, code-only Cortex for OpenCode

> **metis-0** proved the system with borrowed weights: RNT (docs 03/04), GVS (doc 06), the
> deterministic code verifier (doc 11). **metis-1** is the first model whose *weights we train
> ourselves, from zero* — a code-only Cortex designed from step 0 for exactly one job: operating
> OpenCode inside the Metis system, and fighting frontier coding agents on a defined surface while
> being 100–1000× smaller and cheaper.
>
> The bet is NOT "a 0.7B bare model beats a frontier model at chat." The bet, pre-registered and
> falsifiable, is: **metis-1 + Library + deterministic verifier + search ≥ frontier-agent quality on
> a defined code surface, at a fraction of the cost** — because everything a frontier coder wastes
> parameters on (memorized APIs, world knowledge, 100 languages, chat) is either externalized or
> deleted from the objective.

---

## 1. Why from scratch now (and not fine-tuning a base)

Doc 11 said "stand on giants." That was correct for proving the *system*. But every borrowed base
pays for capabilities metis-1 must not pay for, and — decisively — **no existing base was trained
retrieval-native**, so RNT's anti-memorization objective fights the base's own pretraining instead
of shaping it from step 0. What we now know that makes from-scratch viable and *cheap*:

1. **Scope collapse is the biggest compression there is.** metis-1 needs: English + code
   (TypeScript-first), reading retrieved context, emitting edits/diffs/tool-calls. It does NOT need
   world knowledge, multilinguality, chat personality, or safety-tuned prose. Each deleted
   capability is deleted training tokens and deleted parameters.
2. **RNT works and we know its one hard rule** (doc 04, Round 4): *dense supervision or the recall
   circuit starves.* Every training sequence is built retrieval-native from step 0.
3. **Code has a free, deterministic reward signal** (doc 11): compile ∧ typecheck ∧ tests. That
   powers both data filtering (only verified trajectories enter SFT) and RLVR at the end.
4. **Training-efficiency levers we've specified** (the METIS training brief): high-value token
   selection (only ~30% of tokens get a backward pass), seed-and-grow (train small, grow by
   function-preserving stacking at ~half budget), knowledge externalized to the Library so the
   parameter budget buys reasoning only.
5. **It is genuinely affordable.** A 0.7B model on ~200B tokens is ~8.4e20 FLOPs ≈ 4–5 days on one
   8×H100 node ≈ **US$2–4k** at spot prices — before the token-selection and growth savings. The
   whole program including ablations and RLVR fits under ~US$10k.

## 2. What metis-1 is

**Sizing philosophy (decided 2026-07-07): param-poor, token-rich.** The Qwen small-dense lesson is
not "small models are magic" — it is that parameters shrink when you overwhelm them with tokens
(Qwen3-0.6B saw ~36T). metis-1 takes that recipe and narrows the distribution to one job: the same
token flood, but *all of it code, all of it edit-shaped*. We deliberately do NOT train a monster;
we over-train a tiny dense model 30–100× past Chinchilla on a curated GitHub distribution, and
distill from a strong open coder (teacher logits) while doing it — the same strong-to-weak recipe
the capable small models themselves are made with.

| Component | Choice | Why |
|---|---|---|
| **Family** | `metis-1-nano` (~125M, ablation/pilot) → `metis-1` (~0.3B dense, over-trained) | nano de-risks every decision cheaply; ~0.3B × 300B+ tokens beats 0.7B × 100B at equal FLOPs for a narrow domain, and deploys at ~200 MB Q4 — instant everywhere |
| **Architecture** | decoder-only: 24L × 1536d, GQA (12 q / 4 kv heads), SwiGLU, RMSNorm, RoPE | boring on purpose — the novelty is in objective + data, not the block |
| **Tokenizer** | ~48k BPE trained on code+English only | code-tuned vocab ≈ 15–25% fewer tokens per file than general vocabs |
| **Context** | 8k pretrain → 32k extension phase | repo work needs 32k; paying for it from step 0 wastes FLOPs |
| **Precision** | bf16 training; Q4 GGUF for deploy; ternary QAT as a stretch (doc 03 §4) | deploy target is still CPU boxes |
| **Languages** | TypeScript/JavaScript first-class; Python second; nothing else | matches the Phase-5 verifier; depth over breadth |

**Retrieval-native by construction (X4, now at scale).** Every pretraining sequence has the shape
`<lib> retrieved blocks (API sigs, type stubs, imported symbols, doc snippets) </lib> <ctx> repo
context </ctx> <task> issue/test/instruction </task> <edit> target diff/code </edit>` — with
next-token supervision **dense across the whole sequence** (Round-4 rule), target span weighted
higher. The anti-memorization penalty (selective weight decay on MLP down-projections, the RNT
`decayMask` knob) discourages storing API facts in weights: the fact is always in `<lib>`, so the
only way down for the loss is *learning to read it*.

**Edit-native output (X2).** The supervised target is diffs/AST-edits + tool calls, mined from
real commits and verified trajectories — not whole-file generation. The action space the model
practices is exactly the action space OpenCode gives it.

## 3. Data — the actual moat (300B+ tokens, TS-heavy; GitHub is the whole ocean)

**GitHub gives us four supervision signals that generic pretrains barely use, and they are exactly
the four an agent model needs:** (1) **commits/diffs** — millions of real edit-native training
pairs; (2) **issues linked to merging PRs** — natural (task → patch) pairs, the agent objective
itself; (3) **CI status on every PR** — free verified/unverified labels at planetary scale, the
same signal our sandbox produces locally; (4) **review comments** — critique data for the
verify/repair loop. Raw volume is not the moat (everyone has The Stack); *mining these
relational signals into edit-shaped, verified sequences* is.

| Slice | Source | Share | Construction |
|---|---|---|---|
| **RNT code corpus** | The Stack v2 (dedup) TS/JS + Python, permissive | ~55% | for each file, statically resolve imports → retrieve real signatures/stubs/docs into `<lib>`; supervise the file given its true dependencies |
| **Commit/PR diffs** | GitHub archives: (pre-state, issue/CI failure, diff) | ~20% | free edit-native supervision; retrieved context = the files/symbols the diff touches |
| **Verified agent trajectories** | frontier teacher driving OpenCode in the Phase-5 sandbox; keep only runs where compile∧typecheck∧tests pass (X5) | ~10% | process distillation with a deterministic filter — no unverified imitation |
| **Code-adjacent English** | docs, READMEs, StackOverflow-class Q&A | ~10% | the model must read issues and write commit messages, nothing more |
| **Synthetic selection drills** | many-distractor retrieval tasks over real APIs (doc 04 Rounds 1–4, scaled) | ~5% | directly trains the content-matching circuit that broke at toy scale |

**Token selection (the gradient market, v0).** A ~50M scorer model rates sequences/tokens by
predicted loss-reduction-per-FLOP; only the top ~30–40% get a backward pass (code is massively
redundant — boilerplate, imports, generated files). Pre-registered measurement at nano scale:
selection must beat uniform training by ≥1.3× FLOPs-to-loss or we drop it (it's a lever, not a
religion).

**Seed and grow.** Train 12L at full LR → duplicate-stack to 24L at ~50% of budget →
continue. Function-preserving growth; ~1.5–2× savings per the stacking literature. Validated at
nano→2×nano first.

## 4. Training pipeline (five stages, each with a number attached)

1. **Pretrain (RNT objective + teacher distillation)** — 300B+ tokens as above, dense supervision,
   anti-memorization decay, token selection, seed-and-grow, and cached teacher logits from a strong
   open coder (e.g. Qwen-Coder-32B-class) as soft targets — the strong-to-weak multiplier tiny
   models need. Output: `metis-1-base`.
2. **Context extension** — 8k→32k on repo-shaped long sequences (~5B tokens).
3. **SFT** — verified OpenCode trajectories only (the X5 set + tool-call formatting). Small: ~1–2B
   tokens. Output: `metis-1-sft`.
4. **RLVR** — on-policy in the Phase-5 sandbox against the dense reward
   `compiles → typechecks → lint → tests_partial → tests_full`, anti-reward-hacking rules already
   built (tests patch-protected, held-out test split, no network). This is where the model learns to
   *use* the verifier loop instead of merely imitating it. Output: `metis-1`.
5. **Deploy** — Q4 GGUF, served OpenAI-compatible (llama.cpp/ollama); OpenCode points at it as a
   custom provider; the Conductor (GVS + parallel rollouts) wraps it exactly as doc 11 §3 specifies.

## 5. Execution phases, budgets, kill criteria (pre-registered, house style)

### M1.0 — Tokenizer + data factory ⏱ ~1–2 weeks · ~US$100 (CPU + storage)
Build the RNT construction pipeline (import-resolution → `<lib>` blocks), the commit-diff miner,
the scorer model, the tokenizer. **Done when:** 10B tokens of RNT-shaped TS data stream
deterministically. **Kill:** import-resolution yields usable `<lib>` blocks for <60% of files →
fix extraction before any GPU is rented.

### M1.1 — metis-1-nano (125M) ⏱ ~1 week · ~US$100–200 (1×H100, ~2 days)
The scientific gate for everything. Train nano on ~15B tokens, three arms: (a) RNT objective,
(b) same data non-retrieval baseline, (c) RNT without dense supervision.
**Pre-registered success:** on **held-out APIs** (never in any training weights, only in `<lib>`),
arm (a) beats arm (b) by a wide margin, and (c) confirms the Round-4 rule at scale. Also measure:
token-selection savings ≥1.3×, growth 12L→24L(nano×2) stable.
**Kill:** if retrieval doesn't beat baseline on held-out APIs at 125M, RNT does not scale as built —
stop, diagnose against doc 04, do NOT proceed to 0.7B.

### M1.2 — metis-1-base (~0.3B, over-trained) ⏱ ~2–3 weeks · ~US$2–3k (8×H100 spot, ~3–4 days)
Full pretrain + context extension with every lever that survived M1.1.
**Success gate:** beats `Qwen2.5-Coder-1.5B-base` on TS HumanEval-style evals *when given `<lib>`
context*, and shows near-zero degradation on held-out-API tasks (the knowledge is provably outside
the weights). **Kill:** loses to open 0.5B bases at equal context → our data factory, not the
thesis, is the problem; iterate data before scale.

### M1.3 — SFT + RLVR ⏱ ~2 weeks · ~US$1–2k
Verified-trajectory SFT, then RLVR in the sandbox (rollouts are CPU-cheap; the policy updates are
the GPU cost). **Success:** pass@1 on the 20-task H2 set doubles vs `metis-1-base`; fab% stays <2%
(the verifier premise, doc 11 §4). **Kill:** RLVR reward-hacks despite the sandbox rules → freeze
at SFT, strengthen sandbox, log it.

### M1.4 — The fight ⏱ ongoing
`metis-1` inside the full system (Library + GVS + parallel rollouts on the 32-core box) vs frontier
agents, on the pre-registered surface: **SWE-bench Verified (TS subset) + LiveCodeBench (TS/JS) +
a 50-task OpenCode suite**. Report the honest triple: quality, wall-clock, $/task.
**Double down if:** system-metis-1 reaches ≥80% of a frontier agent's score on the TS surface at
≤1% of its $/task. **Kill (the thesis):** if the full system can't beat a *bare* dense 7B coder,
separability failed at product scale — write it up as the negative result, per house rules.

**Total program: ~US$5–8k, ~2 months part-time.** Every phase ends with a bitácora entry in doc 07.

## 6. Why this can actually fight frontier models (and where it honestly can't)

**Where the fight is winnable.** On the defined surface, every hard sub-problem has been moved to a
component that doesn't need parameters: API knowledge → Library (disk); correctness judgment →
compiler/tests (exact, free); depth → search over verified edits (32 CPU cores); breadth of
languages → deleted. What remains for the weights — read context, propose plausible edits, use
tools — is exactly what 0.5–1.5B models already demonstrably do when specialized. The frontier
model carries a datacenter on its back; metis-1 carries a battery. On *this* track, with the
verifier as equalizer, quality-per-dollar is winnable by orders of magnitude, and quality-parity is
the pre-registered experiment.

**Where it isn't.** Open-ended architecture design, cross-language monorepos, tasks without tests
or types — the surface where verification is weak is the surface where scale still wins (doc 03
§6). metis-1 does not claim that ground; it claims the verifiable core of everyday agentic coding,
and measures the claim.

## 7. Relationship to the rest of the project

- **metis-0** stays the system shell (Cortex·Library·Hands·Conductor) — metis-1 slots in as the
  Cortex via the same `METIS_MODEL` override; nothing in the Conductor changes.
- Doc 11's phases 5.0–5.2 (sandbox, H2, verifier-guided search) are **prerequisites already built
  or in flight**; metis-1 is the fulfillment of 5.4's "trajectory distillation" with the missing
  radical step: our own retrieval-native weights.
- Doc 03 §5's scale-up pipeline is exactly what M1.0–M1.2 implements; doc 04's Round-4 dense-
  supervision rule is baked into the data factory as a hard invariant.

## 8. First concrete actions (this week, no GPU needed)

1. `train/` workspace: tokenizer training script on a Stack-v2 TS sample (~10GB).
2. RNT data factory v0: TS import-resolver (`ts-morph`) → `<lib>` block builder → sequence packer;
   golden tests on 100 real repos.
3. Commit-diff miner v0 on 1k popular TS repos (GH Archive).
4. Scorer model spec + training harness (it trains on nano's own loss deltas — no extra data).
5. Reserve the H2/H1 eval sets now as **held-out**, so no training slice ever touches them.
