# metis-1 — what to steal from Agents-A1 and Ornith-1.0 (and the honest bound they set)

> Two recent models are the strongest external evidence yet that metis's bet is right — and the
> clearest map of the mechanisms that make "system beats scale" actually work. Both are also a
> reality check: they prove the direction at 9B–35B, not at sub-1B. This doc extracts what transfers
> to a tiny local Cortex, and sharpens where metis's *unique* claim lives.

---

## 1. Agents-A1 — "Scaling the Horizon, Not the Parameters" (arXiv 2606.30616)

A 35B MoE agent reaches trillion-parameter-class results (competitive with Kimi-K2.6, DeepSeek-V4-pro)
by scaling the *agent horizon*, not weights. This is metis's thesis, validated by a real lab. The
transferable machinery:

- **Knowledge-Action Graph (KAG) — the training substrate is the trajectory, not the final patch.**
  Training data is linked records `(state, action, observation, verifier outcome)`, averaging 45K
  tokens/trajectory. Verification is woven *into* the supervision, not bolted on. This is the same
  object VERA-R (doc 15 §2) calls a "repair transition" and Aletheia stored as a causal graph — A1
  shows it is the *core* data, not a side feature.
- **Self-play proposer-solver-verifier loop** expands the graph: generate task → solve → verify →
  keep only verified, evidence-backed trajectories. Training data manufactured without human labels.
  This is metis's flywheel, at scale.
- **Salient Vocabulary Alignment (SVA) — cheap distillation.** Multi-teacher on-policy distillation
  that computes KL divergence **only over the teacher's top/salient tokens**, renormalized, then
  aggregates per-domain so no domain dominates. The lever for us: distilling from a strong code
  teacher becomes affordable when you only match the top-k token distribution, not the full 100k
  vocab — directly the cheap-distillation path metis-1 needs (doc 12 stage 1).
- **Step-level verification for credit assignment.** Dense process reward across the trajectory
  (constraint satisfaction on partial failures), not just final binary pass/fail — richer signal from
  the same verifier.

## 2. Ornith-1.0 — self-scaffolding (DeepReinforce, MIT; 9B ≈ 30B)

Instead of engineers hard-coding one agent scaffold per task category, the model **learns its own
scaffold** during RL. Each RL step is two stages:

1. **Scaffold stage:** read the task + the scaffold used last time → propose a refined scaffold
   (memory layout, retry logic, tool orchestration, task decomposition).
2. **Solution stage:** condition on that scaffold + task → produce a solution rollout → verify.
   Reward flows back to optimize **both** stages jointly; better scaffolds are selected and mutated.

- **Algorithm:** token-level GRPO, asynchronous pipeline-RL, staleness-weighted off-policy tokens.
- **Anti-reward-hacking triple defense:** (1) immutable trust boundary, (2) deterministic monitors
  flagging banned actions, (3) a frozen LLM-judge veto. metis's Phase-5 sandbox already has (1) and
  (2); (3) is a cheap add.
- **The lesson for metis:** we were about to *hard-code* the GVS/search loop. Ornith says the
  scaffold — how the model orchestrates search, retries, memory — should be **learnable, not fixed**,
  so it doesn't ossify. (metis-0's README already prototyped a zero-training heuristic version,
  `Scaffold::select`; Ornith is the trained version of that idea.)

## 3. The honest bound both set

The smallest Ornith is **9B** (SWE-bench Verified 69.4); Agents-A1 is **35B**. These techniques buy a
**~3–30× parameter reduction** (1T→35B; matching a 30B with a 9B) — **not** the ~1000× reduction to a
200 MB laptop model. **No one has shown a sub-1B model fighting frontier agents, even with horizon
scaling or self-scaffolding.** Any claim that these papers prove "tiny beats frontier" is false; they
prove "system-scale beats parameter-scale" in the 9B–35B band.

So metis must not pretend A1/Ornith did our job. They did something adjacent and load-bearing: they
proved the *direction* pays, and handed us the mechanisms. The sub-1B frontier is still open.

## 4. Where metis's unique claim lives (the axis A1 and Ornith cannot occupy)

Both A1 and Ornith ship **generic, frozen** weights — identical for every user, best-on-day-one,
unable to specialize to your repo. Their horizon-scaling makes a *generalist* smaller; it does not
make it *yours*. metis's differentiator (doc 14) is untouched by them:

> **Local specialization + compounding.** A sub-1B model taught by *your* compiler on *your* repo,
> distilling *its own* verified trajectories nightly, does not need to match a frontier generalist at
> all code — only to beat it on your code, which a frozen generic model structurally cannot do.

A1 and Ornith are the proof that the *system* around a smaller model is where the wins are. metis
pushes the same lever one octave further down (sub-1B) and adds the one thing a datacenter model can
never do: run on, and specialize to, the machine that owns the code.

## 5. What metis-1 adopts (concrete, ordered, and gated by the Aletheia lesson)

Nothing here happens before the generator can write real code (doc 14 §0 — Aletheia's grave). Then,
in order:

1. **KAG trajectory data (from A1 + VERA-R):** train on `(state, diagnostic, action, verifier
   outcome)` transitions produced by self-play under the Phase-5 compiler — not on final diffs alone.
2. **Self-play proposer-solver-verifier (A1):** the generator proposes edits, the compiler verifies,
   only verified trajectories are kept — this *is* the flywheel's data engine.
3. **SVA cheap distillation (A1):** if we distill from a strong code teacher, match only top-k token
   distributions — makes teacher distillation affordable for a tiny model.
4. **Step-level process reward (A1):** reward compile→typecheck→test progress, not just final green.
5. **Learnable scaffold (Ornith):** keep the GVS/search loop *soft* — start with a heuristic scaffold
   selector (metis-0 already has one), leave the seam to make it trained later; do not hard-freeze it.
6. **Anti-reward-hacking triple defense (Ornith):** add the frozen-judge veto on top of the sandbox's
   immutable boundary + deterministic monitors we already have.

## 6. Honest limits

- Horizon-scaling data (45K-token trajectories) is expensive to produce; at sub-1B on a Mac we run a
  *miniature* version (short trajectories, few steps) and must verify the mechanism still helps.
- Self-scaffolding at sub-1B is riskier (bigger search space, weaker model) — Ornith itself notes
  small models need tighter scaffold search spaces. Adopt as a soft, constrained seam, not day-one RL.
- These validate direction, not our specific sub-1B claim. That claim is still ours to prove or kill,
  and doc 14's pass@1-vs-round curve remains the single experiment that decides it.
