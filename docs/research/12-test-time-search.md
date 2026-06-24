# Research: Test-time search vs scale, and compound systems (2024–2026)

> Does "small model + search + verifier" actually reach big-model quality? Yes — on a bounded surface
> — and crucially the bounds (coverage, latency, verifier ceiling) decide our design.

## TL;DR
- **The exact architecture is published and BEATS frontier on grounded/structured tasks:**
  - **MCTS-RAG (Llama-3.1-8B + MCTS + search) > GPT-4o**: GPQA **71.3 vs 54.9**, ComplexWebQA 67.3 vs 59.4.
  - **Search-R1 (Qwen-7B + RL + search)**: Musique EM +238% over RAG.
  - **MinionS (3B local + cloud decomposition)**: **93.4% of GPT-4o at 16.6% of the cost**.
  - **Speculative RAG (drafter+verifier)**: **+13% accuracy AND −51% latency at once**.
  - RankRAG-8B > GPT-4-0613 on NQ/PopQA. (Caveat: most "beats GPT-4" = GPT-4-0613/2023, not o3/GPT-4o.)
- **Search scaling is real:** 5 samples from a cheap model beat 1× GPT-4o at 3.25× lower cost
  (SWE-bench). 7B + tree-search = 34B + sampling at 8× less compute. rStar-Math: 7B 58.8→90.0% MATH.
- **The unifying principle is NOT yet formalized** in any single paper (Duo-LLM / MoD are closest) —
  "allocate compute ∝ output unpredictability" across spec-decode/MoD/early-exit/MoE/KV-reuse/RAG is an
  *original framing opportunity*, not prior art.

## The four walls (each one shapes our design)
1. **Coverage floor (hard).** Search only selects among what the model CAN generate. If pass@1≈0, no N
   helps — *mathematically proven* (Schaeffer, ICLR'25, coverage exponent set by the success-prob tail).
   RL raises pass@1 but **does not expand coverage** (Limit-of-RLVR: 23.3% of AIME perma-zero for 7B;
   RL-only-solved = 0.0%). → search amplifies competence, never conjures it.
2. **Latency wall (decisive for us).** N=128 ≈ **134 s** wall-clock; reasoning models hit 127 s TTFT.
   **Interactive use (TTFT<500ms) is categorically incompatible with big N.** On a 4-vCPU CPU box this
   forces **tiny N** → verify-then-maybe-search, not always-best-of-N.
3. **Verifier ceiling.** Even a 72B RM sits 7.7 pp below oracle (10+ on hard); selection plateaus at
   N≈100 while coverage keeps rising (52-pp oracle-vs-consensus gap on OlymMATH unreachable). Top PRM
   **collapses 78.3→37.3%** on the frontier tail — worst exactly where help is needed.
4. **No-verifier domains: no path.** BoN fails entirely on subjective tasks; self-correction is net
   negative; LLM judges recover only ~21% of the oracle gain.

## What this forces for Metis (interactive, CPU, 4 GB)
- **NO big-N search.** Latency kills it. Use **verify-then-rarely-search** (N≤~3 only on verify-fail).
- The quality win is therefore **trustworthiness (grounded verify + abstain)** + a *modest* select gain,
  on the **groundable** surface — not a math-olympiad search engine.
- Compound-system gains (MCTS-RAG etc.) are real but spend more compute/cloud than our budget; we take
  the cheap end (Speculative-RAG's "verify, don't always search" + MinionS-style decomposition if a
  cloud tier is ever allowed).

## Cost crossover (the one number to remember)
5 samples from a cheap model > 1 GPT-4o at **3.25× lower cost** (SWE-bench Lite) — the small+search
crossover is real **when a verifier/checker exists and latency is amortizable (batch)**, not interactive.

### Sources
- Large Language Monkeys https://arxiv.org/abs/2407.21787 · Snell TTS https://arxiv.org/abs/2408.03314 · Wu inference-scaling https://arxiv.org/abs/2408.00724
- Limit-of-RLVR https://arxiv.org/abs/2504.13837 · coverage power-law https://arxiv.org/abs/2502.17578 · FrontierMath https://arxiv.org/abs/2411.04872
- MCTS-RAG https://arxiv.org/abs/2503.20757 · Search-R1 https://arxiv.org/abs/2503.09516 · MinionS https://arxiv.org/abs/2502.15964 · Speculative-RAG https://arxiv.org/abs/2407.08223 · RankRAG https://arxiv.org/abs/2407.02485
- rStar-Math https://arxiv.org/abs/2501.04519 · ToT https://arxiv.org/abs/2305.10601 · Duo-LLM https://arxiv.org/abs/2410.10846 · MoD https://arxiv.org/abs/2404.02258
- Inference-scaling-FLaws (optimal K) https://arxiv.org/abs/2411.17501 · false-positives https://arxiv.org/abs/2502.06217 · compound-AI (BAIR) https://bair.berkeley.edu/blog/2024/02/18/compound-ai-systems/
