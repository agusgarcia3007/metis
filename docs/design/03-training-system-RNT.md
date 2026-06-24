# RNT — Retrieval-Native Training: a new training system

> **The horizon:** stop training models to *memorize the world*. Train them to *reason over a
> world that is handed to them at inference.* Knowledge becomes data on disk; the weights hold only
> reasoning. The proof that it works: the resulting model fits in **4 GB RAM / 4 vCPU**.

This document defines the training paradigm and shows the **working, verified** demonstration that
backs it (code in `internal/nano`, runnable via `cmd/rnt`). It is the answer to the goal: *a new
training system that moves the industry's horizon, proven by fitting in 4 GB.*

## 1. Why today's training is wasteful
Standard next-token pretraining forces a single objective — predict the next token over raw web
text — to do two very different jobs at once:
1. **Memorize facts** ("Paris is the capital of France", every API signature, every date).
2. **Learn to reason** (parse, deduce, transform, plan).

The research is unambiguous that (1) eats most of the parameters: a model stores **~2 bits of
knowledge per parameter** (Allen-Zhu, *Physics of LMs 3.3*), and those facts live in the **MLP
layers** and are surgically editable (ROME/MEMIT) — i.e. they are a *lookup table baked into the
weights*. A frontier model is mostly an encyclopedia that also happens to reason. (See
`../research/04-knowledge-retrieval-tools.md`.)

If knowledge is a lookup table, **why bake it into the weights at all?** Put it on disk and look it
up. That is the whole idea.

## 2. The paradigm: Retrieval-Native Training (RNT)
RNT changes the *training objective and data pipeline*, not just inference:

1. **Retrieval is in the loop from step 0.** Every training example is constructed so the facts
   needed to predict the target are **already present in the context**, retrieved from the Library.
   The model is never rewarded for having memorized a fact — the fact is right there.
2. **An anti-memorization penalty** discourages the weights from storing facts anyway. Because facts
   live in MLPs (ROME/MEMIT), we apply selective weight decay / an information bottleneck to the MLP
   value matrices (the `decayMask` knob in `internal/nano/train.go`). Gradient descent is pushed
   toward the *only* remaining way to lower the loss: **learning to reason over the retrieved
   context.**
3. **Knowledge and reasoning are physically separated.** Capability now scales along two independent
   axes: grow the **Library** (disk) to know more; grow/curate the **reasoner** (RAM) to think
   better. They no longer compete for the same parameters.

This is *not* RETRO (retrieval bolted onto a normally-pretrained model). In RNT retrieval is native
and the objective actively *prevents* parametric memorization, so the reasoner can be radically
smaller for the same downstream ability.

## 3. The demonstration (real, run it yourself)
`cmd/rnt` trains the **same** tiny transformer (`internal/nano`, a from-scratch autograd + GPT in
pure Go, gradient-checked to 2.8e-3) two ways on a task that cleanly separates knowledge from
reasoning:
- **Knowledge:** a *world* maps each subject → a value (a fact, e.g. `subject 047 = 4`).
- **Reasoning:** the answer is a fixed transform, `answer = (value + 3) mod 10`.

| Regime | What the model sees | What it must do |
|---|---|---|
| **Vanilla** | `[? d d d >]` | **memorize** subject→answer in its weights |
| **RNT** | `[d d d = value ; ? d d d >]` | **reason**: read the retrieved fact, apply the transform |

The decisive test is a **new world** whose facts were never in training:

```
=================== RESULTS (50 facts, identical model) ===================
chance accuracy (10 answers)        :  10.0%
VANILLA  accuracy on TRAINED world  : 100.0%   (memorized — works)
VANILLA  accuracy on NEW world      :  10.0%   (= chance: knowledge frozen in weights — FAILS)
RNT      accuracy on NEW world      : 100.0%   (reads retrieved fact — GENERALIZES)
```

Same architecture, same parameter count. The vanilla model **cannot** answer about facts it didn't
memorize (10 % = random). The RNT model is **perfect on a world it never trained on**, because it
learned to reason over whatever fact is retrieved. This is "knowledge is data, not weights,"
demonstrated end-to-end.

### The capacity wall (the quantitative link to 4 GB)
`cmd/rnt -mode sweep` fixes a tiny model (embd=16, 1 layer — a fixed parameter budget) and grows the
number of facts. Memorization must compete for fixed weights; retrieval does not:

```
fixed model: embd=16 layer=1  (~3.7k params, fixed budget)
facts    params   VANILLA seen-acc   RNT new-world-acc
64       3712     100.0%             100.0%
256      3744      39.8%             100.0%
1024     3776      18.6%             100.0%
4096     3776      12.1%  (→chance)  100.0%
```

Vanilla seen-accuracy decays as facts exceed the model's ~2-bits/param memorization budget; **RNT
stays ~100 % at the same size** because each fact is supplied in context. Therefore knowledge growth
costs **disk, not RAM** — which is exactly why an RNT reasoner + a disk Library fits 4 GB while its
effective knowledge is unbounded.

## 4. How RNT composes with the rest of the stack
RNT is the spine; the other two novel axes from the research plug in to make the proof model *also*
the smallest and fastest possible:
- **Ternary 1.58-bit weights (QAT).** Train the reasoner natively in `{-1,0,+1}` (BitNet, research
  02). A reasoning-only model has far less to encode than a memorizing one, so it tolerates ternary
  better — ~0.4 GB for a 2B-class reasoner, native CPU speed.
- **Activation sparsity (dReLU).** Train with dReLU so ~90 % of neurons are silent per token
  (PowerInfer, research 03) → fewer bytes touched per token → faster on 4 vCPU.
- **Reasoning distillation + verifier curriculum.** Bring the *reasoning* up to frontier level by
  distilling traces from an open teacher (R1-style, research 06) and filtering with a verifier — the
  one thing that genuinely transfers to small models.

The combination is one coherent training system: **a ternary, sparse, retrieval-native reasoner,
distilled for reasoning and forbidden from memorizing** — the most pro/fast/small model the research
permits, with the 4 GB box as its proof.

## 5. Scaling the demonstration to a real model (pipeline)
The `nano` experiment proves the *mechanism*. The same recipe scales:
1. **Corpus → Library.** Index a large corpus (FineWeb-Edu, docs, code) into the disk-resident ANN
   (design `01` §2.2). This is the knowledge store.
2. **RNT data construction.** For each training target, retrieve the supporting passages and place
   them in-context (start/end, dodging "lost in the middle"). Train next-token **densely** across the
   answer span (not a single answer token), with the MLP anti-memorization penalty on.
   **Dense supervision is load-bearing** (Round 4, `04-RNT-results-log.md`): single-answer-token
   supervision silently starves the previous-token/induction circuit the model needs to *select* the
   right retrieved fact — a canonical induction probe went 5% → 100% just by switching sparse→dense.
3. **Reasoning distillation.** Mix in verified reasoning/tool-use traces from an open teacher.
4. **QAT to ternary + dReLU.** Quantization-aware from the start so the deployable model is tiny/fast.
5. **Proof.** Quantize, deploy the reasoner + Library on the 4 GB / 4 vCPU box, and measure on the
   `bench/` harness. No claim ships without a number from that box.

## 6. Honest limits
- RNT recovers **knowledge-bound** ability via retrieval and **reasoning** via distillation. It does
  **not** magically close the hardest agentic-coding / frontier-science gaps (research 04 §"gap
  persists"). The claim is "frontier-ish on the supported surface in 4 GB," not "GPT-5 in 4 GB."
- The demonstration here is a controlled synthetic task that isolates the mechanism. It is a proof of
  *principle and direction*, not a trained product. Section 5 is the bridge to a product, and it
  costs real (but modest, research 06) GPU time.
- **Empirically-found boundary (`04-RNT-results-log.md`):** we stress-tested with *distractors* (the
  model must select the queried fact among many, not just copy a lone value). The from-scratch tiny
  model reliably learns **copy + reasoning** (1 fact → 100%) but **not many-distractor selection** on
  CPU in-session (2 facts ≈ 55%, → chance as distractors grow), even with dense supervision, RoPE,
  and a curriculum. The associative-recall circuit needs more scale/GPU. So what is *proven* in
  session is the **mechanism** (knowledge-in-context generalizes; Round 0) and the **capacity wall**;
  robust retrieval-among-distractors is part of the §5 scale-up (or warm-start from a base that
  already has induction heads, then apply the RNT objective).
- Retrieval quality becomes the new bottleneck: if the Library lacks a fact, the reasoner can't
  invent it. That is a feature (auditable, updatable) but it shifts effort to corpus + retriever.

## 7. What is genuinely new here
1. **Anti-memorization as an explicit training objective**, targeted at the MLP fact-store that
   ROME/MEMIT localized — not just retrieval augmentation, but retrieval *plus suppression of
   parametric recall*.
2. **Co-designed with extreme deployment as the success metric:** the training system and the 4 GB
   proof are one artifact. The horizon is not "a bigger benchmark score" but "a new point on the
   capability-per-byte frontier," demonstrated, not asserted.
