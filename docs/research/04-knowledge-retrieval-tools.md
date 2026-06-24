# Research: Knowledge/Reasoning Split, Retrieval & Tools (2024–2026)

> The keystone: justifies the whole "small reasoner + external knowledge" architecture.

## TL;DR for tiny-llm

- **Most params memorize facts, not reasoning.** Allen-Zhu "Physics of LMs 3.3": ceiling ~**2 bits
  of knowledge per parameter**. ROME/MEMIT: facts live in **MLP layers of middle blocks** and are
  surgically editable. ⇒ Move knowledge OUT of weights → the resident model can be small.
- **Retrieval collapses a >25× param gap on knowledge tasks:**
  - RETRO 7.5B + 2T-token DB ≈ **GPT-3 175B** perplexity.
  - **Atlas-11B + retrieval beats PaLM-540B** on TriviaQA (84.7 vs ~81) and GPT-3 on MMLU (65.6 vs 60).
  - kNN-LM: pure index, no training, −13.5% perplexity.
- **Tool use lets tiny models punch up:** TinyAgent **1.1B = 80% task success**, matching GPT-4-Turbo
  (79%) on a tool benchmark — via **ToolRAG** (retrieve relevant tool schemas per query). Llama-3.1-8B
  ~76% on Berkeley BFCL; gap to frontier <10pts on structured function-calling.
- **Reasoning distills into small models** (see also research 06): R1-Distill-Qwen-1.5B = 83.9 MATH-500
  (> GPT-4o 74.6). s1-32B test-time "budget forcing" beats o1-preview on competition math.
- **Retrieval infra fits cheap hardware:** embeddings tiny + fast on CPU; **DiskANN searches 1B
  vectors from SSD on 64GB RAM at <3ms, 95% recall** → disk-based ANN is the pattern for our 4GB box.

## Knowledge vs reasoning (the architectural premise)
- 2 bits/param ceiling, holds under int8 (arXiv:2404.05405). 7B ⇒ ~14Gbit budget.
- ROME/MEMIT: edit thousands of facts by touching only MLP weights → facts ≈ key-value memories
  in MLPs; attention does in-context reasoning. (arXiv:2202.05262, memit.baulab.info)

## RAG can replace parameters
| System | Params | Retrieval | Task | Score |
|---|---|---|---|---|
| GPT-3 | 175B | no | Pile ppl | baseline |
| RETRO | 7.5B | 2T DB | Pile ppl | ≈ GPT-3 |
| PaLM | 540B | no | TriviaQA-64 | ~81 |
| Atlas | 11B | yes | TriviaQA-64 | **84.7** |
| GPT-3 | 175B | no | MMLU | ~60 |
| Atlas | 11B | yes | MMLU | **65.6** |
- Smaller open models benefit MORE from RAG (weak long-context). "Lost in the middle": put retrieved
  chunks at prompt start/end. RAG also dodges the U-shaped long-context degradation.

## Retrieval infrastructure on cheap hardware
- Embedding models (CPU): all-MiniLM-L6-v2 22M/80MB, **14k sentences/s**, MTEB ~56; nomic-embed 137M
  MTEB 62; query embed <5ms. The embedder is NOT the bottleneck.
- Vector index:
  - FAISS flat 1M×768 f32 ≈ 3GB; IVF+PQ compresses Wikipedia 49GB→9.2GB (5×).
  - HNSW: ~1ms but must be in RAM (degrades 10× if it spills).
  - **DiskANN: 1B vectors from SSD, 64GB RAM, >5000 QPS @ <3ms, 95% recall@1.** SSD-resident →
    perfect for tiny-RAM boxes. We adapt this: index on disk, tiny RAM footprint.
- End-to-end retrieval overhead ≈ 5ms; LLM generation dominates total latency.

## Tool use / function calling
- TinyAgent (arXiv:2409.00608): 1.1B→80.06%, 7B→84.95%, GPT-4-Turbo 79.08%. ToolRAG is the trick.
- BFCL: Llama-3.1-8B ~76%; schema-enforced SLMs match frontier on structured calls.
- Coding agents (SWE-bench Verified): **7-8B = 19-23% vs frontier 77%** — gap still LARGE here.

## KV compression for cheap long context
- SnapKV: 92% KV compression @ negligible loss; 3.6× faster decode, 8.2× mem at 16K (Mistral-7B).
- Mamba/SSM hybrids: 8× faster gen, constant memory, but weak on exact retrieval/ICL/MMLU.

## Where the gap PERSISTS (be honest)
| Capability | Small | Frontier |
|---|---|---|
| SWE-bench agentic coding | 19-23% (7B) | 77% |
| GPQA Diamond science | ~49% (7B) | 65% |
| Hardest math AIME | 55% (7B) | ~80% (R1 full) |
| Open-ended multi-turn, novel generalist reasoning | degrades | strong |

**One-line thesis:** a 1.5–7B RL-reasoner recovers most *reasoning*; dense retrieval recovers most
*knowledge*; tools offload *exact compute/live data*. The residual gap = tasks needing many reasoning
steps fused with implicit world knowledge at once (agentic coding, frontier science).

### Sources
- Knowledge capacity https://arxiv.org/abs/2404.05405 · ROME https://arxiv.org/abs/2202.05262 · MEMIT https://memit.baulab.info/
- RETRO https://arxiv.org/pdf/2112.04426 · Atlas https://arxiv.org/abs/2208.03299 · kNN-LM https://arxiv.org/abs/1911.00172 · REALM https://arxiv.org/abs/2002.08909
- TinyAgent https://arxiv.org/abs/2409.00608 · BFCL https://gorilla.cs.berkeley.edu/leaderboard.html
- DiskANN https://suhasjs.github.io/files/diskann_neurips19.pdf · FAISS https://arxiv.org/html/2401.08281v4
- Lost-in-middle https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00638/119630 · SnapKV https://arxiv.org/pdf/2404.14469
- s1 https://arxiv.org/abs/2501.19393 · R1 https://arxiv.org/html/2501.12948v1
