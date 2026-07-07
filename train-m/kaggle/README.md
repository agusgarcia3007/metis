# metis-1 on Kaggle — free GPU, zero load on your Mac

Training on the MacBook was killing it (18 GB shared between the OS and the trainer).
This moves training to Kaggle's **free T4 GPU** (~9× the M3 Pro's GPU) and leaves the Mac
only for editing code and serving the finished 50 MB model on CPU.

Every lever we measured locally comes along: **Muon** (~12× data efficiency), a **code BPE
tokenizer** (~3.3 bytes/token), and the speedrun architecture (RoPE, QK-norm, ReLU², zero-init,
untied head, logit soft-cap).

## Train (once, in the browser — nothing runs on your Mac)

1. Go to **kaggle.com → Create → New Notebook** (free account, no card).
2. Right sidebar → **Session options → Accelerator = "GPU T4 x2"**.
3. Right sidebar → **Internet = ON** (the notebook clones the training repos).
4. Paste all of **`kaggle_train.py`** into one cell and **Run**.
   - ~2,000 steps of a 50M model finishes in roughly **15–25 min** on the T4.
   - Watch `val` fall in the logs; it should pass our local best (~0.7) quickly.
5. In the **Output** panel (right side), download three files:
   - `metis-torch.pt` — the weights
   - `metis-torch.config.json` — the shape
   - `metis-bpe.json` — the tokenizer

Put those three next to `serve_torch.py`.

## Serve (locally, cool, CPU-only)

```sh
pip install torch tokenizers          # CPU torch is plenty for a 50M model
python serve_torch.py --weights metis-torch.pt
```

OpenCode already has the `metis` provider on `:8484`, so:

```sh
opencode run -m metis/metis-1-mvp "export function sum(a: number, b: number)"
```

The server caps its own thread count and runs on CPU — inference of a small model is light and
will not choke the machine the way training did.

## Why this is the cheap/fast answer

| | MacBook (before) | Kaggle T4 (now) |
|---|---|---|
| cost | your hardware, thermal wear | **free**, 30 h/week |
| speed | ~7 TFLOPS, fp32 | ~65 TFLOPS fp16 → **~9×** |
| load on your Mac | 100%, machine unusable | **0%** |
| session | until it overheats | 9–12 h, background exec |

Same code, same trainer, same trucos — just not on your machine.
