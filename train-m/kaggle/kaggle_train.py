"""metis-1 — Kaggle trainer (paste this whole file into ONE Kaggle notebook cell).

Runs on Kaggle's FREE T4 GPU (~9x an M3 Pro) with ZERO load on your Mac.
Carries over every lever we measured locally:
  - Muon optimizer for block matrices (~12x data efficiency, Night 1)
  - code-tuned BPE tokenizer, ~3.3 bytes/token (Night 2)
  - RoPE + QK-norm + ReLU^2 + zero-init + untied head + logit soft-cap

Setup in Kaggle (once):
  1. kaggle.com -> Create -> New Notebook
  2. Right sidebar: Session options -> Accelerator = "GPU T4 x2"
  3. Right sidebar: Internet = ON  (needed to clone the training repos)
  4. Paste this file into a cell, Run.
  5. When done, download /kaggle/working/metis-torch.pt and metis-bpe.json
     + metis-torch.config.json from the Output panel.

Then locally (cool, CPU-only):  python serve_torch.py --weights metis-torch.pt
"""

import json, math, os, subprocess, time
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)

# ----------------------------------------------------------------- data
REPOS = ["vercel/swr", "pmndrs/zustand", "tanstack/query", "colinhacks/zod",
         "trpc/trpc", "honojs/hono", "drizzle-team/drizzle-orm",
         "microsoft/TypeScript-Website", "remix-run/react-router", "vuejs/core"]

def fetch_corpus(root="/kaggle/working/repos"):
    os.makedirs(root, exist_ok=True)
    for r in REPOS:
        d = os.path.join(root, r.replace("/", "_"))
        if not os.path.isdir(d):
            subprocess.run(["git", "clone", "--depth", "1", "-q",
                            f"https://github.com/{r}.git", d], check=False)
    texts = []
    for dp, _, fns in os.walk(root):
        if any(s in dp for s in ("node_modules", ".git", "dist", "build")):
            continue
        for fn in fns:
            if fn.endswith((".ts", ".tsx")):
                fp = os.path.join(dp, fn)
                try:
                    if os.path.getsize(fp) < 200_000:
                        texts.append(open(fp, errors="replace").read())
                except OSError:
                    pass
    print(f"corpus: {len(texts)} TS files, {sum(len(t) for t in texts)/1e6:.1f}M chars")
    return texts

def build_ids(texts, vocab=16384):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    tok = Tokenizer(models.BPE(byte_fallback=True))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tr = trainers.BpeTrainer(vocab_size=vocab, special_tokens=["<pad>", "<eos>"],
                             initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(texts, trainer=tr)
    tok.save("/kaggle/working/metis-bpe.json")
    eos = tok.token_to_id("<eos>")
    ids = []
    for t in texts:
        ids.extend(tok.encode(t).ids); ids.append(eos)
    comp = sum(len(t.encode()) for t in texts) / max(1, len(ids))
    print(f"BPE vocab {tok.get_vocab_size()}, {len(ids)/1e6:.1f}M tokens, {comp:.2f} bytes/token")
    return np.array(ids, dtype=np.uint16), tok.get_vocab_size()

# ----------------------------------------------------------------- model
def rms(x, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)

class Rope(nn.Module):
    def __init__(self, hd, base=10000):
        super().__init__()
        self.register_buffer("inv", 1.0 / base ** (torch.arange(0, hd, 2).float() / hd))
    def forward(self, x):  # x: (B,H,L,hd)
        L = x.shape[2]
        t = torch.arange(L, device=x.device).float()
        f = torch.outer(t, self.inv)
        cos, sin = f.cos()[None, None], f.sin()[None, None]
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)

class Attn(nn.Module):
    def __init__(self, d, nh):
        super().__init__(); self.nh, self.hd = nh, d // nh
        self.q = nn.Linear(d, d, bias=False); self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False); self.o = nn.Linear(d, d, bias=False)
        self.rope = Rope(self.hd)
    def forward(self, x):
        B, L, D = x.shape
        q = self.q(x).view(B, L, self.nh, self.hd).transpose(1, 2)
        k = self.k(x).view(B, L, self.nh, self.hd).transpose(1, 2)
        v = self.v(x).view(B, L, self.nh, self.hd).transpose(1, 2)
        q, k = rms(self.rope(q)), rms(self.rope(k))
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(o.transpose(1, 2).reshape(B, L, D))

class Block(nn.Module):
    def __init__(self, d, nh):
        super().__init__()
        self.n1 = nn.RMSNorm(d); self.at = Attn(d, nh)
        self.n2 = nn.RMSNorm(d)
        self.w1 = nn.Linear(d, 4 * d, bias=False); self.w2 = nn.Linear(4 * d, d, bias=False)
    def forward(self, x):
        x = x + self.at(self.n1(x))
        return x + self.w2(F.relu(self.w1(self.n2(x))).pow(2))

class GPT(nn.Module):
    def __init__(self, vocab, d, nl, nh):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList(Block(d, nh) for _ in range(nl))
        self.norm = nn.RMSNorm(d); self.head = nn.Linear(d, vocab, bias=False)
        for b in self.blocks:                       # zero-init output projections
            nn.init.zeros_(b.at.o.weight); nn.init.zeros_(b.w2.weight)
    def forward(self, idx):
        x = self.tok(idx)
        for b in self.blocks: x = b(x)
        return 15.0 * torch.tanh(self.head(self.norm(x)) / 15.0)

# ----------------------------------------------------------------- muon
def zeropower(G, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float(); X /= X.norm() + 1e-7
    tr = X.size(0) > X.size(1)
    if tr: X = X.T
    for _ in range(steps):
        A = X @ X.T; X = a * X + (b * A + c * A @ A) @ X
    return (X.T if tr else X)

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95):
        super().__init__(params, dict(lr=lr, momentum=momentum))
    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None: continue
                s = self.state[p]
                buf = s.get("buf")
                if buf is None: buf = s["buf"] = torch.zeros_like(p.grad)
                buf.mul_(g["momentum"]).add_(p.grad)
                u = zeropower(p.grad + g["momentum"] * buf)
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                p.add_(u.type_as(p), alpha=-g["lr"] * scale)

# ----------------------------------------------------------------- train
def main(dim=512, layers=8, heads=8, seq=1024, batch=32,
         steps=2000, muon_lr=0.02, adam_lr=3e-4, warmup=50, val_every=100):
    texts = fetch_corpus()
    ids, vocab = build_ids(texts)
    n_val = len(ids) // 20
    tr, va = torch.from_numpy(ids[:-n_val].astype(np.int64)), torch.from_numpy(ids[-n_val:].astype(np.int64))

    def get(split):
        d = tr if split == "t" else va
        ix = torch.randint(len(d) - seq - 1, (batch,))
        x = torch.stack([d[i:i+seq] for i in ix]); y = torch.stack([d[i+1:i+seq+1] for i in ix])
        return x.to(DEV), y.to(DEV)

    model = GPT(vocab, dim, layers, heads).to(DEV)
    N = sum(p.numel() for p in model.parameters())
    print(f"model {layers}L x {dim}d -> {N/1e6:.1f}M params on {DEV} ({torch.cuda.get_device_name() if DEV=='cuda' else 'cpu'})")

    muon_p = [p for n, p in model.named_parameters() if p.ndim == 2 and "blocks" in n]
    adam_p = [p for n, p in model.named_parameters() if not (p.ndim == 2 and "blocks" in n)]
    muon = Muon(muon_p, lr=muon_lr); adam = torch.optim.AdamW(adam_p, lr=adam_lr, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEV == "cuda"))

    @torch.no_grad()
    def val():
        model.eval(); ls = []
        for _ in range(10):
            x, y = get("v")
            with torch.autocast(DEV, torch.float16, enabled=(DEV == "cuda")):
                ls.append(F.cross_entropy(model(x).flatten(0, 1), y.flatten()).item())
        model.train(); return sum(ls) / len(ls)

    t0 = time.time()
    for step in range(1, steps + 1):
        lr = muon_lr * min(1.0, step / warmup)
        for pg in muon.param_groups: pg["lr"] = lr
        x, y = get("t")
        with torch.autocast(DEV, torch.float16, enabled=(DEV == "cuda")):
            loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        adam.zero_grad(); muon.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(adam); scaler.step(muon); scaler.update()
        if step % val_every == 0:
            el = time.time() - t0
            print(f"step {step:4d}  train {loss.item():.3f}  val {val():.3f}  "
                  f"{step*batch*seq/el/1e3:.0f}k tok/s  {el/60:.1f} min", flush=True)

    torch.save(model.state_dict(), "/kaggle/working/metis-torch.pt")
    json.dump({"vocab": vocab, "dim": dim, "layers": layers, "heads": heads, "seq": seq},
              open("/kaggle/working/metis-torch.config.json", "w"))
    print("saved metis-torch.pt + config + metis-bpe.json — download these 3 from the Output panel")

if __name__ == "__main__":
    main()
