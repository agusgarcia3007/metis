"""metis-1 — edit-native repair: output ONE line, not the whole file (VERA-R §4).

Two whole-file experiments proved a 14M byte model can't faithfully copy a multi-
line file while editing one token — it blends input with training patterns (valid
TS, wrong function). The fix the ruler pointed to: don't regenerate the file.
The compiler diagnostic already gives the line number; the model only has to
produce the ONE corrected line, which we splice back in.

This collapses the task from "copy N lines + edit" to "emit 1 correct line". Same
data, same warm-started 14M Cortex, new output contract. train + eval in one file.

    python edit_repair.py train 500
    python edit_repair.py eval
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "night1"))
from train import TrunkLet, zeropower, is_muon_key

HERE = Path(__file__).parent
NIGHT2 = HERE.parent / "night2"
DATA = HERE / "data" / "transitions.jsonl"
WEIGHTS = HERE / "metis-edit.safetensors"
LINE_RE = re.compile(r"\((\d+),\d+\)")


def changed_line(broken: str, gold: str):
    """Return (0-based index in broken, corrected line text) for the single edit."""
    b, g = broken.splitlines(), gold.splitlines()
    for i in range(min(len(b), len(g))):
        if b[i] != g[i]:
            return i, g[i]
    return None, None


def build_examples(seq_len=384):
    """sequence: <state>broken</state><diag>...</diag><fix>corrected line</fix>."""
    seqs, wmasks = [], []
    for l in open(DATA):
        r = json.loads(l)
        idx, fixline = changed_line(r["broken"], r["gold"])
        if idx is None:
            continue
        s = (f"<state>\n{r['broken']}\n</state>\n"
             f"<diagnostic>\n{r['diagnostic'][:300]}\n</diagnostic>\n"
             f"<fix>\n{fixline}\n</fix>\n")
        b = s.encode()[:seq_len]
        arr = np.frombuffer(b, np.uint8).astype(np.int64)
        if len(arr) < 16:
            continue
        w = np.ones(len(arr), np.float32)
        fi = s.encode().find(b"<fix>")
        if 0 <= fi < len(arr):
            w[fi:] = 4.0                       # the corrected line is the whole skill
        seqs.append(arr); wmasks.append(w)
    return seqs, wmasks


def train(steps=500):
    cfg = json.load(open(NIGHT2 / "metis-fim.config.json"))
    seq_len, batch = 384, 8
    model = TrunkLet(cfg["vocab"], cfg["dim"], cfg["layers"], cfg["heads"])
    model.load_weights(str(NIGHT2 / "metis-fim.safetensors"))
    model.apply(lambda p: p.astype(mx.bfloat16)); mx.eval(model.parameters())
    seqs, wmasks = build_examples(seq_len)
    print(f"edit trainer: {len(seqs)} line-edit examples, warm from FIM, {steps} steps")

    mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len).astype(mx.bfloat16)
    adam = optim.AdamW(learning_rate=3e-4, weight_decay=0.01)
    mom = {k: mx.zeros_like(v) for k, v in tree_flatten(model.trainable_parameters()) if is_muon_key(k)}
    rng = np.random.default_rng(0)

    def batch_iter():
        while True:
            idx = rng.integers(0, len(seqs), size=batch)
            xs, ys, ws = [], [], []
            for i in idx:
                a, w = seqs[i], wmasks[i]
                if len(a) < seq_len + 1:
                    pad = seq_len + 1 - len(a)
                    a = np.concatenate([a, np.zeros(pad, np.int64)])
                    w = np.concatenate([w, np.zeros(pad, np.float32)])
                xs.append(a[:seq_len]); ys.append(a[1:seq_len + 1]); ws.append(w[1:seq_len + 1])
            yield mx.array(np.stack(xs)), mx.array(np.stack(ys)), mx.array(np.stack(ws))

    def loss_fn(m, x, y, w):
        ce = nn.losses.cross_entropy(m(x, mask).astype(mx.float32), y, reduction="none")
        return (ce * w).sum() / (w.sum() + 1e-6)

    lg = nn.value_and_grad(model, loss_fn)

    def step_fn(x, y, w, mlr):
        loss, grads = lg(model, x, y, w)
        flat = tree_flatten(grads)
        adam.update(model, tree_unflatten([(k, g) for k, g in flat if not is_muon_key(k)]))
        params = dict(tree_flatten(model.trainable_parameters()))
        upd = []
        for k, g in flat:
            if not is_muon_key(k):
                continue
            buf = 0.95 * mom[k] + g; mom[k] = buf
            u = zeropower((g + 0.95 * buf).astype(mx.bfloat16))
            sc = max(1.0, g.shape[0] / g.shape[1]) ** 0.5
            upd.append((k, params[k] - (mlr * sc) * u.astype(params[k].dtype)))
        model.update(tree_unflatten(upd))
        return loss

    gen = batch_iter(); t0 = time.time()
    for step in range(1, steps + 1):
        loss = step_fn(*next(gen), mx.array(0.01 * min(1.0, step / 40)))
        mx.eval(model.state, adam.state, mom, loss)
        if step % 100 == 0:
            print(f"step {step:4d}  loss {loss.item():.3f}  {(time.time()-t0)/60:.1f}m", flush=True)
    model.save_weights(str(WEIGHTS))
    json.dump(cfg, open(str(WEIGHTS).replace(".safetensors", ".config.json"), "w"))
    print(f"saved -> {WEIGHTS}")


def make_edit_generator(temperature=0.4, max_new=60):
    cfg = json.load(open(str(WEIGHTS).replace(".safetensors", ".config.json")))
    model = TrunkLet(cfg["vocab"], cfg["dim"], cfg["layers"], cfg["heads"])
    model.load_weights(str(WEIGHTS)); model.apply(lambda p: p.astype(mx.bfloat16)); model.eval()
    seq = cfg["seq"]; rng = np.random.default_rng(0)

    def gen(task, i):
        m = LINE_RE.search(task.diagnostic)
        lineno = int(m.group(1)) if m else 1          # 1-based line the compiler flags
        prompt = (f"<state>\n{task.broken}\n</state>\n"
                  f"<diagnostic>\n{task.diagnostic.strip()[:300]}\n</diagnostic>\n<fix>\n")
        ids = list(np.frombuffer(prompt.encode(), np.uint8).astype(np.int64))[-(seq - max_new - 1):]
        out = []
        for _ in range(max_new):
            c = mx.array(np.array(ids, np.int64)[None])
            mk = nn.MultiHeadAttention.create_additive_causal_mask(len(ids)).astype(mx.bfloat16)
            lg = model(c, mk)[0, -1].astype(mx.float32)
            nxt = int(mx.argmax(lg)) if temperature <= 0 else \
                int(rng.choice(len(lg), p=(lambda p: p/p.sum())(np.array(mx.softmax(lg/temperature)))))
            if nxt == ord("\n") or nxt >= 256:
                break
            out.append(nxt); ids.append(nxt)
        fixline = bytes(out).decode("utf-8", "replace")
        # splice the produced line into the broken file at the flagged line number
        lines = task.broken.splitlines()
        if 1 <= lineno <= len(lines):
            lines[lineno - 1] = fixline
        return "\n".join(lines) + "\n"

    return gen


def evaluate():
    from breaker import make_transitions
    from passk import eval_pass_at_k, gold_generator
    gold = (HERE / "fixture/src/calc.ts").read_text()
    tasks = make_transitions(gold)
    g = make_edit_generator()
    print("edit-native Cortex (outputs ONE line), pass@k on HELD-OUT calc.ts:\n")
    for k in (1, 4, 8):
        r = eval_pass_at_k(tasks, g, k=k)
        print(f"  metis-edit-14M  pass@{k}={r.pass_at_k}  mean_best_score={r.mean_best_score}")
    print(f"\n  (reference) gold  pass@1={eval_pass_at_k(tasks, gold_generator, 1).pass_at_k}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "eval"
    if cmd == "train":
        train(int(sys.argv[2]) if len(sys.argv) > 2 else 500)
    else:
        evaluate()
