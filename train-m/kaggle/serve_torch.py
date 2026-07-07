"""metis — local CPU server for weights trained on Kaggle.

Loads metis-torch.pt + metis-bpe.json and serves an OpenAI-compatible endpoint
on the CPU. A ~50M model decoding short completions on CPU is light — it will
NOT peg your Mac the way training did. KV-cached for snappy generation.

    pip install torch tokenizers          # CPU torch is enough
    python serve_torch.py --weights metis-torch.pt --port 8484

OpenCode already has the `metis` provider pointing at :8484, so:
    opencode run -m metis/metis-1-mvp "export function sum("
"""

import argparse, codecs, json, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch, torch.nn.functional as F
from tokenizers import Tokenizer

from kaggle_train import GPT, rms  # same model definition, no drift

MODEL_ID = "metis-1-mvp"
model = None; tok = None; CFG = {}


def _rope_at(inv, positions, hd, device):
    f = torch.outer(positions.float(), inv)
    return f.cos(), f.sin()


def _apply_rope(x, cos, sin):  # x: (B,H,L,hd)
    cos, sin = cos[None, None], sin[None, None]
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)


@torch.no_grad()
def generate(prompt, max_tokens, temperature):
    seq = CFG["seq"]
    ids = tok.encode(prompt).ids[-(seq - max_tokens - 1):]
    x = torch.tensor([ids])
    caches = [None] * len(model.blocks)

    def forward(idx, pos):
        h = model.tok(idx)
        for i, b in enumerate(model.blocks):
            hn = b.n1(h); a = b.at
            B, L, D = hn.shape
            q = a.q(hn).view(B, L, a.nh, a.hd).transpose(1, 2)
            k = a.k(hn).view(B, L, a.nh, a.hd).transpose(1, 2)
            v = a.v(hn).view(B, L, a.nh, a.hd).transpose(1, 2)
            cos, sin = _rope_at(a.rope.inv, torch.arange(pos, pos + L), a.hd, idx.device)
            q, k = rms(_apply_rope(q, cos, sin)), rms(_apply_rope(k, cos, sin))
            if caches[i] is not None:
                k = torch.cat([caches[i][0], k], 2); v = torch.cat([caches[i][1], v], 2)
            caches[i] = (k, v)
            causal = caches[i][0].shape[2] == L  # only mask during prefill
            o = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
            h = h + a.o(o.transpose(1, 2).reshape(B, L, D))
            h = h + b.w2(F.relu(b.w1(b.n2(h))).pow(2))
        return 15.0 * torch.tanh(model.head(model.norm(h)) / 15.0)

    logits = forward(x, 0)[0, -1]
    dec = codecs.getincrementaldecoder("utf-8")("replace")
    pos = len(ids)
    for _ in range(max_tokens):
        if temperature <= 0:
            nxt = int(logits.argmax())
        else:
            p = F.softmax(logits / temperature, -1)
            tk = torch.topk(p, 40); nxt = int(tk.indices[torch.multinomial(tk.values, 1)])
        piece = tok.id_to_token(nxt) or ""
        txt = dec.decode(bytes(tok.decoder.decode([piece]).encode())) if False else _detok(nxt)
        if txt:
            yield txt
        if pos >= seq - 1:
            break
        logits = forward(torch.tensor([[nxt]]), pos)[0, -1]
        pos += 1


_buf = []
def _detok(tid):
    """Decode incrementally through the tokenizer's byte-level decoder."""
    _buf.append(tid)
    s = tok.decode(_buf)
    return s if s and not s.endswith("�") else ""


def prompt_from_messages(messages):
    parts = []
    for m in messages:
        c = m.get("content", "")
        if isinstance(c, list): c = "".join(p.get("text", "") for p in c)
        parts.append(c)
    return "\n".join(parts)


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def _j(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            self._j(200, {"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]})
        else: self._j(404, {"error": "nf"})
    def do_POST(self):
        if not self.path.rstrip("/").endswith("/chat/completions"):
            return self._j(404, {"error": "nf"})
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = prompt_from_messages(req.get("messages", []))
        mx = min(int(req.get("max_tokens") or 256), 512); temp = float(req.get("temperature", 0.8))
        _buf.clear()
        rid = f"chatcmpl-{uuid.uuid4().hex[:12]}"; created = int(time.time())
        if req.get("stream"):
            self.send_response(200); self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked"); self.end_headers()
            def ch(pl):
                d = f"data: {json.dumps(pl)}\n\n".encode()
                self.wfile.write(f"{len(d):x}\r\n".encode() + d + b"\r\n"); self.wfile.flush()
            base = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": MODEL_ID}
            ch({**base, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
            for frag in generate(prompt, mx, temp):
                ch({**base, "choices": [{"index": 0, "delta": {"content": frag}, "finish_reason": None}]})
            ch({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
            done = b"data: [DONE]\n\n"
            self.wfile.write(f"{len(done):x}\r\n".encode() + done + b"\r\n"); self.wfile.write(b"0\r\n\r\n"); self.wfile.flush()
        else:
            text = "".join(generate(prompt, mx, temp))
            self._j(200, {"id": rid, "object": "chat.completion", "created": created, "model": MODEL_ID,
                          "choices": [{"index": 0, "finish_reason": "stop",
                                       "message": {"role": "assistant", "content": text}}],
                          "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}})


def main():
    global model, tok, CFG
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="metis-torch.pt")
    ap.add_argument("--tokenizer", default="metis-bpe.json")
    ap.add_argument("--port", type=int, default=8484)
    args = ap.parse_args()
    torch.set_num_threads(max(1, (torch.get_num_threads() or 4) // 2))  # stay gentle on the Mac
    CFG.update(json.load(open(args.weights.replace(".pt", ".config.json"))))
    tok = Tokenizer.from_file(args.tokenizer)
    model = GPT(CFG["vocab"], CFG["dim"], CFG["layers"], CFG["heads"])
    model.load_state_dict(torch.load(args.weights, map_location="cpu")); model.eval()
    print(f"metis (torch/CPU) serving {MODEL_ID} on http://127.0.0.1:{args.port}/v1")
    ThreadingHTTPServer(("127.0.0.1", args.port), H).serve_forever()


if __name__ == "__main__":
    main()
