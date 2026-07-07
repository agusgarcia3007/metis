"""metis night1 server — OpenAI-compatible endpoint for the Muon-trained trunk-let.

Same protocol as night0/serve.py, adapted to the night1 architecture
(RoPE + QK-norm + ReLU^2 + soft-cap) with a KV cache that tracks RoPE offsets.

Usage:
    python serve.py --weights metis-n1.safetensors --port 8484
"""

import argparse
import codecs
import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from train import TrunkLet, rms

MODEL_ID = "metis-1-mvp"
model = None
CFG = {}


def sample_next(logits, temperature: float, top_k: int) -> int:
    if temperature <= 0:
        return int(mx.argmax(logits).item())
    probs = np.array(mx.softmax(logits.astype(mx.float32) / temperature))
    idx = np.argsort(probs)[-top_k:]
    p = probs[idx] / probs[idx].sum()
    return int(np.random.choice(idx, p=p))


def _split_heads(t, n_heads, hd):
    B, L, D = t.shape
    return t.reshape(B, L, n_heads, hd).transpose(0, 2, 1, 3)


def _block_step(b, x, pos, k_cache, v_cache):
    """One-position forward through a block, RoPE applied at absolute `pos`."""
    h = b.norm1(x)
    a = b.attn
    q = _split_heads(a.q_proj(h), a.n_heads, a.hd)
    k = _split_heads(a.k_proj(h), a.n_heads, a.hd)
    v = _split_heads(a.v_proj(h), a.n_heads, a.hd)
    q, k = rms(a.rope(q, offset=pos)), rms(a.rope(k, offset=pos))
    k_cache = mx.concatenate([k_cache, k], axis=2)
    v_cache = mx.concatenate([v_cache, v], axis=2)
    o = mx.fast.scaled_dot_product_attention(q, k_cache, v_cache, scale=a.hd ** -0.5, mask=None)
    B, nh, L, hd = o.shape
    x = x + a.o_proj(o.transpose(0, 2, 1, 3).reshape(B, L, nh * hd))
    return x + b.mlp(b.norm2(x)), k_cache, v_cache


def generate(prompt: str, max_tokens: int, temperature: float):
    seq = CFG["seq"]
    ids = list(prompt.encode("utf-8", errors="replace"))[-(seq - max_tokens - 1):]
    decoder = codecs.getincrementaldecoder("utf-8")("replace")

    # prefill: full forward, capture caches (RoPE offset 0)
    mask = nn.MultiHeadAttention.create_additive_causal_mask(len(ids)).astype(mx.bfloat16)
    x = model.tok(mx.array(np.array(ids, dtype=np.uint8)[None, :]))
    caches = []
    for b in model.blocks:
        h = b.norm1(x)
        a = b.attn
        q = _split_heads(a.q_proj(h), a.n_heads, a.hd)
        k = _split_heads(a.k_proj(h), a.n_heads, a.hd)
        v = _split_heads(a.v_proj(h), a.n_heads, a.hd)
        q, k = rms(a.rope(q)), rms(a.rope(k))
        caches.append([k, v])
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=a.hd ** -0.5, mask=mask)
        B, nh, L, hd = o.shape
        x = x + a.o_proj(o.transpose(0, 2, 1, 3).reshape(B, L, nh * hd))
        x = x + b.mlp(b.norm2(x))
    logits = 15.0 * mx.tanh(model.head(model.norm(x))[0, -1] / 15.0)
    mx.eval(logits)

    pos = len(ids)
    for _ in range(max_tokens):
        nxt = sample_next(logits, temperature, top_k=40)
        text = decoder.decode(bytes([nxt]))
        if text:
            yield text
        if pos >= seq - 1:
            break
        x = model.tok(mx.array([[nxt]]))
        for i, b in enumerate(model.blocks):
            x, caches[i][0], caches[i][1] = _block_step(b, x, pos, *caches[i])
        logits = 15.0 * mx.tanh(model.head(model.norm(x))[0, -1] / 15.0)
        mx.eval(logits)
        pos += 1


def prompt_from_messages(messages) -> str:
    parts = []
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, list):
            content = "".join(p.get("text", "") for p in content)
        parts.append(content)
    return "\n".join(parts)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        print(f"[metis] {args[0] if args else ''}")

    def _json(self, code: int, obj: dict):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") in ("/v1/models", "/models"):
            self._json(200, {"object": "list", "data": [
                {"id": MODEL_ID, "object": "model", "owned_by": "metis"}]})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._json(404, {"error": "not found"})
            return
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = prompt_from_messages(req.get("messages", []))
        max_tokens = min(int(req.get("max_tokens") or 256), 512)
        temperature = float(req.get("temperature", 0.8))
        rid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        if req.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            def chunk(payload: dict):
                data = f"data: {json.dumps(payload)}\n\n".encode()
                self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
                self.wfile.flush()

            base = {"id": rid, "object": "chat.completion.chunk",
                    "created": created, "model": MODEL_ID}
            chunk({**base, "choices": [{"index": 0, "delta": {"role": "assistant"},
                                        "finish_reason": None}]})
            for frag in generate(prompt, max_tokens, temperature):
                chunk({**base, "choices": [{"index": 0, "delta": {"content": frag},
                                            "finish_reason": None}]})
            chunk({**base, "choices": [{"index": 0, "delta": {},
                                        "finish_reason": "stop"}]})
            done = b"data: [DONE]\n\n"
            self.wfile.write(f"{len(done):x}\r\n".encode() + done + b"\r\n")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        else:
            text = "".join(generate(prompt, max_tokens, temperature))
            self._json(200, {
                "id": rid, "object": "chat.completion", "created": created,
                "model": MODEL_ID,
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": text}}],
                "usage": {"prompt_tokens": len(prompt.encode()),
                          "completion_tokens": len(text.encode()),
                          "total_tokens": len(prompt.encode()) + len(text.encode())},
            })


def main():
    global model, CFG
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--port", type=int, default=8484)
    args = ap.parse_args()

    CFG.update(json.load(open(args.weights.replace(".safetensors", ".config.json"))))
    model = TrunkLet(CFG["vocab"], CFG["dim"], CFG["layers"], CFG["heads"])
    model.load_weights(args.weights)
    model.apply(lambda p: p.astype(mx.bfloat16))
    mx.eval(model.parameters())
    model.eval()
    print(f"metis night1 serving {MODEL_ID} on http://127.0.0.1:{args.port}/v1")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
