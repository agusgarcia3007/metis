#!/usr/bin/env python3
"""
Metis NLI sidecar — dedicated entailment verifier for Phase 2.

Uses typeform/distilbert-base-uncased-mnli (67M params, ~267MB) to check whether a
CLAIM is entailed by a body of EVIDENCE. This is the Phase 2 specialized verifier:
a small model trained exclusively on Natural Language Inference (MNLI) that is
specialized for the exact task — unlike a 1.7B generalist doing NLI via instruction.

Label ordering for this model: ENTAILMENT=0, NEUTRAL=1, CONTRADICTION=2

POST /verify  {"claim": "...", "evidence": "..."}
           -> {"verdict": "SUPPORTED"|"UNSUPPORTED"|"UNCERTAIN",
               "scores": {"entailment": f, "neutral": f, "contradiction": f}}
GET  /healthz -> 200 "ok"
"""
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = os.environ.get("NLI_MODEL", "typeform/distilbert-base-uncased-mnli")
PORT = int(os.environ.get("PORT", "9090"))

print(f"loading {MODEL_NAME} ...", flush=True)
_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
_model.eval()
_id2label = _model.config.id2label  # {0: 'ENTAILMENT', 1: 'NEUTRAL', 2: 'CONTRADICTION'}
print(f"NLI verifier ready — labels: {_id2label}", flush=True)


def classify(evidence: str, claim: str) -> dict:
    inputs = _tokenizer(evidence, claim, return_tensors="pt",
                        truncation=True, max_length=512)
    with torch.no_grad():
        logits = _model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    best_idx = int(probs.argmax().item())
    best_label = _id2label[best_idx].upper()
    if best_label == "ENTAILMENT":
        verdict = "SUPPORTED"
    elif best_label == "CONTRADICTION":
        verdict = "UNSUPPORTED"
    else:
        verdict = "UNCERTAIN"
    scores = {_id2label[i].lower(): float(p) for i, p in enumerate(probs)}
    return {"verdict": verdict, "scores": scores}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # silence request logs; errors still go to stderr

    def do_GET(self):
        if self.path == "/healthz":
            self._ok(b"ok", "text/plain")
        else:
            self._err(404, b"not found")

    def do_POST(self):
        if self.path != "/verify":
            self._err(404, b"not found")
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._err(400, b'{"error":"invalid JSON"}')
            return
        claim = (body.get("claim") or "").strip()
        evidence = (body.get("evidence") or "").strip()
        if not claim or not evidence:
            self._err(400, b'{"error":"claim and evidence are required"}')
            return
        result = classify(evidence, claim)
        self._ok(json.dumps(result).encode(), "application/json")

    def _ok(self, body: bytes, ctype: str):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _err(self, code: int, body: bytes):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", PORT), Handler)
    print(f"NLI sidecar listening on :{PORT}", flush=True)
    server.serve_forever()
