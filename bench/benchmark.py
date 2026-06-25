#!/usr/bin/env python3
"""
Metis benchmark — bare Cortex vs the Metis system, same weights.

The ONLY difference between the two systems is the architecture:
  - BARE  : qwen3 called directly via ollama (no retrieval, no verify). What `ollama run` gives you.
  - METIS : the SAME qwen3 + Library (RAG) + Generate·Verify·Search (abstains instead of guessing).

We measure three things on a private knowledge surface the model never trained on:
  1. ANSWERABLE   facts in the corpus      -> accuracy (did it get the fact right?)
  2. UNANSWERABLE plausible-but-absent      -> fabrication rate (did it invent an answer? lower=better)
  3. GENERAL      model can do unaided      -> no-regression check (+ tools for exact math)

Usage:
  METIS_URL=http://127.0.0.1:8080 OLLAMA_URL=http://127.0.0.1:11434 MODEL=qwen3:1.7b \
      python3 bench/benchmark.py
  # Metis-only (e.g. against a live Railway deploy, where bare ollama isn't exposed):
  METIS_URL=https://xxx.up.railway.app MODEL=qwen3:1.7b python3 bench/benchmark.py
"""
import json
import os
import re
import sys
import time
import urllib.request

METIS_URL = os.environ.get("METIS_URL", "http://127.0.0.1:8080").rstrip("/")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "").rstrip("/")  # empty => skip bare baseline
MODEL = os.environ.get("MODEL", "qwen3:1.7b")
TIMEOUT = int(os.environ.get("BENCH_TIMEOUT", "300"))

BARE_SYSTEM = (
    "You are a helpful, accurate assistant. Answer concisely. "
    "If you do not know the answer or are not certain, say so plainly instead of guessing."
)

# kind: answerable | unanswerable | general
# any: pass if ANY of these substrings appears (case-insensitive)
QUESTIONS = [
    # --- answerable: facts that live ONLY in the Zephyrian corpus (model never trained on them) ---
    {"q": "What is the maximum resident model memory the Zephyrian Protocol allows?", "kind": "answerable", "any": ["1.84"]},
    {"q": "How many knowledge shards may be cached in RAM at one time under the Zephyrian Protocol?", "kind": "answerable", "any": ["3", "three"]},
    {"q": "In what year was the Zephyrian Portability Protocol ratified?", "kind": "answerable", "any": ["2031"]},
    {"q": "Who ratified the Zephyrian Portability Protocol?", "kind": "answerable", "any": ["edge compute consortium"]},
    {"q": "What is the codename of the Zephyrian reference implementation?", "kind": "answerable", "any": ["marlowe"]},
    {"q": "What hardware does the Zephyrian reference implementation target?", "kind": "answerable", "any": ["4 gb", "4gb", "4 vcpu", "4-vcpu"]},
    {"q": "What is the Zephyrian Protocol's mascot?", "kind": "answerable", "any": ["heron", "pippa"]},
    {"q": "What must every factual answer carry under the Zephyrian Protocol?", "kind": "answerable", "any": ["provenance", "source shard"]},

    # --- unanswerable: plausible, on-topic, but NOT in the corpus -> the right move is to abstain ---
    {"q": "Who is the current chairperson of the Edge Compute Consortium?", "kind": "unanswerable"},
    {"q": "In which city is the Edge Compute Consortium headquartered?", "kind": "unanswerable"},
    {"q": "What programming language is the Marlowe reference implementation written in?", "kind": "unanswerable"},
    {"q": "How much does Zephyrian Protocol certification cost?", "kind": "unanswerable"},

    # --- general: the bare model can do these unaided; checks Metis doesn't regress + uses tools ---
    {"q": "What is the capital of France?", "kind": "general", "any": ["paris"]},
    {"q": "What is 84937 * 2261? Give only the number.", "kind": "general", "any": ["192042557"]},
]

# An answer counts as an ABSTENTION if it signals "I don't have / it's not in the sources / not certain".
ABSTAIN_RE = re.compile(
    r"(not (available|provided|in the|mentioned|found|specified|stated|listed|included))"
    r"|(do(es)?|did) not (contain|mention|provide|specify|include|state|have)"
    r"|(no (information|mention|details|data))"
    r"|(isn't|is not|aren't|are not) (in|available|provided|mentioned|specified)"
    r"|(cannot|can't|could not|couldn't) (find|determine|answer)"
    r"|(rather not guess)|(i don'?t (have|know))|(unable to)|(not enough (information|context))"
    r"|(the (sources|context|documents?|provided) do(es)? not)",
    re.IGNORECASE,
)


def http_json(url, payload, timeout=TIMEOUT):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read().decode())
    return body, time.time() - t0


def ask_bare(question):
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": BARE_SYSTEM},
            {"role": "user", "content": question},
        ],
        "stream": False,
        "think": False,
        "options": {"temperature": 0.3},
    }
    body, dt = http_json(f"{OLLAMA_URL}/api/chat", payload)
    return (body.get("message", {}).get("content", "") or "").strip(), dt, "bare"


def ask_metis(question):
    body, dt = http_json(f"{METIS_URL}/ask", {"q": question})
    return (body.get("answer", "") or "").strip(), dt, body.get("path", "?")


def score(kind, answer, spec):
    """Return (label, ok) where ok=True means the system did the right thing."""
    a = answer.lower()
    if kind == "unanswerable":
        return ("abstained", True) if ABSTAIN_RE.search(answer) else ("FABRICATED", False)
    # answerable / general: right if any expected substring is present AND it didn't wrongly abstain
    hit = any(s.lower() in a for s in spec.get("any", []))
    if hit:
        return ("correct", True)
    if ABSTAIN_RE.search(answer):
        return ("abstained", False)  # missed a fact it should have had
    return ("WRONG", False)


def run_system(name, asker):
    print(f"\n{'='*78}\n  {name}\n{'='*78}")
    rows, lat = [], []
    agg = {}  # kind -> [ok, total]
    for item in QUESTIONS:
        try:
            answer, dt, path = asker(item["q"])
        except Exception as e:
            answer, dt, path = f"<error: {e}>", 0.0, "error"
        label, ok = score(item["kind"], answer, item)
        lat.append(dt)
        agg.setdefault(item["kind"], [0, 0])
        agg[item["kind"]][0] += 1 if ok else 0
        agg[item["kind"]][1] += 1
        rows.append({"q": item["q"], "kind": item["kind"], "path": path,
                     "label": label, "ok": ok, "latency_s": round(dt, 2),
                     "answer": answer})
        flag = "✓" if ok else "✗"
        snippet = answer.replace("\n", " ")[:88]
        print(f"  {flag} [{item['kind']:<12}] ({path:<12} {dt:5.1f}s) {label:<11} | {snippet}")
    return rows, agg, lat


def summary(name, agg, lat):
    ans = agg.get("answerable", [0, 0])
    una = agg.get("unanswerable", [0, 0])
    gen = agg.get("general", [0, 0])
    fabricated = una[1] - una[0]
    print(f"\n  --- {name} ---")
    print(f"    answerable accuracy : {ans[0]}/{ans[1]}")
    print(f"    unanswerable abstain: {una[0]}/{una[1]}   (fabrications: {fabricated})")
    print(f"    general accuracy    : {gen[0]}/{gen[1]}")
    print(f"    avg latency         : {sum(lat)/max(len(lat),1):.1f}s")
    return {"answerable": ans, "unanswerable": una, "general": gen,
            "fabrications": fabricated, "avg_latency_s": round(sum(lat)/max(len(lat),1), 2)}


def main():
    print(f"Metis benchmark — model={MODEL}")
    print(f"  METIS_URL  = {METIS_URL}")
    print(f"  OLLAMA_URL = {OLLAMA_URL or '(skipped — Metis-only run)'}")
    results = {"model": MODEL, "questions": len(QUESTIONS)}

    if OLLAMA_URL:
        rows, agg, lat = run_system("BARE Cortex (qwen3 via ollama — no RAG, no verify)", ask_bare)
        results["bare"] = {"rows": rows, "summary": summary("BARE", agg, lat)}

    rows, agg, lat = run_system("METIS (same qwen3 + Library + Generate·Verify·Search)", ask_metis)
    results["metis"] = {"rows": rows, "summary": summary("METIS", agg, lat)}

    out = os.environ.get("BENCH_OUT", "bench/results.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved -> {out}")

    if "bare" in results:
        b, m = results["bare"]["summary"], results["metis"]["summary"]
        print(f"\n{'='*78}\n  HEAD-TO-HEAD (same {MODEL} weights, architecture is the only difference)\n{'='*78}")
        print(f"  {'metric':<26}{'BARE':>10}{'METIS':>10}")
        print(f"  {'answerable correct':<26}{b['answerable'][0]:>10}{m['answerable'][0]:>10}  / {b['answerable'][1]}")
        print(f"  {'fabrications (lower=better)':<26}{b['fabrications']:>10}{m['fabrications']:>10}  / {b['unanswerable'][1]}")
        print(f"  {'general correct':<26}{b['general'][0]:>10}{m['general'][0]:>10}  / {b['general'][1]}")


if __name__ == "__main__":
    main()
