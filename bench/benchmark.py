#!/usr/bin/env python3
"""
Metis benchmark — bare Cortex vs Metis, and Phase 2 multi-variant comparison.

The ONLY difference between systems is the architecture:
  BARE    : qwen3 called directly via ollama (no retrieval, no verify)
  METIS   : same qwen3 + Library (RAG) + Generate·Verify·Search

Phase 2 adds multi-variant support to compare E0 / E1 / E2 in one run:
  E0 : 1.7B generator + 1.7B LLM verifier  (Phase 1 baseline)
  E1 : 0.6B generator + 0.6B LLM verifier  (naive downgrade)
  E2 : 0.6B generator + 22M NLI verifier   (Phase 2 cascade)

Single-variant usage (backward-compatible):
  METIS_URL=http://127.0.0.1:8080 OLLAMA_URL=http://127.0.0.1:11434 MODEL=qwen3:1.7b \\
      python3 bench/benchmark.py

Multi-variant usage:
  METIS_VARIANTS='[
    {"name":"E0-1.7B-llm", "url":"http://localhost:8080"},
    {"name":"E1-0.6B-llm", "url":"http://localhost:8081"},
    {"name":"E2-0.6B-nli", "url":"http://localhost:8082"}
  ]' python3 bench/benchmark.py

Each variant should be a Metis server configured appropriately (different model/verifier).
See docs/design/08-phase2-cascade.md for the experiment definitions.
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

# Multi-variant: JSON array of {"name": str, "url": str}. Overrides METIS_URL if set.
_variants_json = os.environ.get("METIS_VARIANTS", "")
VARIANTS = json.loads(_variants_json) if _variants_json else [{"name": "METIS", "url": METIS_URL}]

# Question set: load from BENCH_QUESTIONS file (a JSON array) if set, else use the built-in set.
# A loaded question may carry a "tier" field for per-difficulty breakdown.
_questions_file = os.environ.get("BENCH_QUESTIONS", "")

BARE_SYSTEM = (
    "You are a helpful, accurate assistant. Answer concisely. "
    "If you do not know the answer or are not certain, say so plainly instead of guessing."
)

# kind: answerable | unanswerable | general
# any: pass if ANY of these substrings appears (case-insensitive)
QUESTIONS = [
    # --- answerable: facts that live ONLY in the Zephyrian corpus ---
    {"q": "What is the maximum resident model memory the Zephyrian Protocol allows?",             "kind": "answerable",   "any": ["1.84"]},
    {"q": "How many knowledge shards may be cached in RAM at one time under the Zephyrian Protocol?", "kind": "answerable", "any": ["3", "three"]},
    {"q": "In what year was the Zephyrian Portability Protocol ratified?",                        "kind": "answerable",   "any": ["2031"]},
    {"q": "Who ratified the Zephyrian Portability Protocol?",                                     "kind": "answerable",   "any": ["edge compute consortium"]},
    {"q": "What is the codename of the Zephyrian reference implementation?",                      "kind": "answerable",   "any": ["marlowe"]},
    {"q": "What hardware does the Zephyrian reference implementation target?",                    "kind": "answerable",   "any": ["4 gb", "4gb", "4 vcpu", "4-vcpu"]},
    {"q": "What is the Zephyrian Protocol's mascot?",                                             "kind": "answerable",   "any": ["heron", "pippa"]},
    {"q": "What must every factual answer carry under the Zephyrian Protocol?",                   "kind": "answerable",   "any": ["provenance", "source shard"]},

    # --- unanswerable: plausible but NOT in the corpus ---
    {"q": "Who is the current chairperson of the Edge Compute Consortium?",                       "kind": "unanswerable"},
    {"q": "In which city is the Edge Compute Consortium headquartered?",                          "kind": "unanswerable"},
    {"q": "What programming language is the Marlowe reference implementation written in?",        "kind": "unanswerable"},
    {"q": "How much does Zephyrian Protocol certification cost?",                                 "kind": "unanswerable"},

    # --- general: model can do unaided; checks no regression + tools ---
    {"q": "What is the capital of France?",                                                       "kind": "general", "any": ["paris"]},
    {"q": "What is 84937 * 2261? Give only the number.",                                          "kind": "general", "any": ["192042557"]},
]

if _questions_file:
    with open(_questions_file) as _f:
        QUESTIONS = json.load(_f)
    print(f"loaded {len(QUESTIONS)} questions from {_questions_file}")

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


def ask_metis(base_url, question):
    body, dt = http_json(f"{base_url}/ask", {"q": question})
    return (body.get("answer", "") or "").strip(), dt, body.get("path", "?")


def score(kind, answer, spec):
    a = answer.lower()
    if kind == "unanswerable":
        return ("abstained", True) if ABSTAIN_RE.search(answer) else ("FABRICATED", False)
    hit = any(s.lower() in a for s in spec.get("any", []))
    if hit:
        return ("correct", True)
    if ABSTAIN_RE.search(answer):
        return ("abstained", False)
    return ("WRONG", False)


def run_system(name, asker):
    print(f"\n{'='*78}\n  {name}\n{'='*78}")
    rows, lat = [], []
    agg = {}        # kind -> [ok, total]
    tier_agg = {}   # tier -> [ok, total]
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
        tier = item.get("tier", "-")
        tier_agg.setdefault(tier, [0, 0])
        tier_agg[tier][0] += 1 if ok else 0
        tier_agg[tier][1] += 1
        rows.append({"q": item["q"], "kind": item["kind"], "tier": tier, "path": path,
                     "label": label, "ok": ok, "latency_s": round(dt, 2), "answer": answer})
        flag = "✓" if ok else "✗"
        snippet = answer.replace("\n", " ")[:80]
        print(f"  {flag} [{tier:<14}] ({path:<12} {dt:5.1f}s) {label:<11} | {snippet}")
    return rows, agg, lat, tier_agg


def summary(name, agg, lat, tier_agg=None):
    ans = agg.get("answerable", [0, 0])
    una = agg.get("unanswerable", [0, 0])
    gen = agg.get("general", [0, 0])
    fabricated = una[1] - una[0]
    print(f"\n  --- {name} ---")
    print(f"    answerable accuracy : {ans[0]}/{ans[1]}")
    print(f"    unanswerable abstain: {una[0]}/{una[1]}   (fabrications: {fabricated})")
    print(f"    general accuracy    : {gen[0]}/{gen[1]}")
    print(f"    avg latency         : {sum(lat)/max(len(lat),1):.1f}s")
    tiers = {}
    if tier_agg:
        print(f"    --- by difficulty tier ---")
        for tier in sorted(tier_agg):
            ok, tot = tier_agg[tier]
            print(f"      {tier:<16} {ok}/{tot}")
            tiers[tier] = [ok, tot]
    return {"answerable": ans, "unanswerable": una, "general": gen,
            "fabrications": fabricated, "avg_latency_s": round(sum(lat)/max(len(lat),1), 2),
            "tiers": tiers}


def print_comparison(all_summaries):
    """Print a side-by-side table for multi-variant runs."""
    if len(all_summaries) < 2:
        return
    names = [n for n, _ in all_summaries]
    sums  = [s for _, s in all_summaries]
    col = 12
    header = f"  {'metric':<30}" + "".join(f"{n:>{col}}" for n in names)
    print(f"\n{'='*78}\n  PHASE 2 COMPARISON — architecture is the only difference\n{'='*78}")
    print(header)
    print(f"  {'─'*28}" + ("─" * col * len(names)))

    def row(label, vals):
        return f"  {label:<30}" + "".join(f"{v:>{col}}" for v in vals)

    print(row("answerable correct",
              [f"{s['answerable'][0]}/{s['answerable'][1]}" for s in sums]))
    print(row("fabrications (lower=better)",
              [str(s['fabrications']) for s in sums]))
    print(row("general correct",
              [f"{s['general'][0]}/{s['general'][1]}" for s in sums]))
    print(row("avg latency (s)",
              [str(s['avg_latency_s']) for s in sums]))

    # per-tier comparison (only if tiers were tracked)
    all_tiers = sorted({t for s in sums for t in s.get("tiers", {})})
    if all_tiers:
        print(f"  {'─'*28}" + ("─" * col * len(names)))
        for tier in all_tiers:
            vals = []
            for s in sums:
                tv = s.get("tiers", {}).get(tier)
                vals.append(f"{tv[0]}/{tv[1]}" if tv else "-")
            print(row(f"  {tier}", vals))


def main():
    print(f"Metis benchmark — model={MODEL}")
    results = {"model": MODEL, "questions": len(QUESTIONS)}
    all_summaries = []

    if OLLAMA_URL:
        rows, agg, lat, tier_agg = run_system(
            f"BARE Cortex ({MODEL} via ollama — no RAG, no verify)", ask_bare)
        s = summary("BARE", agg, lat, tier_agg)
        results["bare"] = {"rows": rows, "summary": s}
        all_summaries.append(("BARE", s))

    for variant in VARIANTS:
        vname = variant["name"]
        vurl  = variant["url"].rstrip("/")
        rows, agg, lat, tier_agg = run_system(
            f"{vname}  ({vurl})",
            lambda q, u=vurl: ask_metis(u, q),
        )
        s = summary(vname, agg, lat, tier_agg)
        results.setdefault("variants", {})[vname] = {"rows": rows, "summary": s}
        all_summaries.append((vname, s))

    if "bare" in results and len(VARIANTS) == 1:
        b, m = results["bare"]["summary"], list(results["variants"].values())[0]["summary"]
        print(f"\n{'='*78}\n  HEAD-TO-HEAD — same weights, architecture is the only difference\n{'='*78}")
        print(f"  {'metric':<30}{'BARE':>12}{VARIANTS[0]['name']:>12}")
        print(f"  {'answerable correct':<30}{b['answerable'][0]:>12}{m['answerable'][0]:>12}  / {b['answerable'][1]}")
        print(f"  {'fabrications (lower=better)':<30}{b['fabrications']:>12}{m['fabrications']:>12}  / {b['unanswerable'][1]}")
        print(f"  {'general correct':<30}{b['general'][0]:>12}{m['general'][0]:>12}  / {b['general'][1]}")
    else:
        print_comparison(all_summaries)

    out = os.environ.get("BENCH_OUT", "bench/results.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
