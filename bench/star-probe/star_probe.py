#!/usr/bin/env python3
"""
STaR coverage probe — the decisive first measurement for the local-self-training thesis.

For each target function in the metis repo: blank its body, ask qwen3:1.7b to reimplement it
given the file context, splice the candidate back, and run the REAL `cargo test` for that module.
The compiler+tests are the free, local, teacher-free verifier.

We measure:
  pass@1  : does the model's best single shot (low temp) compile+pass?
  pass@k  : does ANY of k samples compile+pass?  <- this is the coverage signal that decides
            whether STaR self-training can even bootstrap (nothing to train on if pass@k≈0).
"""
import json, subprocess, urllib.request, sys, time, os

REPO = "/Users/agustin/projects/personal/metis/metis-0"
OLLAMA = "http://127.0.0.1:11434/api/generate"
MODEL = os.environ.get("PROBE_MODEL", "qwen3:1.7b")
K = int(os.environ.get("PROBE_K", "5"))

# (file, fn_name, signature_line_ending_in_brace, cargo test module filter)
TARGETS = [
    ("src/conductor.rs", "parse_verdict", "fn parse_verdict(s: &str) -> Verdict {", "conductor::tests"),
    ("src/conductor.rs", "evidence_text", "pub fn evidence_text(hits: &[Hit]) -> String {", "conductor::tests"),
    ("src/library/extractive.rs", "split_sentences", "pub fn split_sentences(text: &str) -> Vec<String> {", "library::extractive::tests"),
    ("src/hands/calc.rs", "parse_unary", "fn parse_unary(&mut self) -> Result<f64, String> {", "hands::calc::tests"),
]


def extract_fn(src, sig_line):
    """Return exact source text of the fn whose body opens at sig_line (string/char/comment aware)."""
    start = src.index(sig_line)
    i = start + len(sig_line) - 1  # at the opening '{'
    depth = 0
    in_str = in_char = in_line_comment = False
    while i < len(src):
        c = src[i]
        if in_line_comment:
            if c == "\n":
                in_line_comment = False
        elif in_str:
            if c == "\\":
                i += 1
            elif c == '"':
                in_str = False
        elif in_char:
            if c == "\\":
                i += 1
            elif c == "'":
                in_char = False
        else:
            if c == "/" and i + 1 < len(src) and src[i + 1] == "/":
                in_line_comment = True
            elif c == '"':
                in_str = True
            elif c == "'":
                in_char = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return src[start : i + 1]
        i += 1
    raise RuntimeError("no matching brace for " + sig_line)


def ollama(prompt, temp):
    body = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {"temperature": temp, "num_predict": 1200},
    }).encode()
    req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["response"]


def extract_code(resp, fn_name):
    """Pull the rust code block; then isolate the target fn if the block has extras."""
    block = resp
    if "```" in resp:
        parts = resp.split("```")
        # take the longest fenced segment that mentions the fn
        cands = [p for i, p in enumerate(parts) if i % 2 == 1]
        cands = [c[4:] if c.lower().startswith("rust") else c for c in cands]
        block = max((c for c in cands if "fn " in c), key=len, default=cands[0] if cands else resp)
    sig = f"fn {fn_name}"
    if sig in block:
        try:
            anchor = block.index(sig)
            ln_start = block.rfind("\n", 0, anchor) + 1
            return extract_fn(block[ln_start:], block[ln_start:].split("{")[0].strip() + " {")
        except Exception:
            pass
    return block.strip()


def cargo_pass(filt):
    p = subprocess.run(
        ["cargo", "test", "--lib", filt, "--quiet"],
        cwd=REPO, capture_output=True, text=True, timeout=300,
    )
    return p.returncode == 0


def restore(path):
    subprocess.run(["git", "checkout", "--", path], cwd=REPO, check=True)


def run_one(path, fn, sig, filt, temp):
    full = os.path.join(REPO, path)
    src = open(full).read()
    orig = extract_fn(src, sig)
    stub = sig + "\n    todo!()\n}"
    masked = src.replace(orig, stub, 1)
    prompt = (
        f"You are completing a Rust function in a real project. Below is the file `{path}` "
        f"with the body of `{fn}` replaced by `todo!()`. Implement `{fn}` so the project's "
        f"unit tests pass. Reply with ONLY the complete `{fn}` function in a ```rust code block, "
        f"no prose.\n\n```rust\n{masked}\n```"
    )
    try:
        resp = ollama(prompt, temp)
    except Exception as e:
        return False, f"ollama-error:{e}"
    cand = extract_code(resp, fn)
    if "fn " not in cand:
        return False, "no-fn-in-response"
    spliced = src.replace(orig, cand, 1)
    if spliced == src:
        return False, "splice-failed"
    open(full, "w").write(spliced)
    try:
        ok = cargo_pass(filt)
    except subprocess.TimeoutExpired:
        ok = False
    finally:
        restore(path)
    return ok, "pass" if ok else "test-fail"


def main():
    # warm the build cache once
    print("warming cargo cache...", flush=True)
    subprocess.run(["cargo", "test", "--lib", "--quiet", "--no-run"], cwd=REPO, capture_output=True)
    results = []
    for path, fn, sig, filt in TARGETS:
        print(f"\n=== {fn} ({path}) ===", flush=True)
        outcomes = []
        for j in range(K):
            temp = 0.2 if j == 0 else 0.8
            t0 = time.time()
            ok, why = run_one(path, fn, sig, filt, temp)
            outcomes.append(ok)
            tag = "PASS" if ok else "fail"
            print(f"  sample {j} (t={temp}): {tag:4} [{why}] {time.time()-t0:.0f}s", flush=True)
        p1 = outcomes[0]
        pk = any(outcomes)
        results.append((fn, p1, pk, sum(outcomes), K))
        print(f"  -> pass@1={int(p1)}  pass@{K}={int(pk)}  ({sum(outcomes)}/{K} samples passed)", flush=True)
    print("\n================ SUMMARY ================", flush=True)
    print(f"model={MODEL}  k={K}")
    n = len(results)
    print(f"{'fn':22} pass@1 pass@{K} samples")
    for fn, p1, pk, s, k in results:
        print(f"{fn:22} {int(p1):6} {int(pk):6} {s}/{k}")
    print(f"\npass@1 total: {sum(r[1] for r in results)}/{n}")
    print(f"pass@{K} total: {sum(r[2] for r in results)}/{n}  <- coverage / STaR-bootstrap signal")
    # leave repo clean
    subprocess.run(["git", "checkout", "--", "src/"], cwd=REPO)


if __name__ == "__main__":
    main()
