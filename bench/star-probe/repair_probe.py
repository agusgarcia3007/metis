#!/usr/bin/env python3
"""Repair probe: does execution feedback (the cargo error fed back) cross the near-misses?
Tests the proposed loop: generate -> run -> if fail, feed REAL error back -> regenerate. """
import sys, subprocess, os
sys.path.insert(0, os.path.dirname(__file__))
from star_probe import extract_fn, ollama, extract_code, REPO, TARGETS

REPAIRS = 2  # feedback rounds after the first shot

def test_with(path, sig, orig, cand, filt):
    full = os.path.join(REPO, path)
    src = open(full).read()
    open(full, "w").write(src.replace(orig, cand, 1))
    try:
        p = subprocess.run(["cargo","test","--lib",filt,"--quiet"], cwd=REPO, capture_output=True, text=True, timeout=300)
    finally:
        subprocess.run(["git","checkout","--",path], cwd=REPO, check=True)
    return p.returncode == 0, (p.stdout + p.stderr)

def main():
    subprocess.run(["cargo","test","--lib","--quiet","--no-run"], cwd=REPO, capture_output=True)
    crossed = 0
    for path, fn, sig, filt in TARGETS:
        full = os.path.join(REPO, path); src = open(full).read(); orig = extract_fn(src, sig)
        masked = src.replace(orig, sig + "\n    todo!()\n}", 1)
        prompt = (f"Complete the Rust function `{fn}` in this file so its unit tests pass. "
                  f"Reply ONLY with the complete `{fn}` in a ```rust block.\n\n```rust\n{masked}\n```")
        cand = extract_code(ollama(prompt, 0.2), fn)
        ok, err = test_with(path, sig, orig, cand, filt)
        history = f"  [{fn}] shot 0: {'PASS' if ok else 'fail'}"
        round_passed = ok
        for r in range(REPAIRS):
            if round_passed: break
            tail = "\n".join(err.splitlines()[-15:])
            rp = (f"Your implementation of `{fn}` FAILED the tests. Here is your code:\n```rust\n{cand}\n```\n"
                  f"Here is the exact cargo test error:\n```\n{tail}\n```\n"
                  f"Fix it. Reply ONLY with the corrected `{fn}` in a ```rust block.")
            cand = extract_code(ollama(rp, 0.3), fn)
            round_passed, err = test_with(path, sig, orig, cand, filt)
            history += f" | repair {r+1}: {'PASS' if round_passed else 'fail'}"
        crossed += int(round_passed)
        print(history, flush=True)
    print(f"\ncrossed with execution-feedback repair: {crossed}/{len(TARGETS)}", flush=True)
    subprocess.run(["git","checkout","--","src/"], cwd=REPO)

if __name__ == "__main__":
    main()
