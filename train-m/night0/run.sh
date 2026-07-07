#!/bin/zsh
# metis-1 MVP — one command to (re)start the local server for OpenCode.
#
#   ./run.sh              start server on :8484 with metis-mvp.safetensors
#   ./run.sh corpus       rebuild the training corpus from local projects
#   ./run.sh train        retrain the MVP checkpoint (~50 min on M3 Pro)
#
# OpenCode is already configured (provider "metis" in ~/.config/opencode/opencode.json):
#   opencode run -m metis/metis-1-mvp "function add("
set -e
cd "$(dirname "$0")"

VENV=.venv
if [[ ! -x $VENV/bin/python ]]; then
  echo "[metis] bootstrapping venv…"
  python3 -m venv $VENV
  $VENV/bin/pip install -q mlx numpy
fi

CORPUS=corpus.txt
build_corpus() {
  echo "[metis] building corpus from ~/projects…"
  find ~/projects -type f \( -name "*.ts" -o -name "*.tsx" -o -name "*.rs" -o -name "*.py" -o -name "*.go" -o -name "*.md" \) \
    -not -path "*/node_modules/*" -not -path "*/.git/*" -not -path "*/target/*" \
    -not -path "*/dist/*" -not -path "*/build/*" -not -path "*/.next/*" \
    -not -path "*/venv/*" -not -path "*/.venv/*" -size -200k 2>/dev/null \
    | head -8000 | xargs cat > $CORPUS 2>/dev/null
  ls -lh $CORPUS
}

case "${1:-serve}" in
  corpus) build_corpus ;;
  train)
    [[ -f $CORPUS ]] || build_corpus
    $VENV/bin/python train.py --data $CORPUS --steps 2500 \
      --out results-mvp.json --save-weights metis-mvp.safetensors
    ;;
  serve)
    if [[ ! -f metis-mvp.safetensors ]]; then
      echo "[metis] no weights found — run ./run.sh train first"; exit 1
    fi
    pkill -f "serve.py --weights" 2>/dev/null || true
    echo "[metis] serving on http://127.0.0.1:8484/v1  (Ctrl-C to stop)"
    exec $VENV/bin/python serve.py --weights metis-mvp.safetensors --port 8484
    ;;
  *) echo "usage: ./run.sh [serve|train|corpus]"; exit 1 ;;
esac
