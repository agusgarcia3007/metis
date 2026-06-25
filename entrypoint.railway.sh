#!/bin/sh
# Single-container boot for Railway: start ollama, pull the Cortex + embedder, build the Library
# from the bundled docs, then serve the Metis HTTP API. /healthz answers immediately (so Railway
# marks the deploy live) while the model pull finishes in the background path below.
set -e

echo "[entrypoint] starting ollama..."
ollama serve &

echo "[entrypoint] waiting for ollama..."
i=0
while [ "$i" -lt 60 ]; do
  if ollama list >/dev/null 2>&1; then
    echo "[entrypoint] ollama is up"
    break
  fi
  i=$((i + 1))
  sleep 2
done

echo "[entrypoint] pulling models (idempotent; may take a few minutes on first boot)..."
metis setup || echo "[entrypoint] WARN: model pull reported an issue; continuing"

if [ -n "$METIS_INDEX_DIR" ] && [ -d "$METIS_INDEX_DIR" ] && [ -n "$(ls -A "$METIS_INDEX_DIR" 2>/dev/null)" ]; then
  echo "[entrypoint] indexing $METIS_INDEX_DIR into the Library..."
  metis index "$METIS_INDEX_DIR" || echo "[entrypoint] WARN: index failed; serving ungrounded"
fi

echo "[entrypoint] serving Metis on :$PORT"
exec metis serve
