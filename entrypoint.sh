#!/bin/sh
# Self-provision so the container "just works" on a PaaS (Dokploy) or plain Docker:
#  1. wait for the ollama Cortex backend and pull the model + embedder (idempotent)
#  2. if METIS_INDEX_DIR points to a non-empty folder, (re)build the Library from it
#  3. run the requested metis command (default: serve)
set -e

metis setup

if [ -n "$METIS_INDEX_DIR" ] && [ -d "$METIS_INDEX_DIR" ] && [ -n "$(ls -A "$METIS_INDEX_DIR" 2>/dev/null)" ]; then
  echo "[entrypoint] indexing $METIS_INDEX_DIR into the Library ..."
  metis index "$METIS_INDEX_DIR" || echo "[entrypoint] index failed; serving without grounding"
fi

exec metis "$@"
