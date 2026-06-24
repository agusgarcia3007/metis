#!/bin/sh
# Self-provision: wait for the ollama Cortex backend and pull the model + embedder (idempotent),
# then run the requested metis command (default: serve).
set -e
metis setup
exec metis "$@"
