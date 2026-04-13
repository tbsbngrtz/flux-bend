#!/usr/bin/env bash
set -euo pipefail
if [ ! -d ".venv" ]; then
    echo "ERROR: No .venv found. Run setup.sh first."
    exit 1
fi
uv run flux-bend "$@"
