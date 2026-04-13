#!/usr/bin/env bash
set -euo pipefail
echo "=== flux-bend setup ==="

if ! command -v uv &> /dev/null; then
    echo "ERROR: uv is not installed. Install from https://docs.astral.sh/uv/"
    exit 1
fi

echo "Creating venv with Python 3.12..."
uv venv --python 3.12

echo "Installing dependencies..."
uv pip install -e ".[analysis]" --extra-index-url https://download.pytorch.org/whl/cu128
uv pip install git+https://github.com/huggingface/diffusers.git

echo ""
echo "Setup complete. Run with: ./run.sh [command] [args]"
echo "Example: ./run.sh test --model black-forest-labs/FLUX.2-klein-base-4B --prompt 'test' --seed 42"
echo ""
echo "NOTE: run 'huggingface-cli login' to authenticate with HuggingFace before first use."
