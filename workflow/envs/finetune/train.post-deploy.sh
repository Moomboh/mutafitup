#!/usr/bin/env bash
# Post-deploy: install pip-only packages into the conda env.
#   - esm: not on conda-forge, requires Python >=3.12
#   - mutafitup + wfutils: editable install (full deps already in conda)
#   - flash-attn: optional, CUDA-only (linux-64), needs nvcc from conda pytorch
set -euo pipefail

pip install esm==3.2.3
pip install --no-deps -e .

# flash-attn: only installable on linux with CUDA (needs nvcc to compile)
if [[ "$(uname -s)" == "Linux" ]]; then
    pip install flash-attn || echo "Warning: flash-attn install failed (optional)"
fi
