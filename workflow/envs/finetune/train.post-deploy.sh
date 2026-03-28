#!/usr/bin/env bash
# Post-deploy: install pip-only packages into the conda env.
#   - pytorch: installed via pip to get CUDA 12.1 builds (compatible with
#     LRZ cluster driver 535 / CUDA 12.2). conda-forge only has cuda120
#     builds up to PyTorch 2.5.1 which can't satisfy our other deps.
#   - esm: not on conda-forge, requires Python >=3.12
#   - mutafitup + wfutils: editable install (full deps already in conda)
#   - flash-attn: optional, CUDA-only (linux-64), needs nvcc
set -euo pipefail

# Install esm first (pulls in torch + all its deps), then force-reinstall
# torch to 2.5.1 to match the cluster's CUDA 12.2 driver (cu121 builds).
pip install esm==3.2.3

if [[ "$(uname -s)" == "Linux" ]]; then
    pip install --force-reinstall torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
else
    pip install --force-reinstall torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
fi
pip install --no-deps -e .

# flash-attn: only installable on linux with CUDA (needs nvcc to compile)
if [[ "$(uname -s)" == "Linux" ]]; then
    pip install --no-deps flash-attn || echo "Warning: flash-attn install failed (optional)"
fi
