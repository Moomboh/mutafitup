#!/usr/bin/env bash
# Post-deploy: install wfutils (stdlib-only) into the conda env.
set -euo pipefail
pip install --no-deps -e .
