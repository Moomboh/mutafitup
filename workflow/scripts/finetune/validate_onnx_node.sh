#!/usr/bin/env bash
# Run the Node.js ONNX validation script.
#
# Usage: validate_onnx_node.sh <export_dir> <predictions_dir> <report_path>
#
# The conda environment provides nodejs and pnpm; npm packages
# (onnxruntime-node, @huggingface/transformers, tsx, …) are installed
# via pnpm from the workspace lockfile.
set -euo pipefail

# Resolve all paths to absolute before changing directories.
EXPORT_DIR="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
PREDICTIONS_DIR="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"

REPORT_DIR="$(dirname "$3")"
mkdir -p "$REPORT_DIR"
REPORT_PATH="$(cd "$REPORT_DIR" && pwd)/$(basename "$3")"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PACKAGES_DIR="$(cd "$SCRIPT_DIR/../../../packages" && pwd)"
PKG_DIR="$PACKAGES_DIR/mutafitup-node"

echo "Installing Node.js dependencies …"
cd "$PACKAGES_DIR"
pnpm install --frozen-lockfile

echo "Running ONNX validation …"
cd "$PKG_DIR"
npx tsx "$SCRIPT_DIR/validate_onnx_node.ts" \
    "$EXPORT_DIR" \
    "$PREDICTIONS_DIR" \
    "$REPORT_PATH"
