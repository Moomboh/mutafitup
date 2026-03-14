#!/usr/bin/env bash
# Pin all conda environments for osx-arm64 and linux-64.
#
# Uses conda-lock to generate cross-platform @EXPLICIT lock files from a
# single machine.  No pip stripping needed — env YAMLs contain only conda
# deps; pip-only packages (esm, flash-attn) are installed via post-deploy
# scripts.
#
# Output files follow Snakemake's naming convention:
#   <env_basename>.<platform>.pin.txt
set -euo pipefail

PLATFORMS=(osx-arm64 linux-64)

# Collect all env YAMLs: workflow envs + top-level environment.yml
ENV_FILES=()
while IFS= read -r f; do
    ENV_FILES+=("$f")
done < <(find workflow/envs -name '*.yml' | sort)
ENV_FILES+=("environment.yml")

for envfile in "${ENV_FILES[@]}"; do
    # Determine output directory and basename
    if [[ "$envfile" == "environment.yml" ]]; then
        outdir="."
        basename="environment"
    else
        relpath="${envfile#workflow/envs/}"
        outdir="workflow/envs/$(dirname "$relpath")"
        basename="${relpath%.yml}"
    fi

    for platform in "${PLATFORMS[@]}"; do
        dst="${outdir:+$outdir/}$(basename "${basename}").${platform}.pin.txt"
        # For nested paths, reconstruct properly
        if [[ "$envfile" != "environment.yml" ]]; then
            dst="workflow/envs/${basename}.${platform}.pin.txt"
        fi

        echo "Locking $envfile -> $dst ..."

        # Use --with-cuda for linux-64, --without-cuda for osx-arm64
        cuda_flag=""
        if [[ "$platform" == "linux-64" ]]; then
            cuda_flag="--with-cuda=12.6"
        else
            cuda_flag="--without-cuda"
        fi

        if conda-lock lock \
            --no-mamba \
            --no-micromamba \
            -f "$envfile" \
            -p "$platform" \
            -k explicit \
            $cuda_flag \
            --filename-template "$dst" \
            2>&1; then
            echo "  OK"
        else
            echo "  FAILED (skipping $platform for $envfile)"
        fi
    done
done
