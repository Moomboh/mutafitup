default:
    @just --list

help:
    @just --list

# Snakemake commands (production)
run:
    snakemake --profile workflow/profiles/local

run-force:
    snakemake --profile workflow/profiles/local --forceall

run-trigger:
    snakemake --profile workflow/profiles/local --rerun-triggers code input mtime params

dry-run:
    snakemake --profile workflow/profiles/local --dry-run

dry-run-force:
    snakemake --profile workflow/profiles/local --dry-run --forceall

dry-run-trigger:
    snakemake --profile workflow/profiles/local --rerun-triggers code input mtime params --dry-run

# Snakemake commands (dev - small models, subsampled data)
run-dev:
    snakemake --profile workflow/profiles/local-dev

run-dev-force:
    snakemake --profile workflow/profiles/local-dev --forceall

dry-run-dev:
    snakemake --profile workflow/profiles/local-dev --dry-run

dash:
    snakemake --cores all --use-conda dash

# Development commands
install-dev:
    pip install esm==3.2.3
    pip install --no-deps -e .

lint:
    pyright src

format:
    ruff format src

test:
    pytest -n auto --cov=src/mutafitup --cov-report=term-missing

pin-envs:
    bash scripts/pin-conda-envs.sh

clean:
    python -c "from pathlib import Path; import shutil; roots=['.pytest_cache','.ruff_cache','.pyrefly','site']; [shutil.rmtree(p, ignore_errors=True) for p in roots]; [shutil.rmtree(p, ignore_errors=True) for p in Path('.').rglob('__pycache__')]"