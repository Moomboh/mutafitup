"""Subsample or symlink a single dataset parquet file.

Reads from ``results/datasets/`` and writes to
``results/datasets_subsampled/``.  When all fraction params are ``None``
(prod mode), the output is a symlink to the full parquet -- zero copy,
zero overhead.  Otherwise the matching fraction for the current split is
used to subsample.
"""

import os
import sys
from pathlib import Path

import pandas as pd


def main(snakemake):
    full_path = Path(snakemake.input.full)
    out_path = Path(snakemake.output.dataset)

    train_frac = snakemake.params.train_frac
    valid_frac = snakemake.params.valid_frac
    test_frac = snakemake.params.test_frac
    min_rows = snakemake.params.min_rows
    random_seed = snakemake.params.random_seed

    # Determine which frac applies based on the split wildcard
    split = snakemake.wildcards.split  # e.g. "train", "valid", "test"
    frac_map = {
        "train": train_frac,
        "valid": valid_frac,
        "test": test_frac,
    }
    frac = frac_map.get(split)

    # Prod mode: no subsample config at all → symlink
    if train_frac is None and valid_frac is None and test_frac is None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Use a relative symlink so the results dir stays relocatable
        target = os.path.relpath(full_path.resolve(), out_path.parent.resolve())
        out_path.symlink_to(target)
        return

    # Dev mode: subsample if frac is set for this split, else copy as-is
    df = pd.read_parquet(full_path)

    if frac is not None:
        n = max(min_rows, int(len(df) * frac))
        n = min(n, len(df))
        df = df.sample(n=n, random_state=random_seed)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)


# Snakemake script entry-point
with open(snakemake.log[0], "w") as log_fh:
    sys.stderr = log_fh
    sys.stdout = log_fh
    try:
        main(snakemake)
    except Exception:
        import traceback

        traceback.print_exc(file=log_fh)
        raise
