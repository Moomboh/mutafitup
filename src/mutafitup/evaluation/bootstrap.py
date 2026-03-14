"""Bootstrap index generation and loading for protein-level evaluation.

Bootstrap indices are generated at the protein (sequence) level and stored
as wide-format parquet files.  Each column ``b_0``, ``b_1``, ... contains
the integer count of how many times a sequence was drawn in that bootstrap
iteration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd


def generate_bootstrap_indices(
    sequence_ids: np.ndarray,
    num_bootstraps: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate protein-level bootstrap count matrices.

    Parameters
    ----------
    sequence_ids : array-like
        Unique sequence identifiers (e.g. from the dataset parquet index).
    num_bootstraps : int
        Number of bootstrap iterations.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame with ``sequence_id`` index and columns
        ``b_0`` through ``b_{num_bootstraps - 1}``, each containing the
        integer count of how many times that sequence was sampled.
    """
    rng = np.random.default_rng(seed)
    n = len(sequence_ids)

    counts = np.zeros((n, num_bootstraps), dtype=np.int32)
    for i in range(num_bootstraps):
        drawn = rng.integers(0, n, size=n)
        for idx in drawn:
            counts[idx, i] += 1

    columns = [f"b_{i}" for i in range(num_bootstraps)]
    df = pd.DataFrame(counts, index=sequence_ids, columns=columns)
    df.index.name = "sequence_id"
    return df


def load_bootstrap_indices(path: Union[str, Path]) -> pd.DataFrame:
    """Load bootstrap indices from a parquet file.

    Parameters
    ----------
    path : str or Path
        Path to the bootstrap parquet file.

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame with ``sequence_id`` index and ``b_*`` columns.
    """
    return pd.read_parquet(path)
