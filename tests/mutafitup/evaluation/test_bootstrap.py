"""Tests for mutafitup.evaluation.bootstrap."""

import numpy as np
import pandas as pd
import pytest

from mutafitup.evaluation.bootstrap import (
    generate_bootstrap_indices,
    load_bootstrap_indices,
)


def test_generate_bootstrap_indices_shape():
    """Each bootstrap column sums to n (total draws)."""
    ids = np.array(["prot_A", "prot_B", "prot_C"])
    df = generate_bootstrap_indices(ids, num_bootstraps=50, seed=0)

    assert isinstance(df, pd.DataFrame)
    assert df.index.name == "sequence_id"
    assert list(df.index) == ["prot_A", "prot_B", "prot_C"]
    assert df.shape == (3, 50)
    # Each column should sum to n=3
    assert (df.sum(axis=0) == 3).all()


def test_generate_bootstrap_indices_deterministic():
    """Same seed produces identical results."""
    ids = np.array(["A", "B", "C", "D"])
    df1 = generate_bootstrap_indices(ids, num_bootstraps=100, seed=42)
    df2 = generate_bootstrap_indices(ids, num_bootstraps=100, seed=42)
    pd.testing.assert_frame_equal(df1, df2)


def test_generate_bootstrap_indices_different_seeds():
    """Different seeds produce different results."""
    ids = np.array(["A", "B", "C", "D", "E"])
    df1 = generate_bootstrap_indices(ids, num_bootstraps=100, seed=1)
    df2 = generate_bootstrap_indices(ids, num_bootstraps=100, seed=2)
    # Overwhelmingly likely to differ
    assert not df1.equals(df2)


def test_generate_bootstrap_indices_single_protein():
    """With one protein, all counts are 1."""
    ids = np.array(["only_one"])
    df = generate_bootstrap_indices(ids, num_bootstraps=10, seed=0)
    assert df.shape == (1, 10)
    assert (df.values == 1).all()


def test_column_names():
    """Columns should be b_0, b_1, ..."""
    ids = np.array(["X", "Y"])
    df = generate_bootstrap_indices(ids, num_bootstraps=5, seed=0)
    assert list(df.columns) == ["b_0", "b_1", "b_2", "b_3", "b_4"]


def test_dtypes():
    """Counts should be int32."""
    ids = np.array(["A", "B", "C"])
    df = generate_bootstrap_indices(ids, num_bootstraps=3, seed=0)
    assert df.dtypes.unique().tolist() == [np.dtype("int32")]


def test_save_and_load(tmp_path):
    """Round-trip through parquet preserves data."""
    ids = np.array(["prot1", "prot2", "prot3", "prot4"])
    df = generate_bootstrap_indices(ids, num_bootstraps=20, seed=123)

    path = tmp_path / "bootstraps.parquet"
    df.to_parquet(path)

    loaded = load_bootstrap_indices(path)
    pd.testing.assert_frame_equal(df, loaded)
