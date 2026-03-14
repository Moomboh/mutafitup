"""Tests for mutafitup.evaluation.auprc_metrics."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mutafitup.evaluation.bootstrap import generate_bootstrap_indices
from mutafitup.evaluation.auprc_metrics import compute_auprc


def _write_prob_predictions_jsonl(path: Path, records: list):
    """Helper to write probability prediction JSONL files."""
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


@pytest.fixture
def binary_per_residue_fixtures(tmp_path):
    """Create binary per-residue probability prediction and bootstrap files.

    Three proteins with per-residue binary labels and predicted probabilities.
    """
    records = [
        {
            "id": "p1",
            "probability": [0.9, 0.1, 0.8],
            "target": [1, 0, 1],
        },
        {
            "id": "p2",
            "probability": [0.2, 0.7, 0.3],
            "target": [0, 1, 0],
        },
        {
            "id": "p3",
            "probability": [0.6, 0.4, 0.95],
            "target": [1, 0, 1],
        },
    ]

    predictions_path = tmp_path / "prob_predictions.jsonl"
    _write_prob_predictions_jsonl(predictions_path, records)

    ids = np.array(["p1", "p2", "p3"])
    bootstrap_df = generate_bootstrap_indices(ids, num_bootstraps=50, seed=42)
    bootstrap_path = tmp_path / "bootstrap.parquet"
    bootstrap_df.to_parquet(bootstrap_path)

    return predictions_path, bootstrap_path


@pytest.fixture
def perfect_prediction_fixtures(tmp_path):
    """Create fixtures with perfect predictions (proba=1 for pos, 0 for neg)."""
    records = [
        {
            "id": "p1",
            "probability": [1.0, 0.0, 1.0],
            "target": [1, 0, 1],
        },
        {
            "id": "p2",
            "probability": [0.0, 1.0, 0.0],
            "target": [0, 1, 0],
        },
    ]

    predictions_path = tmp_path / "perfect_predictions.jsonl"
    _write_prob_predictions_jsonl(predictions_path, records)

    ids = np.array(["p1", "p2"])
    bootstrap_df = generate_bootstrap_indices(ids, num_bootstraps=30, seed=42)
    bootstrap_path = tmp_path / "bootstrap.parquet"
    bootstrap_df.to_parquet(bootstrap_path)

    return predictions_path, bootstrap_path


def test_auprc_structure(binary_per_residue_fixtures):
    """Output has correct structure with metrics and bootstrap_values."""
    pred_path, boot_path = binary_per_residue_fixtures
    result = compute_auprc(
        prob_predictions_path=pred_path,
        bootstrap_path=boot_path,
    )

    assert "metrics" in result
    assert "bootstrap_values" in result
    assert "auprc" in result["metrics"]
    assert "value" in result["metrics"]["auprc"]
    assert "std" in result["metrics"]["auprc"]
    assert "ci95" in result["metrics"]["auprc"]
    assert "auprc" in result["bootstrap_values"]
    assert len(result["bootstrap_values"]["auprc"]) == 50


def test_auprc_value_in_range(binary_per_residue_fixtures):
    """AUPRC should be in [0, 1]."""
    pred_path, boot_path = binary_per_residue_fixtures
    result = compute_auprc(
        prob_predictions_path=pred_path,
        bootstrap_path=boot_path,
    )
    value = result["metrics"]["auprc"]["value"]
    assert 0.0 <= value <= 1.0


def test_auprc_perfect_predictions(perfect_prediction_fixtures):
    """Perfect predictions should yield AUPRC close to 1.0."""
    pred_path, boot_path = perfect_prediction_fixtures
    result = compute_auprc(
        prob_predictions_path=pred_path,
        bootstrap_path=boot_path,
    )
    value = result["metrics"]["auprc"]["value"]
    assert value > 0.95, f"Expected near-perfect AUPRC, got {value}"


def test_auprc_ci95_is_196_times_std(binary_per_residue_fixtures):
    """CI95 should be exactly 1.96 * std."""
    pred_path, boot_path = binary_per_residue_fixtures
    result = compute_auprc(
        prob_predictions_path=pred_path,
        bootstrap_path=boot_path,
    )
    m = result["metrics"]["auprc"]
    assert abs(m["ci95"] - 1.96 * m["std"]) < 1e-10


def test_auprc_bootstrap_values_count(binary_per_residue_fixtures):
    """Bootstrap values list should match num_bootstraps."""
    pred_path, boot_path = binary_per_residue_fixtures
    result = compute_auprc(
        prob_predictions_path=pred_path,
        bootstrap_path=boot_path,
    )
    assert len(result["bootstrap_values"]["auprc"]) == 50


def test_auprc_reasonable_for_good_predictions(binary_per_residue_fixtures):
    """Our fixture data has reasonably correlated predictions — AUPRC > 0.5."""
    pred_path, boot_path = binary_per_residue_fixtures
    result = compute_auprc(
        prob_predictions_path=pred_path,
        bootstrap_path=boot_path,
    )
    value = result["metrics"]["auprc"]["value"]
    assert value > 0.5, f"Expected reasonable AUPRC, got {value}"
