"""Tests for mutafitup.evaluation.metrics."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mutafitup.evaluation.bootstrap import generate_bootstrap_indices
from mutafitup.evaluation.metrics import compute_metrics


def _write_predictions_jsonl(path: Path, records: list):
    """Helper to write prediction JSONL files."""
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


@pytest.fixture
def classification_fixtures(tmp_path):
    """Create classification prediction and bootstrap files."""
    # 5 proteins, 3 classes, per-protein classification
    ids = ["p1", "p2", "p3", "p4", "p5"]
    preds = [0, 1, 2, 0, 1]
    targets = [0, 1, 2, 1, 1]

    predictions_path = tmp_path / "predictions.jsonl"
    records = [
        {
            "id": ids[i],
            "prediction": preds[i],
            "target": targets[i],
            "sequence": f"SEQ{i}",
        }
        for i in range(5)
    ]
    _write_predictions_jsonl(predictions_path, records)

    bootstrap_df = generate_bootstrap_indices(np.array(ids), num_bootstraps=50, seed=42)
    bootstrap_path = tmp_path / "bootstrap.parquet"
    bootstrap_df.to_parquet(bootstrap_path)

    return predictions_path, bootstrap_path


@pytest.fixture
def regression_fixtures(tmp_path):
    """Create regression prediction and bootstrap files."""
    ids = ["p1", "p2", "p3", "p4", "p5"]
    preds = [1.0, 2.0, 3.0, 4.0, 5.0]
    targets = [1.1, 2.1, 2.9, 4.2, 4.8]

    predictions_path = tmp_path / "predictions.jsonl"
    records = [
        {
            "id": ids[i],
            "prediction": preds[i],
            "target": targets[i],
            "sequence": f"SEQ{i}",
        }
        for i in range(5)
    ]
    _write_predictions_jsonl(predictions_path, records)

    bootstrap_df = generate_bootstrap_indices(np.array(ids), num_bootstraps=50, seed=42)
    bootstrap_path = tmp_path / "bootstrap.parquet"
    bootstrap_df.to_parquet(bootstrap_path)

    return predictions_path, bootstrap_path


@pytest.fixture
def per_residue_fixtures(tmp_path):
    """Create per-residue classification prediction and bootstrap files."""
    ids = ["p1", "p2", "p3"]
    # Each protein has a list of residue predictions/targets
    records = [
        {"id": "p1", "prediction": [0, 1, 0], "target": [0, 1, 0], "sequence": "ABC"},
        {"id": "p2", "prediction": [1, 1, 0], "target": [1, 0, 0], "sequence": "DEF"},
        {"id": "p3", "prediction": [0, 0, 1], "target": [0, 0, 1], "sequence": "GHI"},
    ]

    predictions_path = tmp_path / "predictions.jsonl"
    _write_predictions_jsonl(predictions_path, records)

    bootstrap_df = generate_bootstrap_indices(np.array(ids), num_bootstraps=30, seed=42)
    bootstrap_path = tmp_path / "bootstrap.parquet"
    bootstrap_df.to_parquet(bootstrap_path)

    return predictions_path, bootstrap_path


def test_classification_metrics_structure(classification_fixtures):
    """Output has correct structure for classification metrics."""
    pred_path, boot_path = classification_fixtures
    result = compute_metrics(
        predictions_path=pred_path,
        bootstrap_path=boot_path,
        metric_names=["accuracy", "f1_macro"],
        subset_type="per_protein_classification",
        num_classes=3,
    )

    assert "metrics" in result
    assert "bootstrap_values" in result

    for name in ("accuracy", "f1_macro"):
        assert name in result["metrics"]
        assert "value" in result["metrics"][name]
        assert "std" in result["metrics"][name]
        assert "ci95" in result["metrics"][name]
        assert name in result["bootstrap_values"]
        assert len(result["bootstrap_values"][name]) == 50  # num_bootstraps


def test_classification_metrics_reasonable(classification_fixtures):
    """Accuracy should be in [0, 1]."""
    pred_path, boot_path = classification_fixtures
    result = compute_metrics(
        predictions_path=pred_path,
        bootstrap_path=boot_path,
        metric_names=["accuracy"],
        subset_type="per_protein_classification",
        num_classes=3,
    )
    value = result["metrics"]["accuracy"]["value"]
    assert 0.0 <= value <= 1.0


def test_regression_metrics_structure(regression_fixtures):
    """Output has correct structure for regression metrics."""
    pred_path, boot_path = regression_fixtures
    result = compute_metrics(
        predictions_path=pred_path,
        bootstrap_path=boot_path,
        metric_names=["spearman", "pearson"],
        subset_type="per_protein_regression",
    )

    assert "metrics" in result
    assert "bootstrap_values" in result
    for name in ("spearman", "pearson"):
        assert name in result["metrics"]
        assert len(result["bootstrap_values"][name]) == 50


def test_regression_metrics_reasonable(regression_fixtures):
    """Strong positive correlation expected for our test data."""
    pred_path, boot_path = regression_fixtures
    result = compute_metrics(
        predictions_path=pred_path,
        bootstrap_path=boot_path,
        metric_names=["spearman"],
        subset_type="per_protein_regression",
    )
    value = result["metrics"]["spearman"]["value"]
    assert value > 0.5, f"Expected high correlation, got {value}"


def test_per_residue_classification(per_residue_fixtures):
    """Per-residue classification metrics work with list predictions."""
    pred_path, boot_path = per_residue_fixtures
    result = compute_metrics(
        predictions_path=pred_path,
        bootstrap_path=boot_path,
        metric_names=["accuracy"],
        subset_type="per_residue_classification",
        num_classes=2,
    )
    value = result["metrics"]["accuracy"]["value"]
    assert 0.0 <= value <= 1.0
    assert len(result["bootstrap_values"]["accuracy"]) == 30


def test_ci95_is_196_times_std(classification_fixtures):
    """CI95 should be exactly 1.96 * std."""
    pred_path, boot_path = classification_fixtures
    result = compute_metrics(
        predictions_path=pred_path,
        bootstrap_path=boot_path,
        metric_names=["accuracy"],
        subset_type="per_protein_classification",
        num_classes=3,
    )
    m = result["metrics"]["accuracy"]
    assert abs(m["ci95"] - 1.96 * m["std"]) < 1e-10
