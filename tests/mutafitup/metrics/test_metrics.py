import math

import pytest
import torch

from mutafitup.metrics import (
    compute_classification_metrics,
    compute_regression_metrics,
    get_metric_fn,
    ndcg_at_k,
)


def test_ndcg_at_k_perfect_ranking_is_one():
    preds = torch.tensor([3.0, 2.0, 1.0, 0.0])
    targets = torch.tensor([3.0, 2.0, 1.0, 0.0])

    score = ndcg_at_k(preds, targets, k=4).item()

    assert math.isclose(score, 1.0, rel_tol=1e-6)


def test_ndcg_at_k_matches_manual_computation_binary():
    targets = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0])
    preds = targets.clone()

    k = 3
    score = ndcg_at_k(preds, targets, k=k).item()

    gains = []
    discounts = []
    for i in range(k):
        rel = targets[i].item()
        gain = (2.0**rel) - 1.0
        discount = math.log2(i + 2)
        gains.append(gain)
        discounts.append(discount)

    dcg = sum(g / d for g, d in zip(gains, discounts))
    idcg = dcg

    expected = dcg / idcg if idcg > 0 else 0.0

    assert math.isclose(score, expected, rel_tol=1e-6)


def test_compute_classification_metrics_returns_all_with_ci():
    device = torch.device("cpu")

    targets = torch.tensor([0, 1, 0, 1, 0, 1])
    preds = targets.clone()

    metrics = compute_classification_metrics(
        preds, targets, num_classes=2, device=device, num_bootstraps=100
    )

    assert isinstance(metrics, dict)

    for key in ["accuracy", "f1_macro", "mcc"]:
        assert key in metrics
        assert 0.0 <= metrics[key]["value"] <= 1.0
        assert metrics[key]["std"] >= 0.0


def test_compute_regression_metrics_returns_all_with_ci():
    device = torch.device("cpu")

    targets = torch.tensor([3.0, 2.0, 1.0, 0.0])
    preds = targets.clone()

    metrics = compute_regression_metrics(
        preds, targets, device=device, num_bootstraps=100, ndcg_k=4
    )

    assert isinstance(metrics, dict)

    for key in ["spearman", "pearson", "ndcg_at_10"]:
        assert key in metrics
        assert -1.0 <= metrics[key]["value"] <= 1.0
        assert metrics[key]["std"] >= 0.0


def test_compute_classification_metrics_no_bootstrap():
    device = torch.device("cpu")

    targets = torch.tensor([0, 1, 0, 1, 0, 1])
    preds = targets.clone()

    metrics = compute_classification_metrics(
        preds, targets, num_classes=2, device=device, num_bootstraps=0
    )

    assert isinstance(metrics, dict)

    for key in ["accuracy", "f1_macro", "mcc"]:
        assert key in metrics
        assert 0.0 <= metrics[key]["value"] <= 1.0
        assert metrics[key]["std"] == 0.0


def test_compute_regression_metrics_no_bootstrap():
    device = torch.device("cpu")

    targets = torch.tensor([3.0, 2.0, 1.0, 0.0])
    preds = targets.clone()

    metrics = compute_regression_metrics(
        preds, targets, device=device, num_bootstraps=0, ndcg_k=4
    )

    assert isinstance(metrics, dict)

    for key in ["spearman", "pearson", "ndcg_at_10"]:
        assert key in metrics
        assert -1.0 <= metrics[key]["value"] <= 1.0
        assert metrics[key]["std"] == 0.0


def test_get_metric_fn_accuracy():
    fn = get_metric_fn("accuracy", num_classes=3)
    preds = torch.tensor([0, 1, 2, 0, 1])
    targets = torch.tensor([0, 1, 2, 0, 1])
    result = fn(preds, targets)
    assert result.item() == 1.0


def test_get_metric_fn_f1_macro():
    fn = get_metric_fn("f1_macro", num_classes=3)
    preds = torch.tensor([0, 1, 2, 0, 1])
    targets = torch.tensor([0, 1, 2, 0, 1])
    result = fn(preds, targets)
    assert result.item() > 0.0


def test_get_metric_fn_spearman():
    fn = get_metric_fn("spearman")
    preds = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    targets = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    result = fn(preds, targets)
    assert abs(result.item() - 1.0) < 1e-5


def test_get_metric_fn_pearson():
    fn = get_metric_fn("pearson")
    preds = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    targets = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    result = fn(preds, targets)
    assert abs(result.item() - 1.0) < 1e-5


def test_get_metric_fn_mcc():
    fn = get_metric_fn("mcc", num_classes=2)
    preds = torch.tensor([0, 1, 0, 1])
    targets = torch.tensor([0, 1, 0, 1])
    result = fn(preds, targets)
    assert result.item() > 0.0


def test_get_metric_fn_unknown_raises():
    with pytest.raises(ValueError, match="Unknown metric name"):
        get_metric_fn("nonexistent_metric")
