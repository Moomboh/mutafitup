import torch

from mutafitup.metrics import (
    bootstrap_paired_metric_delta,
    get_metric_fn,
    get_primary_metric_fn,
)


def test_bootstrap_paired_metric_delta_identical():
    """When baseline and approach are the same, delta should be ~0."""
    preds = torch.randn(100)
    targets = torch.randn(100)

    def dummy_metric(p, t):
        return p.mean()

    result = bootstrap_paired_metric_delta(
        preds, targets, preds, targets, dummy_metric, num_bootstraps=200
    )

    assert abs(result["mean"]) < 0.15
    assert result["ci95"] >= 0


def test_bootstrap_paired_metric_delta_positive():
    """Approach is strictly better: delta should be positive."""
    preds_base = torch.zeros(50)
    targets = torch.ones(50)
    preds_app = torch.ones(50)

    def simple_metric(p, t):
        return (p == t).float().mean()

    result = bootstrap_paired_metric_delta(
        preds_base, targets, preds_app, targets, simple_metric, num_bootstraps=200
    )

    assert result["mean"] > 0.5
    assert result["significant"]


def test_bootstrap_paired_metric_delta_empty():
    """Empty tensors should return zeros."""
    result = bootstrap_paired_metric_delta(
        torch.tensor([]),
        torch.tensor([]),
        torch.tensor([]),
        torch.tensor([]),
        lambda p, t: p.mean(),
    )

    assert result["mean"] == 0.0
    assert result["std"] == 0.0
    assert not result["significant"]


def test_get_primary_metric_fn_classification():
    """Backward compat alias still works for classification."""
    fn = get_primary_metric_fn("classification", num_classes=3)
    preds = torch.tensor([0, 1, 2, 0, 1])
    targets = torch.tensor([0, 1, 2, 0, 1])
    result = fn(preds, targets)
    assert result.item() == 1.0


def test_get_primary_metric_fn_regression():
    """Backward compat alias still works for regression."""
    fn = get_primary_metric_fn("regression")
    preds = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    targets = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    result = fn(preds, targets)
    assert abs(result.item() - 1.0) < 1e-5


def test_get_metric_fn_accuracy():
    fn = get_metric_fn("accuracy", num_classes=3)
    preds = torch.tensor([0, 1, 2, 0, 1])
    targets = torch.tensor([0, 1, 2, 0, 1])
    result = fn(preds, targets)
    assert result.item() == 1.0


def test_get_metric_fn_spearman():
    fn = get_metric_fn("spearman")
    preds = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    targets = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    result = fn(preds, targets)
    assert abs(result.item() - 1.0) < 1e-5


def test_get_metric_fn_f1_macro():
    fn = get_metric_fn("f1_macro", num_classes=2)
    preds = torch.tensor([0, 1, 0, 1])
    targets = torch.tensor([0, 1, 0, 1])
    result = fn(preds, targets)
    assert result.item() > 0.0
