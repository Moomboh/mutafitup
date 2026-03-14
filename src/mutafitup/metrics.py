import math
from typing import Callable, Dict

import torch
from torchmetrics import PearsonCorrCoef, SpearmanCorrCoef
from torchmetrics.classification import (
    MatthewsCorrCoef,
    MulticlassAccuracy,
    MulticlassF1Score,
)


def ndcg_at_k(preds: torch.Tensor, targets: torch.Tensor, k: int = 10) -> torch.Tensor:
    if preds.numel() == 0:
        return torch.tensor(0.0, device=preds.device, dtype=preds.dtype)

    preds = preds.view(-1)
    targets = targets.view(-1)

    k = min(k, preds.numel())

    _, indices = torch.topk(preds, k)
    rel = torch.relu(targets[indices])
    discounts = torch.log2(
        torch.arange(2, k + 2, device=preds.device, dtype=preds.dtype)
    )
    gains = (
        torch.pow(torch.tensor(2.0, device=preds.device, dtype=preds.dtype), rel) - 1.0
    )

    dcg = torch.sum(gains / discounts)

    _, ideal_indices = torch.topk(targets, k)
    ideal_rel = torch.relu(targets[ideal_indices])
    ideal_gains = (
        torch.pow(torch.tensor(2.0, device=preds.device, dtype=preds.dtype), ideal_rel)
        - 1.0
    )
    ideal_dcg = torch.sum(ideal_gains / discounts)

    if ideal_dcg <= 0:
        return torch.tensor(0.0, device=preds.device, dtype=preds.dtype)

    return dcg / ideal_dcg


def _bootstrap_scalar_metric(
    preds: torch.Tensor,
    targets: torch.Tensor,
    metric_fn,
    num_bootstraps: int = 1000,
) -> dict:
    preds = preds.view(-1)
    targets = targets.view(-1)

    n = preds.numel()
    if n == 0:
        return {"mean": 0.0, "std": 0.0}

    # Point estimate only — no resampling
    if num_bootstraps <= 0:
        v = metric_fn(preds, targets)
        if isinstance(v, torch.Tensor):
            v = v.to(device=torch.device(preds.device), dtype=torch.float32)
            if torch.isnan(v):
                return {"mean": 0.0, "std": 0.0}
            v = v.item()
        elif math.isnan(v):
            return {"mean": 0.0, "std": 0.0}
        return {"mean": float(v), "std": 0.0}

    device = preds.device
    values = []

    for _ in range(num_bootstraps):
        indices = torch.randint(0, n, (n,), device=device)
        v = metric_fn(preds[indices], targets[indices])
        if isinstance(v, torch.Tensor):
            v = v.to(device=torch.device(device), dtype=torch.float32)
            if torch.isnan(v):
                continue
            v = v.item()
        else:
            if math.isnan(v):
                continue
        values.append(float(v))

    if not values:
        return {"mean": 0.0, "std": 0.0}

    samples = torch.tensor(values, device=device, dtype=torch.float32)
    mean = samples.mean().item()
    std = samples.std(unbiased=False).item()

    return {"mean": mean, "std": std}


def compute_classification_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    device: torch.device,
    num_bootstraps: int = 1000,
) -> Dict[str, Dict[str, float]]:
    """Compute bootstrapped classification metrics.

    Returns a dict mapping metric names to {value, std} dicts.
    The caller decides which metric is primary via config.
    """
    preds = preds.to(device)
    targets = targets.to(device)

    acc_metric = MulticlassAccuracy(num_classes=num_classes).to(device)

    def acc_fn(p, t):
        acc_metric.reset()
        return acc_metric(p, t)

    acc_stats = _bootstrap_scalar_metric(preds, targets, acc_fn, num_bootstraps)

    f1_metric = MulticlassF1Score(num_classes=num_classes, average="macro").to(device)

    def f1_fn(p, t):
        f1_metric.reset()
        return f1_metric(p, t)

    f1_stats = _bootstrap_scalar_metric(preds, targets, f1_fn, num_bootstraps)

    mcc_metric = MatthewsCorrCoef(task="multiclass", num_classes=num_classes).to(device)

    def mcc_fn(p, t):
        mcc_metric.reset()
        return mcc_metric(p, t)

    mcc_stats = _bootstrap_scalar_metric(preds, targets, mcc_fn, num_bootstraps)

    metrics = {
        "accuracy": {"value": acc_stats["mean"], "std": acc_stats["std"]},
        "f1_macro": {"value": f1_stats["mean"], "std": f1_stats["std"]},
        "mcc": {"value": mcc_stats["mean"], "std": mcc_stats["std"]},
    }

    return metrics


def compute_regression_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    num_bootstraps: int = 1000,
    ndcg_k: int = 10,
) -> Dict[str, Dict[str, float]]:
    """Compute bootstrapped regression metrics.

    Returns a dict mapping metric names to {value, std} dicts.
    The caller decides which metric is primary via config.
    """
    preds = preds.to(device)
    targets = targets.to(device)

    spear_metric = SpearmanCorrCoef().to(device)

    def spear_fn(p, t):
        spear_metric.reset()
        return spear_metric(p, t)

    spear_stats = _bootstrap_scalar_metric(preds, targets, spear_fn, num_bootstraps)

    pearson_metric = PearsonCorrCoef().to(device)

    def pearson_fn(p, t):
        pearson_metric.reset()
        return pearson_metric(p, t)

    pearson_stats = _bootstrap_scalar_metric(preds, targets, pearson_fn, num_bootstraps)

    def ndcg_fn(p, t):
        return ndcg_at_k(p, t, k=ndcg_k)

    ndcg_stats = _bootstrap_scalar_metric(preds, targets, ndcg_fn, num_bootstraps)

    metrics = {
        "spearman": {"value": spear_stats["mean"], "std": spear_stats["std"]},
        "pearson": {"value": pearson_stats["mean"], "std": pearson_stats["std"]},
        "ndcg_at_10": {"value": ndcg_stats["mean"], "std": ndcg_stats["std"]},
    }

    return metrics


def bootstrap_paired_metric_delta(
    preds_baseline: torch.Tensor,
    targets_baseline: torch.Tensor,
    preds_approach: torch.Tensor,
    targets_approach: torch.Tensor,
    metric_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    num_bootstraps: int = 1000,
    percent: bool = False,
) -> Dict[str, float]:
    """Compute the bootstrapped paired delta between two sets of predictions.

    Both prediction sets must share the same sample ordering so that
    bootstrap index resampling is paired.  The delta is computed as
    ``metric(approach) - metric(baseline)`` on each bootstrap resample.

    When *percent* is True the delta is expressed as a percentage
    relative to the baseline value:
    ``(metric(approach) - metric(baseline)) / |metric(baseline)| * 100``.

    Parameters
    ----------
    preds_baseline, targets_baseline : Tensor
        Predictions and targets for the baseline run (1-D).
    preds_approach, targets_approach : Tensor
        Predictions and targets for the approach run (1-D).
    metric_fn : callable
        ``(preds, targets) -> scalar Tensor``.
    num_bootstraps : int
    percent : bool
        If True, return the percent difference relative to baseline.

    Returns
    -------
    dict with keys ``mean``, ``std``, ``ci95`` (= 1.96 * std),
    ``significant`` (bool, True if CI excludes 0).
    """
    preds_baseline = preds_baseline.view(-1)
    targets_baseline = targets_baseline.view(-1)
    preds_approach = preds_approach.view(-1)
    targets_approach = targets_approach.view(-1)

    n_baseline = preds_baseline.numel()
    n_approach = preds_approach.numel()

    if n_baseline == 0 or n_approach == 0:
        return {"mean": 0.0, "std": 0.0, "ci95": 0.0, "significant": False}

    device = preds_baseline.device
    deltas = []

    for _ in range(num_bootstraps):
        idx_b = torch.randint(0, n_baseline, (n_baseline,), device=device)
        idx_a = torch.randint(0, n_approach, (n_approach,), device=device)

        val_b = metric_fn(preds_baseline[idx_b], targets_baseline[idx_b])
        val_a = metric_fn(preds_approach[idx_a], targets_approach[idx_a])

        if isinstance(val_b, torch.Tensor):
            val_b = val_b.float()
            if torch.isnan(val_b):
                continue
            val_b = val_b.item()
        elif math.isnan(val_b):
            continue

        if isinstance(val_a, torch.Tensor):
            val_a = val_a.float()
            if torch.isnan(val_a):
                continue
            val_a = val_a.item()
        elif math.isnan(val_a):
            continue

        if percent:
            if abs(val_b) < 1e-12:
                continue
            deltas.append((val_a - val_b) / abs(val_b) * 100.0)
        else:
            deltas.append(val_a - val_b)

    if not deltas:
        return {"mean": 0.0, "std": 0.0, "ci95": 0.0, "significant": False}

    samples = torch.tensor(deltas, dtype=torch.float32)
    mean = samples.mean().item()
    std = samples.std(unbiased=False).item()
    ci95 = 1.96 * std
    significant = abs(mean) > ci95

    return {
        "mean": mean,
        "std": std,
        "ci95": ci95,
        "significant": significant,
    }


def get_metric_fn(
    metric_name: str,
    num_classes: int = 2,
    device: torch.device = torch.device("cpu"),
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Return a metric function for the given metric name.

    Supported metric names: 'accuracy', 'f1_macro', 'mcc', 'spearman', 'pearson'.
    """
    if metric_name == "accuracy":
        acc_metric = MulticlassAccuracy(num_classes=num_classes).to(device)

        def acc_fn(p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            acc_metric.reset()
            return acc_metric(p, t)

        return acc_fn

    if metric_name == "f1_macro":
        f1_metric = MulticlassF1Score(num_classes=num_classes, average="macro").to(
            device
        )

        def f1_fn(p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            f1_metric.reset()
            return f1_metric(p, t)

        return f1_fn

    if metric_name == "mcc":
        mcc_metric = MatthewsCorrCoef(task="multiclass", num_classes=num_classes).to(
            device
        )

        def mcc_fn(p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            mcc_metric.reset()
            return mcc_metric(p, t)

        return mcc_fn

    if metric_name == "spearman":
        spear_metric = SpearmanCorrCoef().to(device)

        def spear_fn(p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            spear_metric.reset()
            return spear_metric(p, t)

        return spear_fn

    if metric_name == "pearson":
        pearson_metric = PearsonCorrCoef().to(device)

        def pearson_fn(p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            pearson_metric.reset()
            return pearson_metric(p, t)

        return pearson_fn

    raise ValueError(f"Unknown metric name: {metric_name!r}")


# Backward compatibility alias
def get_primary_metric_fn(
    problem_type: str,
    num_classes: int = 2,
    device: torch.device = torch.device("cpu"),
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Return the primary metric function for the given problem type.

    Deprecated: use get_metric_fn() instead.
    """
    if problem_type == "classification":
        return get_metric_fn("accuracy", num_classes=num_classes, device=device)
    return get_metric_fn("spearman", device=device)
