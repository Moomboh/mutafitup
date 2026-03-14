"""Per-run metric computation with protein-level bootstraps.

Loads prediction JSONL files and bootstrap index parquets, then computes
metrics on each bootstrap sample (resampling at the protein level).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from torchmetrics import PearsonCorrCoef, SpearmanCorrCoef
from torchmetrics.classification import (
    MatthewsCorrCoef,
    MulticlassAccuracy,
    MulticlassF1Score,
)


def _load_predictions_jsonl(path: Union[str, Path]) -> List[dict]:
    """Load prediction records from a JSONL file."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _make_metric_fn(metric_name: str, num_classes: int = 2):
    """Create a torchmetrics callable for *metric_name*."""
    if metric_name == "accuracy":
        m = MulticlassAccuracy(num_classes=num_classes)

        def fn(p, t):
            m.reset()
            return m(p, t)

        return fn

    if metric_name == "f1_macro":
        m = MulticlassF1Score(num_classes=num_classes, average="macro")

        def fn(p, t):
            m.reset()
            return m(p, t)

        return fn

    if metric_name == "mcc":
        m = MatthewsCorrCoef(task="multiclass", num_classes=num_classes)

        def fn(p, t):
            m.reset()
            return m(p, t)

        return fn

    if metric_name == "spearman":
        m = SpearmanCorrCoef()

        def fn(p, t):
            m.reset()
            return m(p, t)

        return fn

    if metric_name == "pearson":
        m = PearsonCorrCoef()

        def fn(p, t):
            m.reset()
            return m(p, t)

        return fn

    raise ValueError(f"Unknown metric: {metric_name!r}")


def _safe_compute(fn, preds: torch.Tensor, targets: torch.Tensor) -> Optional[float]:
    """Compute a metric, returning None on NaN."""
    val = fn(preds, targets)
    if isinstance(val, torch.Tensor):
        val = val.float()
        if torch.isnan(val):
            return None
        return val.item()
    if math.isnan(val):
        return None
    return float(val)


def compute_metrics(
    predictions_path: Union[str, Path],
    bootstrap_path: Union[str, Path],
    metric_names: Sequence[str],
    subset_type: str,
    num_classes: int = 2,
) -> Dict[str, Any]:
    """Compute bootstrapped metrics for a single run / task / variant / split.

    Parameters
    ----------
    predictions_path : str or Path
        Path to the predictions JSONL file.
    bootstrap_path : str or Path
        Path to the bootstrap indices parquet file.
    metric_names : sequence of str
        Which metrics to compute (e.g. ``["accuracy", "f1_macro", "mcc"]``).
    subset_type : str
        One of ``per_residue_classification``, ``per_residue_regression``,
        ``per_protein_classification``, ``per_protein_regression``.
    num_classes : int
        Number of classes (for classification metrics).

    Returns
    -------
    dict
        ``{"metrics": {...}, "bootstrap_values": {...}}`` where
        ``metrics[name]`` has keys ``value``, ``std``, ``ci95`` and
        ``bootstrap_values[name]`` is a list of floats (one per bootstrap).
    """
    records = _load_predictions_jsonl(predictions_path)
    bootstrap_df = pd.read_parquet(bootstrap_path)

    is_per_residue = "per_residue" in subset_type

    # Build per-protein structures
    # protein_ids[i] = sequence id for sample i
    # For per-residue: preds_flat[i] and targets_flat[i] are lists
    # For per-protein: preds_flat[i] and targets_flat[i] are scalars
    protein_ids = []
    preds_per_protein = []
    targets_per_protein = []

    for rec in records:
        protein_ids.append(rec["id"])
        preds_per_protein.append(rec["prediction"])
        targets_per_protein.append(rec["target"])

    # Map protein IDs to indices in the records list
    id_to_indices: Dict[str, List[int]] = {}
    for idx, pid in enumerate(protein_ids):
        id_to_indices.setdefault(pid, []).append(idx)

    num_bootstraps = len([c for c in bootstrap_df.columns if c.startswith("b_")])
    boot_columns = [f"b_{i}" for i in range(num_bootstraps)]

    # Build metric functions
    metric_fns = {}
    for name in metric_names:
        metric_fns[name] = _make_metric_fn(name, num_classes=num_classes)

    # Compute metrics for each bootstrap
    bootstrap_values: Dict[str, List[float]] = {name: [] for name in metric_names}

    for col in boot_columns:
        counts = bootstrap_df[col]

        # Collect all predictions/targets for this bootstrap (protein-level resampling)
        all_preds = []
        all_targets = []

        for seq_id, count in counts.items():
            if count == 0:
                continue
            indices = id_to_indices.get(seq_id, [])
            for _ in range(int(count)):
                for idx in indices:
                    p = preds_per_protein[idx]
                    t = targets_per_protein[idx]
                    if is_per_residue:
                        all_preds.extend(p)
                        all_targets.extend(t)
                    else:
                        all_preds.append(p)
                        all_targets.append(t)

        if not all_preds:
            for name in metric_names:
                bootstrap_values[name].append(float("nan"))
            continue

        preds_tensor = torch.tensor(
            all_preds,
            dtype=torch.long if "classification" in subset_type else torch.float32,
        )
        targets_tensor = torch.tensor(
            all_targets,
            dtype=torch.long if "classification" in subset_type else torch.float32,
        )

        for name in metric_names:
            val = _safe_compute(metric_fns[name], preds_tensor, targets_tensor)
            if val is not None:
                bootstrap_values[name].append(val)
            else:
                bootstrap_values[name].append(float("nan"))

    # Aggregate
    result_metrics: Dict[str, Dict[str, float]] = {}
    result_bootstrap: Dict[str, List[float]] = {}

    for name in metric_names:
        vals = [v for v in bootstrap_values[name] if not math.isnan(v)]
        if vals:
            arr = np.array(vals)
            mean = float(arr.mean())
            std = float(arr.std(ddof=0))
            ci95 = 1.96 * std
        else:
            mean = 0.0
            std = 0.0
            ci95 = 0.0

        result_metrics[name] = {"value": mean, "std": std, "ci95": ci95}
        result_bootstrap[name] = bootstrap_values[name]

    return {
        "metrics": result_metrics,
        "bootstrap_values": result_bootstrap,
    }
