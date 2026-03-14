"""AUPRC metric computation with protein-level bootstraps.

Loads probability prediction JSONL files and bootstrap index parquets,
then computes AUPRC (Area Under Precision-Recall Curve) on each bootstrap
sample (resampling at the protein level).

This is a completely independent pipeline from the existing metrics module
and does NOT modify or interfere with cached prediction/metric outputs.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from torchmetrics.classification import BinaryAveragePrecision


def _load_prob_predictions_jsonl(path: Union[str, Path]) -> List[dict]:
    """Load probability prediction records from a JSONL file.

    Each record has keys: ``id``, ``probability`` (list of floats),
    ``target`` (list of ints).
    """
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_auprc(
    prob_predictions_path: Union[str, Path],
    bootstrap_path: Union[str, Path],
) -> Dict[str, Any]:
    """Compute bootstrapped AUPRC for a single run / task / variant / split.

    Parameters
    ----------
    prob_predictions_path : str or Path
        Path to the probability predictions JSONL file (from predict_proba).
    bootstrap_path : str or Path
        Path to the bootstrap indices parquet file.

    Returns
    -------
    dict
        ``{"metrics": {"auprc": {"value": ..., "std": ..., "ci95": ...}},
          "bootstrap_values": {"auprc": [...]}}``
    """
    records = _load_prob_predictions_jsonl(prob_predictions_path)
    bootstrap_df = pd.read_parquet(bootstrap_path)

    # Build per-protein structures
    protein_ids: List[str] = []
    probs_per_protein: List[List[float]] = []
    targets_per_protein: List[List[int]] = []

    for rec in records:
        protein_ids.append(rec["id"])
        probs_per_protein.append(rec["probability"])
        targets_per_protein.append(rec["target"])

    # Map protein IDs to indices in the records list
    id_to_indices: Dict[str, List[int]] = {}
    for idx, pid in enumerate(protein_ids):
        id_to_indices.setdefault(pid, []).append(idx)

    num_bootstraps = len([c for c in bootstrap_df.columns if c.startswith("b_")])
    boot_columns = [f"b_{i}" for i in range(num_bootstraps)]

    metric = BinaryAveragePrecision()

    bootstrap_values: List[float] = []

    for col in boot_columns:
        counts = bootstrap_df[col]

        all_probs: List[float] = []
        all_targets: List[int] = []

        for seq_id, count in counts.items():
            if count == 0:
                continue
            indices = id_to_indices.get(seq_id, [])
            for _ in range(int(count)):
                for idx in indices:
                    all_probs.extend(probs_per_protein[idx])
                    all_targets.extend(targets_per_protein[idx])

        if not all_probs:
            bootstrap_values.append(float("nan"))
            continue

        probs_tensor = torch.tensor(all_probs, dtype=torch.float32)
        targets_tensor = torch.tensor(all_targets, dtype=torch.long)

        metric.reset()
        val = metric(probs_tensor, targets_tensor)

        if isinstance(val, torch.Tensor):
            val = val.float()
            if torch.isnan(val):
                bootstrap_values.append(float("nan"))
            else:
                bootstrap_values.append(val.item())
        elif math.isnan(val):
            bootstrap_values.append(float("nan"))
        else:
            bootstrap_values.append(float(val))

    # Aggregate
    valid_vals = [v for v in bootstrap_values if not math.isnan(v)]
    if valid_vals:
        arr = np.array(valid_vals)
        mean = float(arr.mean())
        std = float(arr.std(ddof=0))
        ci95 = 1.96 * std
    else:
        mean = 0.0
        std = 0.0
        ci95 = 0.0

    return {
        "metrics": {
            "auprc": {"value": mean, "std": std, "ci95": ci95},
        },
        "bootstrap_values": {
            "auprc": bootstrap_values,
        },
    }
