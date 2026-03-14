from __future__ import annotations

from typing import Any, Dict, Iterable, Sequence, cast

import pandas as pd
from torch import nn

from mutafitup.models import build_backbone_and_tokenizer
from mutafitup.models.multitask_heads import (
    PerProteinClassificationHead,
    PerProteinRegressionHead,
    PerResidueClassificationHead,
    PerResidueRegressionHead,
)


SECTION_ORDER = {
    "heads_only": 0,
    "lora": 1,
    "accgrad_lora": 2,
    "align_lora": 3,
}


def make_run_key(section: str, run_id: str) -> str:
    return f"{section}/{run_id}"


def filter_run_entries(
    run_entries: Sequence[Dict[str, Any]],
    include_run_keys: Sequence[str] | None = None,
) -> list[Dict[str, Any]]:
    if not include_run_keys:
        return list(run_entries)

    include_set = set(include_run_keys)
    filtered = [
        entry
        for entry in run_entries
        if make_run_key(entry["section"], entry["id"]) in include_set
    ]

    found_keys = {make_run_key(entry["section"], entry["id"]) for entry in filtered}
    missing = [key for key in include_run_keys if key not in found_keys]
    if missing:
        raise ValueError(f"Unknown parameter_counts.include_runs entries: {missing}")

    return filtered


def count_module_parameters(module: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def build_head(task: Dict[str, Any], dropout: float, hidden_size: int) -> nn.Module:
    subset_type = task["subset_type"]
    if subset_type == "per_residue_classification":
        return PerResidueClassificationHead(
            dropout,
            hidden_size,
            hidden_size,
            task["num_labels"],
        )
    if subset_type == "per_residue_regression":
        return PerResidueRegressionHead(dropout, hidden_size, hidden_size)
    if subset_type == "per_protein_classification":
        return PerProteinClassificationHead(
            dropout,
            hidden_size,
            hidden_size,
            task["num_labels"],
        )
    if subset_type == "per_protein_regression":
        return PerProteinRegressionHead(dropout, hidden_size, hidden_size)
    raise ValueError(f"Unsupported subset_type: {subset_type}")


def count_heads(
    tasks: Sequence[Dict[str, Any]],
    dropout: float,
    hidden_size: int,
) -> tuple[int, int]:
    total = 0
    trainable = 0
    for task in tasks:
        head = build_head(task, dropout, hidden_size)
        t, tr = count_module_parameters(head)
        total += t
        trainable += tr
    return total, trainable


def summarize_parameter_counts(run_entries: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    backbone_cache: dict[tuple[str, Any, Any], dict[str, int]] = {}
    rows = []

    for entry in run_entries:
        checkpoint = entry["checkpoint"]
        lora_rank = entry.get("lora_rank")
        lora_alpha = entry.get("lora_alpha")
        cache_key = (checkpoint, lora_rank, lora_alpha)

        if cache_key not in backbone_cache:
            backbone, _ = build_backbone_and_tokenizer(
                checkpoint, lora_rank, lora_alpha
            )
            backbone_total, backbone_trainable = count_module_parameters(backbone)
            hidden_size = cast(int, getattr(backbone, "hidden_size"))
            backbone_cache[cache_key] = {
                "hidden_size": hidden_size,
                "total": backbone_total,
                "trainable": backbone_trainable,
            }

        backbone_info = backbone_cache[cache_key]
        head_total, head_trainable = count_heads(
            entry["tasks"],
            entry["dropout"],
            backbone_info["hidden_size"],
        )
        total_params = backbone_info["total"] + head_total
        trainable_params = backbone_info["trainable"] + head_trainable

        rows.append(
            {
                "section": entry["section"],
                "run_id": entry["id"],
                "checkpoint": checkpoint,
                "num_tasks": len(entry["tasks"]),
                "task_names": [t["name"] for t in entry["tasks"]],
                "lora_rank": lora_rank,
                "lora_alpha": lora_alpha,
                "total_params": total_params,
                "trainable_params": trainable_params,
                "frozen_params": total_params - trainable_params,
                "trainable_fraction": trainable_params / total_params
                if total_params
                else 0.0,
                "backbone_total_params": backbone_info["total"],
                "backbone_trainable_params": backbone_info["trainable"],
                "head_total_params": head_total,
                "head_trainable_params": head_trainable,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["section_order"] = df["section"].apply(
        lambda section: SECTION_ORDER.get(section, 99)
    )
    return df.sort_values(["section_order", "run_id"]).drop(columns=["section_order"])
