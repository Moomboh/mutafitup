#!/usr/bin/env python3
"""Extract history.json and best_checkpoints.json from cached training checkpoints.

One-off workaround for runs that are still training but already have useful
best-overall models rsynced locally. This lets downstream Snakemake rules
(evaluation, plotting, representation-eval) see the runs as complete.

Usage:
    python scripts/extract_checkpoint_history.py [--force]

The three incomplete align-lora runs are hardcoded below.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

RUNS = [
    {
        "run_id": "esmc_300m_all_r4",
        "checkpoint": ".cache/training_checkpoints/align_lora/esmc_300m_all_r4/training_checkpoint.pt",
        "output_dir": "results/train/align_lora/esmc_300m_all_r4",
    },
    {
        "run_id": "esmc_300m_all_r16",
        "checkpoint": ".cache/training_checkpoints/align_lora/esmc_300m_all_r16/training_checkpoint.pt",
        "output_dir": "results/train/align_lora/esmc_300m_all_r16",
    },
]


def _reconstruct_best_checkpoints(
    history: list[dict[str, Any]],
    task_metrics: dict[str, list[float]],
    best_metrics: dict[str, float],
    best_epochs: dict[str, int],
    task_val_losses: dict[str, list[float]],
    best_val_losses: dict[str, float],
    best_loss_epochs: dict[str, int],
) -> dict[str, Any]:
    """Replicate the best_checkpoints logic from base_trainer.py lines 928-1000."""
    task_names = list(best_epochs.keys())

    # --- Best overall by relative metric improvement sum ---
    eps = 1e-6
    baseline_metrics = {name: eps for name in task_names}
    best_rel_improve_idx = 0
    for i in range(len(history)):
        total_rel_improvement = 0.0
        current_metrics: dict[str, float] = {}
        for name in task_names:
            metric = max(float(task_metrics[name][i]), eps)
            current_metrics[name] = metric
            baseline = max(baseline_metrics[name], eps)
            total_rel_improvement += (metric - baseline) / baseline
        if total_rel_improvement >= 0.0:
            baseline_metrics = dict(current_metrics)
            best_rel_improve_idx = i

    # --- Best overall by relative loss improvement sum ---
    best_loss_rel_improve_idx = 0
    loss_baseline: dict[str, float] = {}
    for i in range(len(history)):
        current_losses_at_i: dict[str, float] = {}
        for name in task_names:
            current_losses_at_i[name] = float(task_val_losses[name][i])
        if not loss_baseline:
            loss_baseline = dict(current_losses_at_i)
            best_loss_rel_improve_idx = i
        else:
            total_loss_rel_imp = 0.0
            for name in task_names:
                bl = loss_baseline[name]
                if bl > 0:
                    total_loss_rel_imp += (bl - current_losses_at_i[name]) / bl
                else:
                    if current_losses_at_i[name] <= 0:
                        pass
                    else:
                        total_loss_rel_imp -= 1.0
            if total_loss_rel_imp >= 0.0:
                loss_baseline = dict(current_losses_at_i)
                best_loss_rel_improve_idx = i

    return {
        "tasks": {
            name: {
                "epoch": int(best_epochs[name]),
                "metric": float(best_metrics[name]),
            }
            for name in task_names
        },
        "best_overall": {
            "epoch": history[best_rel_improve_idx]["epoch"],
            "metrics": {
                name: float(task_metrics[name][best_rel_improve_idx])
                for name in task_names
            },
        },
        "best_loss_tasks": {
            name: {
                "epoch": int(best_loss_epochs[name]),
                "loss": float(best_val_losses[name]),
            }
            for name in task_names
        },
        "best_loss_overall": {
            "epoch": history[best_loss_rel_improve_idx]["epoch"],
            "losses": {
                name: float(task_val_losses[name][best_loss_rel_improve_idx])
                for name in task_names
            },
        },
    }


def process_run(run: dict[str, str], *, force: bool) -> bool:
    """Extract history and best_checkpoints for one run. Returns True on success."""
    run_id = run["run_id"]
    checkpoint_path = Path(run["checkpoint"])
    output_dir = Path(run["output_dir"])

    history_path = output_dir / "history.json"
    best_checkpoints_path = output_dir / "best_checkpoints.json"

    if not checkpoint_path.exists():
        print(f"  SKIP {run_id}: checkpoint not found at {checkpoint_path}")
        return False

    if not output_dir.exists():
        print(f"  SKIP {run_id}: output directory does not exist at {output_dir}")
        return False

    if not force:
        existing = []
        if history_path.exists():
            existing.append("history.json")
        if best_checkpoints_path.exists():
            existing.append("best_checkpoints.json")
        if existing:
            print(
                f"  SKIP {run_id}: {', '.join(existing)} already exist (use --force to overwrite)"
            )
            return False

    print(f"  Loading {checkpoint_path} ...")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    history: list[dict[str, Any]] = ckpt["history"]
    task_metrics: dict[str, list[float]] = ckpt["task_metrics"]
    best_metrics: dict[str, float] = ckpt["best_metrics"]
    best_epochs: dict[str, int] = ckpt["best_epochs"]
    task_val_losses: dict[str, list[float]] = ckpt["task_val_losses"]
    best_val_losses: dict[str, float] = ckpt["best_val_losses"]
    best_loss_epochs: dict[str, int] = ckpt["best_loss_epochs"]

    task_names = list(best_epochs.keys())
    n_validations = len(history)
    last_epoch = history[-1]["epoch"] if history else "?"

    print(
        f"  {run_id}: {n_validations} validation steps, last epoch {last_epoch}, tasks: {task_names}"
    )

    # Write history.json
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"  Wrote {history_path} ({len(history)} entries)")

    # Reconstruct and write best_checkpoints.json
    best_checkpoints = _reconstruct_best_checkpoints(
        history=history,
        task_metrics=task_metrics,
        best_metrics=best_metrics,
        best_epochs=best_epochs,
        task_val_losses=task_val_losses,
        best_val_losses=best_val_losses,
        best_loss_epochs=best_loss_epochs,
    )
    best_checkpoints_path.write_text(
        json.dumps(best_checkpoints, indent=2), encoding="utf-8"
    )
    print(f"  Wrote {best_checkpoints_path}")
    print(f"  Best overall epoch: {best_checkpoints['best_overall']['epoch']}")

    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract history.json and best_checkpoints.json from cached training checkpoints."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing history.json / best_checkpoints.json files.",
    )
    args = parser.parse_args()

    print(f"Processing {len(RUNS)} runs ...\n")
    successes = 0
    for run in RUNS:
        print(f"[{run['run_id']}]")
        if process_run(run, force=args.force):
            successes += 1
        print()

    print(f"Done: {successes}/{len(RUNS)} runs extracted.")
    if successes < len(RUNS):
        sys.exit(1)


if __name__ == "__main__":
    main()
