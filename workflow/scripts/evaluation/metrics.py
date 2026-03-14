"""Snakemake script: Compute bootstrapped per-run metrics."""

import json
import os

from snakemake.script import snakemake

from mutafitup.evaluation.metrics import compute_metrics
from wfutils.logging import get_logger, log_snakemake_info

logger = get_logger(__name__)
log_snakemake_info(logger)

predictions_path = snakemake.params["predictions_path"]
bootstrap_path = snakemake.input["bootstraps"]
output_path = snakemake.output["metrics"]

metric_names = snakemake.params["metric_names"]
subset_type = snakemake.params["subset_type"]
num_classes = snakemake.params["num_classes"]
section = snakemake.params["section"]
run_id = snakemake.params["run_id"]
variant = snakemake.params["variant"]
split = snakemake.params["split"]
task = snakemake.params["task"]

if not os.path.exists(predictions_path):
    # The run was not trained on this task — write a sentinel JSON with null
    # metrics so downstream evaluation plots can show a neutral "N/A" cell.
    logger.warning(
        f"Predictions file not found: {predictions_path} — "
        f"writing null metrics (run likely not trained on task '{task}')"
    )
    output = {
        "dataset": task,
        "subset_type": subset_type,
        "variant": variant,
        "split": split,
        "run": {"section": section, "id": run_id},
        "metrics": None,
        "bootstrap_values": None,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Wrote null metrics to {output_path}")
else:
    logger.info(
        f"Computing metrics for {section}/{run_id}/{variant}/{split}/{task} "
        f"(metrics: {metric_names})"
    )

    result = compute_metrics(
        predictions_path=predictions_path,
        bootstrap_path=bootstrap_path,
        metric_names=metric_names,
        subset_type=subset_type,
        num_classes=num_classes,
    )

    # Add metadata
    output = {
        "dataset": task,
        "subset_type": subset_type,
        "variant": variant,
        "split": split,
        "run": {"section": section, "id": run_id},
        "metrics": result["metrics"],
        "bootstrap_values": result["bootstrap_values"],
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Wrote metrics to {output_path}")
    for name, m in result["metrics"].items():
        logger.info(
            f"  {name}: {m['value']:.4f} +/- {m['std']:.4f} (ci95={m['ci95']:.4f})"
        )
