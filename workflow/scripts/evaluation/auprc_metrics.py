"""Snakemake script: Compute bootstrapped AUPRC for a single run/task."""

import json
import os

from snakemake.script import snakemake

from mutafitup.evaluation.auprc_metrics import compute_auprc
from wfutils.logging import get_logger, log_snakemake_info

logger = get_logger(__name__)
log_snakemake_info(logger)

prob_predictions_path = snakemake.params["prob_predictions_path"]
bootstrap_path = snakemake.input["bootstraps"]
output_path = snakemake.output["metrics"]

section = snakemake.params["section"]
run_id = snakemake.params["run_id"]
variant = snakemake.params["variant"]
split = snakemake.params["split"]
task = snakemake.params["task"]

if not os.path.exists(prob_predictions_path):
    # The run was not trained on this task — write a sentinel JSON with null
    # metrics so downstream AUPRC summary can show a neutral "N/A" cell.
    logger.warning(
        f"Probability predictions file not found: {prob_predictions_path} — "
        f"writing null AUPRC metrics (run likely not trained on task '{task}')"
    )
    output = {
        "dataset": task,
        "subset_type": "per_residue_classification",
        "variant": variant,
        "split": split,
        "run": {"section": section, "id": run_id},
        "metrics": None,
        "bootstrap_values": None,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Wrote null AUPRC metrics to {output_path}")
else:
    logger.info(f"Computing AUPRC for {section}/{run_id}/{variant}/{split}/{task}")

    result = compute_auprc(
        prob_predictions_path=prob_predictions_path,
        bootstrap_path=bootstrap_path,
    )

    # Add metadata
    output = {
        "dataset": task,
        "subset_type": "per_residue_classification",
        "variant": variant,
        "split": split,
        "run": {"section": section, "id": run_id},
        "metrics": result["metrics"],
        "bootstrap_values": result["bootstrap_values"],
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Wrote AUPRC metrics to {output_path}")
    m = result["metrics"]["auprc"]
    logger.info(f"  auprc: {m['value']:.4f} +/- {m['std']:.4f} (ci95={m['ci95']:.4f})")
