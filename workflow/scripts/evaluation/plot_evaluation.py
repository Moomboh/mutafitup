"""Snakemake script: Generate evaluation plots and summary."""

import json
import os
from pathlib import Path

from snakemake.script import snakemake

from mutafitup.evaluation.plotting import (
    generate_evaluation_summary,
    load_metric_json,
    plot_delta_heatmap,
    plot_delta_summary,
)
from wfutils.logging import get_logger, log_snakemake_info

logger = get_logger(__name__)
log_snakemake_info(logger)

evaluation = snakemake.params["evaluation"]
eval_id = evaluation["id"]
split = evaluation["split"]
variant = evaluation["variant"]

# Load all metric JSONs
metric_json_paths = snakemake.input["metric_jsons"]
if isinstance(metric_json_paths, str):
    metric_json_paths = [metric_json_paths]

metric_jsons = {}
for path in metric_json_paths:
    # Extract the key from the path: {section}/{run_id}/{variant}/{split}/{task}
    # Path is: results/metrics/{section}/{run_id}/{variant}/{split}/{task}.json
    p = Path(path)
    task = p.stem  # task name without .json
    split_dir = p.parent.name
    variant_dir = p.parent.parent.name
    run_id = p.parent.parent.parent.name
    section = p.parent.parent.parent.parent.name
    key = f"{section}/{run_id}/{variant_dir}/{split_dir}/{task}"
    logger.info(f"Loading metric JSON: {key}")
    metric_jsons[key] = load_metric_json(path)

# Generate heatmap
heatmap_path = snakemake.output["heatmap"]
os.makedirs(os.path.dirname(heatmap_path), exist_ok=True)
logger.info(f"Generating delta heatmap: {heatmap_path}")
plot_delta_heatmap(
    evaluation=evaluation,
    metric_jsons=metric_jsons,
    output_path=heatmap_path,
    train_config=snakemake.params.get("train_config", {}),
)

# Generate delta summary plot
delta_summary_plot_path = snakemake.output["delta_summary_plot"]
logger.info(f"Generating delta summary plot: {delta_summary_plot_path}")
plot_delta_summary(
    evaluation=evaluation,
    metric_jsons=metric_jsons,
    output_path=delta_summary_plot_path,
    train_config=snakemake.params.get("train_config", {}),
)

# Generate summary JSON
summary_path = snakemake.output["summary"]
logger.info(f"Generating summary: {summary_path}")
summary = generate_evaluation_summary(
    evaluation=evaluation,
    metric_jsons=metric_jsons,
)
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

logger.info("Evaluation plotting complete")
