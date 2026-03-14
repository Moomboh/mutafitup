"""Snakemake script: Generate one evaluation per-task bar chart."""

import os
from pathlib import Path

from snakemake.script import snakemake

from mutafitup.evaluation.plotting import load_metric_json, plot_absolute_bars
from wfutils.logging import get_logger, log_snakemake_info

logger = get_logger(__name__)
log_snakemake_info(logger)

evaluation = snakemake.params["evaluation"]
task = str(snakemake.wildcards["task"])
metric_name = snakemake.params["metric_name"]

metric_json_paths = snakemake.input["metric_jsons"]
if isinstance(metric_json_paths, str):
    metric_json_paths = [metric_json_paths]

metric_jsons = {}
for path in metric_json_paths:
    p = Path(path)
    task_name = p.stem
    split_dir = p.parent.name
    variant_dir = p.parent.parent.name
    run_id = p.parent.parent.parent.name
    section = p.parent.parent.parent.parent.name
    key = f"{section}/{run_id}/{variant_dir}/{split_dir}/{task_name}"
    logger.info("Loading metric JSON: %s", key)
    metric_jsons[key] = load_metric_json(path)

output_path = snakemake.output["plot"]
os.makedirs(os.path.dirname(output_path), exist_ok=True)
logger.info("Generating bar chart: %s", output_path)
plot_absolute_bars(
    evaluation=evaluation,
    metric_jsons=metric_jsons,
    task=task,
    metric_name=metric_name,
    output_path=output_path,
    train_config=snakemake.params.get("train_config", {}),
)

logger.info("Evaluation bar chart complete")
