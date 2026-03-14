"""Snakemake script: Generate SOTA comparison plots (one per test-set group)."""

import os
import yaml
from pathlib import Path

from snakemake.script import snakemake

from mutafitup.evaluation.plotting import (
    group_sota_tasks_by_test_set,
    load_metric_json,
    plot_sota_comparison,
)
from wfutils.logging import get_logger, log_snakemake_info

logger = get_logger(__name__)
log_snakemake_info(logger)

evaluation = snakemake.params["evaluation"]
train_config = snakemake.params.get("train_config", {})
test_set_group = snakemake.wildcards.test_set_group

# Load SOTA database.
sota_path = snakemake.input.sota_db
logger.info("Loading SOTA metrics from %s", sota_path)
with open(sota_path) as f:
    sota_data = yaml.safe_load(f)["tasks"]

# Determine which tasks belong to this test_set_group.
groups = group_sota_tasks_by_test_set(sota_data)
tasks = groups.get(test_set_group, [])
logger.info("Test-set group %r -> tasks: %s", test_set_group, tasks)

# Load all metric JSONs (same pattern as plot_evaluation.py).
metric_json_paths = snakemake.input.metric_jsons
if isinstance(metric_json_paths, str):
    metric_json_paths = [metric_json_paths]

metric_jsons = {}
for path in metric_json_paths:
    p = Path(path)
    task = p.stem
    split_dir = p.parent.name
    variant_dir = p.parent.parent.name
    run_id = p.parent.parent.parent.name
    section = p.parent.parent.parent.parent.name
    key = f"{section}/{run_id}/{variant_dir}/{split_dir}/{task}"
    logger.info("Loading metric JSON: %s", key)
    metric_jsons[key] = load_metric_json(path)

# Plot.
out_path = snakemake.output.plot
os.makedirs(os.path.dirname(out_path), exist_ok=True)
logger.info("Plotting SOTA comparison for group %r to %s", test_set_group, out_path)
plot_sota_comparison(
    sota_data=sota_data,
    tasks=tasks,
    evaluation=evaluation,
    metric_jsons=metric_jsons,
    output_path=out_path,
    train_config=train_config,
    title=snakemake.params.get("title"),
)

logger.info("SOTA comparison plot complete")
