"""Snakemake script: Export evaluation summaries as Typst table fragments."""

import json
from pathlib import Path

from snakemake.script import snakemake

from mutafitup.evaluation.plotting import load_metric_json
from mutafitup.evaluation.typst_tables import (
    export_delta_summary_typst,
    export_full_results_typst,
    write_text,
)
from wfutils.logging import get_logger, log_snakemake_info

logger = get_logger(__name__)
log_snakemake_info(logger)

evaluation = snakemake.params["evaluation"]
metric_json_paths = snakemake.input["metric_jsons"]
summary_path = Path(str(snakemake.input["summary"]))

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
    metric_jsons[key] = load_metric_json(path)

with summary_path.open() as f:
    summary = json.load(f)

write_text(
    snakemake.output["full_results_table"],
    export_full_results_typst(evaluation, metric_jsons),
)
write_text(
    snakemake.output["delta_summary_table"],
    export_delta_summary_typst(summary),
)

logger.info("Exported Typst tables for evaluation %s", evaluation["id"])
