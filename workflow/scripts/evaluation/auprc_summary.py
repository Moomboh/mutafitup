"""Snakemake script: Generate AUPRC summary JSON and bar chart."""

import json

from snakemake.script import snakemake

from mutafitup.evaluation.auprc_summary import (
    generate_auprc_summary,
    plot_auprc_bar_chart,
)
from wfutils.logging import get_logger, log_snakemake_info

logger = get_logger(__name__)
log_snakemake_info(logger)

evaluation = snakemake.params["evaluation"]
summary_path = snakemake.output["summary"]
bar_chart_path = snakemake.output["bar_chart"]

logger.info(f"Generating AUPRC summary for evaluation '{evaluation['id']}'")

summary = generate_auprc_summary(
    evaluation=evaluation,
    metrics_dir="results/auprc_metrics",
)

with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
logger.info(f"Wrote AUPRC summary to {summary_path}")

plot_auprc_bar_chart(
    summary=summary,
    output_path=bar_chart_path,
)
logger.info(f"Wrote AUPRC bar chart to {bar_chart_path}")
