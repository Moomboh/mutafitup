"""Snakemake script: compute normalization statistics for regression tasks.

Reads training parquets and writes a ``normalization_stats.json`` file
co-located with the ONNX export. Delegates to
:func:`mutafitup.normalization_stats.compute_normalization_stats`.
"""

import json

from snakemake.script import snakemake

from mutafitup.normalization_stats import compute_normalization_stats
from wfutils.logging import get_logger, log_snakemake_info


logger = get_logger(__name__)
log_snakemake_info(logger)

tasks = snakemake.params["tasks"]
datasets_resplit_dir = snakemake.params["datasets_resplit_dir"]
output_path = str(snakemake.output["normalization_stats"])

stats = compute_normalization_stats(tasks, datasets_resplit_dir)

with open(output_path, "w") as f:
    json.dump(stats, f, indent=2)
    f.write("\n")

logger.info("Wrote normalization stats to %s", output_path)
