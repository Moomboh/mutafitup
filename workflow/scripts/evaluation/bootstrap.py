"""Snakemake script: Generate protein-level bootstrap indices."""

import pandas as pd
from snakemake.script import snakemake

from mutafitup.evaluation.bootstrap import generate_bootstrap_indices
from wfutils.logging import get_logger, log_snakemake_info

logger = get_logger(__name__)
log_snakemake_info(logger)

dataset_path = snakemake.input["dataset"]
output_path = snakemake.output["bootstraps"]
num_bootstraps = snakemake.params["num_bootstraps"]
bootstrap_seed = snakemake.params["bootstrap_seed"]

logger.info(f"Loading dataset from {dataset_path}")
df = pd.read_parquet(dataset_path)

# Use the dataframe index as sequence IDs
sequence_ids = df.index.values
logger.info(f"Found {len(sequence_ids)} unique sequences")

logger.info(f"Generating {num_bootstraps} bootstrap samples (seed={bootstrap_seed})")
bootstrap_df = generate_bootstrap_indices(
    sequence_ids=sequence_ids,
    num_bootstraps=num_bootstraps,
    seed=bootstrap_seed,
)

bootstrap_df.to_parquet(output_path)
logger.info(f"Wrote bootstrap indices to {output_path}")
