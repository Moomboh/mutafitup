"""Split NetSurfP-2.0 training data into train/valid based on MMseqs2 clusters.

Reads the full training parquets (SecStr8, SecStr, RSA) and cluster
assignments, then assigns a fraction of clusters to the validation set
(ensuring no sequence-similarity leakage between train and valid).  Also
copies the test parquets to the final output locations.

All three dataset variants (SecStr8, SecStr, RSA) share the same protein
sequences and cluster assignments; they differ only in the label column.
The same index-based split is applied to all.

The split is deterministic: clusters are sorted by representative ID and
a fixed random seed controls the assignment.
"""

import random
import shutil
from pathlib import Path

import pandas as pd
from snakemake.script import snakemake

from wfutils import get_logger
from wfutils.logging import log_snakemake_info

logger = get_logger()
log_snakemake_info(logger)


cluster_tsv = Path(str(snakemake.input["cluster_tsv"]))

valid_fraction = float(snakemake.params["valid_cluster_fraction"])
seed = int(snakemake.params["random_seed"])

logger.info(
    "Splitting NetSurfP-2.0: %.0f%% of clusters -> valid, seed=%d",
    valid_fraction * 100,
    seed,
)

# Read cluster assignments
cluster_df = pd.read_csv(
    cluster_tsv, sep="\t", header=None, names=["representative", "member"]
)

# Build cluster -> member indices mapping
# Member column contains the 0-based DataFrame indices
clusters: dict[str, list[int]] = {}
for _, row in cluster_df.iterrows():
    rep = str(row["representative"])
    member = int(row["member"])
    clusters.setdefault(rep, []).append(member)

# Sort cluster representatives for deterministic ordering
sorted_reps = sorted(clusters.keys(), key=lambda r: int(r))
n_valid = max(1, round(len(sorted_reps) * valid_fraction))

# Randomly select clusters for validation
rng = random.Random(seed)
valid_reps = set(rng.sample(sorted_reps, n_valid))

# Compute index sets (shared across all dataset variants)
valid_indices: set[int] = set()
for rep in valid_reps:
    valid_indices.update(clusters[rep])

logger.info(
    "Cluster split: %d clusters total, %d valid clusters, %d valid sequences",
    len(sorted_reps),
    n_valid,
    len(valid_indices),
)

# Apply the same split to each dataset variant
VARIANTS = ["secstr8", "secstr", "rsa"]

for variant in VARIANTS:
    train_full_parquet = Path(str(snakemake.input[f"{variant}_train_full"]))
    test_parquet = Path(str(snakemake.input[f"{variant}_test"]))

    out_train = Path(str(snakemake.output[f"{variant}_train"]))
    out_valid = Path(str(snakemake.output[f"{variant}_valid"]))
    out_test = Path(str(snakemake.output[f"{variant}_test"]))

    train_df = pd.read_parquet(train_full_parquet)

    train_indices = [i for i in range(len(train_df)) if i not in valid_indices]
    valid_indices_sorted = sorted(valid_indices)

    train_split = train_df.iloc[train_indices].reset_index(drop=True)
    valid_split = train_df.iloc[valid_indices_sorted].reset_index(drop=True)

    logger.info(
        "[%s] Split: %d train, %d valid (from %d total)",
        variant,
        len(train_split),
        len(valid_split),
        len(train_df),
    )

    # Write outputs
    for path in (out_train, out_valid, out_test):
        path.parent.mkdir(parents=True, exist_ok=True)

    train_split.to_parquet(out_train)
    valid_split.to_parquet(out_valid)

    # Copy test set
    shutil.copy2(test_parquet, out_test)

    logger.info("[%s] Wrote train/valid/test parquets to %s", variant, out_train.parent)
