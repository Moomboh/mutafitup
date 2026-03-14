"""Split GPSite training data into train/valid based on MMseqs2 clusters.

Reads the full training parquet and cluster assignments, then assigns a
fraction of clusters to the validation set (ensuring no sequence-similarity
leakage between train and valid). Also copies the test parquet to the
final output location.

The split is deterministic: clusters are sorted by representative ID and
a fixed random seed controls the assignment.
"""

import random
from pathlib import Path

import pandas as pd
from snakemake.script import snakemake

from wfutils import get_logger
from wfutils.logging import log_snakemake_info

logger = get_logger()
log_snakemake_info(logger)


ligand = str(snakemake.params["ligand"])
dataset_name = str(snakemake.params["dataset_name"])

train_full_parquet = Path(str(snakemake.input["train_full"]))
test_parquet = Path(str(snakemake.input["test"]))
cluster_tsv = Path(str(snakemake.input["cluster_tsv"]))

out_train = Path(str(snakemake.output["train"]))
out_valid = Path(str(snakemake.output["valid"]))
out_test = Path(str(snakemake.output["test"]))

valid_fraction = float(snakemake.params["valid_cluster_fraction"])
seed = int(snakemake.params["random_seed"])

logger.info(
    "Splitting GPSite %s (%s): %.0f%% of clusters -> valid, seed=%d",
    ligand,
    dataset_name,
    valid_fraction * 100,
    seed,
)

# Read data
train_df = pd.read_parquet(train_full_parquet)
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

# Assign sequences to train or valid
valid_indices: set[int] = set()
for rep in valid_reps:
    valid_indices.update(clusters[rep])

train_indices = [i for i in range(len(train_df)) if i not in valid_indices]
valid_indices_sorted = sorted(valid_indices)

train_split = train_df.iloc[train_indices].reset_index(drop=True)
valid_split = train_df.iloc[valid_indices_sorted].reset_index(drop=True)

logger.info(
    "%s split: %d train, %d valid (from %d total, %d clusters, %d valid clusters)",
    ligand,
    len(train_split),
    len(valid_split),
    len(train_df),
    len(sorted_reps),
    n_valid,
)

# Write outputs
for path in (out_train, out_valid, out_test):
    path.parent.mkdir(parents=True, exist_ok=True)

train_split.to_parquet(out_train)
valid_split.to_parquet(out_valid)

# Copy test set
import shutil

shutil.copy2(test_parquet, out_test)

logger.info(
    "Wrote train/valid/test parquets for %s to %s", dataset_name, out_train.parent
)
