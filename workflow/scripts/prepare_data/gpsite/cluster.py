"""Cluster GPSite training sequences using MMseqs2 easy-cluster.

Reads the ``train_full.parquet`` for a ligand type, writes sequences to a
temporary FASTA file, runs ``mmseqs easy-cluster`` at the configured sequence
identity threshold, and writes the cluster assignments to a TSV file.

The output TSV has two columns (no header):
- ``representative``: the 0-based index of the cluster representative
- ``member``: the 0-based index of a cluster member
"""

import tempfile
from pathlib import Path
from subprocess import CalledProcessError, run

import pandas as pd
from snakemake.script import snakemake

from wfutils import get_logger
from wfutils.logging import log_snakemake_info

logger = get_logger()
log_snakemake_info(logger)


def write_fasta(sequences: list[str], fasta_path: Path):
    """Write sequences to a FASTA file with 0-based integer IDs."""
    with open(fasta_path, "w") as fh:
        for idx, seq in enumerate(sequences):
            fh.write(f">{idx}\n{seq}\n")


def run_mmseqs_cluster(
    fasta_path: Path,
    output_prefix: Path,
    min_seq_id: float,
    tmp_dir: str,
):
    """Run ``mmseqs easy-cluster`` and return the cluster TSV path."""
    cmd = [
        "mmseqs",
        "easy-cluster",
        str(fasta_path),
        str(output_prefix),
        tmp_dir,
        "--min-seq-id",
        str(min_seq_id),
    ]
    logger.info("Running MMseqs2: %s", " ".join(cmd))
    try:
        run(cmd, check=True)
    except CalledProcessError as exc:
        raise RuntimeError(
            f"MMseqs2 easy-cluster failed with exit code {exc.returncode}"
        ) from exc

    # easy-cluster produces {prefix}_cluster.tsv with two columns:
    # representative_header \t member_header
    cluster_tsv = Path(f"{output_prefix}_cluster.tsv")
    if not cluster_tsv.exists():
        raise FileNotFoundError(
            f"Expected cluster output at {cluster_tsv} but it does not exist"
        )
    return cluster_tsv


ligand = str(snakemake.wildcards["ligand"])
train_full_parquet = Path(str(snakemake.input["train_full"]))
output_tsv = Path(str(snakemake.output["cluster_tsv"]))

min_seq_id = float(snakemake.params["cluster_min_seq_id"])

logger.info(
    "Clustering GPSite %s training sequences at %.0f%% identity",
    ligand,
    min_seq_id * 100,
)

df = pd.read_parquet(train_full_parquet)
sequences = df["sequence"].tolist()
logger.info("Read %d training sequences for %s", len(sequences), ligand)

output_tsv.parent.mkdir(parents=True, exist_ok=True)

with tempfile.TemporaryDirectory() as tmp_dir:
    fasta_path = Path(tmp_dir) / "train.fasta"
    write_fasta(sequences, fasta_path)

    mmseqs_prefix = Path(tmp_dir) / "mmseqs_out"
    cluster_tsv = run_mmseqs_cluster(fasta_path, mmseqs_prefix, min_seq_id, tmp_dir)

    # Read and copy cluster assignments to output
    # The raw MMseqs2 output has: representative_id \t member_id
    # Our IDs are 0-based integers, so we just copy the file
    import shutil

    shutil.copy2(cluster_tsv, output_tsv)

logger.info("Cluster assignments written to %s", output_tsv)

# Log cluster statistics
cluster_df = pd.read_csv(
    output_tsv, sep="\t", header=None, names=["representative", "member"]
)
n_clusters = cluster_df["representative"].nunique()
logger.info(
    "%s: %d sequences -> %d clusters (%.1f%% reduction)",
    ligand,
    len(sequences),
    n_clusters,
    (1 - n_clusters / len(sequences)) * 100,
)
