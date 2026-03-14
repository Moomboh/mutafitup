"""Run MMseqs2 easy-cluster on all sequences.

Produces a two-column TSV (representative, member) with one row per member.
"""

import tempfile
from collections import Counter
from pathlib import Path
from subprocess import CalledProcessError, run

from snakemake.script import snakemake

from wfutils import get_logger
from wfutils.logging import log_snakemake_info

logger = get_logger()
log_snakemake_info(logger)


def _count_fasta_sequences(fasta: Path) -> int:
    """Count the number of sequences in a FASTA file."""
    count = 0
    with fasta.open() as fh:
        for line in fh:
            if line.startswith(">"):
                count += 1
    return count


def _log_cluster_stats(clusters_tsv: Path, n_sequences: int) -> None:
    """Read the output TSV and log cluster size statistics."""
    cluster_sizes: Counter[str] = Counter()
    with clusters_tsv.open() as fh:
        header = fh.readline()  # skip header
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                cluster_sizes[parts[0]] += 1

    n_clusters = len(cluster_sizes)
    if n_clusters == 0:
        logger.info("MMseqs2: 0 clusters produced")
        return

    sizes = sorted(cluster_sizes.values(), reverse=True)
    n_singletons = sum(1 for s in sizes if s == 1)
    largest = sizes[0]

    logger.info(
        "MMseqs2: %d sequences -> %d clusters "
        "(%d singletons, %s; largest cluster: %d members)",
        n_sequences,
        n_clusters,
        n_singletons,
        f"{n_singletons / n_clusters * 100:.1f}%" if n_clusters else "0.0%",
        largest,
    )


def run_mmseqs_cluster(
    fasta: Path,
    clusters_tsv: Path,
    min_seq_id: float,
    gpu: int = 0,
):
    """Run ``mmseqs easy-cluster`` and reformat output to a simple TSV."""

    clusters_tsv.parent.mkdir(parents=True, exist_ok=True)
    n_sequences = _count_fasta_sequences(fasta)
    logger.info(
        "Clustering %d sequences (min_seq_id=%s, gpu=%d)",
        n_sequences,
        min_seq_id,
        gpu,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        prefix = Path(tmp_dir) / "cluster"

        cmd = [
            "mmseqs",
            "easy-cluster",
            str(fasta),
            str(prefix),
            tmp_dir,
            "--min-seq-id",
            str(min_seq_id),
            "--gpu",
            str(gpu),
        ]
        logger.info("Running: %s", " ".join(cmd))
        try:
            run(cmd, check=True)
        except CalledProcessError as exc:
            raise RuntimeError(
                f"MMseqs2 easy-cluster failed with exit code {exc.returncode}"
            ) from exc

        # easy-cluster produces <prefix>_cluster.tsv with columns:
        # representative_id \t member_id
        raw_tsv = Path(f"{prefix}_cluster.tsv")
        if not raw_tsv.exists():
            raise FileNotFoundError(
                f"Expected MMseqs2 cluster output at {raw_tsv} but file not found. "
                f"Contents of tmp_dir: {list(Path(tmp_dir).iterdir())}"
            )

        # Add header and copy to final location
        with raw_tsv.open() as src, clusters_tsv.open("w") as dst:
            dst.write("representative\tmember\n")
            for line in src:
                dst.write(line)

    _log_cluster_stats(clusters_tsv, n_sequences)


def main():
    fasta = Path(snakemake.input.fasta)
    clusters_tsv = Path(snakemake.output.clusters)
    min_seq_id = float(snakemake.params.min_seq_id)
    gpu = int(snakemake.params.get("gpu", 0))

    run_mmseqs_cluster(fasta, clusters_tsv, min_seq_id, gpu=gpu)


main()
