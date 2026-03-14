"""Run Foldseek clustering on a pre-built Foldseek database.

Reads the database created by ``foldseek_createdb.py`` (which includes
ProstT5 structure prediction) and runs ``foldseek cluster`` +
``foldseek createtsv`` to produce a two-column TSV (representative, member).

Separating DB creation from clustering means changing ``min_seq_id``
does not require re-running the expensive ProstT5 inference step.
"""

import tempfile
from collections import Counter
from pathlib import Path
from subprocess import CalledProcessError, run

from snakemake.script import snakemake

from wfutils import get_logger
from wfutils.foldseek import get_foldseek_bin
from wfutils.logging import log_snakemake_info

logger = get_logger()
log_snakemake_info(logger)


def _log_cluster_stats(clusters_tsv: Path, label: str = "Foldseek") -> None:
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
        logger.info("%s: 0 clusters produced", label)
        return

    n_members = sum(cluster_sizes.values())
    sizes = sorted(cluster_sizes.values(), reverse=True)
    n_singletons = sum(1 for s in sizes if s == 1)
    largest = sizes[0]

    logger.info(
        "%s: %d sequences -> %d clusters "
        "(%d singletons, %s; largest cluster: %d members)",
        label,
        n_members,
        n_clusters,
        n_singletons,
        f"{n_singletons / n_clusters * 100:.1f}%" if n_clusters else "0.0%",
        largest,
    )


def run_foldseek_cluster(
    db_dir: Path,
    clusters_tsv: Path,
    min_seq_id: float,
    gpu: int = 0,
):
    """Cluster a pre-built Foldseek DB and convert results to TSV.

    Parameters
    ----------
    db_dir:
        Directory containing the Foldseek DB files (``seqDB*`` prefix).
        If *gpu* is truthy the padded DB (``seqDB_pad``) is used instead.
    clusters_tsv:
        Output path for the two-column (representative, member) TSV.
    min_seq_id:
        Minimum sequence identity threshold for clustering.
    gpu:
        Whether to use GPU-accelerated clustering.
    """

    foldseek_bin = get_foldseek_bin(gpu)

    clusters_tsv.parent.mkdir(parents=True, exist_ok=True)

    # Select the correct DB prefix (padded for GPU, normal otherwise)
    if gpu:
        db = db_dir / "seqDB_pad"
    else:
        db = db_dir / "seqDB"

    # The original (unpadded) DB is always needed for createtsv
    db_orig = db_dir / "seqDB"

    logger.info(
        "Clustering (min_seq_id=%s, gpu=%d, db=%s)",
        min_seq_id,
        gpu,
        db,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        cluster_db = tmp / "clusterDB"
        cluster_tmp = tmp / "cluster_tmp"
        cluster_tmp.mkdir()

        # Step 1: cluster
        cmd_cluster = [
            foldseek_bin,
            "cluster",
            str(db),
            str(cluster_db),
            str(cluster_tmp),
            "--min-seq-id",
            str(min_seq_id),
        ]
        logger.info("Step 1/2: Clustering: %s", " ".join(cmd_cluster))
        try:
            run(cmd_cluster, check=True)
        except CalledProcessError as exc:
            raise RuntimeError(
                f"Foldseek cluster failed with exit code {exc.returncode}"
            ) from exc

        # Step 2: createtsv to get representative-member pairs
        raw_tsv = tmp / "raw_clusters.tsv"
        cmd_tsv = [
            foldseek_bin,
            "createtsv",
            str(db_orig),
            str(db_orig),
            str(cluster_db),
            str(raw_tsv),
        ]
        logger.info("Step 2/2: Converting to TSV: %s", " ".join(cmd_tsv))
        try:
            run(cmd_tsv, check=True)
        except CalledProcessError as exc:
            raise RuntimeError(
                f"Foldseek createtsv failed with exit code {exc.returncode}"
            ) from exc

        # Add header and copy to final location
        with raw_tsv.open() as src, clusters_tsv.open("w") as dst:
            dst.write("representative\tmember\n")
            for line in src:
                # createtsv produces: representative \t member
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    dst.write(f"{parts[0]}\t{parts[1]}\n")

    _log_cluster_stats(clusters_tsv)


def main():
    db_dir = Path(snakemake.input.db)
    clusters_tsv = Path(snakemake.output.clusters)
    min_seq_id = float(snakemake.params.min_seq_id)
    gpu = int(snakemake.params.get("gpu", 0))

    run_foldseek_cluster(db_dir, clusters_tsv, min_seq_id, gpu=gpu)


main()
