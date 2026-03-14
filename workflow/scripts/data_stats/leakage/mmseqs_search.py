"""Run MMseqs2 easy-search between a query FASTA and a target FASTA.

Produces a tab-separated result file with columns: query, target, fident.
Only hits with sequence identity >= the configured threshold are reported.
"""

import shutil
import tempfile
from pathlib import Path
from subprocess import CalledProcessError, run

from snakemake.script import snakemake

from wfutils import get_logger
from wfutils.logging import log_snakemake_info

logger = get_logger()
log_snakemake_info(logger)


def run_mmseqs_search(
    query_fasta: Path,
    target_fasta: Path,
    result_tsv: Path,
    min_seq_id: float,
    gpu: int = 0,
):
    """Run ``mmseqs easy-search`` and write hits to *result_tsv*."""

    with tempfile.TemporaryDirectory() as tmp_dir:
        cmd = [
            "mmseqs",
            "easy-search",
            str(query_fasta),
            str(target_fasta),
            str(result_tsv),
            tmp_dir,
            "--min-seq-id",
            str(min_seq_id),
            "--format-output",
            "query,target,fident",
            "--gpu",
            str(gpu),
        ]
        logger.info("Running MMseqs2: %s", " ".join(cmd))
        try:
            run(cmd, check=True)
        except CalledProcessError as exc:
            raise RuntimeError(
                f"MMseqs2 easy-search failed with exit code {exc.returncode}"
            ) from exc

    logger.info("MMseqs2 results written to %s", result_tsv)


def main():
    query_fasta = Path(snakemake.input["query"])
    target_fasta = Path(snakemake.input["target"])
    result_tsv = Path(snakemake.output["result"])
    min_seq_id = float(snakemake.params["min_seq_id"])
    gpu = int(snakemake.params.get("gpu", 0))

    result_tsv.parent.mkdir(parents=True, exist_ok=True)
    run_mmseqs_search(query_fasta, target_fasta, result_tsv, min_seq_id, gpu=gpu)


if __name__ == "__main__":
    main()
