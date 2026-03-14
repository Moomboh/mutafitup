"""Run Foldseek structural similarity search with ProstT5 between two FASTA files.

Uses ``foldseek createdb --prostt5-model`` to predict 3Di structural tokens
directly from amino-acid sequences (no PDB files needed), then runs
``foldseek search`` followed by ``foldseek convertalis`` to produce a
tab-separated result file with columns: query, target, fident.

Only hits with sequence identity >= the configured threshold are reported.

Note: ``foldseek easy-search`` cannot accept plain FASTA for *both* query
and target inputs — it expects PDB/mmCIF files and fails with
"No structures found in given input" during ``createdb``.  The multi-step
approach (createdb → search → convertalis) works correctly with FASTA +
``--prostt5-model``.
"""

import tempfile
from pathlib import Path
from subprocess import CalledProcessError, run

from snakemake.script import snakemake

from wfutils import get_logger
from wfutils.foldseek import get_foldseek_bin
from wfutils.logging import log_snakemake_info

logger = get_logger()
log_snakemake_info(logger)


def run_foldseek_search(
    query_fasta: Path,
    target_fasta: Path,
    result_tsv: Path,
    min_seq_id: float,
    prostt5_model: Path,
    gpu: int = 0,
):
    """Build Foldseek DBs with ProstT5, search, and convert results to TSV."""

    foldseek_bin = get_foldseek_bin(gpu)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        query_db = tmp / "queryDB"
        target_db = tmp / "targetDB"
        result_db = tmp / "resultDB"
        search_tmp = tmp / "search_tmp"
        search_tmp.mkdir()

        # The --prostt5-model flag must point to the *prefix* used by
        # ``foldseek databases ProstT5``, not the .gguf file itself.
        # ``foldseek databases ProstT5 <prefix> <tmp>`` creates a directory
        # at <prefix>/ containing ``prostt5-f16.gguf``, so the prefix is
        # simply the parent directory of the .gguf file.
        prostt5_prefix = str(prostt5_model.parent)

        # Step 1: createdb for query FASTA
        cmd_query_db = [
            foldseek_bin,
            "createdb",
            str(query_fasta),
            str(query_db),
            "--prostt5-model",
            prostt5_prefix,
            "--gpu",
            str(gpu),
        ]
        logger.info("Creating query DB: %s", " ".join(cmd_query_db))
        try:
            run(cmd_query_db, check=True)
        except CalledProcessError as exc:
            raise RuntimeError(
                f"Foldseek createdb (query) failed with exit code {exc.returncode}"
            ) from exc

        # Step 2: createdb for target FASTA
        cmd_target_db = [
            foldseek_bin,
            "createdb",
            str(target_fasta),
            str(target_db),
            "--prostt5-model",
            prostt5_prefix,
            "--gpu",
            str(gpu),
        ]
        logger.info("Creating target DB: %s", " ".join(cmd_target_db))
        try:
            run(cmd_target_db, check=True)
        except CalledProcessError as exc:
            raise RuntimeError(
                f"Foldseek createdb (target) failed with exit code {exc.returncode}"
            ) from exc

        # Step 2.5: pad target DB for GPU search (required when --gpu 1)
        if gpu:
            target_db_pad = tmp / "targetDB_pad"
            cmd_pad = [
                foldseek_bin,
                "makepaddedseqdb",
                str(target_db),
                str(target_db_pad),
            ]
            logger.info("Padding target DB for GPU: %s", " ".join(cmd_pad))
            try:
                run(cmd_pad, check=True)
            except CalledProcessError as exc:
                raise RuntimeError(
                    f"Foldseek makepaddedseqdb failed with exit code {exc.returncode}"
                ) from exc
            search_target_db = target_db_pad
        else:
            search_target_db = target_db

        # Step 3: search
        cmd_search = [
            foldseek_bin,
            "search",
            str(query_db),
            str(search_target_db),
            str(result_db),
            str(search_tmp),
            "--min-seq-id",
            str(min_seq_id),
            "--gpu",
            str(gpu),
        ]
        logger.info("Running Foldseek search: %s", " ".join(cmd_search))
        try:
            run(cmd_search, check=True)
        except CalledProcessError as exc:
            raise RuntimeError(
                f"Foldseek search failed with exit code {exc.returncode}"
            ) from exc

        # Step 4: convertalis to TSV
        cmd_convert = [
            foldseek_bin,
            "convertalis",
            str(query_db),
            str(target_db),
            str(result_db),
            str(result_tsv),
            "--format-output",
            "query,target,fident",
        ]
        logger.info("Converting results: %s", " ".join(cmd_convert))
        try:
            run(cmd_convert, check=True)
        except CalledProcessError as exc:
            raise RuntimeError(
                f"Foldseek convertalis failed with exit code {exc.returncode}"
            ) from exc

    logger.info("Foldseek results written to %s", result_tsv)


def main():
    query_fasta = Path(snakemake.input["query"])
    target_fasta = Path(snakemake.input["target"])
    prostt5_model = Path(snakemake.input["prostt5_model"])
    result_tsv = Path(snakemake.output["result"])
    min_seq_id = float(snakemake.params["min_seq_id"])
    gpu = int(snakemake.params.get("gpu", 0))

    result_tsv.parent.mkdir(parents=True, exist_ok=True)
    run_foldseek_search(
        query_fasta, target_fasta, result_tsv, min_seq_id, prostt5_model, gpu=gpu
    )


if __name__ == "__main__":
    main()
