"""Create a Foldseek database with ProstT5 structure prediction.

This is the expensive step (ProstT5 inference on all sequences) and is
separated from clustering so the DB can be cached and reused when only
``min_seq_id`` changes.

Uses ``foldseek createdb --prostt5-model`` and optionally
``foldseek makepaddedseqdb`` for GPU clustering.

Produces a Foldseek DB directory containing the ``seqDB*`` files.
"""

from pathlib import Path
from subprocess import CalledProcessError, run

from snakemake.script import snakemake

from wfutils import get_logger
from wfutils.foldseek import get_foldseek_bin
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


def run_foldseek_createdb(
    fasta: Path,
    db_dir: Path,
    prostt5_model: Path,
    gpu: int = 0,
):
    """Build Foldseek DB with ProstT5 and persist to *db_dir*.

    The DB prefix inside *db_dir* is ``seqDB``.  When *gpu* is truthy
    a padded copy (``seqDB_pad``) is also created.
    """

    foldseek_bin = get_foldseek_bin(gpu)

    db_dir.mkdir(parents=True, exist_ok=True)
    n_sequences = _count_fasta_sequences(fasta)
    logger.info(
        "Creating Foldseek DB for %d sequences (gpu=%d)",
        n_sequences,
        gpu,
    )

    # The --prostt5-model flag takes the *directory* containing
    # prostt5-f16.gguf (i.e., the parent of the .gguf file path).
    prostt5_prefix = str(prostt5_model.parent)

    db = db_dir / "seqDB"

    # Step 1: createdb with ProstT5
    cmd_createdb = [
        foldseek_bin,
        "createdb",
        str(fasta),
        str(db),
        "--prostt5-model",
        prostt5_prefix,
        "--gpu",
        str(gpu),
    ]
    logger.info("Step 1: Creating Foldseek DB: %s", " ".join(cmd_createdb))
    try:
        run(cmd_createdb, check=True)
    except CalledProcessError as exc:
        raise RuntimeError(
            f"Foldseek createdb failed with exit code {exc.returncode}"
        ) from exc

    # Step 2 (GPU only): pad DB for GPU clustering
    if gpu:
        db_pad = db_dir / "seqDB_pad"
        cmd_pad = [
            foldseek_bin,
            "makepaddedseqdb",
            str(db),
            str(db_pad),
        ]
        logger.info("Step 2: Padding DB for GPU: %s", " ".join(cmd_pad))
        try:
            run(cmd_pad, check=True)
        except CalledProcessError as exc:
            raise RuntimeError(
                f"Foldseek makepaddedseqdb failed with exit code {exc.returncode}"
            ) from exc

    logger.info("Foldseek DB created in %s (%d sequences)", db_dir, n_sequences)


def main():
    fasta = Path(snakemake.input.fasta)
    prostt5_model = Path(snakemake.input.prostt5_model)
    db_dir = Path(snakemake.output.db)
    gpu = int(snakemake.params.get("gpu", 0))

    run_foldseek_createdb(fasta, db_dir, prostt5_model, gpu=gpu)


main()
