"""Download and extract the pinned Swiss-Prot FASTA."""

import gzip
import shutil
from pathlib import Path

import pooch
from snakemake.script import snakemake

from wfutils import get_logger
from wfutils.logging import log_snakemake_info

logger = get_logger()
log_snakemake_info(logger)


def main() -> None:
    url = str(snakemake.params["url"])
    known_hash = str(snakemake.params["hash"])

    fasta_gz = Path(str(snakemake.output["fasta_gz"]))
    fasta = Path(str(snakemake.output["fasta"]))

    fasta_gz.parent.mkdir(parents=True, exist_ok=True)
    fasta.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading Swiss-Prot FASTA: %s", url)
    pooch.retrieve(
        url,
        known_hash=known_hash,
        fname=fasta_gz.name,
        path=str(fasta_gz.parent),
    )

    logger.info("Extracting %s -> %s", fasta_gz, fasta)
    with gzip.open(fasta_gz, "rb") as src, fasta.open("wb") as dst:
        shutil.copyfileobj(src, dst)

    logger.info("Swiss-Prot FASTA ready at %s", fasta)


if __name__ == "__main__":
    main()
