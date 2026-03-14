"""Extract sequences from a parquet dataset split into a FASTA file.

Each sequence gets a unique ID of the form ``{dataset}_{split}_{index}``
so that downstream search results can be traced back to their source.
"""

from pathlib import Path

import pandas as pd
from snakemake.script import snakemake

from wfutils import get_logger
from wfutils.logging import log_snakemake_info

logger = get_logger()
log_snakemake_info(logger)


def extract_fasta(parquet_path: Path, fasta_path: Path, dataset: str, split: str):
    """Read *parquet_path* and write sequences to *fasta_path*."""

    df = pd.read_parquet(parquet_path)

    if "sequence" not in df.columns:
        raise ValueError(
            f"Parquet file {parquet_path} does not contain a 'sequence' column. "
            f"Available columns: {list(df.columns)}"
        )

    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with fasta_path.open("w") as fh:
        for idx, seq in enumerate(df["sequence"]):
            seq = str(seq).strip()
            if not seq:
                continue
            header = f"{dataset}_{split}_{idx}"
            fh.write(f">{header}\n{seq}\n")
            n_written += 1

    logger.info(
        "Wrote %d sequences from %s/%s to %s", n_written, dataset, split, fasta_path
    )


def main():
    parquet_path = Path(snakemake.input["parquet"])
    fasta_path = Path(snakemake.output["fasta"])
    dataset = str(snakemake.wildcards.dataset)
    split = str(snakemake.wildcards.split)

    extract_fasta(parquet_path, fasta_path, dataset, split)


if __name__ == "__main__":
    main()
