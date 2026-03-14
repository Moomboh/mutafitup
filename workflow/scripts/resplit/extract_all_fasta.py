"""Extract all sequences from resplit-configured datasets into one FASTA + metadata TSV.

Reads all ``results/datasets_subsampled/{type}/{name}/{split}.parquet`` files
for configured datasets x {train, valid, test}.  Deduplicates by exact
sequence match across datasets (same sequence appearing in e.g. meltome and
gpsite_ca gets one FASTA entry but is tracked as belonging to both).

Outputs:
- ``results/resplit/all_sequences.fasta`` — headers are ``>{integer_id}``
- ``results/resplit/sequence_metadata.tsv`` — columns: seq_id, sequence,
  datasets (comma-separated ``dataset:split`` pairs)
"""

from pathlib import Path

import pandas as pd
from snakemake.script import snakemake

from wfutils import get_logger
from wfutils.logging import log_snakemake_info

logger = get_logger()
log_snakemake_info(logger)


def _pct(part: int, total: int) -> str:
    """Format a percentage string, safe against zero division."""
    return f"{part / total * 100:.1f}%" if total else "0.0%"


def main():
    datasets = list(snakemake.params.datasets)
    fasta_path = Path(snakemake.output.fasta)
    metadata_path = Path(snakemake.output.metadata)

    # Collect all sequences with their dataset:split memberships.
    # Key: sequence string -> list of "dataset:split" tags
    seq_memberships: dict[str, list[str]] = {}
    total_raw = 0

    for ds in datasets:
        ds_raw = 0
        for split in ("train", "valid", "test"):
            key = f"{ds}_{split}"
            parquet_path = Path(snakemake.input[key])
            df = pd.read_parquet(parquet_path)

            if "sequence" not in df.columns:
                raise ValueError(
                    f"Parquet {parquet_path} does not contain a 'sequence' column. "
                    f"Available columns: {list(df.columns)}"
                )

            n_split = len(df)
            ds_raw += n_split
            logger.info("[%s/%s] %d sequences", ds, split, n_split)

            for seq in df["sequence"]:
                seq = str(seq).strip()
                if not seq:
                    continue
                tag = f"{ds}:{split}"
                if seq not in seq_memberships:
                    seq_memberships[seq] = []
                # Avoid duplicate tags (same dataset:split can appear if
                # the parquet has duplicate sequences)
                if tag not in seq_memberships[seq]:
                    seq_memberships[seq].append(tag)

        total_raw += ds_raw
        logger.info("[%s] %d raw sequences across 3 splits", ds, ds_raw)

    n_unique = len(seq_memberships)
    n_shared = total_raw - n_unique
    logger.info(
        "Deduplicated: %d raw -> %d unique (%d shared across datasets, %s)",
        total_raw,
        n_unique,
        n_shared,
        _pct(n_shared, total_raw),
    )

    # Assign integer IDs and write FASTA + metadata
    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    metadata_rows = []
    with fasta_path.open("w") as fh:
        for seq_id, (seq, tags) in enumerate(seq_memberships.items()):
            fh.write(f">{seq_id}\n{seq}\n")
            metadata_rows.append(
                {
                    "seq_id": seq_id,
                    "sequence": seq,
                    "datasets": ",".join(tags),
                }
            )

    metadata_df = pd.DataFrame(metadata_rows)
    metadata_df.to_csv(metadata_path, sep="\t", index=False)

    logger.info(
        "Wrote %d unique sequences from %d datasets to FASTA + metadata",
        n_unique,
        len(datasets),
    )


main()
