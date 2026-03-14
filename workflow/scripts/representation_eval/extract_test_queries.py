"""Extract one deduplicated FASTA of all benchmark test sequences."""

from pathlib import Path

import pandas as pd
from snakemake.script import snakemake

from wfutils import get_logger
from wfutils.logging import log_snakemake_info

logger = get_logger()
log_snakemake_info(logger)


def main() -> None:
    datasets = list(snakemake.params["datasets"])
    fasta_path = Path(str(snakemake.output["fasta"]))
    metadata_path = Path(str(snakemake.output["metadata"]))

    seq_memberships: dict[str, list[str]] = {}
    total_raw = 0

    for dataset in datasets:
        parquet_path = Path(snakemake.input[f"{dataset}_test"])
        df = pd.read_parquet(parquet_path)
        if "sequence" not in df.columns:
            raise ValueError(
                f"Parquet {parquet_path} does not contain a 'sequence' column. "
                f"Available columns: {list(df.columns)}"
            )

        logger.info("[%s/test] %d sequences", dataset, len(df))
        total_raw += len(df)
        for seq in df["sequence"]:
            seq = str(seq).strip()
            if not seq:
                continue
            tag = f"{dataset}:test"
            seq_memberships.setdefault(seq, [])
            if tag not in seq_memberships[seq]:
                seq_memberships[seq].append(tag)

    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with fasta_path.open("w") as fasta_fh:
        for seq_id, (sequence, memberships) in enumerate(seq_memberships.items()):
            header = f"testseq_{seq_id}"
            fasta_fh.write(f">{header}\n{sequence}\n")
            rows.append(
                {
                    "query_id": header,
                    "sequence": sequence,
                    "datasets": ",".join(memberships),
                }
            )

    pd.DataFrame(rows).to_csv(metadata_path, sep="\t", index=False)
    logger.info(
        "Wrote %d unique test sequences (%d raw) to %s and %s",
        len(rows),
        total_raw,
        fasta_path,
        metadata_path,
    )


if __name__ == "__main__":
    main()
