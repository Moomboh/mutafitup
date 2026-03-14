"""Join retained Swiss-Prot hits with fetched UniProt entry annotations."""

from pathlib import Path

import pandas as pd
from snakemake.script import snakemake

from wfutils import get_logger
from wfutils.logging import log_snakemake_info

logger = get_logger()
log_snakemake_info(logger)


def main() -> None:
    best_hits_path = Path(str(snakemake.input["best_hits"]))
    entries_path = Path(str(snakemake.input["entries"]))
    matched_entries_path = Path(str(snakemake.output["matched_entries"]))

    best_hits = pd.read_csv(best_hits_path, sep="\t")
    entries = pd.read_csv(entries_path, sep="\t")

    matched = best_hits.merge(
        entries, on="accession", how="left", validate="many_to_one"
    )
    missing = matched[matched["entry_name"].isna()]
    if not missing.empty:
        raise ValueError(
            f"Missing UniProt entry rows for {len(missing)} matched accessions; "
            f"first few: {missing['accession'].tolist()[:10]}"
        )

    column_order = [
        "query_id",
        "datasets",
        "sequence",
        "accession",
        "entry_name",
        "protein_names",
        "gene_names",
        "organism",
        "go_ids",
        "ec_numbers",
        "interpro_ids",
        "pfam_ids",
        "target_id",
        "fident",
        "alnlen",
        "qcov",
        "tcov",
        "evalue",
        "bits",
    ]
    matched = matched[column_order]

    matched_entries_path.parent.mkdir(parents=True, exist_ok=True)
    matched.to_csv(matched_entries_path, sep="\t", index=False)
    logger.info(
        "Wrote %d joined matched-entry rows to %s", len(matched), matched_entries_path
    )


if __name__ == "__main__":
    main()
