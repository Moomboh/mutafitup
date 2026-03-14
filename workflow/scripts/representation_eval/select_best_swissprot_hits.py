"""Filter MMseqs hits to one best Swiss-Prot homolog per query."""

import re
from pathlib import Path

import pandas as pd
from snakemake.script import snakemake

from wfutils import get_logger
from wfutils.logging import log_snakemake_info

logger = get_logger()
log_snakemake_info(logger)


_UNIPROT_ACCESSION_RE = re.compile(
    r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})$"
)


def _extract_accession(target: str) -> str:
    value = str(target).strip()
    if not value:
        raise ValueError("Cannot parse UniProt accession from empty target id")

    parts = value.split("|")
    if len(parts) >= 3 and parts[0] in {"sp", "tr"}:
        return parts[1]
    if len(parts) == 1 and _UNIPROT_ACCESSION_RE.fullmatch(value):
        return value
    raise ValueError(f"Could not parse UniProt accession from target id: {target!r}")


def main() -> None:
    hits_path = Path(str(snakemake.input["hits"]))
    metadata_path = Path(str(snakemake.input["metadata"]))
    best_hits_path = Path(str(snakemake.output["best_hits"]))
    accessions_path = Path(str(snakemake.output["accessions"]))
    min_query_coverage = float(snakemake.params["min_query_coverage"])
    min_target_coverage = float(snakemake.params["min_target_coverage"])

    hits = pd.read_csv(hits_path, sep="\t")
    metadata = pd.read_csv(metadata_path, sep="\t")

    if hits.empty:
        raise ValueError("MMseqs hits file is empty; no Swiss-Prot matches were found")

    numeric_cols = ["fident", "alnlen", "qcov", "tcov", "evalue", "bits"]
    for col in numeric_cols:
        hits[col] = pd.to_numeric(hits[col], errors="coerce")

    hits = hits[
        (hits["qcov"] >= min_query_coverage) & (hits["tcov"] >= min_target_coverage)
    ]
    if hits.empty:
        raise ValueError(
            "No MMseqs hits remain after applying query/target coverage thresholds"
        )

    hits["accession"] = [
        _extract_accession(target) for target in hits["target"].tolist()
    ]
    hit_records = list(hits.itertuples(index=False, name=None))
    hit_records.sort(
        key=lambda row: (
            str(row[hits.columns.get_loc("query")]),
            -float(row[hits.columns.get_loc("bits")]),
            float(row[hits.columns.get_loc("evalue")]),
            -float(row[hits.columns.get_loc("fident")]),
        )
    )
    hits = pd.DataFrame(hit_records, columns=hits.columns)
    best = hits.drop_duplicates(subset=["query"], keep="first").copy()

    best = best.merge(metadata, left_on="query", right_on="query_id", how="left")
    missing = best[best["sequence"].isna()]
    if not missing.empty:
        raise ValueError(
            f"Missing query metadata for {len(missing)} retained hits: "
            f"{missing['query'].tolist()[:5]}"
        )

    best = best.rename(columns={"query": "query_id", "target": "target_id"})
    columns = [
        "query_id",
        "datasets",
        "sequence",
        "accession",
        "target_id",
        "fident",
        "alnlen",
        "qcov",
        "tcov",
        "evalue",
        "bits",
    ]
    best = best[columns]

    best_hits_path.parent.mkdir(parents=True, exist_ok=True)
    best.to_csv(best_hits_path, sep="\t", index=False)

    accession_values = [
        str(value) for value in best["accession"].tolist() if pd.notna(value)
    ]
    accessions = sorted(set(accession_values))
    accessions_path.write_text("".join(f"{acc}\n" for acc in accessions))

    logger.info(
        "Retained %d best Swiss-Prot hits for %d unique accessions",
        len(best),
        len(accessions),
    )


if __name__ == "__main__":
    main()
