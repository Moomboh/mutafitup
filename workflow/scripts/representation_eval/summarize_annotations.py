"""Summarize annotation coverage and term support for matched Swiss-Prot entries."""

from pathlib import Path

import pandas as pd
from snakemake.script import snakemake

from wfutils import get_logger
from wfutils.logging import log_snakemake_info

logger = get_logger()
log_snakemake_info(logger)


FAMILIES = {
    "go": "go_ids",
    "ec": "ec_numbers",
    "interpro": "interpro_ids",
    "pfam": "pfam_ids",
}


def _parse_terms(value: object) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).strip().strip(";")
    if not text:
        return []
    return [term.strip() for term in text.split(";") if term.strip()]


def main() -> None:
    matched_entries_path = Path(str(snakemake.input["matched_entries"]))
    coverage_summary_path = Path(str(snakemake.output["coverage_summary"]))
    term_supports_path = Path(str(snakemake.output["term_supports"]))

    matched_entries = pd.read_csv(matched_entries_path, sep="\t")
    unique_accessions = matched_entries.drop_duplicates(subset=["accession"]).copy()

    summary_rows = []
    for scope_name, frame in [
        ("matched_queries", matched_entries),
        ("unique_accessions", unique_accessions),
    ]:
        row = {"scope": scope_name, "n_records": int(len(frame))}
        any_annotation = pd.Series(False, index=frame.index)
        for family, column in FAMILIES.items():
            parsed = frame[column].map(_parse_terms)
            has_family = parsed.map(bool)
            any_annotation = any_annotation | has_family
            row[f"with_{family}"] = int(has_family.sum())
            row[f"unique_{family}_terms"] = int(
                len({t for terms in parsed for t in terms})
            )
        row["with_any_annotation"] = int(any_annotation.sum())
        summary_rows.append(row)

    term_rows = []
    for family, column in FAMILIES.items():
        support: dict[str, int] = {}
        for terms in unique_accessions[column].map(_parse_terms):
            for term in terms:
                support[term] = support.get(term, 0) + 1
        for term, count in sorted(
            support.items(), key=lambda item: (-item[1], item[0])
        ):
            term_rows.append({"family": family, "term": term, "n_accessions": count})

    coverage_summary_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(coverage_summary_path, sep="\t", index=False)
    pd.DataFrame(term_rows).to_csv(term_supports_path, sep="\t", index=False)
    logger.info("Wrote annotation coverage summary to %s", coverage_summary_path)
    logger.info("Wrote annotation term supports to %s", term_supports_path)


if __name__ == "__main__":
    main()
