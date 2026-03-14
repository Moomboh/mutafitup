"""Fetch exact Swiss-Prot entry annotations for retained accessions."""

import csv
import io
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd
from snakemake.script import snakemake

from wfutils import get_logger
from wfutils.logging import log_snakemake_info

logger = get_logger()
log_snakemake_info(logger)


MAX_BATCH_SIZE = 100
MAX_RETRIES = 5
BASE_BACKOFF_SECONDS = 1.0
RETRYABLE_HTTP_CODES = {429, 500, 502, 503, 504}


FIELDS = [
    "accession",
    "id",
    "protein_name",
    "gene_names",
    "organism_name",
    "go_id",
    "ec",
    "xref_interpro",
    "xref_pfam",
]


def _batched(items: list[str], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _fetch_batch(accessions: list[str]) -> list[dict[str, str]]:
    query = (
        "("
        + " OR ".join(f"accession:{acc}" for acc in accessions)
        + ") AND reviewed:true"
    )
    params = {
        "query": query,
        "format": "tsv",
        "fields": ",".join(FIELDS),
        "size": str(len(accessions)),
    }
    url = "https://rest.uniprot.org/uniprotkb/search?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "mutafitup-representation-eval/1.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        text = response.read().decode("utf-8")

    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    rows = []
    for row in reader:
        rows.append(
            {
                "accession": row.get("Entry", ""),
                "entry_name": row.get("Entry Name", ""),
                "protein_names": row.get("Protein names", ""),
                "gene_names": row.get("Gene Names", ""),
                "organism": row.get("Organism", ""),
                "go_ids": row.get("Gene Ontology IDs", ""),
                "ec_numbers": row.get("EC number", ""),
                "interpro_ids": row.get("InterPro", ""),
                "pfam_ids": row.get("Pfam", ""),
            }
        )
    return rows


def _sleep_with_backoff(attempt: int, retry_after: str | None = None) -> None:
    if retry_after is not None:
        try:
            delay = max(float(retry_after), 0.0)
        except ValueError:
            delay = BASE_BACKOFF_SECONDS * (2**attempt)
    else:
        delay = BASE_BACKOFF_SECONDS * (2**attempt)

    delay += random.uniform(0.0, 0.5)
    logger.info("Sleeping %.2f seconds before retry", delay)
    time.sleep(delay)


def _fetch_batch_with_retries(accessions: list[str]) -> list[dict[str, str]]:
    for attempt in range(MAX_RETRIES + 1):
        try:
            return _fetch_batch(accessions)
        except urllib.error.HTTPError as exc:
            if exc.code not in RETRYABLE_HTTP_CODES or attempt == MAX_RETRIES:
                raise
            logger.warning(
                "UniProt REST batch of %d accessions failed with HTTP %d on attempt %d/%d",
                len(accessions),
                exc.code,
                attempt + 1,
                MAX_RETRIES + 1,
            )
            _sleep_with_backoff(attempt, exc.headers.get("Retry-After"))
        except (TimeoutError, urllib.error.URLError) as exc:
            if attempt == MAX_RETRIES:
                raise
            logger.warning(
                "UniProt REST batch of %d accessions failed with %s on attempt %d/%d",
                len(accessions),
                type(exc).__name__,
                attempt + 1,
                MAX_RETRIES + 1,
            )
            _sleep_with_backoff(attempt)

    raise RuntimeError("Retry loop exited unexpectedly")


def _fetch_batch_resilient(accessions: list[str]) -> list[dict[str, str]]:
    try:
        return _fetch_batch_with_retries(accessions)
    except urllib.error.HTTPError as exc:
        if exc.code in {400, 404, 414} and len(accessions) > 1:
            midpoint = len(accessions) // 2
            left = accessions[:midpoint]
            right = accessions[midpoint:]
            logger.warning(
                "UniProt REST batch of %d accessions failed with HTTP %d; retrying as %d + %d",
                len(accessions),
                exc.code,
                len(left),
                len(right),
            )
            time.sleep(0.1)
            return _fetch_batch_resilient(left) + _fetch_batch_resilient(right)
        raise


def main() -> None:
    accessions_path = Path(str(snakemake.input["accessions"]))
    entries_path = Path(str(snakemake.output["entries"]))
    configured_batch_size = int(snakemake.params["batch_size"])
    batch_size = min(configured_batch_size, MAX_BATCH_SIZE)

    accessions = [
        line.strip()
        for line in accessions_path.read_text().splitlines()
        if line.strip()
    ]
    if not accessions:
        raise ValueError("No accessions provided for UniProt entry fetch")

    logger.info(
        "Fetching %d accessions with configured batch size %d (effective %d)",
        len(accessions),
        configured_batch_size,
        batch_size,
    )

    all_rows: list[dict[str, str]] = []
    for batch in _batched(accessions, batch_size):
        logger.info("Fetching %d Swiss-Prot entries from UniProt REST", len(batch))
        all_rows.extend(_fetch_batch_resilient(batch))
        time.sleep(0.1)

    df = pd.DataFrame(all_rows)
    if df.empty:
        raise ValueError("UniProt REST fetch returned no Swiss-Prot entries")

    df = df.drop_duplicates(subset=["accession"]).sort_values("accession")
    missing = sorted(set(accessions) - set(df["accession"]))
    if missing:
        raise ValueError(
            f"Failed to fetch {len(missing)} Swiss-Prot accessions; first few: {missing[:10]}"
        )

    entries_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(entries_path, sep="\t", index=False)
    logger.info("Wrote %d Swiss-Prot entries to %s", len(df), entries_path)


if __name__ == "__main__":
    main()
