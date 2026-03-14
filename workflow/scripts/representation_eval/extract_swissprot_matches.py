"""Extract matched Swiss-Prot sequences for retained homolog hits."""

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


def _iter_fasta_records(path: Path):
    header = None
    seq_lines: list[str] = []
    with path.open() as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_lines)
                header = line[1:]
                seq_lines = []
            else:
                seq_lines.append(line.strip())
    if header is not None:
        yield header, "".join(seq_lines)


def _extract_accession(header: str) -> str:
    primary = header.split(None, 1)[0]
    parts = primary.split("|")
    if len(parts) >= 3 and parts[0] in {"sp", "tr"}:
        return parts[1]
    if len(parts) == 1 and _UNIPROT_ACCESSION_RE.fullmatch(primary):
        return primary
    raise ValueError(f"Could not parse UniProt accession from FASTA header: {header!r}")


def main() -> None:
    swissprot_fasta = Path(str(snakemake.input["swissprot_fasta"]))
    best_hits_path = Path(str(snakemake.input["best_hits"]))
    entries_path = Path(str(snakemake.input["entries"]))
    output_fasta = Path(str(snakemake.output["fasta"]))

    best_hits = pd.read_csv(best_hits_path, sep="\t")
    entries = pd.read_csv(entries_path, sep="\t")
    entry_names = dict(zip(entries["accession"], entries["entry_name"]))
    wanted = set(best_hits["accession"].tolist())

    sequences: dict[str, tuple[str, str]] = {}
    for header, sequence in _iter_fasta_records(swissprot_fasta):
        accession = _extract_accession(header)
        if accession in wanted and accession not in sequences:
            sequences[accession] = (header, sequence)

    missing = sorted(wanted - set(sequences))
    if missing:
        raise ValueError(
            f"Could not find {len(missing)} retained accessions in Swiss-Prot FASTA; "
            f"first few: {missing[:10]}"
        )

    output_fasta.parent.mkdir(parents=True, exist_ok=True)
    with output_fasta.open("w") as fh:
        for accession in sorted(wanted):
            entry_name = entry_names.get(accession, "")
            _header, sequence = sequences[accession]
            fh.write(f">{accession}|{entry_name}\n{sequence}\n")

    logger.info(
        "Wrote %d matched Swiss-Prot sequences to %s", len(wanted), output_fasta
    )


if __name__ == "__main__":
    main()
