"""Parse GPSite raw text files into standardized parquet format.

Each GPSite ``.txt`` file has entries in a 3-line format::

    >PDBID_CHAIN
    AMINO_ACID_SEQUENCE
    BINARY_LABEL_STRING   (0/1 per residue)

This script reads the raw files for a single ligand type and produces
``train_full.parquet`` (all original training data) and ``test.parquet``
with columns: ``sequence``, ``label``, ``resolved``.
"""

from pathlib import Path

import pandas as pd
from snakemake.script import snakemake

from wfutils import get_logger
from wfutils.logging import log_snakemake_info

logger = get_logger()
log_snakemake_info(logger)


def parse_gpsite_txt(txt_path: Path) -> pd.DataFrame:
    """Parse a GPSite text file into a DataFrame.

    Returns a DataFrame with columns:
    - ``sequence`` (str): amino acid sequence
    - ``label`` (list[int]): per-residue binary labels (0 or 1)
    - ``resolved`` (list[int]): all 1s (all residues are from PDB structures)
    """
    sequences = []
    labels = []
    resolved = []

    with open(txt_path) as fh:
        lines = [line.rstrip("\n") for line in fh if line.strip()]

    if len(lines) % 3 != 0:
        raise ValueError(
            f"Expected number of non-empty lines to be a multiple of 3, "
            f"got {len(lines)} in {txt_path}"
        )

    for i in range(0, len(lines), 3):
        header = lines[i]
        seq = lines[i + 1]
        label_str = lines[i + 2]

        if not header.startswith(">"):
            raise ValueError(
                f"Expected header line starting with '>' at line {i + 1}, "
                f"got: {header!r}"
            )

        if len(seq) != len(label_str):
            raise ValueError(
                f"Sequence length ({len(seq)}) != label length ({len(label_str)}) "
                f"for {header}"
            )

        label_list = [int(c) for c in label_str]

        # Validate labels are binary
        if not all(v in (0, 1) for v in label_list):
            raise ValueError(
                f"Labels must be 0 or 1, got unexpected values for {header}"
            )

        sequences.append(seq)
        labels.append(label_list)
        resolved.append([1] * len(seq))

    return pd.DataFrame({"sequence": sequences, "label": labels, "resolved": resolved})


ligand = str(snakemake.wildcards["ligand"])

raw_dir = Path(str(snakemake.input["raw_dir"]))
train_txt = raw_dir / f"{ligand}_train.txt"
test_txt = raw_dir / f"{ligand}_test.txt"

train_full_parquet = Path(str(snakemake.output["train_full"]))
test_parquet = Path(str(snakemake.output["test"]))

train_full_parquet.parent.mkdir(parents=True, exist_ok=True)
test_parquet.parent.mkdir(parents=True, exist_ok=True)

logger.info("Parsing GPSite %s training data from %s", ligand, train_txt)
train_df = parse_gpsite_txt(train_txt)
logger.info("Parsed %d training sequences for %s", len(train_df), ligand)
train_df.to_parquet(train_full_parquet)

logger.info("Parsing GPSite %s test data from %s", ligand, test_txt)
test_df = parse_gpsite_txt(test_txt)
logger.info("Parsed %d test sequences for %s", len(test_df), ligand)
test_df.to_parquet(test_parquet)
