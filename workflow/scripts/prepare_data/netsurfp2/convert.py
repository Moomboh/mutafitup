"""Parse NetSurfP-2.0 data into standardized parquet format.

Reads the original DTU NPZ files (containing full structural annotations)
and the ProtTrans NEW364 CSV (used as the test set for secondary structure).

Produces parquets for three dataset variants:

**SecStr8** — 8-state secondary structure (per-residue classification)

    NPZ one-hot order [57:65]: G, H, I, B, E, S, T, C
    Integer labels (argmax):   G=0, H=1, I=2, B=3, E=4, S=5, T=6, C=7

    Parquet columns: ``sequence``, ``label`` (list[int]), ``resolved`` (list[int])

**SecStr** — 3-state secondary structure (per-residue classification)

    Derived from Q8 via reduction:
        G, H, I  (helix)  -> H = 2
        B, E     (strand) -> E = 1
        S, T, C  (coil)   -> C = 0

    Parquet columns: ``sequence``, ``label`` (list[int]), ``resolved`` (list[int])

**RSA** — Relative solvent accessibility, isolated chain (per-residue regression)

    NPZ index [55], float in [0.0, 1.0].
    Disordered residues (disorder mask = 0) are set to 999.0 (unresolved sentinel).

    Parquet columns: ``sequence``, ``score`` (list[float])

The disorder mask (NPZ index [51]) determines the ``resolved`` column for
classification datasets: disorder=1 -> resolved=1, disorder=0 -> resolved=0.
For the RSA regression dataset, disordered positions use the 999.0 sentinel
value in the ``score`` column instead of a separate resolved column.

**Data sources:**

- Training data: ``Train_HHblits.npz`` from DTU (10,848 sequences)
- RSA test set: ``CB513_HHblits.npz`` from DTU (513 sequences)
- SecStr/SecStr8 test set: ``NEW364.csv`` from ProtTrans Dropbox (364 sequences)

NEW364 is only available as a CSV (no NPZ on DTU), so it cannot be used
for RSA testing.  CB513 is the standard RSA benchmark in the literature.
"""

import csv
from pathlib import Path

import numpy as np
import pandas as pd
from snakemake.script import snakemake

from wfutils import get_logger
from wfutils.logging import log_snakemake_info

logger = get_logger()
log_snakemake_info(logger)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Amino acid one-hot encoding order in NPZ (alphabetical, 20 standard AAs)
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"

# NPZ feature indices
IDX_AA_START = 0
IDX_AA_END = 20
IDX_SEQ_MASK = 50
IDX_DISORDER_MASK = 51
IDX_RSA_ISOLATED = 55
IDX_Q8_START = 57
IDX_Q8_END = 65

# Q8 one-hot order in NPZ [57:65]
Q8_CLASSES = ["G", "H", "I", "B", "E", "S", "T", "C"]

# Q8 -> Q3 reduction (applied to Q8 integer labels)
# G=0, H=1, I=2 -> Helix -> Q3=2
# B=3, E=4      -> Strand -> Q3=1
# S=5, T=6, C=7 -> Coil   -> Q3=0
Q8_TO_Q3 = {0: 2, 1: 2, 2: 2, 3: 1, 4: 1, 5: 0, 6: 0, 7: 0}

# CSV Q8 character -> NPZ-native integer mapping (for NEW364 test set)
CSV_DSSP8_TO_INT = {
    "G": 0,
    "H": 1,
    "I": 2,
    "B": 3,
    "E": 4,
    "S": 5,
    "T": 6,
    "C": 7,
}

# CSV Q3 character -> integer mapping
CSV_DSSP3_TO_INT = {
    "C": 0,
    "E": 1,
    "H": 2,
}

# Sentinel value for unresolved regression targets
UNRESOLVED_SCORE = 999.0


# ---------------------------------------------------------------------------
# NPZ parsing
# ---------------------------------------------------------------------------


def parse_netsurfp2_npz(npz_path: Path) -> dict:
    """Parse a NetSurfP-2.0 NPZ file into structured data.

    Returns a dict with keys:
    - ``sequences``: list[str]
    - ``q8_labels``: list[list[int]] — 8-state DSSP labels (0-7)
    - ``q3_labels``: list[list[int]] — 3-state DSSP labels (0-2)
    - ``rsa_scores``: list[list[float]] — RSA values (999.0 for unresolved)
    - ``resolved``: list[list[int]] — per-residue resolved mask (1/0)
    """
    npz = np.load(npz_path, allow_pickle=True)
    data = npz["data"]  # shape: (n_samples, max_seq_len, 68)

    sequences: list[str] = []
    q8_labels: list[list[int]] = []
    q3_labels: list[list[int]] = []
    rsa_scores: list[list[float]] = []
    resolved_lists: list[list[int]] = []

    for sample_idx in range(len(data)):
        sample = data[sample_idx]

        # Determine sequence length from sequence mask
        seq_mask = sample[:, IDX_SEQ_MASK]
        seq_len = int(seq_mask.sum())

        # Extract amino acid sequence from one-hot encoding
        aa_onehot = sample[:seq_len, IDX_AA_START:IDX_AA_END]
        aa_sum = aa_onehot.sum(axis=1)
        aa_indices = aa_onehot.argmax(axis=1)

        chars = []
        for pos in range(seq_len):
            if aa_sum[pos] == 0:
                # All-zero one-hot = unknown residue
                chars.append("X")
            else:
                chars.append(AA_ALPHABET[aa_indices[pos]])
        sequence = "".join(chars)

        # Extract disorder mask -> resolved
        disorder = sample[:seq_len, IDX_DISORDER_MASK]
        resolved = [int(d) for d in disorder]

        # Extract Q8 labels from one-hot
        q8_onehot = sample[:seq_len, IDX_Q8_START:IDX_Q8_END]
        q8 = q8_onehot.argmax(axis=1).tolist()

        # Derive Q3 from Q8
        q3 = [Q8_TO_Q3[label] for label in q8]

        # Extract RSA (isolated), set to sentinel where unresolved
        rsa_raw = sample[:seq_len, IDX_RSA_ISOLATED]
        rsa = []
        for pos in range(seq_len):
            if disorder[pos] == 0:
                rsa.append(UNRESOLVED_SCORE)
            else:
                rsa.append(float(rsa_raw[pos]))

        sequences.append(sequence)
        q8_labels.append(q8)
        q3_labels.append(q3)
        rsa_scores.append(rsa)
        resolved_lists.append(resolved)

    return {
        "sequences": sequences,
        "q8_labels": q8_labels,
        "q3_labels": q3_labels,
        "rsa_scores": rsa_scores,
        "resolved": resolved_lists,
    }


# ---------------------------------------------------------------------------
# CSV parsing (for NEW364 test set — only has Q3/Q8/disorder, no RSA)
# ---------------------------------------------------------------------------


def parse_netsurfp2_csv(csv_path: Path) -> dict:
    """Parse a ProtTrans-format CSV into structured data.

    Returns a dict with keys:
    - ``sequences``: list[str]
    - ``q8_labels``: list[list[int]]
    - ``q3_labels``: list[list[int]]
    - ``resolved``: list[list[int]]

    Note: No RSA data is available in the CSV format.
    """
    sequences: list[str] = []
    q8_labels: list[list[int]] = []
    q3_labels: list[list[int]] = []
    resolved_lists: list[list[int]] = []

    with open(csv_path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        header = [h.strip() for h in header]

        required = ["input", "dssp3", "dssp8", "disorder"]
        assert header[:4] == required, (
            f"Unexpected CSV header: {header} (expected {required} as first 4 columns)"
        )

        col_indices = {name: idx for idx, name in enumerate(header)}
        col_input = col_indices["input"]
        col_dssp3 = col_indices["dssp3"]
        col_dssp8 = col_indices["dssp8"]
        col_disorder = col_indices["disorder"]

        for row_idx, row in enumerate(reader):
            seq_tokens = row[col_input].strip().split()
            dssp3_tokens = row[col_dssp3].strip().split()
            dssp8_tokens = row[col_dssp8].strip().split()
            disorder_tokens = row[col_disorder].strip().split()

            seq = "".join(seq_tokens)

            if len(seq) != len(dssp8_tokens):
                raise ValueError(
                    f"Row {row_idx}: sequence length ({len(seq)}) != "
                    f"dssp8 length ({len(dssp8_tokens)})"
                )
            if len(seq) != len(dssp3_tokens):
                raise ValueError(
                    f"Row {row_idx}: sequence length ({len(seq)}) != "
                    f"dssp3 length ({len(dssp3_tokens)})"
                )
            if len(seq) != len(disorder_tokens):
                raise ValueError(
                    f"Row {row_idx}: sequence length ({len(seq)}) != "
                    f"disorder length ({len(disorder_tokens)})"
                )

            # Map Q8 characters to integers (NPZ-native order)
            q8 = []
            for char in dssp8_tokens:
                if char not in CSV_DSSP8_TO_INT:
                    raise ValueError(f"Row {row_idx}: unexpected dssp8 class '{char}'")
                q8.append(CSV_DSSP8_TO_INT[char])

            # Map Q3 characters to integers
            q3 = []
            for char in dssp3_tokens:
                if char not in CSV_DSSP3_TO_INT:
                    raise ValueError(f"Row {row_idx}: unexpected dssp3 class '{char}'")
                q3.append(CSV_DSSP3_TO_INT[char])

            # Convert disorder to resolved mask
            resolved = []
            for val_str in disorder_tokens:
                val = float(val_str)
                if val == 1.0:
                    resolved.append(1)
                elif val == 0.0:
                    resolved.append(0)
                else:
                    raise ValueError(
                        f"Row {row_idx}: unexpected disorder value {val} "
                        f"(expected 0.0 or 1.0)"
                    )

            sequences.append(seq)
            q8_labels.append(q8)
            q3_labels.append(q3)
            resolved_lists.append(resolved)

    return {
        "sequences": sequences,
        "q8_labels": q8_labels,
        "q3_labels": q3_labels,
        "resolved": resolved_lists,
    }


# ---------------------------------------------------------------------------
# DataFrame construction helpers
# ---------------------------------------------------------------------------


def make_classification_df(
    sequences: list[str],
    labels: list[list[int]],
    resolved: list[list[int]],
) -> pd.DataFrame:
    """Build a per-residue classification parquet DataFrame."""
    return pd.DataFrame(
        {
            "sequence": sequences,
            "label": labels,
            "resolved": resolved,
        }
    )


def make_regression_df(
    sequences: list[str],
    scores: list[list[float]],
) -> pd.DataFrame:
    """Build a per-residue regression parquet DataFrame."""
    return pd.DataFrame(
        {
            "sequence": sequences,
            "score": scores,
        }
    )


# ---------------------------------------------------------------------------
# Snakemake I/O
# ---------------------------------------------------------------------------

raw_dir = Path(str(snakemake.input["raw_dir"]))

train_npz_path = raw_dir / "Train_HHblits.npz"
cb513_npz_path = raw_dir / "CB513_HHblits.npz"
new364_csv_path = raw_dir / "NEW364.csv"

# ---- Parse training data from NPZ ----
logger.info("Parsing training data from %s", train_npz_path)
train_data = parse_netsurfp2_npz(train_npz_path)
logger.info("Parsed %d training sequences", len(train_data["sequences"]))

# Write SecStr8 train_full
secstr8_train_full = Path(str(snakemake.output["secstr8_train_full"]))
secstr8_train_full.parent.mkdir(parents=True, exist_ok=True)
df = make_classification_df(
    train_data["sequences"], train_data["q8_labels"], train_data["resolved"]
)
df.to_parquet(secstr8_train_full)
logger.info("Wrote SecStr8 train_full: %d rows", len(df))

# Write SecStr train_full
secstr_train_full = Path(str(snakemake.output["secstr_train_full"]))
secstr_train_full.parent.mkdir(parents=True, exist_ok=True)
df = make_classification_df(
    train_data["sequences"], train_data["q3_labels"], train_data["resolved"]
)
df.to_parquet(secstr_train_full)
logger.info("Wrote SecStr train_full: %d rows", len(df))

# Write RSA train_full
rsa_train_full = Path(str(snakemake.output["rsa_train_full"]))
rsa_train_full.parent.mkdir(parents=True, exist_ok=True)
df = make_regression_df(train_data["sequences"], train_data["rsa_scores"])
df.to_parquet(rsa_train_full)
logger.info("Wrote RSA train_full: %d rows", len(df))

# ---- Parse CB513 test data from NPZ (for RSA test) ----
logger.info("Parsing CB513 test data from %s", cb513_npz_path)
cb513_data = parse_netsurfp2_npz(cb513_npz_path)
logger.info("Parsed %d CB513 test sequences", len(cb513_data["sequences"]))

rsa_test = Path(str(snakemake.output["rsa_test"]))
rsa_test.parent.mkdir(parents=True, exist_ok=True)
df = make_regression_df(cb513_data["sequences"], cb513_data["rsa_scores"])
df.to_parquet(rsa_test)
logger.info("Wrote RSA test: %d rows", len(df))

# ---- Parse NEW364 test data from CSV (for SecStr/SecStr8 test) ----
logger.info("Parsing NEW364 test data from %s", new364_csv_path)
new364_data = parse_netsurfp2_csv(new364_csv_path)
logger.info("Parsed %d NEW364 test sequences", len(new364_data["sequences"]))

# Write SecStr8 test
secstr8_test = Path(str(snakemake.output["secstr8_test"]))
secstr8_test.parent.mkdir(parents=True, exist_ok=True)
df = make_classification_df(
    new364_data["sequences"], new364_data["q8_labels"], new364_data["resolved"]
)
df.to_parquet(secstr8_test)
logger.info("Wrote SecStr8 test: %d rows", len(df))

# Write SecStr test
secstr_test = Path(str(snakemake.output["secstr_test"]))
secstr_test.parent.mkdir(parents=True, exist_ok=True)
df = make_classification_df(
    new364_data["sequences"], new364_data["q3_labels"], new364_data["resolved"]
)
df.to_parquet(secstr_test)
logger.info("Wrote SecStr test: %d rows", len(df))
