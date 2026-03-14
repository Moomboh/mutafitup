"""Tests for the FASTA extraction step of the leakage analysis pipeline."""

import importlib
import sys
from pathlib import Path

import pandas as pd
import pytest

# Import the extract_fasta function directly from the script module.
# The script uses ``from snakemake.script import snakemake`` at module level,
# which would fail outside Snakemake. We therefore import only the function we
# need by loading the module manually and patching the snakemake import.

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "workflow" / "scripts"


@pytest.fixture()
def extract_fasta_func():
    """Import and return the ``extract_fasta`` helper function."""
    script_path = SCRIPTS_DIR / "data_stats" / "leakage" / "extract_fasta.py"
    assert script_path.exists(), f"Script not found: {script_path}"

    import types

    # Provide a minimal snakemake stub so the top-level import succeeds.
    fake_snakemake_mod = types.ModuleType("snakemake")
    fake_script_mod = types.ModuleType("snakemake.script")
    fake_script_mod.snakemake = None
    fake_snakemake_mod.script = fake_script_mod
    sys.modules.setdefault("snakemake", fake_snakemake_mod)
    sys.modules.setdefault("snakemake.script", fake_script_mod)

    spec = importlib.util.spec_from_file_location("extract_fasta", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.extract_fasta


def test_basic_extraction(tmp_path, extract_fasta_func):
    """Sequences are correctly written to a FASTA file."""
    df = pd.DataFrame({"sequence": ["ACDE", "FGHI", "KLMN"]})
    parquet_path = tmp_path / "data.parquet"
    df.to_parquet(parquet_path)

    fasta_path = tmp_path / "output" / "test.fasta"
    extract_fasta_func(parquet_path, fasta_path, dataset="TestDS", split="train")

    assert fasta_path.exists()

    records = _parse_fasta(fasta_path)
    assert len(records) == 3
    assert records["TestDS_train_0"] == "ACDE"
    assert records["TestDS_train_1"] == "FGHI"
    assert records["TestDS_train_2"] == "KLMN"


def test_empty_sequences_skipped(tmp_path, extract_fasta_func):
    """Empty / whitespace-only sequences are not written."""
    df = pd.DataFrame({"sequence": ["ACDE", "", "  ", "FGHI"]})
    parquet_path = tmp_path / "data.parquet"
    df.to_parquet(parquet_path)

    fasta_path = tmp_path / "out.fasta"
    extract_fasta_func(parquet_path, fasta_path, dataset="DS", split="valid")

    records = _parse_fasta(fasta_path)
    assert len(records) == 2
    assert "DS_valid_0" in records
    assert "DS_valid_3" in records


def test_missing_sequence_column(tmp_path, extract_fasta_func):
    """Raise ValueError when parquet has no 'sequence' column."""
    df = pd.DataFrame({"label": [1, 2, 3]})
    parquet_path = tmp_path / "bad.parquet"
    df.to_parquet(parquet_path)

    fasta_path = tmp_path / "out.fasta"
    with pytest.raises(ValueError, match="sequence"):
        extract_fasta_func(parquet_path, fasta_path, dataset="X", split="test")


def test_output_directory_created(tmp_path, extract_fasta_func):
    """Output parent directories are created automatically."""
    df = pd.DataFrame({"sequence": ["ACDE"]})
    parquet_path = tmp_path / "data.parquet"
    df.to_parquet(parquet_path)

    fasta_path = tmp_path / "deep" / "nested" / "dir" / "out.fasta"
    extract_fasta_func(parquet_path, fasta_path, dataset="D", split="train")

    assert fasta_path.exists()
    records = _parse_fasta(fasta_path)
    assert len(records) == 1


def test_unique_headers(tmp_path, extract_fasta_func):
    """All FASTA headers are unique."""
    df = pd.DataFrame({"sequence": ["ACDE", "ACDE", "FGHI"]})
    parquet_path = tmp_path / "dup.parquet"
    df.to_parquet(parquet_path)

    fasta_path = tmp_path / "out.fasta"
    extract_fasta_func(parquet_path, fasta_path, dataset="Dup", split="train")

    records = _parse_fasta(fasta_path)
    # Even if sequences are the same, headers should be unique (different indices)
    assert len(records) == 3
    headers = list(records.keys())
    assert len(headers) == len(set(headers))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_fasta(path: Path) -> dict[str, str]:
    """Parse a FASTA file into a dict of header -> sequence."""
    records = {}
    header = None
    chunks = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records[header] = "".join(chunks)
                header = line[1:]
                chunks = []
            else:
                chunks.append(line)
    if header is not None:
        records[header] = "".join(chunks)
    return records
