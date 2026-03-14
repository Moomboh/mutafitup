"""Tests for the heatmap plotting / leakage matrix computation logic."""

import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest

try:
    import seaborn  # noqa: F401

    _HAS_SEABORN = True
except ImportError:
    _HAS_SEABORN = False

pytestmark = pytest.mark.skipif(not _HAS_SEABORN, reason="seaborn not installed")

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "workflow" / "scripts"


@pytest.fixture()
def heatmap_module():
    """Import the plot_heatmap module with a stubbed snakemake dependency."""
    script_path = SCRIPTS_DIR / "data_stats" / "leakage" / "plot_heatmap.py"
    assert script_path.exists(), f"Script not found: {script_path}"

    fake_snakemake_mod = types.ModuleType("snakemake")
    fake_script_mod = types.ModuleType("snakemake.script")
    fake_script_mod.snakemake = None
    fake_snakemake_mod.script = fake_script_mod
    sys.modules.setdefault("snakemake", fake_snakemake_mod)
    sys.modules.setdefault("snakemake.script", fake_script_mod)

    spec = importlib.util.spec_from_file_location("plot_heatmap", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# count_fasta_sequences
# ---------------------------------------------------------------------------


class TestCountFastaSequences:
    def test_counts_sequences(self, tmp_path, heatmap_module):
        fasta = tmp_path / "test.fasta"
        fasta.write_text(">seq1\nACDE\n>seq2\nFGHI\n>seq3\nKLMN\n")
        assert heatmap_module.count_fasta_sequences(fasta) == 3

    def test_empty_file(self, tmp_path, heatmap_module):
        fasta = tmp_path / "empty.fasta"
        fasta.write_text("")
        assert heatmap_module.count_fasta_sequences(fasta) == 0

    def test_single_sequence(self, tmp_path, heatmap_module):
        fasta = tmp_path / "single.fasta"
        fasta.write_text(">only\nACDE\n")
        assert heatmap_module.count_fasta_sequences(fasta) == 1


# ---------------------------------------------------------------------------
# count_query_hits
# ---------------------------------------------------------------------------


class TestCountQueryHits:
    def test_counts_unique_queries(self, tmp_path, heatmap_module):
        tsv = tmp_path / "results.tsv"
        tsv.write_text(
            "query1\ttarget1\t0.95\nquery1\ttarget2\t0.80\nquery2\ttarget1\t0.60\n"
        )
        assert heatmap_module.count_query_hits(tsv) == 2

    def test_empty_file(self, tmp_path, heatmap_module):
        tsv = tmp_path / "empty.tsv"
        tsv.write_text("")
        assert heatmap_module.count_query_hits(tsv) == 0

    def test_missing_file(self, tmp_path, heatmap_module):
        tsv = tmp_path / "nonexistent.tsv"
        assert heatmap_module.count_query_hits(tsv) == 0

    def test_single_hit(self, tmp_path, heatmap_module):
        tsv = tmp_path / "single.tsv"
        tsv.write_text("q1\tt1\t0.99\n")
        assert heatmap_module.count_query_hits(tsv) == 1

    def test_blank_lines_ignored(self, tmp_path, heatmap_module):
        tsv = tmp_path / "blanks.tsv"
        tsv.write_text("q1\tt1\t0.5\n\n\nq2\tt1\t0.6\n\n")
        assert heatmap_module.count_query_hits(tsv) == 2


# ---------------------------------------------------------------------------
# compute_leakage_matrix
# ---------------------------------------------------------------------------


class TestComputeLeakageMatrix:
    def _setup_files(
        self, tmp_path, datasets, query_split, target_split, n_seqs_per_query, hits
    ):
        """Create FASTA and TSV fixtures.

        Args:
            datasets: list of dataset names
            query_split: "valid" or "test"
            target_split: "train", "valid", or "test"
            n_seqs_per_query: dict mapping dataset -> number of query sequences
            hits: dict mapping (query_dataset, target_dataset) -> set of query IDs with hits
        """
        fasta_dir = tmp_path / "fasta"
        results_dir = tmp_path / "results"
        fasta_dir.mkdir()
        results_dir.mkdir()

        for ds in datasets:
            n = n_seqs_per_query.get(ds, 0)
            fasta_path = fasta_dir / f"{ds}_{query_split}.fasta"
            with fasta_path.open("w") as fh:
                for i in range(n):
                    fh.write(f">{ds}_{query_split}_{i}\nACDE\n")

        for q_ds in datasets:
            for t_ds in datasets:
                tsv_path = (
                    results_dir / f"{q_ds}_{query_split}_vs_{t_ds}_{target_split}.tsv"
                )
                hit_ids = hits.get((q_ds, t_ds), set())
                with tsv_path.open("w") as fh:
                    for qid in hit_ids:
                        fh.write(f"{qid}\tsome_target\t0.99\n")

        return fasta_dir, results_dir

    def test_full_leakage(self, tmp_path, heatmap_module):
        """All queries match -> 100%."""
        datasets = ["a", "b"]
        fasta_dir, results_dir = self._setup_files(
            tmp_path,
            datasets,
            "valid",
            "train",
            n_seqs_per_query={"a": 3, "b": 2},
            hits={
                ("a", "a"): {"a_valid_0", "a_valid_1", "a_valid_2"},
                ("a", "b"): {"a_valid_0", "a_valid_1", "a_valid_2"},
                ("b", "a"): {"b_valid_0", "b_valid_1"},
                ("b", "b"): {"b_valid_0", "b_valid_1"},
            },
        )

        matrix, row_labels, col_labels = heatmap_module.compute_leakage_matrix(
            datasets, "valid", "train", results_dir, fasta_dir
        )

        assert matrix.shape == (2, 2)
        np.testing.assert_allclose(matrix, 100.0)

    def test_no_leakage(self, tmp_path, heatmap_module):
        """No queries match -> 0%."""
        datasets = ["a", "b"]
        fasta_dir, results_dir = self._setup_files(
            tmp_path,
            datasets,
            "test",
            "train",
            n_seqs_per_query={"a": 5, "b": 3},
            hits={},
        )

        matrix, _, _ = heatmap_module.compute_leakage_matrix(
            datasets, "test", "train", results_dir, fasta_dir
        )

        assert matrix.shape == (2, 2)
        np.testing.assert_allclose(matrix, 0.0)

    def test_partial_leakage(self, tmp_path, heatmap_module):
        """Partial overlap -> correct percentages."""
        datasets = ["x", "y"]
        fasta_dir, results_dir = self._setup_files(
            tmp_path,
            datasets,
            "valid",
            "train",
            n_seqs_per_query={"x": 10, "y": 4},
            hits={
                ("x", "x"): {f"x_valid_{i}" for i in range(5)},  # 50%
                ("x", "y"): {f"x_valid_{i}" for i in range(2)},  # 20%
                ("y", "x"): {"y_valid_0"},  # 25%
                ("y", "y"): {
                    "y_valid_0",
                    "y_valid_1",
                    "y_valid_2",
                    "y_valid_3",
                },  # 100%
            },
        )

        matrix, _, _ = heatmap_module.compute_leakage_matrix(
            datasets, "valid", "train", results_dir, fasta_dir
        )

        assert matrix.shape == (2, 2)
        np.testing.assert_allclose(matrix[0, 0], 50.0)
        np.testing.assert_allclose(matrix[0, 1], 20.0)
        np.testing.assert_allclose(matrix[1, 0], 25.0)
        np.testing.assert_allclose(matrix[1, 1], 100.0)

    def test_asymmetric_matrix(self, tmp_path, heatmap_module):
        """Leakage from a->b may differ from b->a."""
        datasets = ["p", "q"]
        fasta_dir, results_dir = self._setup_files(
            tmp_path,
            datasets,
            "valid",
            "train",
            n_seqs_per_query={"p": 100, "q": 10},
            hits={
                ("p", "q"): {f"p_valid_{i}" for i in range(10)},  # 10%
                ("q", "p"): {f"q_valid_{i}" for i in range(9)},  # 90%
            },
        )

        matrix, _, _ = heatmap_module.compute_leakage_matrix(
            datasets, "valid", "train", results_dir, fasta_dir
        )

        np.testing.assert_allclose(matrix[0, 1], 10.0)
        np.testing.assert_allclose(matrix[1, 0], 90.0)

    def test_test_vs_valid_comparison(self, tmp_path, heatmap_module):
        """Test that test vs valid comparison works correctly."""
        datasets = ["a", "b"]
        fasta_dir, results_dir = self._setup_files(
            tmp_path,
            datasets,
            "test",
            "valid",
            n_seqs_per_query={"a": 4, "b": 2},
            hits={
                ("a", "a"): {"a_test_0", "a_test_1"},  # 50%
                ("b", "b"): {"b_test_0"},  # 50%
            },
        )

        matrix, _, _ = heatmap_module.compute_leakage_matrix(
            datasets, "test", "valid", results_dir, fasta_dir
        )

        assert matrix.shape == (2, 2)
        np.testing.assert_allclose(matrix[0, 0], 50.0)
        np.testing.assert_allclose(matrix[0, 1], 0.0)
        np.testing.assert_allclose(matrix[1, 0], 0.0)
        np.testing.assert_allclose(matrix[1, 1], 50.0)


# ---------------------------------------------------------------------------
# plot_heatmap (smoke test – just verifies file is created)
# ---------------------------------------------------------------------------


class TestPlotHeatmap:
    def test_creates_png(self, tmp_path, heatmap_module):
        matrix = np.array([[100.0, 50.0], [25.0, 80.0]])
        output_path = tmp_path / "heatmap.png"
        heatmap_module.plot_heatmap(
            matrix,
            row_labels=["ds1", "ds2"],
            col_labels=["ds1", "ds2"],
            title="Test leakage heatmap",
            output_path=output_path,
        )
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_single_dataset(self, tmp_path, heatmap_module):
        """1x1 matrix should still produce a valid plot."""
        matrix = np.array([[42.0]])
        output_path = tmp_path / "single.png"
        heatmap_module.plot_heatmap(
            matrix,
            row_labels=["only"],
            col_labels=["only"],
            title="Single dataset",
            output_path=output_path,
        )
        assert output_path.exists()

    def test_five_datasets(self, tmp_path, heatmap_module):
        """4x4 matrix matching the actual use case."""
        rng = np.random.default_rng(42)
        matrix = rng.uniform(0, 100, size=(4, 4))
        labels = ["secstr", "disorder", "meltome", "subloc"]
        output_path = tmp_path / "five.png"
        heatmap_module.plot_heatmap(
            matrix,
            row_labels=labels,
            col_labels=labels,
            title="MMseqs2 leakage: valid vs train (min identity 30%)",
            output_path=output_path,
        )
        assert output_path.exists()
