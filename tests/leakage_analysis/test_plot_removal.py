"""Tests for the removal heatmap / bar plot computation and plotting logic."""

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
def removal_module():
    """Import the plot_removal module with a stubbed snakemake dependency."""
    script_path = SCRIPTS_DIR / "data_stats" / "leakage" / "plot_removal.py"
    assert script_path.exists(), f"Script not found: {script_path}"

    fake_snakemake_mod = types.ModuleType("snakemake")
    fake_script_mod = types.ModuleType("snakemake.script")
    fake_script_mod.snakemake = None
    fake_snakemake_mod.script = fake_script_mod
    sys.modules.setdefault("snakemake", fake_snakemake_mod)
    sys.modules.setdefault("snakemake.script", fake_script_mod)

    spec = importlib.util.spec_from_file_location("plot_removal", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# collect_target_ids
# ---------------------------------------------------------------------------


class TestCollectTargetIds:
    def test_unique_targets(self, tmp_path, removal_module):
        tsv = tmp_path / "results.tsv"
        tsv.write_text(
            "query1\ttarget1\t0.95\nquery1\ttarget2\t0.80\nquery2\ttarget1\t0.60\n"
        )
        ids = removal_module.collect_target_ids(tsv)
        assert ids == {"target1", "target2"}

    def test_empty_file(self, tmp_path, removal_module):
        tsv = tmp_path / "empty.tsv"
        tsv.write_text("")
        ids = removal_module.collect_target_ids(tsv)
        assert ids == set()

    def test_missing_file(self, tmp_path, removal_module):
        tsv = tmp_path / "nonexistent.tsv"
        ids = removal_module.collect_target_ids(tsv)
        assert ids == set()

    def test_single_hit(self, tmp_path, removal_module):
        tsv = tmp_path / "single.tsv"
        tsv.write_text("q1\tt1\t0.99\n")
        ids = removal_module.collect_target_ids(tsv)
        assert ids == {"t1"}

    def test_blank_lines_ignored(self, tmp_path, removal_module):
        tsv = tmp_path / "blanks.tsv"
        tsv.write_text("q1\tt1\t0.5\n\n\nq2\tt2\t0.6\n\n")
        ids = removal_module.collect_target_ids(tsv)
        assert ids == {"t1", "t2"}

    def test_duplicate_targets_counted_once(self, tmp_path, removal_module):
        tsv = tmp_path / "dupes.tsv"
        tsv.write_text("q1\tt1\t0.5\nq2\tt1\t0.6\nq3\tt1\t0.7\n")
        ids = removal_module.collect_target_ids(tsv)
        assert ids == {"t1"}


# ---------------------------------------------------------------------------
# collect_query_ids
# ---------------------------------------------------------------------------


class TestCollectQueryIds:
    def test_unique_queries(self, tmp_path, removal_module):
        tsv = tmp_path / "results.tsv"
        tsv.write_text(
            "query1\ttarget1\t0.95\nquery1\ttarget2\t0.80\nquery2\ttarget1\t0.60\n"
        )
        ids = removal_module.collect_query_ids(tsv)
        assert ids == {"query1", "query2"}

    def test_empty_file(self, tmp_path, removal_module):
        tsv = tmp_path / "empty.tsv"
        tsv.write_text("")
        ids = removal_module.collect_query_ids(tsv)
        assert ids == set()

    def test_missing_file(self, tmp_path, removal_module):
        tsv = tmp_path / "nonexistent.tsv"
        ids = removal_module.collect_query_ids(tsv)
        assert ids == set()

    def test_single_hit(self, tmp_path, removal_module):
        tsv = tmp_path / "single.tsv"
        tsv.write_text("q1\tt1\t0.99\n")
        ids = removal_module.collect_query_ids(tsv)
        assert ids == {"q1"}

    def test_blank_lines_ignored(self, tmp_path, removal_module):
        tsv = tmp_path / "blanks.tsv"
        tsv.write_text("q1\tt1\t0.5\n\n\nq2\tt2\t0.6\n\n")
        ids = removal_module.collect_query_ids(tsv)
        assert ids == {"q1", "q2"}

    def test_duplicate_queries_counted_once(self, tmp_path, removal_module):
        tsv = tmp_path / "dupes.tsv"
        tsv.write_text("q1\tt1\t0.5\nq1\tt2\t0.6\nq1\tt3\t0.7\n")
        ids = removal_module.collect_query_ids(tsv)
        assert ids == {"q1"}


# ---------------------------------------------------------------------------
# Helper for setting up test fixtures
# ---------------------------------------------------------------------------


def _setup_files(
    tmp_path,
    datasets,
    query_split,
    target_split,
    n_target_per_ds,
    target_hits,
    n_query_per_ds=None,
    query_hits=None,
):
    """Create FASTA and TSV fixtures.

    Args:
        datasets: list of dataset names
        query_split: "valid" or "test"
        target_split: "train", "valid", or "test"
        n_target_per_ds: dict mapping dataset -> number of target sequences
        target_hits: dict mapping (query_dataset, target_dataset) -> set of target IDs
        n_query_per_ds: dict mapping dataset -> number of query sequences (optional)
        query_hits: dict mapping (query_dataset, target_dataset) -> set of query IDs (optional)
    """
    fasta_dir = tmp_path / "fasta"
    results_dir = tmp_path / "results"
    fasta_dir.mkdir()
    results_dir.mkdir()

    # Create target FASTAs
    for ds in datasets:
        n = n_target_per_ds.get(ds, 0)
        target_path = fasta_dir / f"{ds}_{target_split}.fasta"
        with target_path.open("w") as fh:
            for i in range(n):
                fh.write(f">{ds}_{target_split}_{i}\nACDE\n")

    # Create query FASTAs
    for ds in datasets:
        n = (n_query_per_ds or {}).get(ds, 1)
        query_path = fasta_dir / f"{ds}_{query_split}.fasta"
        with query_path.open("w") as fh:
            for i in range(n):
                fh.write(f">{ds}_{query_split}_{i}\nACDE\n")

    # Create result TSVs
    for q_ds in datasets:
        for t_ds in datasets:
            tsv_path = (
                results_dir / f"{q_ds}_{query_split}_vs_{t_ds}_{target_split}.tsv"
            )
            with tsv_path.open("w") as fh:
                # Write target-based hits
                t_ids = target_hits.get((q_ds, t_ds), set())
                for tid in t_ids:
                    fh.write(f"some_query\t{tid}\t0.99\n")
                # Write query-based hits (for query removal tests)
                q_ids = (query_hits or {}).get((q_ds, t_ds), set())
                for qid in q_ids:
                    fh.write(f"{qid}\tsome_target\t0.99\n")

    return fasta_dir, results_dir


# ---------------------------------------------------------------------------
# compute_target_removal_matrix
# ---------------------------------------------------------------------------


class TestComputeTargetRemovalMatrix:
    def test_full_removal(self, tmp_path, removal_module):
        """All target seqs are matched -> 100%."""
        datasets = ["a", "b"]
        fasta_dir, results_dir = _setup_files(
            tmp_path,
            datasets,
            "valid",
            "train",
            n_target_per_ds={"a": 3, "b": 2},
            target_hits={
                ("a", "a"): {"a_train_0", "a_train_1", "a_train_2"},
                ("a", "b"): {"b_train_0", "b_train_1"},
                ("b", "a"): {"a_train_0", "a_train_1", "a_train_2"},
                ("b", "b"): {"b_train_0", "b_train_1"},
            },
        )

        pct, absc, row_labels, col_labels = (
            removal_module.compute_target_removal_matrix(
                datasets, "valid", "train", results_dir, fasta_dir
            )
        )

        assert pct.shape == (3, 2)  # 2 datasets + 1 Total row
        assert row_labels == ["a", "b", "Total"]
        assert col_labels == ["a", "b"]
        # Per-dataset rows
        np.testing.assert_allclose(pct[0, 0], 100.0)  # A->A: 3/3
        np.testing.assert_allclose(pct[0, 1], 100.0)  # A->B: 2/2
        np.testing.assert_allclose(pct[1, 0], 100.0)  # B->A: 3/3
        np.testing.assert_allclose(pct[1, 1], 100.0)  # B->B: 2/2
        # Total row (union = same as per-row since all match)
        np.testing.assert_allclose(pct[2, 0], 100.0)
        np.testing.assert_allclose(pct[2, 1], 100.0)

    def test_no_removal(self, tmp_path, removal_module):
        """No targets hit -> 0%."""
        datasets = ["a", "b"]
        fasta_dir, results_dir = _setup_files(
            tmp_path,
            datasets,
            "test",
            "train",
            n_target_per_ds={"a": 10, "b": 5},
            target_hits={},
        )

        pct, absc, _, _ = removal_module.compute_target_removal_matrix(
            datasets, "test", "train", results_dir, fasta_dir
        )

        assert pct.shape == (3, 2)
        np.testing.assert_allclose(pct, 0.0)
        np.testing.assert_allclose(absc, 0.0)

    def test_partial_removal(self, tmp_path, removal_module):
        """Partial target hits -> correct percentages."""
        datasets = ["x", "y"]
        fasta_dir, results_dir = _setup_files(
            tmp_path,
            datasets,
            "valid",
            "train",
            n_target_per_ds={"x": 10, "y": 4},
            target_hits={
                ("x", "x"): {"x_train_0", "x_train_1", "x_train_2"},  # 3/10 = 30%
                ("x", "y"): {"y_train_0"},  # 1/4 = 25%
                ("y", "x"): {"x_train_2", "x_train_3"},  # 2/10 = 20%
                ("y", "y"): {"y_train_0", "y_train_1"},  # 2/4 = 50%
            },
        )

        pct, absc, _, _ = removal_module.compute_target_removal_matrix(
            datasets, "valid", "train", results_dir, fasta_dir
        )

        assert pct.shape == (3, 2)
        np.testing.assert_allclose(pct[0, 0], 30.0)
        np.testing.assert_allclose(pct[0, 1], 25.0)
        np.testing.assert_allclose(pct[1, 0], 20.0)
        np.testing.assert_allclose(pct[1, 1], 50.0)

        # Absolute counts
        np.testing.assert_allclose(absc[0, 0], 3)
        np.testing.assert_allclose(absc[0, 1], 1)
        np.testing.assert_allclose(absc[1, 0], 2)
        np.testing.assert_allclose(absc[1, 1], 2)

    def test_total_row_uses_union(self, tmp_path, removal_module):
        """Total row should be the union of target IDs, not the sum."""
        datasets = ["a", "b"]
        fasta_dir, results_dir = _setup_files(
            tmp_path,
            datasets,
            "valid",
            "train",
            n_target_per_ds={"a": 10, "b": 10},
            target_hits={
                # a->a hits train_0,1,2  (3 targets)
                ("a", "a"): {"a_train_0", "a_train_1", "a_train_2"},
                # b->a hits train_1,2,3  (3 targets, 2 overlap with a->a)
                ("b", "a"): {"a_train_1", "a_train_2", "a_train_3"},
                # union for column a = {0,1,2,3} = 4, not 6
            },
        )

        pct, absc, _, _ = removal_module.compute_target_removal_matrix(
            datasets, "valid", "train", results_dir, fasta_dir
        )

        # Total row for column A
        np.testing.assert_allclose(absc[2, 0], 4)  # union of {0,1,2} and {1,2,3}
        np.testing.assert_allclose(pct[2, 0], 40.0)  # 4/10

    def test_three_datasets(self, tmp_path, removal_module):
        """Matrix shape is correct for 3 datasets (4 rows including Total)."""
        datasets = ["a", "b", "c"]
        fasta_dir, results_dir = _setup_files(
            tmp_path,
            datasets,
            "valid",
            "train",
            n_target_per_ds={"a": 5, "b": 5, "c": 5},
            target_hits={
                ("a", "b"): {"b_train_0"},
                ("c", "b"): {"b_train_0", "b_train_1"},
            },
        )

        pct, absc, row_labels, col_labels = (
            removal_module.compute_target_removal_matrix(
                datasets, "valid", "train", results_dir, fasta_dir
            )
        )

        assert pct.shape == (4, 3)
        assert row_labels == ["a", "b", "c", "Total"]
        assert col_labels == ["a", "b", "c"]

        # Column b total row = union({b_train_0}, {b_train_0, b_train_1}) = {b_train_0, b_train_1} = 2
        np.testing.assert_allclose(absc[3, 1], 2)
        np.testing.assert_allclose(pct[3, 1], 40.0)  # 2/5

    def test_cross_task_only_skips_diagonal(self, tmp_path, removal_module):
        """With cross_task_only=True, diagonal entries are zero."""
        datasets = ["a", "b"]
        fasta_dir, results_dir = _setup_files(
            tmp_path,
            datasets,
            "valid",
            "train",
            n_target_per_ds={"a": 10, "b": 10},
            target_hits={
                ("a", "a"): {"a_train_0", "a_train_1"},  # within-task
                ("a", "b"): {"b_train_0"},  # cross-task
                ("b", "a"): {"a_train_2"},  # cross-task
                ("b", "b"): {"b_train_0", "b_train_1", "b_train_2"},  # within-task
            },
        )

        pct, absc, _, _ = removal_module.compute_target_removal_matrix(
            datasets, "valid", "train", results_dir, fasta_dir, cross_task_only=True
        )

        # Diagonal should be 0
        np.testing.assert_allclose(pct[0, 0], 0.0)
        np.testing.assert_allclose(pct[1, 1], 0.0)
        np.testing.assert_allclose(absc[0, 0], 0.0)
        np.testing.assert_allclose(absc[1, 1], 0.0)

        # Off-diagonal should be computed
        np.testing.assert_allclose(pct[0, 1], 10.0)  # 1/10
        np.testing.assert_allclose(pct[1, 0], 10.0)  # 1/10

        # Total row should only include cross-task targets
        np.testing.assert_allclose(absc[2, 0], 1)  # only a_train_2 from b->a
        np.testing.assert_allclose(absc[2, 1], 1)  # only b_train_0 from a->b

    def test_test_vs_valid_target_removal(self, tmp_path, removal_module):
        """Target removal with test vs valid comparison."""
        datasets = ["a", "b"]
        fasta_dir, results_dir = _setup_files(
            tmp_path,
            datasets,
            "test",
            "valid",
            n_target_per_ds={"a": 5, "b": 4},
            target_hits={
                ("a", "a"): {"a_valid_0", "a_valid_1"},  # 2/5 = 40%
                ("b", "b"): {"b_valid_0"},  # 1/4 = 25%
            },
        )

        pct, absc, row_labels, col_labels = (
            removal_module.compute_target_removal_matrix(
                datasets, "test", "valid", results_dir, fasta_dir
            )
        )

        assert pct.shape == (3, 2)
        assert row_labels == ["a", "b", "Total"]
        np.testing.assert_allclose(pct[0, 0], 40.0)
        np.testing.assert_allclose(pct[1, 1], 25.0)


# ---------------------------------------------------------------------------
# compute_query_removal_matrix
# ---------------------------------------------------------------------------


class TestComputeQueryRemovalMatrix:
    def test_basic_query_removal(self, tmp_path, removal_module):
        """Counts unique query IDs and denominates by query set size."""
        datasets = ["a", "b"]
        fasta_dir, results_dir = _setup_files(
            tmp_path,
            datasets,
            "valid",
            "train",
            n_target_per_ds={"a": 100, "b": 100},
            target_hits={},
            n_query_per_ds={"a": 10, "b": 20},
            query_hits={
                ("a", "a"): {"a_valid_0", "a_valid_1"},  # 2/10 = 20%
                ("a", "b"): {"a_valid_0", "a_valid_2"},  # 2/10 = 20%
                ("b", "a"): {"b_valid_0"},  # 1/20 = 5%
                ("b", "b"): {"b_valid_0", "b_valid_1", "b_valid_2"},  # 3/20 = 15%
            },
        )

        pct, absc, row_labels, col_labels = removal_module.compute_query_removal_matrix(
            datasets, "valid", "train", results_dir, fasta_dir
        )

        # No Total row — shape is (n_datasets, n_datasets + 1)
        assert pct.shape == (2, 3)
        assert row_labels == ["a", "b"]
        assert col_labels == ["a", "b", "Total"]

        np.testing.assert_allclose(pct[0, 0], 20.0)  # a query vs a target: 2/10
        np.testing.assert_allclose(pct[0, 1], 20.0)  # a query vs b target: 2/10
        np.testing.assert_allclose(pct[1, 0], 5.0)  # b query vs a target: 1/20
        np.testing.assert_allclose(pct[1, 1], 15.0)  # b query vs b target: 3/20

        np.testing.assert_allclose(absc[0, 0], 2)
        np.testing.assert_allclose(absc[0, 1], 2)
        np.testing.assert_allclose(absc[1, 0], 1)
        np.testing.assert_allclose(absc[1, 1], 3)

        # Total column: a union across target sets = {a_valid_0, a_valid_1, a_valid_2} = 3/10
        np.testing.assert_allclose(absc[0, 2], 3)
        np.testing.assert_allclose(pct[0, 2], 30.0)  # 3/10
        # Total column: b union across target sets = {b_valid_0, b_valid_1, b_valid_2} = 3/20
        np.testing.assert_allclose(absc[1, 2], 3)
        np.testing.assert_allclose(pct[1, 2], 15.0)  # 3/20

    def test_cross_task_only(self, tmp_path, removal_module):
        """With cross_task_only=True, diagonal entries are zero."""
        datasets = ["a", "b"]
        fasta_dir, results_dir = _setup_files(
            tmp_path,
            datasets,
            "valid",
            "train",
            n_target_per_ds={"a": 100, "b": 100},
            target_hits={},
            n_query_per_ds={"a": 10, "b": 20},
            query_hits={
                ("a", "a"): {"a_valid_0", "a_valid_1"},  # within-task
                ("a", "b"): {"a_valid_0", "a_valid_2"},  # cross-task
                ("b", "a"): {"b_valid_0"},  # cross-task
                ("b", "b"): {"b_valid_0", "b_valid_1"},  # within-task
            },
        )

        pct, absc, _, _ = removal_module.compute_query_removal_matrix(
            datasets, "valid", "train", results_dir, fasta_dir, cross_task_only=True
        )

        assert pct.shape == (2, 3)

        # Diagonal should be 0
        np.testing.assert_allclose(pct[0, 0], 0.0)
        np.testing.assert_allclose(pct[1, 1], 0.0)

        # Off-diagonal should be computed
        np.testing.assert_allclose(pct[0, 1], 20.0)  # a query vs b target: 2/10
        np.testing.assert_allclose(pct[1, 0], 5.0)  # b query vs a target: 1/20

        # Total column should only include cross-task queries
        # Row a: only {a_valid_0, a_valid_2} from a->b cross-task = 2/10
        np.testing.assert_allclose(absc[0, 2], 2)
        np.testing.assert_allclose(pct[0, 2], 20.0)  # 2/10
        # Row b: only {b_valid_0} from b->a cross-task = 1/20
        np.testing.assert_allclose(absc[1, 2], 1)
        np.testing.assert_allclose(pct[1, 2], 5.0)  # 1/20

    def test_total_col_uses_union(self, tmp_path, removal_module):
        """Total column should be the union of query IDs across target datasets."""
        datasets = ["a", "b"]
        fasta_dir, results_dir = _setup_files(
            tmp_path,
            datasets,
            "valid",
            "train",
            n_target_per_ds={"a": 100, "b": 100},
            target_hits={},
            n_query_per_ds={"a": 10, "b": 10},
            query_hits={
                # a leaks into a target via valid_0,1 and into b target via valid_0,2
                ("a", "a"): {"a_valid_0", "a_valid_1"},
                ("a", "b"): {"a_valid_0", "a_valid_2"},
                # union across target sets for a = {a_valid_0, a_valid_1, a_valid_2} = 3
            },
        )

        pct, absc, _, _ = removal_module.compute_query_removal_matrix(
            datasets, "valid", "train", results_dir, fasta_dir
        )

        # Total column for row a: union = {a_valid_0, a_valid_1, a_valid_2} = 3/10
        np.testing.assert_allclose(absc[0, 2], 3)
        np.testing.assert_allclose(pct[0, 2], 30.0)  # 3/10

    def test_total_col_deduplicates_across_target_datasets(
        self, tmp_path, removal_module
    ):
        """A query that leaks into multiple target sets is counted once in Total col."""
        datasets = ["a", "b"]
        fasta_dir, results_dir = _setup_files(
            tmp_path,
            datasets,
            "valid",
            "train",
            n_target_per_ds={"a": 100, "b": 100},
            target_hits={},
            n_query_per_ds={"a": 10, "b": 10},
            query_hits={
                # a_valid_0 leaks into both a target and b target
                ("a", "a"): {"a_valid_0"},
                ("a", "b"): {"a_valid_0"},
            },
        )

        pct, absc, _, _ = removal_module.compute_query_removal_matrix(
            datasets, "valid", "train", results_dir, fasta_dir
        )

        # Per-cell: each shows 1
        np.testing.assert_allclose(absc[0, 0], 1)
        np.testing.assert_allclose(absc[0, 1], 1)
        # Total column for row a: union = {a_valid_0} = 1, NOT 2
        np.testing.assert_allclose(absc[0, 2], 1)
        np.testing.assert_allclose(pct[0, 2], 10.0)  # 1/10

    def test_no_removal(self, tmp_path, removal_module):
        """No query hits -> 0%."""
        datasets = ["a", "b"]
        fasta_dir, results_dir = _setup_files(
            tmp_path,
            datasets,
            "valid",
            "train",
            n_target_per_ds={"a": 10, "b": 10},
            target_hits={},
            n_query_per_ds={"a": 5, "b": 5},
        )

        pct, absc, _, _ = removal_module.compute_query_removal_matrix(
            datasets, "valid", "train", results_dir, fasta_dir
        )

        assert pct.shape == (2, 3)
        np.testing.assert_allclose(pct, 0.0)
        np.testing.assert_allclose(absc, 0.0)

    def test_three_datasets(self, tmp_path, removal_module):
        """Matrix shape is correct for 3 datasets (3 rows, 4 cols including Total)."""
        datasets = ["a", "b", "c"]
        fasta_dir, results_dir = _setup_files(
            tmp_path,
            datasets,
            "valid",
            "train",
            n_target_per_ds={"a": 100, "b": 100, "c": 100},
            target_hits={},
            n_query_per_ds={"a": 10, "b": 10, "c": 10},
            query_hits={
                ("a", "b"): {"a_valid_0"},
                ("c", "b"): {"c_valid_0", "c_valid_1"},
            },
        )

        pct, absc, row_labels, col_labels = removal_module.compute_query_removal_matrix(
            datasets, "valid", "train", results_dir, fasta_dir
        )

        assert pct.shape == (3, 4)
        assert row_labels == ["a", "b", "c"]
        assert col_labels == ["a", "b", "c", "Total"]

        # Total column for row a: {a_valid_0} = 1/10
        np.testing.assert_allclose(absc[0, 3], 1)
        np.testing.assert_allclose(pct[0, 3], 10.0)  # 1/10
        # Total column for row c: {c_valid_0, c_valid_1} = 2/10
        np.testing.assert_allclose(absc[2, 3], 2)
        np.testing.assert_allclose(pct[2, 3], 20.0)  # 2/10
        # Total column for row b: no hits = 0/10
        np.testing.assert_allclose(absc[1, 3], 0)
        np.testing.assert_allclose(pct[1, 3], 0.0)

    def test_test_vs_valid_query_removal(self, tmp_path, removal_module):
        """Query removal with test vs valid comparison."""
        datasets = ["a", "b"]
        fasta_dir, results_dir = _setup_files(
            tmp_path,
            datasets,
            "test",
            "valid",
            n_target_per_ds={"a": 100, "b": 100},
            target_hits={},
            n_query_per_ds={"a": 10, "b": 5},
            query_hits={
                ("a", "a"): {"a_test_0", "a_test_1"},  # 2/10 = 20%
                ("b", "b"): {"b_test_0"},  # 1/5 = 20%
            },
        )

        pct, absc, row_labels, col_labels = removal_module.compute_query_removal_matrix(
            datasets, "test", "valid", results_dir, fasta_dir
        )

        assert pct.shape == (2, 3)
        assert row_labels == ["a", "b"]
        assert col_labels == ["a", "b", "Total"]
        np.testing.assert_allclose(pct[0, 0], 20.0)
        np.testing.assert_allclose(pct[1, 1], 20.0)


# ---------------------------------------------------------------------------
# plot_removal_heatmap (smoke test)
# ---------------------------------------------------------------------------


class TestPlotRemovalHeatmap:
    def test_target_variant(self, tmp_path, removal_module):
        """Target removal: Total row, no Total column — (n+1, n) shape."""
        pct = np.array([[100.0, 50.0], [25.0, 80.0], [100.0, 80.0]])
        absc = np.array([[10, 5], [3, 8], [10, 8]])
        output_path = tmp_path / "target_removal.png"
        removal_module.plot_removal_heatmap(
            pct,
            absc,
            row_labels=["ds1", "ds2", "Total"],
            col_labels=["ds1", "ds2"],
            title="Target removal heatmap",
            output_path=output_path,
            has_total_row=True,
            has_total_col=False,
        )
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_query_variant(self, tmp_path, removal_module):
        """Query removal: no Total row, Total column — (n, n+1) shape."""
        pct = np.array([[20.0, 15.0, 30.0], [5.0, 10.0, 12.0]])
        absc = np.array([[2, 1, 3], [1, 2, 2]])
        output_path = tmp_path / "query_removal.png"
        removal_module.plot_removal_heatmap(
            pct,
            absc,
            row_labels=["ds1", "ds2"],
            col_labels=["ds1", "ds2", "Total"],
            title="Query removal (no Total row)",
            output_path=output_path,
            has_total_row=False,
            has_total_col=True,
        )
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_single_dataset_target(self, tmp_path, removal_module):
        """2x1 matrix (1 dataset + Total row) for target variant."""
        pct = np.array([[42.0], [42.0]])
        absc = np.array([[10], [10]])
        output_path = tmp_path / "single_target.png"
        removal_module.plot_removal_heatmap(
            pct,
            absc,
            row_labels=["only", "Total"],
            col_labels=["only"],
            title="Single dataset (target)",
            output_path=output_path,
            has_total_row=True,
            has_total_col=False,
        )
        assert output_path.exists()

    def test_four_datasets_target(self, tmp_path, removal_module):
        """5x4 matrix (4 datasets + Total row) matching target use case."""
        rng = np.random.default_rng(42)
        pct = rng.uniform(0, 100, size=(5, 4))
        absc = rng.integers(0, 1000, size=(5, 4)).astype(float)
        labels = ["secstr", "disorder", "meltome", "subloc"]
        output_path = tmp_path / "five_target.png"
        removal_module.plot_removal_heatmap(
            pct,
            absc,
            row_labels=labels + ["Total"],
            col_labels=labels,
            title="MMseqs2 target removal: valid vs train (min identity 20%)",
            output_path=output_path,
            has_total_row=True,
            has_total_col=False,
        )
        assert output_path.exists()

    def test_four_datasets_query(self, tmp_path, removal_module):
        """4x5 matrix (4 datasets, no Total row, Total col) matching query use case."""
        rng = np.random.default_rng(42)
        pct = rng.uniform(0, 100, size=(4, 5))
        absc = rng.integers(0, 1000, size=(4, 5)).astype(float)
        labels = ["secstr", "disorder", "meltome", "subloc"]
        output_path = tmp_path / "five_query.png"
        removal_module.plot_removal_heatmap(
            pct,
            absc,
            row_labels=labels,
            col_labels=labels + ["Total"],
            title="MMseqs2 query removal: test vs train (min identity 20%)",
            output_path=output_path,
            has_total_row=False,
            has_total_col=True,
        )
        assert output_path.exists()

    def test_custom_labels(self, tmp_path, removal_module):
        """Custom xlabel, ylabel, cbar_label are accepted."""
        pct = np.array([[50.0, 25.0, 37.5], [30.0, 60.0, 45.0]])
        absc = np.array([[5, 3, 8], [3, 6, 9]])
        output_path = tmp_path / "custom_labels.png"
        removal_module.plot_removal_heatmap(
            pct,
            absc,
            row_labels=["ds1", "ds2"],
            col_labels=["ds1", "ds2", "Total"],
            title="Query removal",
            output_path=output_path,
            xlabel="Train set",
            ylabel="Valid set (query)",
            cbar_label="% valid sequences to remove",
            has_total_row=False,
            has_total_col=True,
        )
        assert output_path.exists()


# ---------------------------------------------------------------------------
# plot_removal_barplot (smoke test)
# ---------------------------------------------------------------------------


class TestPlotRemovalBarplot:
    def test_creates_png(self, tmp_path, removal_module):
        pct = np.array([80.0, 50.0, 10.0])
        absc = np.array([800, 250, 50])
        output_path = tmp_path / "bar.png"
        removal_module.plot_removal_barplot(
            pct,
            absc,
            col_labels=["ds1", "ds2", "ds3"],
            title="Test removal bar",
            output_path=output_path,
        )
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_single_bar(self, tmp_path, removal_module):
        pct = np.array([42.0])
        absc = np.array([100])
        output_path = tmp_path / "single_bar.png"
        removal_module.plot_removal_barplot(
            pct,
            absc,
            col_labels=["only"],
            title="Single bar",
            output_path=output_path,
        )
        assert output_path.exists()

    def test_four_bars(self, tmp_path, removal_module):
        rng = np.random.default_rng(42)
        pct = rng.uniform(0, 100, size=4)
        absc = rng.integers(0, 1000, size=4).astype(float)
        labels = ["secstr", "disorder", "meltome", "subloc"]
        output_path = tmp_path / "five_bar.png"
        removal_module.plot_removal_barplot(
            pct,
            absc,
            col_labels=labels,
            title="Overall removal",
            output_path=output_path,
        )
        assert output_path.exists()

    def test_custom_ylabel(self, tmp_path, removal_module):
        """Custom ylabel is accepted."""
        pct = np.array([80.0, 50.0])
        absc = np.array([800, 250])
        output_path = tmp_path / "custom_bar.png"
        removal_module.plot_removal_barplot(
            pct,
            absc,
            col_labels=["ds1", "ds2"],
            title="Query removal bar",
            output_path=output_path,
            ylabel="% valid sequences to remove",
        )
        assert output_path.exists()
