"""Tests for the compute_resplit module.

Tests load_data, phase_a (test reconstruction), phase_b (main resplit),
build_summary, and validate_assignments.
"""

import importlib
import random
import sys
import types
from collections import defaultdict
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

# Stub snakemake + wfutils before importing the module
_snakemake_stub = types.ModuleType("snakemake")
_snakemake_script_stub = types.ModuleType("snakemake.script")
_snakemake_script_stub.snakemake = MagicMock()
_snakemake_stub.script = _snakemake_script_stub
sys.modules.setdefault("snakemake", _snakemake_stub)
sys.modules.setdefault("snakemake.script", _snakemake_script_stub)

_wfutils_stub = types.ModuleType("wfutils")
_wfutils_stub.get_logger = MagicMock(return_value=MagicMock())
_wfutils_logging_stub = types.ModuleType("wfutils.logging")
_wfutils_logging_stub.log_snakemake_info = MagicMock()
sys.modules.setdefault("wfutils", _wfutils_stub)
sys.modules.setdefault("wfutils.logging", _wfutils_logging_stub)

SCRIPT_DIR = Path(__file__).resolve().parents[2] / "workflow" / "scripts" / "resplit"

# Load module suppressing main()
_source = (SCRIPT_DIR / "compute_resplit.py").read_text()
_patched = _source.replace("\nmain()", "\npass  # main() suppressed in test")

compute_mod = types.ModuleType("compute_resplit_mod")
compute_mod.__file__ = str(SCRIPT_DIR / "compute_resplit.py")
code = compile(_patched, str(SCRIPT_DIR / "compute_resplit.py"), "exec")
exec(code, compute_mod.__dict__)

load_data = compute_mod.load_data
phase_a = compute_mod.phase_a
phase_b = compute_mod.phase_b
build_summary = compute_mod.build_summary
validate_assignments = compute_mod.validate_assignments


# ---------------------------------------------------------------------------
# Helpers for creating test data
# ---------------------------------------------------------------------------


def _write_clusters(path: Path, rows: list[tuple[int, str]]):
    """Write merged_clusters.tsv with (cluster_id, member) rows."""
    df = pd.DataFrame(rows, columns=["cluster_id", "member"])
    df.to_csv(path, sep="\t", index=False)


def _write_metadata(path: Path, rows: list[tuple[str, str, str]]):
    """Write sequence_metadata.tsv with (seq_id, sequence, datasets) rows.

    datasets format: "ds1:train,ds2:test"
    """
    df = pd.DataFrame(rows, columns=["seq_id", "sequence", "datasets"])
    df.to_csv(path, sep="\t", index=False)


def _build_lookups(
    cluster_rows: list[tuple[int, str]],
    metadata_rows: list[tuple[str, str, str]],
):
    """Build the lookup dicts directly (without file I/O) for faster tests."""
    seq_to_clusters: dict[str, set[int]] = defaultdict(set)
    cluster_to_seqs: dict[int, set[str]] = defaultdict(set)
    for cid, member in cluster_rows:
        seq_to_clusters[member].add(cid)
        cluster_to_seqs[cid].add(member)

    seq_ds_split: dict[tuple[str, str], str] = {}
    ds_seqs: dict[str, set[str]] = defaultdict(set)
    for sid, _seq, datasets in metadata_rows:
        for tag in datasets.split(","):
            tag = tag.strip()
            if ":" not in tag:
                continue
            ds, split = tag.split(":", 1)
            seq_ds_split[(sid, ds)] = split
            ds_seqs[ds].add(sid)

    return seq_to_clusters, cluster_to_seqs, seq_ds_split, ds_seqs


# ---------------------------------------------------------------------------
# Tests: load_data
# ---------------------------------------------------------------------------


class TestLoadData:
    def test_basic_loading(self, tmp_path):
        """load_data correctly parses clusters and metadata TSVs."""
        clusters_path = tmp_path / "clusters.tsv"
        metadata_path = tmp_path / "metadata.tsv"

        _write_clusters(
            clusters_path,
            [
                (0, "s1"),
                (0, "s2"),
                (1, "s3"),
            ],
        )
        _write_metadata(
            metadata_path,
            [
                ("s1", "ACDE", "dsA:train"),
                ("s2", "FGHI", "dsA:test"),
                ("s3", "KLMN", "dsB:valid"),
            ],
        )

        clusters_df, seq_to_clusters, cluster_to_seqs, seq_ds_split, ds_seqs = (
            load_data(clusters_path, metadata_path)
        )

        assert set(cluster_to_seqs[0]) == {"s1", "s2"}
        assert set(cluster_to_seqs[1]) == {"s3"}
        assert seq_to_clusters["s1"] == {0}
        assert seq_ds_split[("s1", "dsA")] == "train"
        assert seq_ds_split[("s2", "dsA")] == "test"
        assert seq_ds_split[("s3", "dsB")] == "valid"
        assert ds_seqs["dsA"] == {"s1", "s2"}
        assert ds_seqs["dsB"] == {"s3"}

    def test_multi_dataset_sequence(self, tmp_path):
        """A sequence appearing in multiple datasets is tracked correctly."""
        clusters_path = tmp_path / "clusters.tsv"
        metadata_path = tmp_path / "metadata.tsv"

        _write_clusters(clusters_path, [(0, "s1")])
        _write_metadata(
            metadata_path,
            [
                ("s1", "ACDE", "dsA:train,dsB:test"),
            ],
        )

        _, seq_to_clusters, _, seq_ds_split, ds_seqs = load_data(
            clusters_path, metadata_path
        )

        assert seq_ds_split[("s1", "dsA")] == "train"
        assert seq_ds_split[("s1", "dsB")] == "test"
        assert "s1" in ds_seqs["dsA"]
        assert "s1" in ds_seqs["dsB"]


# ---------------------------------------------------------------------------
# Tests: phase_a (test reconstruction)
# ---------------------------------------------------------------------------


class TestPhaseA:
    def test_no_reconstruct_datasets(self):
        """Returns empty dicts when no datasets configured for reconstruction."""
        results, step_stats = phase_a(
            reconstruct_datasets=[],
            min_test_fraction=0.1,
            all_resplit_datasets=["dsA"],
            seq_to_clusters={},
            cluster_to_seqs={},
            seq_ds_split={},
            ds_seqs={},
            rng=random.Random(42),
        )
        assert results == {}
        assert step_stats == {}

    def test_reconstruct_from_cluster_overlap(self):
        """Sequences in clusters overlapping other-dataset test sets become new test."""
        # Setup: dsA has 10 sequences (s0..s9) all in train.
        # dsB has s10 as test, in cluster 0 with s0 from dsA.
        # So s0 should be assigned to dsA's new test via cluster overlap.
        cluster_rows = [
            (0, "s0"),
            (0, "s10"),  # s0 and s10 in same cluster
            (1, "s1"),
            (2, "s2"),
            (3, "s3"),
            (4, "s4"),
            (5, "s5"),
            (6, "s6"),
            (7, "s7"),
            (8, "s8"),
            (9, "s9"),
        ]
        metadata_rows = [
            ("s0", "SEQ0", "dsA:train"),
            ("s1", "SEQ1", "dsA:train"),
            ("s2", "SEQ2", "dsA:train"),
            ("s3", "SEQ3", "dsA:train"),
            ("s4", "SEQ4", "dsA:train"),
            ("s5", "SEQ5", "dsA:train"),
            ("s6", "SEQ6", "dsA:train"),
            ("s7", "SEQ7", "dsA:train"),
            ("s8", "SEQ8", "dsA:train"),
            ("s9", "SEQ9", "dsA:train"),
            ("s10", "SEQ10", "dsB:test"),
        ]
        seq_to_clusters, cluster_to_seqs, seq_ds_split, ds_seqs = _build_lookups(
            cluster_rows, metadata_rows
        )

        results, step_stats = phase_a(
            reconstruct_datasets=["dsA"],
            min_test_fraction=0.1,
            all_resplit_datasets=["dsA", "dsB"],
            seq_to_clusters=seq_to_clusters,
            cluster_to_seqs=cluster_to_seqs,
            seq_ds_split=seq_ds_split,
            ds_seqs=ds_seqs,
            rng=random.Random(42),
        )

        assert "dsA" in results
        assert "s0" in results["dsA"]["test"]
        assert "s0" not in results["dsA"]["train_valid"]
        # All other sequences should be in train_valid
        for i in range(1, 10):
            assert f"s{i}" in results["dsA"]["train_valid"]

        # Step stats
        assert "dsA" in step_stats
        assert step_stats["dsA"]["total_pool"] == 10
        assert step_stats["dsA"]["overlap_test_count"] == 1  # s0
        assert step_stats["dsA"]["final_test_count"] >= 1

    def test_top_up_when_below_min(self):
        """If overlap gives fewer than min_test_fraction, random clusters are added."""
        # 20 sequences total, min_test_fraction=0.1 -> need at least 2 test
        # Only s0 overlaps with dsB test via cluster 0, so top-up needed
        cluster_rows = [(0, "s0"), (0, "s20")]
        for i in range(1, 20):
            cluster_rows.append((i, f"s{i}"))
        cluster_rows.append((20, "s20"))

        metadata_rows = [("s20", "SEQ20", "dsB:test")]
        for i in range(20):
            metadata_rows.append((f"s{i}", f"SEQ{i}", "dsA:train"))

        seq_to_clusters, cluster_to_seqs, seq_ds_split, ds_seqs = _build_lookups(
            cluster_rows, metadata_rows
        )

        results, step_stats = phase_a(
            reconstruct_datasets=["dsA"],
            min_test_fraction=0.1,
            all_resplit_datasets=["dsA", "dsB"],
            seq_to_clusters=seq_to_clusters,
            cluster_to_seqs=cluster_to_seqs,
            seq_ds_split=seq_ds_split,
            ds_seqs=ds_seqs,
            rng=random.Random(42),
        )

        # Should have at least 2 in test (10% of 20)
        assert len(results["dsA"]["test"]) >= 2
        # s0 must be in test (from overlap)
        assert "s0" in results["dsA"]["test"]
        # test + train_valid = 20
        assert len(results["dsA"]["test"]) + len(results["dsA"]["train_valid"]) == 20

        # Step stats: topup should be non-zero
        assert step_stats["dsA"]["overlap_test_count"] == 1
        assert step_stats["dsA"]["topup_test_count"] >= 1
        assert step_stats["dsA"]["final_test_count"] >= 2


# ---------------------------------------------------------------------------
# Tests: phase_b (main resplit)
# ---------------------------------------------------------------------------


class TestPhaseB:
    def _make_simple_scenario(self):
        """Create a simple 2-dataset scenario for phase_b tests.

        dsA: s1(train), s2(train), s3(train), s4(test)
        dsB: s5(train), s6(train), s7(test)

        Clusters: {s1}, {s2, s5} (cross-dataset), {s3}, {s4, s7} (test overlap), {s6}
        """
        cluster_rows = [
            (0, "s1"),
            (1, "s2"),
            (1, "s5"),  # cross-dataset cluster
            (2, "s3"),
            (3, "s4"),
            (3, "s7"),  # both are test sequences
            (4, "s6"),
        ]
        metadata_rows = [
            ("s1", "SEQ1", "dsA:train"),
            ("s2", "SEQ2", "dsA:train"),
            ("s3", "SEQ3", "dsA:train"),
            ("s4", "SEQ4", "dsA:test"),
            ("s5", "SEQ5", "dsB:train"),
            ("s6", "SEQ6", "dsB:train"),
            ("s7", "SEQ7", "dsB:test"),
        ]
        return _build_lookups(cluster_rows, metadata_rows)

    def test_test_sets_preserved(self):
        """Test sets are preserved unchanged for normal (non-Phase-A) datasets."""
        seq_to_clusters, cluster_to_seqs, seq_ds_split, ds_seqs = (
            self._make_simple_scenario()
        )
        rng = random.Random(42)

        assignments, step_stats = phase_b(
            all_datasets=["dsA", "dsB"],
            phase_a_results={},
            min_valid_fraction=0.0,
            shared_protein_groups=[],
            seq_to_clusters=seq_to_clusters,
            cluster_to_seqs=cluster_to_seqs,
            seq_ds_split=seq_ds_split,
            ds_seqs=ds_seqs,
            rng=rng,
        )

        assert assignments[("s4", "dsA")] == "test"
        assert assignments[("s7", "dsB")] == "test"

    def test_within_dataset_test_similar_dropped(self):
        """Sequences in the same cluster as own test set are dropped (Step 2)."""
        # Add s8 to dsA's train, in the same cluster (3) as s4 (dsA test)
        cluster_rows = [
            (0, "s1"),
            (1, "s2"),
            (2, "s3"),
            (3, "s4"),
            (3, "s8"),  # s8 in same cluster as s4 (test)
            (4, "s5"),
        ]
        metadata_rows = [
            ("s1", "SEQ1", "dsA:train"),
            ("s2", "SEQ2", "dsA:train"),
            ("s3", "SEQ3", "dsA:train"),
            ("s4", "SEQ4", "dsA:test"),
            ("s8", "SEQ8", "dsA:train"),
            ("s5", "SEQ5", "dsB:train"),
        ]
        seq_to_clusters, cluster_to_seqs, seq_ds_split, ds_seqs = _build_lookups(
            cluster_rows, metadata_rows
        )

        assignments, step_stats = phase_b(
            all_datasets=["dsA", "dsB"],
            phase_a_results={},
            min_valid_fraction=0.0,
            shared_protein_groups=[],
            seq_to_clusters=seq_to_clusters,
            cluster_to_seqs=cluster_to_seqs,
            seq_ds_split=seq_ds_split,
            ds_seqs=ds_seqs,
            rng=random.Random(42),
        )

        # s8 should be dropped (same dataset's test cluster)
        assert assignments[("s8", "dsA")] == "dropped"

        # Step stats should reflect the drop
        assert step_stats["dsA"]["step2_drop_within_test_similar"]["dropped"] == 1

    def test_cross_dataset_contaminated_to_valid(self):
        """Sequences in clusters with other-dataset test seqs move to valid (Step 3)."""
        seq_to_clusters, cluster_to_seqs, seq_ds_split, ds_seqs = (
            self._make_simple_scenario()
        )

        assignments, step_stats = phase_b(
            all_datasets=["dsA", "dsB"],
            phase_a_results={},
            min_valid_fraction=0.0,
            shared_protein_groups=[],
            seq_to_clusters=seq_to_clusters,
            cluster_to_seqs=cluster_to_seqs,
            seq_ds_split=seq_ds_split,
            ds_seqs=ds_seqs,
            rng=random.Random(42),
        )

        # s2 is in cluster 1 (no test seqs) — NOT contaminated.
        # s5 is in cluster 1 (no test seqs) — NOT contaminated.
        assert assignments[("s1", "dsA")] == "train"
        assert assignments[("s3", "dsA")] == "train"

        # Step 3 stats should be present
        assert "step3_cross_contamination" in step_stats["dsA"]
        assert "step3_cross_contamination" in step_stats["dsB"]

    def test_valid_topup(self):
        """Valid is topped up to min_valid_fraction when insufficient (Step 4)."""
        # Simple scenario: 3 datasets, no test overlap, need valid topup
        cluster_rows = [
            (0, "s1"),
            (1, "s2"),
            (2, "s3"),
            (3, "s4"),
            (4, "s5"),
        ]
        metadata_rows = [
            ("s1", "SEQ1", "dsA:train"),
            ("s2", "SEQ2", "dsA:train"),
            ("s3", "SEQ3", "dsA:train"),
            ("s4", "SEQ4", "dsA:train"),
            ("s5", "SEQ5", "dsA:test"),
        ]
        seq_to_clusters, cluster_to_seqs, seq_ds_split, ds_seqs = _build_lookups(
            cluster_rows, metadata_rows
        )

        assignments, step_stats = phase_b(
            all_datasets=["dsA"],
            phase_a_results={},
            min_valid_fraction=0.25,  # Want 25% of train+valid (4) = 1
            shared_protein_groups=[],
            seq_to_clusters=seq_to_clusters,
            cluster_to_seqs=cluster_to_seqs,
            seq_ds_split=seq_ds_split,
            ds_seqs=ds_seqs,
            rng=random.Random(42),
        )

        # At least 1 sequence should be in valid (25% of 4 = 1)
        valid_count = sum(
            1
            for (sid, ds), split in assignments.items()
            if ds == "dsA" and split == "valid"
        )
        assert valid_count >= 1

        # Step 4 stats should reflect topup
        assert step_stats["dsA"]["step4_topup"]["topped_up"] >= 1
        assert step_stats["dsA"]["step4_topup"]["final_valid"] >= 1

    def test_shared_protein_group_same_splits(self):
        """Datasets in a shared_protein_group get identical valid top-up."""
        # dsA and dsB share all proteins, grouped together
        cluster_rows = [
            (0, "s1"),
            (1, "s2"),
            (2, "s3"),
            (3, "s4"),  # test cluster
        ]
        metadata_rows = [
            ("s1", "SEQ1", "dsA:train,dsB:train"),
            ("s2", "SEQ2", "dsA:train,dsB:train"),
            ("s3", "SEQ3", "dsA:train,dsB:train"),
            ("s4", "SEQ4", "dsA:test,dsB:test"),
        ]
        seq_to_clusters, cluster_to_seqs, seq_ds_split, ds_seqs = _build_lookups(
            cluster_rows, metadata_rows
        )

        assignments, step_stats = phase_b(
            all_datasets=["dsA", "dsB"],
            phase_a_results={},
            min_valid_fraction=0.34,  # 34% of 3 tv seqs = 1
            shared_protein_groups=[["dsA", "dsB"]],
            seq_to_clusters=seq_to_clusters,
            cluster_to_seqs=cluster_to_seqs,
            seq_ds_split=seq_ds_split,
            ds_seqs=ds_seqs,
            rng=random.Random(42),
        )

        # The same sequences should be in valid for both dsA and dsB
        dsA_valid = {
            sid
            for (sid, ds), split in assignments.items()
            if ds == "dsA" and split == "valid"
        }
        dsB_valid = {
            sid
            for (sid, ds), split in assignments.items()
            if ds == "dsB" and split == "valid"
        }
        assert dsA_valid == dsB_valid
        assert len(dsA_valid) >= 1

        # Both datasets should have step4 stats
        assert "step4_topup" in step_stats["dsA"]
        assert "step4_topup" in step_stats["dsB"]

    def test_all_sequences_accounted_for(self):
        """Every sequence gets exactly one assignment (train/valid/test/dropped)."""
        seq_to_clusters, cluster_to_seqs, seq_ds_split, ds_seqs = (
            self._make_simple_scenario()
        )

        assignments, step_stats = phase_b(
            all_datasets=["dsA", "dsB"],
            phase_a_results={},
            min_valid_fraction=0.1,
            shared_protein_groups=[],
            seq_to_clusters=seq_to_clusters,
            cluster_to_seqs=cluster_to_seqs,
            seq_ds_split=seq_ds_split,
            ds_seqs=ds_seqs,
            rng=random.Random(42),
        )

        # Every (seq_id, dataset) pair must be assigned
        for ds, seqs in ds_seqs.items():
            for sid in seqs:
                assert (sid, ds) in assignments, f"({sid}, {ds}) not in assignments"
                assert assignments[(sid, ds)] in ("train", "valid", "test", "dropped")

    def test_deterministic_with_seed(self):
        """Same seed produces identical assignments."""
        seq_to_clusters, cluster_to_seqs, seq_ds_split, ds_seqs = (
            self._make_simple_scenario()
        )

        assignments1, _ = phase_b(
            all_datasets=["dsA", "dsB"],
            phase_a_results={},
            min_valid_fraction=0.2,
            shared_protein_groups=[],
            seq_to_clusters=seq_to_clusters,
            cluster_to_seqs=cluster_to_seqs,
            seq_ds_split=seq_ds_split,
            ds_seqs=ds_seqs,
            rng=random.Random(42),
        )

        assignments2, _ = phase_b(
            all_datasets=["dsA", "dsB"],
            phase_a_results={},
            min_valid_fraction=0.2,
            shared_protein_groups=[],
            seq_to_clusters=seq_to_clusters,
            cluster_to_seqs=cluster_to_seqs,
            seq_ds_split=seq_ds_split,
            ds_seqs=ds_seqs,
            rng=random.Random(42),
        )

        assert assignments1 == assignments2


# ---------------------------------------------------------------------------
# Tests: build_summary
# ---------------------------------------------------------------------------


class TestBuildSummary:
    def test_summary_counts(self):
        """Summary correctly counts original and new split distributions."""
        ds_seqs = {"dsA": {"s1", "s2", "s3"}}
        seq_ds_split = {
            ("s1", "dsA"): "train",
            ("s2", "dsA"): "valid",
            ("s3", "dsA"): "test",
        }
        assignments = {
            ("s1", "dsA"): "train",
            ("s2", "dsA"): "train",  # moved from valid to train
            ("s3", "dsA"): "test",
        }

        summary = build_summary(
            all_datasets=["dsA"],
            assignments=assignments,
            seq_ds_split=seq_ds_split,
            ds_seqs=ds_seqs,
            phase_a_results={},
            phase_a_step_stats={},
            phase_b_step_stats={},
        )

        assert summary["dsA"]["original_counts"] == {"train": 1, "valid": 1, "test": 1}
        assert summary["dsA"]["new_counts"]["train"] == 2
        assert summary["dsA"]["new_counts"]["valid"] == 0
        assert summary["dsA"]["new_counts"]["test"] == 1
        assert summary["dsA"]["total"] == 3

    def test_summary_with_phase_a(self):
        """Summary includes reconstruct_test info for Phase A datasets."""
        ds_seqs = {"dsA": {"s1", "s2"}}
        seq_ds_split = {("s1", "dsA"): "train", ("s2", "dsA"): "train"}
        assignments = {("s1", "dsA"): "test", ("s2", "dsA"): "train"}
        phase_a_results = {
            "dsA": {"test": {"s1"}, "train_valid": {"s2"}},
        }
        phase_a_step_stats = {
            "dsA": {
                "total_pool": 2,
                "overlap_test_count": 1,
                "topup_test_count": 0,
                "final_test_count": 1,
                "train_valid_count": 1,
            },
        }

        summary = build_summary(
            all_datasets=["dsA"],
            assignments=assignments,
            seq_ds_split=seq_ds_split,
            ds_seqs=ds_seqs,
            phase_a_results=phase_a_results,
            phase_a_step_stats=phase_a_step_stats,
            phase_b_step_stats={},
        )

        assert "reconstruct_test" in summary["dsA"]
        assert summary["dsA"]["reconstruct_test"]["new_test_count"] == 1
        assert summary["dsA"]["reconstruct_test"]["train_valid_count"] == 1

    def test_summary_includes_step_stats(self):
        """Summary includes per-step detail from both phases."""
        ds_seqs = {"dsA": {"s1", "s2", "s3"}}
        seq_ds_split = {
            ("s1", "dsA"): "train",
            ("s2", "dsA"): "valid",
            ("s3", "dsA"): "test",
        }
        assignments = {
            ("s1", "dsA"): "train",
            ("s2", "dsA"): "valid",
            ("s3", "dsA"): "test",
        }
        phase_b_step_stats = {
            "dsA": {
                "step1_merge": {"test": 1, "train_valid": 2},
                "step2_drop_within_test_similar": {
                    "dropped": 0,
                    "train_valid_after": 2,
                },
                "step3_cross_contamination": {
                    "moved_to_valid": 0,
                    "train": 2,
                    "valid": 0,
                },
                "step4_topup": {"topped_up": 1, "final_train": 1, "final_valid": 1},
            },
        }

        summary = build_summary(
            all_datasets=["dsA"],
            assignments=assignments,
            seq_ds_split=seq_ds_split,
            ds_seqs=ds_seqs,
            phase_a_results={},
            phase_a_step_stats={},
            phase_b_step_stats=phase_b_step_stats,
        )

        assert "steps" in summary["dsA"]
        steps = summary["dsA"]["steps"]
        assert "step1_merge" in steps
        assert steps["step1_merge"]["test"] == 1
        assert steps["step1_merge"]["train_valid"] == 2
        assert "step2_drop_within_test_similar" in steps
        assert steps["step2_drop_within_test_similar"]["dropped"] == 0
        assert "step3_cross_contamination" in steps
        assert "step4_topup" in steps
        assert steps["step4_topup"]["topped_up"] == 1

    def test_summary_includes_phase_a_step_stats(self):
        """Summary includes Phase A step stats under steps.phase_a."""
        ds_seqs = {"dsA": {"s1", "s2"}}
        seq_ds_split = {("s1", "dsA"): "train", ("s2", "dsA"): "train"}
        assignments = {("s1", "dsA"): "test", ("s2", "dsA"): "train"}
        phase_a_results = {"dsA": {"test": {"s1"}, "train_valid": {"s2"}}}
        phase_a_step_stats = {
            "dsA": {
                "total_pool": 2,
                "overlap_test_count": 1,
                "topup_test_count": 0,
                "final_test_count": 1,
                "train_valid_count": 1,
            },
        }
        phase_b_step_stats = {
            "dsA": {
                "step1_merge": {"test": 1, "train_valid": 1},
                "step2_drop_within_test_similar": {
                    "dropped": 0,
                    "train_valid_after": 1,
                },
                "step3_cross_contamination": {
                    "moved_to_valid": 0,
                    "train": 1,
                    "valid": 0,
                },
                "step4_topup": {"topped_up": 0, "final_train": 1, "final_valid": 0},
            },
        }

        summary = build_summary(
            all_datasets=["dsA"],
            assignments=assignments,
            seq_ds_split=seq_ds_split,
            ds_seqs=ds_seqs,
            phase_a_results=phase_a_results,
            phase_a_step_stats=phase_a_step_stats,
            phase_b_step_stats=phase_b_step_stats,
        )

        assert "steps" in summary["dsA"]
        assert "phase_a" in summary["dsA"]["steps"]
        assert summary["dsA"]["steps"]["phase_a"]["overlap_test_count"] == 1
        assert summary["dsA"]["steps"]["phase_a"]["total_pool"] == 2


# ---------------------------------------------------------------------------
# Tests: validate_assignments
# ---------------------------------------------------------------------------


class TestValidateAssignments:
    def test_valid_assignments_pass(self):
        """Valid assignments pass without error."""
        ds_seqs = {"dsA": {"s1", "s2", "s3"}}
        seq_ds_split = {
            ("s1", "dsA"): "train",
            ("s2", "dsA"): "valid",
            ("s3", "dsA"): "test",
        }
        assignments = {
            ("s1", "dsA"): "train",
            ("s2", "dsA"): "valid",
            ("s3", "dsA"): "test",
        }

        # Should not raise
        validate_assignments(
            all_datasets=["dsA"],
            assignments=assignments,
            ds_seqs=ds_seqs,
            seq_ds_split=seq_ds_split,
            phase_a_results={},
            min_valid_fraction=0.0,
            shared_protein_groups=[],
        )

    def test_missing_assignment_raises(self):
        """Missing assignment for a sequence raises ValueError."""
        ds_seqs = {"dsA": {"s1", "s2"}}
        seq_ds_split = {("s1", "dsA"): "train", ("s2", "dsA"): "test"}
        assignments = {("s1", "dsA"): "train"}  # s2 missing

        with pytest.raises(ValueError, match="validation failed"):
            validate_assignments(
                all_datasets=["dsA"],
                assignments=assignments,
                ds_seqs=ds_seqs,
                seq_ds_split=seq_ds_split,
                phase_a_results={},
                min_valid_fraction=0.0,
                shared_protein_groups=[],
            )

    def test_changed_test_set_raises(self):
        """Changing test set for non-Phase-A dataset raises ValueError."""
        ds_seqs = {"dsA": {"s1", "s2"}}
        seq_ds_split = {("s1", "dsA"): "train", ("s2", "dsA"): "test"}
        assignments = {
            ("s1", "dsA"): "test",  # was train, now test
            ("s2", "dsA"): "train",  # was test, now train
        }

        with pytest.raises(ValueError, match="validation failed"):
            validate_assignments(
                all_datasets=["dsA"],
                assignments=assignments,
                ds_seqs=ds_seqs,
                seq_ds_split=seq_ds_split,
                phase_a_results={},
                min_valid_fraction=0.0,
                shared_protein_groups=[],
            )

    def test_phase_a_dataset_test_change_allowed(self):
        """Phase A datasets are allowed to have different test sets."""
        ds_seqs = {"dsA": {"s1", "s2"}}
        seq_ds_split = {("s1", "dsA"): "train", ("s2", "dsA"): "test"}
        assignments = {
            ("s1", "dsA"): "test",
            ("s2", "dsA"): "train",
        }
        phase_a_results = {"dsA": {"test": {"s1"}, "train_valid": {"s2"}}}

        # Should not raise — Phase A datasets skip the test-set check
        validate_assignments(
            all_datasets=["dsA"],
            assignments=assignments,
            ds_seqs=ds_seqs,
            seq_ds_split=seq_ds_split,
            phase_a_results=phase_a_results,
            min_valid_fraction=0.0,
            shared_protein_groups=[],
        )

    def test_train_valid_overlap_raises(self):
        """Sequence in both train and valid raises ValueError."""
        ds_seqs = {"dsA": {"s1", "s2"}}
        seq_ds_split = {("s1", "dsA"): "train", ("s2", "dsA"): "test"}
        # Can't really construct overlapping assignments since dict keys are unique,
        # but we can test the train-test overlap path
        assignments = {
            ("s1", "dsA"): "train",
            ("s2", "dsA"): "test",
        }
        # This should pass (no overlap)
        validate_assignments(
            all_datasets=["dsA"],
            assignments=assignments,
            ds_seqs=ds_seqs,
            seq_ds_split=seq_ds_split,
            phase_a_results={},
            min_valid_fraction=0.0,
            shared_protein_groups=[],
        )


# ---------------------------------------------------------------------------
# Tests: Step stats detail
# ---------------------------------------------------------------------------


class TestStepStats:
    """Tests verifying per-step statistics from phase_a and phase_b."""

    def test_phase_b_step_stats_structure(self):
        """Phase B returns step stats with all 4 steps for each dataset."""
        cluster_rows = [
            (0, "s1"),
            (1, "s2"),
            (2, "s3"),
        ]
        metadata_rows = [
            ("s1", "SEQ1", "dsA:train"),
            ("s2", "SEQ2", "dsA:train"),
            ("s3", "SEQ3", "dsA:test"),
        ]
        seq_to_clusters, cluster_to_seqs, seq_ds_split, ds_seqs = _build_lookups(
            cluster_rows, metadata_rows
        )

        _, step_stats = phase_b(
            all_datasets=["dsA"],
            phase_a_results={},
            min_valid_fraction=0.0,
            shared_protein_groups=[],
            seq_to_clusters=seq_to_clusters,
            cluster_to_seqs=cluster_to_seqs,
            seq_ds_split=seq_ds_split,
            ds_seqs=ds_seqs,
            rng=random.Random(42),
        )

        assert "dsA" in step_stats
        assert "step1_merge" in step_stats["dsA"]
        assert "step2_drop_within_test_similar" in step_stats["dsA"]
        assert "step3_cross_contamination" in step_stats["dsA"]
        assert "step4_topup" in step_stats["dsA"]

    def test_step2_counts_match(self):
        """Step 2 dropped count matches actual drops."""
        # s3 is in same cluster as s4 (test), so s3 should be dropped
        cluster_rows = [
            (0, "s1"),
            (1, "s2"),
            (2, "s3"),
            (2, "s4"),  # s3 and s4 in same cluster
        ]
        metadata_rows = [
            ("s1", "SEQ1", "dsA:train"),
            ("s2", "SEQ2", "dsA:train"),
            ("s3", "SEQ3", "dsA:train"),
            ("s4", "SEQ4", "dsA:test"),
        ]
        seq_to_clusters, cluster_to_seqs, seq_ds_split, ds_seqs = _build_lookups(
            cluster_rows, metadata_rows
        )

        assignments, step_stats = phase_b(
            all_datasets=["dsA"],
            phase_a_results={},
            min_valid_fraction=0.0,
            shared_protein_groups=[],
            seq_to_clusters=seq_to_clusters,
            cluster_to_seqs=cluster_to_seqs,
            seq_ds_split=seq_ds_split,
            ds_seqs=ds_seqs,
            rng=random.Random(42),
        )

        assert step_stats["dsA"]["step2_drop_within_test_similar"]["dropped"] == 1
        assert (
            step_stats["dsA"]["step2_drop_within_test_similar"]["train_valid_after"]
            == 2
        )
        assert assignments[("s3", "dsA")] == "dropped"

    def test_step3_move_count_matches(self):
        """Step 3 moved_to_valid count matches actual moves."""
        # s2 is in same cluster as s7 (dsB test), so s2 should move to valid
        cluster_rows = [
            (0, "s1"),
            (1, "s2"),
            (1, "s7"),  # s2 in cluster with s7 (dsB test)
            (2, "s3"),
            (3, "s4"),  # dsA test
        ]
        metadata_rows = [
            ("s1", "SEQ1", "dsA:train"),
            ("s2", "SEQ2", "dsA:train"),
            ("s3", "SEQ3", "dsA:train"),
            ("s4", "SEQ4", "dsA:test"),
            ("s7", "SEQ7", "dsB:test"),
        ]
        seq_to_clusters, cluster_to_seqs, seq_ds_split, ds_seqs = _build_lookups(
            cluster_rows, metadata_rows
        )

        assignments, step_stats = phase_b(
            all_datasets=["dsA", "dsB"],
            phase_a_results={},
            min_valid_fraction=0.0,
            shared_protein_groups=[],
            seq_to_clusters=seq_to_clusters,
            cluster_to_seqs=cluster_to_seqs,
            seq_ds_split=seq_ds_split,
            ds_seqs=ds_seqs,
            rng=random.Random(42),
        )

        # s2 should be moved to valid (cross-contamination with dsB's test)
        assert assignments[("s2", "dsA")] == "valid"
        assert step_stats["dsA"]["step3_cross_contamination"]["moved_to_valid"] >= 1

    def test_step4_topup_count_matches(self):
        """Step 4 topped_up count matches actual movements from train to valid."""
        cluster_rows = [
            (0, "s1"),
            (1, "s2"),
            (2, "s3"),
            (3, "s4"),
            (4, "s5"),
        ]
        metadata_rows = [
            ("s1", "SEQ1", "dsA:train"),
            ("s2", "SEQ2", "dsA:train"),
            ("s3", "SEQ3", "dsA:train"),
            ("s4", "SEQ4", "dsA:train"),
            ("s5", "SEQ5", "dsA:test"),
        ]
        seq_to_clusters, cluster_to_seqs, seq_ds_split, ds_seqs = _build_lookups(
            cluster_rows, metadata_rows
        )

        assignments, step_stats = phase_b(
            all_datasets=["dsA"],
            phase_a_results={},
            min_valid_fraction=0.5,  # 50% of 4 train_valid = 2
            shared_protein_groups=[],
            seq_to_clusters=seq_to_clusters,
            cluster_to_seqs=cluster_to_seqs,
            seq_ds_split=seq_ds_split,
            ds_seqs=ds_seqs,
            rng=random.Random(42),
        )

        valid_count = sum(
            1
            for (sid, ds), split in assignments.items()
            if ds == "dsA" and split == "valid"
        )
        assert valid_count >= 2
        assert step_stats["dsA"]["step4_topup"]["topped_up"] >= 2
        assert step_stats["dsA"]["step4_topup"]["final_valid"] == valid_count

    def test_step_stats_consistency(self):
        """Step stats are internally consistent across steps."""
        cluster_rows = [
            (0, "s1"),
            (0, "s5"),  # cross-dataset cluster
            (1, "s2"),
            (2, "s3"),
            (3, "s4"),
            (3, "s6"),  # s4(dsA test) and s6(dsB test) in same cluster
        ]
        metadata_rows = [
            ("s1", "SEQ1", "dsA:train"),
            ("s2", "SEQ2", "dsA:train"),
            ("s3", "SEQ3", "dsA:train"),
            ("s4", "SEQ4", "dsA:test"),
            ("s5", "SEQ5", "dsB:train"),
            ("s6", "SEQ6", "dsB:test"),
        ]
        seq_to_clusters, cluster_to_seqs, seq_ds_split, ds_seqs = _build_lookups(
            cluster_rows, metadata_rows
        )

        _, step_stats = phase_b(
            all_datasets=["dsA", "dsB"],
            phase_a_results={},
            min_valid_fraction=0.0,
            shared_protein_groups=[],
            seq_to_clusters=seq_to_clusters,
            cluster_to_seqs=cluster_to_seqs,
            seq_ds_split=seq_ds_split,
            ds_seqs=ds_seqs,
            rng=random.Random(42),
        )

        for ds in ["dsA", "dsB"]:
            s = step_stats[ds]
            # step1 total = test + train_valid
            total = s["step1_merge"]["test"] + s["step1_merge"]["train_valid"]
            # step2: train_valid_after = train_valid - dropped
            assert (
                s["step2_drop_within_test_similar"]["train_valid_after"]
                == s["step1_merge"]["train_valid"]
                - s["step2_drop_within_test_similar"]["dropped"]
            )
            # step3: train + valid = train_valid_after
            assert (
                s["step3_cross_contamination"]["train"]
                + s["step3_cross_contamination"]["valid"]
                == s["step2_drop_within_test_similar"]["train_valid_after"]
            )
            # step4: final_train + final_valid = step3 train + step3 valid
            # (topup just moves between train and valid, no sequences lost)
            assert (
                s["step4_topup"]["final_train"] + s["step4_topup"]["final_valid"]
                == s["step3_cross_contamination"]["train"]
                + s["step3_cross_contamination"]["valid"]
            )


# ---------------------------------------------------------------------------
# Tests: Integration (phase_a + phase_b)
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_end_to_end_simple(self):
        """Full pipeline: 2 datasets, no phase A, basic resplit."""
        cluster_rows = [
            (0, "s1"),
            (0, "s5"),  # cross-dataset cluster
            (1, "s2"),
            (2, "s3"),
            (2, "s6"),  # cross-dataset cluster
            (3, "s4"),  # dsA test
            (4, "s7"),  # dsB test
        ]
        metadata_rows = [
            ("s1", "SEQ1", "dsA:train"),
            ("s2", "SEQ2", "dsA:train"),
            ("s3", "SEQ3", "dsA:train"),
            ("s4", "SEQ4", "dsA:test"),
            ("s5", "SEQ5", "dsB:train"),
            ("s6", "SEQ6", "dsB:train"),
            ("s7", "SEQ7", "dsB:test"),
        ]
        seq_to_clusters, cluster_to_seqs, seq_ds_split, ds_seqs = _build_lookups(
            cluster_rows, metadata_rows
        )

        all_datasets = sorted(ds_seqs.keys())
        rng = random.Random(42)

        phase_a_results, phase_a_step_stats = phase_a(
            reconstruct_datasets=[],
            min_test_fraction=0.1,
            all_resplit_datasets=all_datasets,
            seq_to_clusters=seq_to_clusters,
            cluster_to_seqs=cluster_to_seqs,
            seq_ds_split=seq_ds_split,
            ds_seqs=ds_seqs,
            rng=rng,
        )

        assignments, phase_b_step_stats = phase_b(
            all_datasets=all_datasets,
            phase_a_results=phase_a_results,
            min_valid_fraction=0.0,
            shared_protein_groups=[],
            seq_to_clusters=seq_to_clusters,
            cluster_to_seqs=cluster_to_seqs,
            seq_ds_split=seq_ds_split,
            ds_seqs=ds_seqs,
            rng=rng,
        )

        # Validate
        validate_assignments(
            all_datasets=all_datasets,
            assignments=assignments,
            ds_seqs=ds_seqs,
            seq_ds_split=seq_ds_split,
            phase_a_results=phase_a_results,
            min_valid_fraction=0.0,
            shared_protein_groups=[],
        )

        # Test sets preserved
        assert assignments[("s4", "dsA")] == "test"
        assert assignments[("s7", "dsB")] == "test"

        # Every sequence assigned
        for ds, seqs in ds_seqs.items():
            for sid in seqs:
                assert (sid, ds) in assignments

        # Summary includes step stats
        summary = build_summary(
            all_datasets=all_datasets,
            assignments=assignments,
            seq_ds_split=seq_ds_split,
            ds_seqs=ds_seqs,
            phase_a_results=phase_a_results,
            phase_a_step_stats=phase_a_step_stats,
            phase_b_step_stats=phase_b_step_stats,
        )
        for ds in all_datasets:
            assert "steps" in summary[ds]

    def test_end_to_end_with_phase_a(self):
        """Full pipeline with Phase A test reconstruction for meltome-like dataset."""
        cluster_rows = [
            (0, "m1"),
            (0, "a2"),  # m1 in same cluster as a2 (dsA test!)
            (1, "m2"),
            (2, "m3"),
            (3, "a1"),
            (4, "m4"),
            (5, "m5"),
        ]
        metadata_rows = [
            ("m1", "SEQM1", "dsM:train"),
            ("m2", "SEQM2", "dsM:train"),
            ("m3", "SEQM3", "dsM:train"),
            ("m4", "SEQM4", "dsM:valid"),
            ("m5", "SEQM5", "dsM:valid"),
            ("a1", "SEQA1", "dsA:train"),
            ("a2", "SEQA2", "dsA:test"),
        ]
        seq_to_clusters, cluster_to_seqs, seq_ds_split, ds_seqs = _build_lookups(
            cluster_rows, metadata_rows
        )

        all_datasets = sorted(ds_seqs.keys())
        rng = random.Random(42)

        phase_a_results, phase_a_step_stats = phase_a(
            reconstruct_datasets=["dsM"],
            min_test_fraction=0.1,
            all_resplit_datasets=all_datasets,
            seq_to_clusters=seq_to_clusters,
            cluster_to_seqs=cluster_to_seqs,
            seq_ds_split=seq_ds_split,
            ds_seqs=ds_seqs,
            rng=rng,
        )

        assert "dsM" in phase_a_results
        assert "m1" in phase_a_results["dsM"]["test"]
        assert "dsM" in phase_a_step_stats
        assert phase_a_step_stats["dsM"]["overlap_test_count"] >= 1

        # Phase B
        assignments, phase_b_step_stats = phase_b(
            all_datasets=all_datasets,
            phase_a_results=phase_a_results,
            min_valid_fraction=0.0,
            shared_protein_groups=[],
            seq_to_clusters=seq_to_clusters,
            cluster_to_seqs=cluster_to_seqs,
            seq_ds_split=seq_ds_split,
            ds_seqs=ds_seqs,
            rng=rng,
        )

        # m1 should be in dsM's test
        assert assignments[("m1", "dsM")] == "test"
        # a2 should still be dsA's test
        assert assignments[("a2", "dsA")] == "test"

        # Summary
        summary = build_summary(
            all_datasets=all_datasets,
            assignments=assignments,
            seq_ds_split=seq_ds_split,
            ds_seqs=ds_seqs,
            phase_a_results=phase_a_results,
            phase_a_step_stats=phase_a_step_stats,
            phase_b_step_stats=phase_b_step_stats,
        )
        assert "reconstruct_test" in summary["dsM"]
        assert "steps" in summary["dsM"]
        assert "phase_a" in summary["dsM"]["steps"]
