"""Tests for the merge_clusters module.

Tests the union-of-edges cluster merging algorithm that combines
clustering tool results into a single merged_clusters.tsv
via connected components.  Supports one or more tools (mmseqs, foldseek, etc.).
"""

import importlib
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

# The script imports `from snakemake.script import snakemake` at module level,
# which fails outside Snakemake. We stub the snakemake module before importing.
_snakemake_stub = types.ModuleType("snakemake")
_snakemake_script_stub = types.ModuleType("snakemake.script")
_snakemake_script_stub.snakemake = MagicMock()
_snakemake_stub.script = _snakemake_script_stub
sys.modules.setdefault("snakemake", _snakemake_stub)
sys.modules.setdefault("snakemake.script", _snakemake_script_stub)

# Also stub wfutils.logging to avoid log file side effects
_wfutils_stub = types.ModuleType("wfutils")
_wfutils_stub.get_logger = MagicMock(return_value=MagicMock())
_wfutils_logging_stub = types.ModuleType("wfutils.logging")
_wfutils_logging_stub.log_snakemake_info = MagicMock()
sys.modules.setdefault("wfutils", _wfutils_stub)
sys.modules.setdefault("wfutils.logging", _wfutils_logging_stub)

SCRIPT_DIR = Path(__file__).resolve().parents[2] / "workflow" / "scripts" / "resplit"
spec = importlib.util.spec_from_file_location(
    "merge_clusters_mod",
    SCRIPT_DIR / "merge_clusters.py",
    submodule_search_locations=[],
)
merge_clusters_mod = importlib.util.module_from_spec(spec)

# Prevent the `main()` call at the bottom of the module from executing
_orig_exec = spec.loader.exec_module


def _patched_exec(module):
    """Load the module but suppress the top-level main() invocation."""
    import builtins

    original_source = (SCRIPT_DIR / "merge_clusters.py").read_text()
    # Replace the final main() call with pass
    patched_source = original_source.replace(
        "\nmain()", "\npass  # main() suppressed in test"
    )
    code = compile(patched_source, str(SCRIPT_DIR / "merge_clusters.py"), "exec")
    exec(code, module.__dict__)


_patched_exec(merge_clusters_mod)
merge_clusters_fn = merge_clusters_mod.merge_clusters


def _write_cluster_tsv(path: Path, rows: list[tuple[str, str]]):
    """Write a cluster TSV with 'representative' and 'member' columns."""
    df = pd.DataFrame(rows, columns=["representative", "member"])
    df.to_csv(path, sep="\t", index=False)


def _read_merged(path: Path) -> pd.DataFrame:
    """Read merged_clusters.tsv."""
    return pd.read_csv(path, sep="\t")


class TestMergeClusters:
    """Tests for the merge_clusters() function with two tools."""

    def test_disjoint_clusters(self, tmp_path):
        """Two tools with completely disjoint clusters produce separate components."""
        mmseqs_path = tmp_path / "mmseqs.tsv"
        foldseek_path = tmp_path / "foldseek.tsv"
        output_path = tmp_path / "merged.tsv"

        # MMseqs: cluster of {A, B}
        _write_cluster_tsv(mmseqs_path, [("A", "A"), ("A", "B")])
        # Foldseek: cluster of {C, D}
        _write_cluster_tsv(foldseek_path, [("C", "C"), ("C", "D")])

        merge_clusters_fn(
            {"mmseqs": mmseqs_path, "foldseek": foldseek_path}, output_path
        )

        df = _read_merged(output_path)
        assert set(df.columns) == {"cluster_id", "member"}
        assert set(df["member"]) == {"A", "B", "C", "D"}

        # Should have 2 clusters
        assert df["cluster_id"].nunique() == 2

        # A and B in same cluster
        cluster_ab = df[df["member"] == "A"]["cluster_id"].iloc[0]
        cluster_b = df[df["member"] == "B"]["cluster_id"].iloc[0]
        assert cluster_ab == cluster_b

        # C and D in same cluster
        cluster_cd = df[df["member"] == "C"]["cluster_id"].iloc[0]
        cluster_d = df[df["member"] == "D"]["cluster_id"].iloc[0]
        assert cluster_cd == cluster_d

        # AB and CD in different clusters
        assert cluster_ab != cluster_cd

    def test_overlapping_clusters_union(self, tmp_path):
        """Overlapping clusters across tools are merged via union."""
        mmseqs_path = tmp_path / "mmseqs.tsv"
        foldseek_path = tmp_path / "foldseek.tsv"
        output_path = tmp_path / "merged.tsv"

        # MMseqs: {A, B}
        _write_cluster_tsv(mmseqs_path, [("A", "A"), ("A", "B")])
        # Foldseek: {B, C}  — B bridges the two tools' clusters
        _write_cluster_tsv(foldseek_path, [("B", "B"), ("B", "C")])

        merge_clusters_fn(
            {"mmseqs": mmseqs_path, "foldseek": foldseek_path}, output_path
        )

        df = _read_merged(output_path)
        assert set(df["member"]) == {"A", "B", "C"}

        # All 3 should be in the same cluster (connected via B)
        assert df["cluster_id"].nunique() == 1

    def test_all_singletons(self, tmp_path):
        """When all sequences are singletons, each gets its own cluster."""
        mmseqs_path = tmp_path / "mmseqs.tsv"
        foldseek_path = tmp_path / "foldseek.tsv"
        output_path = tmp_path / "merged.tsv"

        _write_cluster_tsv(mmseqs_path, [("X", "X"), ("Y", "Y")])
        _write_cluster_tsv(foldseek_path, [("X", "X"), ("Y", "Y")])

        merge_clusters_fn(
            {"mmseqs": mmseqs_path, "foldseek": foldseek_path}, output_path
        )

        df = _read_merged(output_path)
        assert set(df["member"]) == {"X", "Y"}
        assert df["cluster_id"].nunique() == 2

    def test_transitive_merge(self, tmp_path):
        """Union merges chains: A-B from mmseqs, B-C from foldseek -> {A,B,C}."""
        mmseqs_path = tmp_path / "mmseqs.tsv"
        foldseek_path = tmp_path / "foldseek.tsv"
        output_path = tmp_path / "merged.tsv"

        # MMseqs: A-B, D-E
        _write_cluster_tsv(
            mmseqs_path,
            [
                ("A", "A"),
                ("A", "B"),
                ("D", "D"),
                ("D", "E"),
            ],
        )
        # Foldseek: B-C, E-F
        _write_cluster_tsv(
            foldseek_path,
            [
                ("B", "B"),
                ("B", "C"),
                ("E", "E"),
                ("E", "F"),
            ],
        )

        merge_clusters_fn(
            {"mmseqs": mmseqs_path, "foldseek": foldseek_path}, output_path
        )

        df = _read_merged(output_path)
        assert set(df["member"]) == {"A", "B", "C", "D", "E", "F"}

        # {A, B, C} in one cluster and {D, E, F} in another
        assert df["cluster_id"].nunique() == 2

        cid_a = df[df["member"] == "A"]["cluster_id"].iloc[0]
        cid_b = df[df["member"] == "B"]["cluster_id"].iloc[0]
        cid_c = df[df["member"] == "C"]["cluster_id"].iloc[0]
        assert cid_a == cid_b == cid_c

        cid_d = df[df["member"] == "D"]["cluster_id"].iloc[0]
        cid_e = df[df["member"] == "E"]["cluster_id"].iloc[0]
        cid_f = df[df["member"] == "F"]["cluster_id"].iloc[0]
        assert cid_d == cid_e == cid_f

        assert cid_a != cid_d

    def test_empty_foldseek(self, tmp_path):
        """When foldseek has only singletons (no non-self edges), mmseqs clusters are preserved."""
        mmseqs_path = tmp_path / "mmseqs.tsv"
        foldseek_path = tmp_path / "foldseek.tsv"
        output_path = tmp_path / "merged.tsv"

        _write_cluster_tsv(mmseqs_path, [("A", "A"), ("A", "B"), ("C", "C")])
        _write_cluster_tsv(foldseek_path, [("A", "A"), ("B", "B"), ("C", "C")])

        merge_clusters_fn(
            {"mmseqs": mmseqs_path, "foldseek": foldseek_path}, output_path
        )

        df = _read_merged(output_path)
        assert set(df["member"]) == {"A", "B", "C"}
        assert df["cluster_id"].nunique() == 2

        # A and B should be in the same cluster
        cid_a = df[df["member"] == "A"]["cluster_id"].iloc[0]
        cid_b = df[df["member"] == "B"]["cluster_id"].iloc[0]
        assert cid_a == cid_b

    def test_large_single_component(self, tmp_path):
        """A chain of pairwise edges across tools merges into one large component."""
        mmseqs_path = tmp_path / "mmseqs.tsv"
        foldseek_path = tmp_path / "foldseek.tsv"
        output_path = tmp_path / "merged.tsv"

        # MMseqs: 0-1, 2-3, 4-5
        _write_cluster_tsv(
            mmseqs_path,
            [
                ("0", "0"),
                ("0", "1"),
                ("2", "2"),
                ("2", "3"),
                ("4", "4"),
                ("4", "5"),
            ],
        )
        # Foldseek: 1-2, 3-4 — bridges everything into one component
        _write_cluster_tsv(
            foldseek_path,
            [
                ("0", "0"),
                ("1", "1"),
                ("1", "2"),
                ("3", "3"),
                ("3", "4"),
                ("5", "5"),
            ],
        )

        merge_clusters_fn(
            {"mmseqs": mmseqs_path, "foldseek": foldseek_path}, output_path
        )

        df = _read_merged(output_path)
        # Numeric IDs round-trip through CSV as integers; compare as strings
        assert set(str(m) for m in df["member"]) == {"0", "1", "2", "3", "4", "5"}
        assert df["cluster_id"].nunique() == 1

    def test_integer_ids(self, tmp_path):
        """Numeric IDs (as in actual pipeline) are handled correctly."""
        mmseqs_path = tmp_path / "mmseqs.tsv"
        foldseek_path = tmp_path / "foldseek.tsv"
        output_path = tmp_path / "merged.tsv"

        _write_cluster_tsv(mmseqs_path, [("0", "0"), ("0", "1"), ("2", "2")])
        _write_cluster_tsv(foldseek_path, [("0", "0"), ("1", "1"), ("2", "2")])

        merge_clusters_fn(
            {"mmseqs": mmseqs_path, "foldseek": foldseek_path}, output_path
        )

        df = _read_merged(output_path)
        # Numeric IDs round-trip through CSV; internally the function uses strings,
        # but the output may be read back as integers. Verify content via string cast.
        assert set(str(m) for m in df["member"]) == {"0", "1", "2"}
        assert df["cluster_id"].nunique() == 2

    def test_cluster_ids_deterministic(self, tmp_path):
        """Cluster IDs are assigned deterministically (sorted by min member)."""
        mmseqs_path = tmp_path / "mmseqs.tsv"
        foldseek_path = tmp_path / "foldseek.tsv"

        _write_cluster_tsv(
            mmseqs_path,
            [
                ("Z", "Z"),
                ("Z", "Y"),
                ("A", "A"),
                ("A", "B"),
            ],
        )
        _write_cluster_tsv(
            foldseek_path,
            [
                ("Z", "Z"),
                ("Y", "Y"),
                ("A", "A"),
                ("B", "B"),
            ],
        )

        # Run twice
        output1 = tmp_path / "merged1.tsv"
        output2 = tmp_path / "merged2.tsv"
        tool_paths = {"mmseqs": mmseqs_path, "foldseek": foldseek_path}
        merge_clusters_fn(tool_paths, output1)
        merge_clusters_fn(tool_paths, output2)

        df1 = _read_merged(output1)
        df2 = _read_merged(output2)

        # Should be identical
        pd.testing.assert_frame_equal(df1, df2)

        # Cluster with {A, B} should have lower cluster_id than {Y, Z}
        # because min("A", "B") = "A" < min("Y", "Z") = "Y"
        cid_a = df1[df1["member"] == "A"]["cluster_id"].iloc[0]
        cid_y = df1[df1["member"] == "Y"]["cluster_id"].iloc[0]
        assert cid_a < cid_y


class TestMergeClustersSingleTool:
    """Tests for merge_clusters() with only one tool enabled."""

    def test_mmseqs_only(self, tmp_path):
        """Single tool (mmseqs) produces correct clusters."""
        mmseqs_path = tmp_path / "mmseqs.tsv"
        output_path = tmp_path / "merged.tsv"

        _write_cluster_tsv(
            mmseqs_path,
            [("A", "A"), ("A", "B"), ("C", "C"), ("C", "D")],
        )

        merge_clusters_fn({"mmseqs": mmseqs_path}, output_path)

        df = _read_merged(output_path)
        assert set(df["member"]) == {"A", "B", "C", "D"}
        assert df["cluster_id"].nunique() == 2

        cid_a = df[df["member"] == "A"]["cluster_id"].iloc[0]
        cid_b = df[df["member"] == "B"]["cluster_id"].iloc[0]
        assert cid_a == cid_b

        cid_c = df[df["member"] == "C"]["cluster_id"].iloc[0]
        cid_d = df[df["member"] == "D"]["cluster_id"].iloc[0]
        assert cid_c == cid_d
        assert cid_a != cid_c

    def test_foldseek_only(self, tmp_path):
        """Single tool (foldseek) produces correct clusters."""
        foldseek_path = tmp_path / "foldseek.tsv"
        output_path = tmp_path / "merged.tsv"

        _write_cluster_tsv(
            foldseek_path,
            [("X", "X"), ("X", "Y"), ("X", "Z"), ("W", "W")],
        )

        merge_clusters_fn({"foldseek": foldseek_path}, output_path)

        df = _read_merged(output_path)
        assert set(df["member"]) == {"W", "X", "Y", "Z"}
        assert df["cluster_id"].nunique() == 2

        # X, Y, Z in same cluster
        cid_x = df[df["member"] == "X"]["cluster_id"].iloc[0]
        cid_y = df[df["member"] == "Y"]["cluster_id"].iloc[0]
        cid_z = df[df["member"] == "Z"]["cluster_id"].iloc[0]
        assert cid_x == cid_y == cid_z

        # W is a singleton
        cid_w = df[df["member"] == "W"]["cluster_id"].iloc[0]
        assert cid_w != cid_x

    def test_single_tool_singletons(self, tmp_path):
        """Single tool with all singletons produces one cluster per sequence."""
        mmseqs_path = tmp_path / "mmseqs.tsv"
        output_path = tmp_path / "merged.tsv"

        _write_cluster_tsv(
            mmseqs_path,
            [("A", "A"), ("B", "B"), ("C", "C")],
        )

        merge_clusters_fn({"mmseqs": mmseqs_path}, output_path)

        df = _read_merged(output_path)
        assert set(df["member"]) == {"A", "B", "C"}
        assert df["cluster_id"].nunique() == 3

    def test_single_tool_one_big_cluster(self, tmp_path):
        """Single tool where everything is in one cluster."""
        foldseek_path = tmp_path / "foldseek.tsv"
        output_path = tmp_path / "merged.tsv"

        _write_cluster_tsv(
            foldseek_path,
            [("A", "A"), ("A", "B"), ("A", "C"), ("A", "D")],
        )

        merge_clusters_fn({"foldseek": foldseek_path}, output_path)

        df = _read_merged(output_path)
        assert set(df["member"]) == {"A", "B", "C", "D"}
        assert df["cluster_id"].nunique() == 1
