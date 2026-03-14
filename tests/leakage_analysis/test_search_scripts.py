"""Tests for mmseqs_search and foldseek_search script functions.

Since we cannot assume mmseqs2 or foldseek are installed in the test
environment, we mock the subprocess.run calls and verify the correct
command lines are constructed.
"""

import importlib
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "workflow" / "scripts"


def _import_script(name: str):
    """Import a leakage script module with stubbed snakemake.

    Ensures the real ``wfutils`` package is in ``sys.modules`` before
    loading the script, because pytest's collection of ``tests/wfutils/``
    may shadow the installed package with a stale namespace entry.
    """
    script_path = SCRIPTS_DIR / "data_stats" / "leakage" / f"{name}.py"
    assert script_path.exists(), f"Script not found: {script_path}"

    fake_snakemake_mod = types.ModuleType("snakemake")
    fake_script_mod = types.ModuleType("snakemake.script")
    fake_script_mod.snakemake = None
    fake_snakemake_mod.script = fake_script_mod
    sys.modules.setdefault("snakemake", fake_snakemake_mod)
    sys.modules.setdefault("snakemake.script", fake_script_mod)

    # Guard against pytest's collection of tests/wfutils/ shadowing
    # the installed wfutils package with a stale namespace entry.
    # Purge any broken wfutils entries so the real package is found.
    _wfutils_mod = sys.modules.get("wfutils")
    if _wfutils_mod is not None and getattr(_wfutils_mod, "__file__", None) is None:
        for key in [
            k for k in sys.modules if k == "wfutils" or k.startswith("wfutils.")
        ]:
            sys.modules.pop(key)

    spec = importlib.util.spec_from_file_location(name, script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def mmseqs_mod():
    return _import_script("mmseqs_search")


@pytest.fixture()
def foldseek_mod():
    return _import_script("foldseek_search")


class TestMMseqsSearch:
    def test_command_construction(self, tmp_path, mmseqs_mod):
        """Verify the correct mmseqs command is assembled."""
        query = tmp_path / "query.fasta"
        target = tmp_path / "target.fasta"
        result = tmp_path / "result.tsv"
        query.write_text(">q1\nACDE\n")
        target.write_text(">t1\nFGHI\n")

        with patch.object(mmseqs_mod, "run") as mock_run:
            mock_run.return_value = None
            mmseqs_mod.run_mmseqs_search(query, target, result, min_seq_id=0.3)

            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]
            assert cmd[0] == "mmseqs"
            assert cmd[1] == "easy-search"
            assert str(query) in cmd
            assert str(target) in cmd
            assert str(result) in cmd
            assert "--min-seq-id" in cmd
            seq_id_idx = cmd.index("--min-seq-id")
            assert cmd[seq_id_idx + 1] == "0.3"
            assert "--format-output" in cmd
            fmt_idx = cmd.index("--format-output")
            assert cmd[fmt_idx + 1] == "query,target,fident"

    def test_different_threshold(self, tmp_path, mmseqs_mod):
        """Threshold value is passed through correctly."""
        query = tmp_path / "q.fasta"
        target = tmp_path / "t.fasta"
        result = tmp_path / "r.tsv"
        query.write_text(">q\nA\n")
        target.write_text(">t\nA\n")

        with patch.object(mmseqs_mod, "run") as mock_run:
            mock_run.return_value = None
            mmseqs_mod.run_mmseqs_search(query, target, result, min_seq_id=0.5)

            cmd = mock_run.call_args[0][0]
            seq_id_idx = cmd.index("--min-seq-id")
            assert cmd[seq_id_idx + 1] == "0.5"

    def test_subprocess_failure_raises(self, tmp_path, mmseqs_mod):
        """CalledProcessError is wrapped in RuntimeError."""
        from subprocess import CalledProcessError

        query = tmp_path / "q.fasta"
        target = tmp_path / "t.fasta"
        result = tmp_path / "r.tsv"
        query.write_text(">q\nA\n")
        target.write_text(">t\nA\n")

        with patch.object(
            mmseqs_mod, "run", side_effect=CalledProcessError(1, "mmseqs")
        ):
            with pytest.raises(RuntimeError, match="MMseqs2"):
                mmseqs_mod.run_mmseqs_search(query, target, result, min_seq_id=0.3)

    def test_gpu_default_off(self, tmp_path, mmseqs_mod):
        """GPU defaults to 0 when not specified."""
        query = tmp_path / "q.fasta"
        target = tmp_path / "t.fasta"
        result = tmp_path / "r.tsv"
        query.write_text(">q\nA\n")
        target.write_text(">t\nA\n")

        with patch.object(mmseqs_mod, "run") as mock_run:
            mock_run.return_value = None
            mmseqs_mod.run_mmseqs_search(query, target, result, min_seq_id=0.3)

            cmd = mock_run.call_args[0][0]
            gpu_idx = cmd.index("--gpu")
            assert cmd[gpu_idx + 1] == "0"

    def test_gpu_enabled(self, tmp_path, mmseqs_mod):
        """GPU flag is passed through when gpu=1."""
        query = tmp_path / "q.fasta"
        target = tmp_path / "t.fasta"
        result = tmp_path / "r.tsv"
        query.write_text(">q\nA\n")
        target.write_text(">t\nA\n")

        with patch.object(mmseqs_mod, "run") as mock_run:
            mock_run.return_value = None
            mmseqs_mod.run_mmseqs_search(query, target, result, min_seq_id=0.3, gpu=1)

            cmd = mock_run.call_args[0][0]
            gpu_idx = cmd.index("--gpu")
            assert cmd[gpu_idx + 1] == "1"


class TestFoldseekSearch:
    """Tests for the multi-step foldseek approach (createdb + search + convertalis)."""

    def test_four_step_command_sequence(self, tmp_path, foldseek_mod):
        """Verify that four foldseek commands are called in order:
        createdb (query), createdb (target), search, convertalis."""
        query = tmp_path / "query.fasta"
        target = tmp_path / "target.fasta"
        result = tmp_path / "result.tsv"
        prostt5 = tmp_path / "prostt5" / "prostt5-f16.gguf"
        query.write_text(">q1\nACDE\n")
        target.write_text(">t1\nFGHI\n")
        prostt5.parent.mkdir(parents=True)
        prostt5.write_text("fake model")

        with patch.object(foldseek_mod, "run") as mock_run:
            mock_run.return_value = None
            foldseek_mod.run_foldseek_search(
                query, target, result, min_seq_id=0.3, prostt5_model=prostt5
            )

            assert mock_run.call_count == 4
            calls = [c[0][0] for c in mock_run.call_args_list]

            # Step 1: createdb query
            assert calls[0][0] == "foldseek"
            assert calls[0][1] == "createdb"
            assert str(query) in calls[0]
            assert "--prostt5-model" in calls[0]

            # Step 2: createdb target
            assert calls[1][0] == "foldseek"
            assert calls[1][1] == "createdb"
            assert str(target) in calls[1]
            assert "--prostt5-model" in calls[1]

            # Step 3: search
            assert calls[2][0] == "foldseek"
            assert calls[2][1] == "search"
            assert "--min-seq-id" in calls[2]
            seq_id_idx = calls[2].index("--min-seq-id")
            assert calls[2][seq_id_idx + 1] == "0.3"

            # Step 4: convertalis
            assert calls[3][0] == "foldseek"
            assert calls[3][1] == "convertalis"
            assert str(result) in calls[3]
            assert "--format-output" in calls[3]
            fmt_idx = calls[3].index("--format-output")
            assert calls[3][fmt_idx + 1] == "query,target,fident"

    def test_prostt5_model_parent_dir_used(self, tmp_path, foldseek_mod):
        """The --prostt5-model flag receives the parent directory of the .gguf file."""
        query = tmp_path / "q.fasta"
        target = tmp_path / "t.fasta"
        result = tmp_path / "r.tsv"
        prostt5 = tmp_path / "models" / "prostt5-f16.gguf"
        query.write_text(">q\nA\n")
        target.write_text(">t\nA\n")
        prostt5.parent.mkdir(parents=True)
        prostt5.write_text("fake model")

        expected_prefix = str(tmp_path / "models")

        with patch.object(foldseek_mod, "run") as mock_run:
            mock_run.return_value = None
            foldseek_mod.run_foldseek_search(
                query, target, result, min_seq_id=0.3, prostt5_model=prostt5
            )

            # Both createdb calls should use the parent directory as prefix
            for call_idx in (0, 1):
                cmd = mock_run.call_args_list[call_idx][0][0]
                idx = cmd.index("--prostt5-model")
                assert cmd[idx + 1] == expected_prefix

    def test_different_threshold(self, tmp_path, foldseek_mod):
        """Threshold value is passed through correctly to the search step."""
        query = tmp_path / "q.fasta"
        target = tmp_path / "t.fasta"
        result = tmp_path / "r.tsv"
        prostt5 = tmp_path / "prostt5-f16.gguf"
        query.write_text(">q\nA\n")
        target.write_text(">t\nA\n")
        prostt5.write_text("fake model")

        with patch.object(foldseek_mod, "run") as mock_run:
            mock_run.return_value = None
            foldseek_mod.run_foldseek_search(
                query, target, result, min_seq_id=0.5, prostt5_model=prostt5
            )

            # The search step (3rd call, index 2) should have the threshold
            search_cmd = mock_run.call_args_list[2][0][0]
            seq_id_idx = search_cmd.index("--min-seq-id")
            assert search_cmd[seq_id_idx + 1] == "0.5"

    def test_gpu_default_off(self, tmp_path, foldseek_mod):
        """GPU defaults to 0 in createdb and search steps when not specified."""
        query = tmp_path / "q.fasta"
        target = tmp_path / "t.fasta"
        result = tmp_path / "r.tsv"
        prostt5 = tmp_path / "prostt5-f16.gguf"
        query.write_text(">q\nA\n")
        target.write_text(">t\nA\n")
        prostt5.write_text("fake model")

        with patch.object(foldseek_mod, "run") as mock_run:
            mock_run.return_value = None
            foldseek_mod.run_foldseek_search(
                query, target, result, min_seq_id=0.3, prostt5_model=prostt5
            )

            # createdb query (step 0), createdb target (step 1), search (step 2)
            # should all have --gpu 0; convertalis (step 3) does not.
            for step_idx in (0, 1, 2):
                cmd = mock_run.call_args_list[step_idx][0][0]
                gpu_idx = cmd.index("--gpu")
                assert cmd[gpu_idx + 1] == "0"

            # convertalis should NOT have --gpu
            convert_cmd = mock_run.call_args_list[3][0][0]
            assert "--gpu" not in convert_cmd

    def test_gpu_enabled(self, tmp_path, foldseek_mod):
        """GPU flag is propagated to createdb and search steps when gpu=1.

        With GPU enabled the flow is 5 steps:
        0: createdb (query)  --gpu 1
        1: createdb (target) --gpu 1
        2: makepaddedseqdb   (no --gpu)
        3: search            --gpu 1  (uses padded target DB)
        4: convertalis       (no --gpu, uses original target DB)

        When gpu=1, the GPU binary is used instead of the conda
        ``foldseek``.  The binary path ends with ``foldseek`` but may
        be an absolute path to the cached GPU build.
        """
        query = tmp_path / "q.fasta"
        target = tmp_path / "t.fasta"
        result = tmp_path / "r.tsv"
        prostt5 = tmp_path / "prostt5-f16.gguf"
        query.write_text(">q\nA\n")
        target.write_text(">t\nA\n")
        prostt5.write_text("fake model")

        with patch.object(foldseek_mod, "run") as mock_run:
            mock_run.return_value = None
            foldseek_mod.run_foldseek_search(
                query,
                target,
                result,
                min_seq_id=0.3,
                prostt5_model=prostt5,
                gpu=1,
            )

            assert mock_run.call_count == 5
            calls = [c[0][0] for c in mock_run.call_args_list]

            # createdb query (step 0) and createdb target (step 1) have --gpu 1
            for step_idx in (0, 1):
                assert "--gpu" in calls[step_idx]
                gpu_idx = calls[step_idx].index("--gpu")
                assert calls[step_idx][gpu_idx + 1] == "1"

            # makepaddedseqdb (step 2) — binary ends with "foldseek"
            assert Path(calls[2][0]).name == "foldseek"
            assert calls[2][1] == "makepaddedseqdb"
            assert "--gpu" not in calls[2]

            # search (step 3) has --gpu 1 and uses padded target DB
            assert calls[3][1] == "search"
            gpu_idx = calls[3].index("--gpu")
            assert calls[3][gpu_idx + 1] == "1"
            # search target DB path should contain "targetDB_pad"
            assert any("targetDB_pad" in arg for arg in calls[3])

            # convertalis (step 4) should NOT have --gpu
            # and should use the original (unpadded) target DB
            assert calls[4][1] == "convertalis"
            assert "--gpu" not in calls[4]
            assert not any("targetDB_pad" in arg for arg in calls[4])

    def test_createdb_query_failure_raises(self, tmp_path, foldseek_mod):
        """CalledProcessError in createdb (query) is wrapped in RuntimeError."""
        from subprocess import CalledProcessError

        query = tmp_path / "q.fasta"
        target = tmp_path / "t.fasta"
        result = tmp_path / "r.tsv"
        prostt5 = tmp_path / "prostt5-f16.gguf"
        query.write_text(">q\nA\n")
        target.write_text(">t\nA\n")
        prostt5.write_text("fake model")

        with patch.object(
            foldseek_mod, "run", side_effect=CalledProcessError(1, "foldseek")
        ):
            with pytest.raises(RuntimeError, match="Foldseek createdb \\(query\\)"):
                foldseek_mod.run_foldseek_search(
                    query, target, result, min_seq_id=0.3, prostt5_model=prostt5
                )

    def test_makepaddedseqdb_failure_raises(self, tmp_path, foldseek_mod):
        """CalledProcessError in makepaddedseqdb step is wrapped in RuntimeError."""
        from subprocess import CalledProcessError

        query = tmp_path / "q.fasta"
        target = tmp_path / "t.fasta"
        result = tmp_path / "r.tsv"
        prostt5 = tmp_path / "prostt5-f16.gguf"
        query.write_text(">q\nA\n")
        target.write_text(">t\nA\n")
        prostt5.write_text("fake model")

        # First two calls (createdb) succeed, third call (makepaddedseqdb) fails
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 3:
                raise CalledProcessError(1, "foldseek")
            return None

        with patch.object(foldseek_mod, "run", side_effect=side_effect):
            with pytest.raises(RuntimeError, match="Foldseek makepaddedseqdb"):
                foldseek_mod.run_foldseek_search(
                    query,
                    target,
                    result,
                    min_seq_id=0.3,
                    prostt5_model=prostt5,
                    gpu=1,
                )

    def test_search_failure_raises(self, tmp_path, foldseek_mod):
        """CalledProcessError in search step is wrapped in RuntimeError."""
        from subprocess import CalledProcessError

        query = tmp_path / "q.fasta"
        target = tmp_path / "t.fasta"
        result = tmp_path / "r.tsv"
        prostt5 = tmp_path / "prostt5-f16.gguf"
        query.write_text(">q\nA\n")
        target.write_text(">t\nA\n")
        prostt5.write_text("fake model")

        # First two calls (createdb) succeed, third call (search) fails
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 3:
                raise CalledProcessError(1, "foldseek")
            return None

        with patch.object(foldseek_mod, "run", side_effect=side_effect):
            with pytest.raises(RuntimeError, match="Foldseek search"):
                foldseek_mod.run_foldseek_search(
                    query, target, result, min_seq_id=0.3, prostt5_model=prostt5
                )

    def test_convertalis_failure_raises(self, tmp_path, foldseek_mod):
        """CalledProcessError in convertalis step is wrapped in RuntimeError."""
        from subprocess import CalledProcessError

        query = tmp_path / "q.fasta"
        target = tmp_path / "t.fasta"
        result = tmp_path / "r.tsv"
        prostt5 = tmp_path / "prostt5-f16.gguf"
        query.write_text(">q\nA\n")
        target.write_text(">t\nA\n")
        prostt5.write_text("fake model")

        # First three calls succeed, fourth (convertalis) fails
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 4:
                raise CalledProcessError(1, "foldseek")
            return None

        with patch.object(foldseek_mod, "run", side_effect=side_effect):
            with pytest.raises(RuntimeError, match="Foldseek convertalis"):
                foldseek_mod.run_foldseek_search(
                    query, target, result, min_seq_id=0.3, prostt5_model=prostt5
                )
