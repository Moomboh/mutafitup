"""Tests for mutafitup.evaluation.resolve."""

import pytest

from mutafitup.evaluation.resolve import (
    _get_run_tasks,
    resolve_all_metric_inputs,
    resolve_bootstrap_inputs,
    resolve_evaluation_inputs,
)


def _make_evaluation(
    *,
    eval_id="test_eval",
    split="valid",
    variant="best_overall",
    tasks=None,
    baseline_run=None,
    baseline_runs=None,
    approaches=None,
):
    if tasks is None:
        tasks = [
            {"task": "secstr", "metric": "accuracy"},
            {"task": "disorder", "metric": "spearman"},
        ]
    evaluation = {
        "id": eval_id,
        "split": split,
        "variant": variant,
        "datasets": tasks,
    }
    if baseline_run:
        evaluation["baseline"] = {"run": baseline_run}
    elif baseline_runs:
        evaluation["baseline"] = {"runs": baseline_runs}
    else:
        evaluation["baseline"] = {
            "runs": {
                "secstr": {"section": "lora", "id": "run_secstr"},
                "disorder": {"section": "lora", "id": "run_disorder"},
            }
        }
    if approaches is None:
        approaches = [
            {
                "label": "MT r4",
                "run": {"section": "accgrad_lora", "id": "run_all_r4"},
            }
        ]
    evaluation["approaches"] = approaches
    return evaluation


class TestResolveEvaluationInputs:
    def test_basic_paths(self):
        evaluation = _make_evaluation()
        paths = resolve_evaluation_inputs(evaluation)
        assert len(paths) > 0
        for p in paths:
            assert p.startswith("results/metrics/") or p.startswith(
                "results/auprc_metrics/"
            )
            assert p.endswith(".json")

    def test_baseline_per_task_mapping(self):
        evaluation = _make_evaluation(
            baseline_runs={
                "secstr": {"section": "lora", "id": "run_secstr"},
                "disorder": {"section": "lora", "id": "run_disorder"},
            }
        )
        paths = resolve_evaluation_inputs(evaluation)
        assert "results/metrics/lora/run_secstr/best_overall/valid/secstr.json" in paths
        assert (
            "results/metrics/lora/run_disorder/best_overall/valid/disorder.json"
            in paths
        )

    def test_baseline_single_run(self):
        evaluation = _make_evaluation(
            baseline_run={"section": "lora", "id": "single_run"}
        )
        paths = resolve_evaluation_inputs(evaluation)
        assert "results/metrics/lora/single_run/best_overall/valid/secstr.json" in paths
        assert (
            "results/metrics/lora/single_run/best_overall/valid/disorder.json" in paths
        )

    def test_approach_single_run(self):
        evaluation = _make_evaluation(
            approaches=[
                {
                    "label": "MT",
                    "run": {"section": "accgrad_lora", "id": "all_r4"},
                }
            ],
        )
        paths = resolve_evaluation_inputs(evaluation)
        assert (
            "results/metrics/accgrad_lora/all_r4/best_overall/valid/secstr.json"
            in paths
        )
        assert (
            "results/metrics/accgrad_lora/all_r4/best_overall/valid/disorder.json"
            in paths
        )

    def test_approach_per_task_mapping(self):
        evaluation = _make_evaluation(
            approaches=[
                {
                    "label": "MT",
                    "runs": {
                        "secstr": {"section": "accgrad_lora", "id": "run_a"},
                        "disorder": {"section": "accgrad_lora", "id": "run_b"},
                    },
                }
            ],
        )
        paths = resolve_evaluation_inputs(evaluation)
        assert (
            "results/metrics/accgrad_lora/run_a/best_overall/valid/secstr.json" in paths
        )
        assert (
            "results/metrics/accgrad_lora/run_b/best_overall/valid/disorder.json"
            in paths
        )

    def test_no_duplicates(self):
        """Same run used as both baseline and approach should appear once."""
        evaluation = _make_evaluation(
            baseline_run={"section": "lora", "id": "same_run"},
            approaches=[
                {
                    "label": "Same",
                    "run": {"section": "lora", "id": "same_run"},
                }
            ],
        )
        paths = resolve_evaluation_inputs(evaluation)
        assert len(paths) == len(set(paths))

    def test_auprc_metric_uses_auprc_metrics_prefix(self):
        """Tasks with metric='auprc' should resolve to results/auprc_metrics/."""
        evaluation = _make_evaluation(
            tasks=[
                {"task": "secstr", "metric": "accuracy"},
                {"task": "gpsite_dna", "metric": "auprc"},
            ],
            baseline_run={"section": "lora", "id": "run_all"},
            approaches=[
                {
                    "label": "MT",
                    "run": {"section": "accgrad_lora", "id": "mt_run"},
                }
            ],
        )
        paths = resolve_evaluation_inputs(evaluation)
        # secstr (accuracy) should use results/metrics/
        assert "results/metrics/lora/run_all/best_overall/valid/secstr.json" in paths
        assert (
            "results/metrics/accgrad_lora/mt_run/best_overall/valid/secstr.json"
            in paths
        )
        # gpsite_dna (auprc) should use results/auprc_metrics/
        assert (
            "results/auprc_metrics/lora/run_all/best_overall/valid/gpsite_dna.json"
            in paths
        )
        assert (
            "results/auprc_metrics/accgrad_lora/mt_run/best_overall/valid/gpsite_dna.json"
            in paths
        )
        # Verify no gpsite_dna paths under results/metrics/
        for p in paths:
            if "gpsite_dna" in p:
                assert p.startswith("results/auprc_metrics/")

    def test_auprc_skipped_when_run_not_trained_on_task(self):
        """AUPRC paths should be skipped when train_config shows run not trained on task."""
        evaluation = _make_evaluation(
            tasks=[
                {"task": "secstr", "metric": "accuracy"},
                {"task": "gpsite_dna", "metric": "auprc"},
            ],
            baseline_run={"section": "lora", "id": "run_all"},
            approaches=[
                {
                    "label": "Structural only",
                    "run": {"section": "accgrad_lora", "id": "structural_r4"},
                }
            ],
        )
        train_config = {
            "lora": [{"id": "run_all", "tasks": ["secstr", "gpsite_dna"]}],
            "accgrad_lora": [
                {"id": "structural_r4", "tasks": ["secstr"]},
            ],
        }
        paths = resolve_evaluation_inputs(evaluation, train_config=train_config)

        # Baseline is trained on gpsite_dna → AUPRC path emitted
        assert (
            "results/auprc_metrics/lora/run_all/best_overall/valid/gpsite_dna.json"
            in paths
        )
        # Approach is NOT trained on gpsite_dna → no path at all
        for p in paths:
            assert not ("structural_r4" in p and "gpsite_dna" in p), (
                f"Unexpected path for untrained task: {p}"
            )
        # Approach IS trained on secstr → normal metrics path emitted
        assert (
            "results/metrics/accgrad_lora/structural_r4/best_overall/valid/secstr.json"
            in paths
        )

    def test_auprc_emitted_when_run_trained_on_task(self):
        """AUPRC paths should still be emitted when train_config confirms training."""
        evaluation = _make_evaluation(
            tasks=[
                {"task": "gpsite_dna", "metric": "auprc"},
            ],
            baseline_run={"section": "lora", "id": "run_gpsite"},
            approaches=[
                {
                    "label": "GPSite model",
                    "run": {"section": "accgrad_lora", "id": "gpsite_r4"},
                }
            ],
        )
        train_config = {
            "lora": [{"id": "run_gpsite", "tasks": ["gpsite_dna"]}],
            "accgrad_lora": [{"id": "gpsite_r4", "tasks": ["gpsite_dna"]}],
        }
        paths = resolve_evaluation_inputs(evaluation, train_config=train_config)
        assert (
            "results/auprc_metrics/lora/run_gpsite/best_overall/valid/gpsite_dna.json"
            in paths
        )
        assert (
            "results/auprc_metrics/accgrad_lora/gpsite_r4/best_overall/valid/gpsite_dna.json"
            in paths
        )

    def test_non_auprc_skipped_when_run_not_trained_on_task(self):
        """Non-AUPRC metrics paths should also be skipped for untrained tasks."""
        evaluation = _make_evaluation(
            tasks=[
                {"task": "secstr", "metric": "accuracy"},
                {"task": "meltome", "metric": "spearman"},
            ],
            baseline_run={"section": "lora", "id": "run_all"},
            approaches=[
                {
                    "label": "Structural only",
                    "run": {"section": "accgrad_lora", "id": "structural_r4"},
                }
            ],
        )
        train_config = {
            "lora": [{"id": "run_all", "tasks": ["secstr", "meltome"]}],
            "accgrad_lora": [
                {"id": "structural_r4", "tasks": ["secstr"]},
            ],
        }
        paths = resolve_evaluation_inputs(evaluation, train_config=train_config)

        # Approach IS trained on secstr → metrics path emitted
        assert (
            "results/metrics/accgrad_lora/structural_r4/best_overall/valid/secstr.json"
            in paths
        )
        # Approach is NOT trained on meltome → no path at all
        for p in paths:
            assert not ("structural_r4" in p and "meltome" in p), (
                f"Unexpected path for untrained task: {p}"
            )
        # Baseline is trained on both → both paths emitted
        assert "results/metrics/lora/run_all/best_overall/valid/secstr.json" in paths
        assert "results/metrics/lora/run_all/best_overall/valid/meltome.json" in paths

    def test_all_families_scenario(self):
        """All-families evaluation: family models skip all paths for tasks they don't cover."""
        evaluation = _make_evaluation(
            tasks=[
                {"task": "secstr", "metric": "accuracy"},
                {"task": "gpsite_dna", "metric": "auprc"},
            ],
            baseline_runs={
                "secstr": {"section": "lora", "id": "st_secstr"},
                "gpsite_dna": {"section": "lora", "id": "st_gpsite_dna"},
            },
            approaches=[
                {
                    "label": "Structural r4",
                    "run": {"section": "accgrad_lora", "id": "structural_r4"},
                },
                {
                    "label": "GPSite r4",
                    "run": {"section": "accgrad_lora", "id": "gpsite_r4"},
                },
                {
                    "label": "All r4",
                    "run": {"section": "accgrad_lora", "id": "all_r4"},
                },
            ],
        )
        train_config = {
            "lora": [
                {"id": "st_secstr", "tasks": ["secstr"]},
                {"id": "st_gpsite_dna", "tasks": ["gpsite_dna"]},
            ],
            "accgrad_lora": [
                {"id": "structural_r4", "tasks": ["secstr"]},
                {"id": "gpsite_r4", "tasks": ["gpsite_dna"]},
                {"id": "all_r4", "tasks": ["secstr", "gpsite_dna"]},
            ],
        }
        paths = resolve_evaluation_inputs(evaluation, train_config=train_config)

        # Structural model: secstr YES, gpsite_dna NO (not trained)
        assert (
            "results/metrics/accgrad_lora/structural_r4/best_overall/valid/secstr.json"
            in paths
        )
        assert all(not ("structural_r4" in p and "gpsite_dna" in p) for p in paths)

        # GPSite model: gpsite_dna YES, secstr NO (not trained)
        assert (
            "results/auprc_metrics/accgrad_lora/gpsite_r4/best_overall/valid/gpsite_dna.json"
            in paths
        )
        assert all(not ("gpsite_r4" in p and "secstr" in p) for p in paths)

        # All-tasks model: both paths emitted
        assert (
            "results/metrics/accgrad_lora/all_r4/best_overall/valid/secstr.json"
            in paths
        )
        assert (
            "results/auprc_metrics/accgrad_lora/all_r4/best_overall/valid/gpsite_dna.json"
            in paths
        )

    def test_multiple_approaches(self):
        evaluation = _make_evaluation(
            approaches=[
                {
                    "label": "MT r4",
                    "run": {"section": "accgrad_lora", "id": "r4"},
                },
                {
                    "label": "MT r16",
                    "run": {"section": "accgrad_lora", "id": "r16"},
                },
            ],
        )
        paths = resolve_evaluation_inputs(evaluation)
        # 2 tasks * 3 run refs (baseline + 2 approaches), but baseline has per-task
        # so up to 2 baseline + 2*2 approach = 6 unique paths
        assert len(paths) >= 4


class TestGetRunTasks:
    def test_found(self):
        train_config = {
            "lora": [{"id": "run_a", "tasks": ["secstr", "rsa"]}],
        }
        assert _get_run_tasks(train_config, "lora", "run_a") == {"secstr", "rsa"}

    def test_not_found_section(self):
        train_config = {"lora": [{"id": "run_a", "tasks": ["secstr"]}]}
        assert _get_run_tasks(train_config, "accgrad_lora", "run_a") is None

    def test_not_found_id(self):
        train_config = {"lora": [{"id": "run_a", "tasks": ["secstr"]}]}
        assert _get_run_tasks(train_config, "lora", "run_b") is None

    def test_empty_tasks(self):
        train_config = {"lora": [{"id": "run_a", "tasks": []}]}
        assert _get_run_tasks(train_config, "lora", "run_a") == set()


class TestResolveBootstrapInputs:
    def test_basic(self):
        config = {
            "standardized_datasets": {
                "per_residue_classification": {"secstr": {}},
                "per_residue_regression": {"disorder": {}},
            },
            "evaluations": [_make_evaluation()],
        }
        result = resolve_bootstrap_inputs(config)
        assert ("per_residue_classification", "secstr", "valid") in result
        assert ("per_residue_regression", "disorder", "valid") in result

    def test_no_duplicates(self):
        eval1 = _make_evaluation(eval_id="e1")
        eval2 = _make_evaluation(eval_id="e2")
        config = {
            "standardized_datasets": {
                "per_residue_classification": {"secstr": {}},
                "per_residue_regression": {"disorder": {}},
            },
            "evaluations": [eval1, eval2],
        }
        result = resolve_bootstrap_inputs(config)
        assert len(result) == len(set(result))

    def test_unknown_task_raises(self):
        config = {
            "standardized_datasets": {},
            "evaluations": [_make_evaluation()],
        }
        with pytest.raises(ValueError, match="not found"):
            resolve_bootstrap_inputs(config)


class TestResolveAllMetricInputs:
    def test_basic(self):
        config = {"evaluations": [_make_evaluation()]}
        result = resolve_all_metric_inputs(config)
        assert len(result) > 0
        for entry in result:
            assert "section" in entry
            assert "run_id" in entry
            assert "variant" in entry
            assert "split" in entry
            assert "task" in entry

    def test_no_duplicates(self):
        config = {"evaluations": [_make_evaluation()]}
        result = resolve_all_metric_inputs(config)
        keys = [
            (r["section"], r["run_id"], r["variant"], r["split"], r["task"])
            for r in result
        ]
        assert len(keys) == len(set(keys))


class TestPerApproachVariantOverride:
    """Tests for the optional per-approach ``variant`` override."""

    def test_approach_variant_override_in_resolve_inputs(self):
        """An approach with variant override should produce paths with that variant."""
        evaluation = _make_evaluation(
            approaches=[
                {
                    "label": "Heads-only",
                    "variant": "best_task",
                    "run": {"section": "heads_only", "id": "ho_run"},
                }
            ],
        )
        paths = resolve_evaluation_inputs(evaluation)
        # Baseline should still use the evaluation-level variant (best_overall)
        assert "results/metrics/lora/run_secstr/best_overall/valid/secstr.json" in paths
        # Approach should use the overridden variant (best_task)
        assert "results/metrics/heads_only/ho_run/best_task/valid/secstr.json" in paths
        assert (
            "results/metrics/heads_only/ho_run/best_task/valid/disorder.json" in paths
        )

    def test_baseline_variant_override_in_resolve_inputs(self):
        """A baseline with variant override should produce paths with that variant."""
        evaluation = _make_evaluation(
            baseline_run={"section": "heads_only", "id": "bl_ho"},
        )
        evaluation["baseline"]["variant"] = "best_task"
        paths = resolve_evaluation_inputs(evaluation)
        assert "results/metrics/heads_only/bl_ho/best_task/valid/secstr.json" in paths

    def test_mixed_variants_in_resolve_all_metric_inputs(self):
        """resolve_all_metric_inputs should respect per-approach variant."""
        evaluation = _make_evaluation(
            approaches=[
                {
                    "label": "Heads-only",
                    "variant": "best_task",
                    "run": {"section": "heads_only", "id": "ho_run"},
                },
                {
                    "label": "MT r4",
                    "run": {"section": "accgrad_lora", "id": "all_r4"},
                },
            ],
        )
        config = {"evaluations": [evaluation]}
        result = resolve_all_metric_inputs(config)

        # Build lookup for quick assertion
        entries = {(r["section"], r["run_id"], r["variant"], r["task"]) for r in result}

        # Heads-only approach should use best_task
        assert ("heads_only", "ho_run", "best_task", "secstr") in entries
        assert ("heads_only", "ho_run", "best_task", "disorder") in entries

        # MT r4 approach should use the default best_overall
        assert ("accgrad_lora", "all_r4", "best_overall", "secstr") in entries
        assert ("accgrad_lora", "all_r4", "best_overall", "disorder") in entries

    def test_no_override_uses_default(self):
        """Without variant override, approach uses evaluation-level variant."""
        evaluation = _make_evaluation(
            approaches=[
                {
                    "label": "MT r4",
                    "run": {"section": "accgrad_lora", "id": "all_r4"},
                }
            ],
        )
        paths = resolve_evaluation_inputs(evaluation)
        assert (
            "results/metrics/accgrad_lora/all_r4/best_overall/valid/secstr.json"
            in paths
        )
