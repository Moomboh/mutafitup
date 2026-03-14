"""Tests for mutafitup.evaluation.auprc_resolve."""

import pytest

from mutafitup.evaluation.auprc_resolve import (
    resolve_auprc_bootstrap_inputs,
    resolve_auprc_evaluation_inputs,
    resolve_auprc_prob_prediction_inputs,
)


GPSITE_TASKS = [
    "gpsite_dna",
    "gpsite_rna",
    "gpsite_pep",
    "gpsite_pro",
    "gpsite_atp",
]


def _make_auprc_evaluation(
    *,
    eval_id="test_auprc",
    split="test",
    variant="best_overall",
    tasks=None,
    baseline_run=None,
    baseline_runs=None,
    approaches=None,
):
    if tasks is None:
        tasks = ["gpsite_dna", "gpsite_rna"]
    evaluation = {
        "id": eval_id,
        "split": split,
        "variant": variant,
        "tasks": tasks,
    }
    if baseline_run:
        evaluation["baseline"] = {"run": baseline_run}
    elif baseline_runs:
        evaluation["baseline"] = {"runs": baseline_runs}
    else:
        evaluation["baseline"] = {
            "runs": {
                "gpsite_dna": {"section": "lora", "id": "run_dna"},
                "gpsite_rna": {"section": "lora", "id": "run_rna"},
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


class TestResolveAuprcEvaluationInputs:
    def test_basic_paths(self):
        evaluation = _make_auprc_evaluation()
        paths = resolve_auprc_evaluation_inputs(evaluation)
        assert len(paths) > 0
        for p in paths:
            assert p.startswith("results/auprc_metrics/")
            assert p.endswith(".json")

    def test_baseline_per_task_mapping(self):
        evaluation = _make_auprc_evaluation(
            baseline_runs={
                "gpsite_dna": {"section": "lora", "id": "run_dna"},
                "gpsite_rna": {"section": "lora", "id": "run_rna"},
            }
        )
        paths = resolve_auprc_evaluation_inputs(evaluation)
        assert (
            "results/auprc_metrics/lora/run_dna/best_overall/test/gpsite_dna.json"
            in paths
        )
        assert (
            "results/auprc_metrics/lora/run_rna/best_overall/test/gpsite_rna.json"
            in paths
        )

    def test_baseline_single_run(self):
        evaluation = _make_auprc_evaluation(
            baseline_run={"section": "lora", "id": "single_run"}
        )
        paths = resolve_auprc_evaluation_inputs(evaluation)
        assert (
            "results/auprc_metrics/lora/single_run/best_overall/test/gpsite_dna.json"
            in paths
        )
        assert (
            "results/auprc_metrics/lora/single_run/best_overall/test/gpsite_rna.json"
            in paths
        )

    def test_approach_variant_override(self):
        evaluation = _make_auprc_evaluation(
            approaches=[
                {
                    "label": "Heads-only",
                    "variant": "best_task",
                    "run": {"section": "heads_only", "id": "ho_run"},
                }
            ],
        )
        paths = resolve_auprc_evaluation_inputs(evaluation)
        # Baseline should use evaluation-level variant (best_overall)
        assert (
            "results/auprc_metrics/lora/run_dna/best_overall/test/gpsite_dna.json"
            in paths
        )
        # Approach should use overridden variant (best_task)
        assert (
            "results/auprc_metrics/heads_only/ho_run/best_task/test/gpsite_dna.json"
            in paths
        )

    def test_no_duplicates(self):
        evaluation = _make_auprc_evaluation(
            baseline_run={"section": "lora", "id": "same_run"},
            approaches=[
                {
                    "label": "Same",
                    "run": {"section": "lora", "id": "same_run"},
                }
            ],
        )
        paths = resolve_auprc_evaluation_inputs(evaluation)
        assert len(paths) == len(set(paths))


class TestResolveAuprcProbPredictionInputs:
    def test_basic(self):
        config = {"auprc_evaluations": [_make_auprc_evaluation()]}
        result = resolve_auprc_prob_prediction_inputs(config)
        assert len(result) > 0
        for entry in result:
            assert "section" in entry
            assert "run_id" in entry
            assert "variant" in entry
            assert "split" in entry
            assert "task" in entry

    def test_no_duplicates(self):
        config = {"auprc_evaluations": [_make_auprc_evaluation()]}
        result = resolve_auprc_prob_prediction_inputs(config)
        keys = [
            (r["section"], r["run_id"], r["variant"], r["split"], r["task"])
            for r in result
        ]
        assert len(keys) == len(set(keys))

    def test_variant_override(self):
        evaluation = _make_auprc_evaluation(
            approaches=[
                {
                    "label": "HO",
                    "variant": "best_task",
                    "run": {"section": "heads_only", "id": "ho"},
                },
                {
                    "label": "MT",
                    "run": {"section": "accgrad_lora", "id": "mt"},
                },
            ],
        )
        config = {"auprc_evaluations": [evaluation]}
        result = resolve_auprc_prob_prediction_inputs(config)
        entries = {(r["section"], r["run_id"], r["variant"], r["task"]) for r in result}
        assert ("heads_only", "ho", "best_task", "gpsite_dna") in entries
        assert ("accgrad_lora", "mt", "best_overall", "gpsite_dna") in entries

    def test_empty_config(self):
        config = {}
        result = resolve_auprc_prob_prediction_inputs(config)
        assert result == []


class TestResolveAuprcBootstrapInputs:
    def test_basic(self):
        config = {"auprc_evaluations": [_make_auprc_evaluation()]}
        result = resolve_auprc_bootstrap_inputs(config)
        assert ("per_residue_classification", "gpsite_dna", "test") in result
        assert ("per_residue_classification", "gpsite_rna", "test") in result

    def test_no_duplicates(self):
        eval1 = _make_auprc_evaluation(eval_id="e1")
        eval2 = _make_auprc_evaluation(eval_id="e2")
        config = {"auprc_evaluations": [eval1, eval2]}
        result = resolve_auprc_bootstrap_inputs(config)
        assert len(result) == len(set(result))

    def test_empty_config(self):
        config = {}
        result = resolve_auprc_bootstrap_inputs(config)
        assert result == []
