from mutafitup.evaluation.plotting import generate_evaluation_summary
from mutafitup.evaluation.typst_tables import (
    export_delta_summary_typst,
    export_full_results_typst,
)


def _make_eval_config():
    return {
        "id": "test_eval",
        "split": "valid",
        "variant": "best_overall",
        "datasets": [
            {"task": "secstr", "metric": "accuracy"},
            {"task": "disorder", "metric": "spearman"},
        ],
        "baseline": {
            "label": "Single-task LoRA r4",
            "runs": {
                "secstr": {"section": "lora", "id": "bl_secstr"},
                "disorder": {"section": "lora", "id": "bl_disorder"},
            },
        },
        "approaches": [
            {
                "label": "MT r4",
                "run": {"section": "accgrad_lora", "id": "all_r4"},
            }
        ],
    }


def _make_metric_jsons():
    base = {
        "metrics": {
            "accuracy": {"value": 0.8, "std": 0.02, "ci95": 0.04},
            "spearman": {"value": 0.7, "std": 0.03, "ci95": 0.06},
        },
        "bootstrap_values": {
            "accuracy": [0.79 + 0.01 * i for i in range(10)],
            "spearman": [0.69 + 0.01 * i for i in range(10)],
        },
    }
    better = {
        "metrics": {
            "accuracy": {"value": 0.85, "std": 0.02, "ci95": 0.04},
            "spearman": {"value": 0.75, "std": 0.03, "ci95": 0.06},
        },
        "bootstrap_values": {
            "accuracy": [0.84 + 0.01 * i for i in range(10)],
            "spearman": [0.74 + 0.01 * i for i in range(10)],
        },
    }
    return {
        "lora/bl_secstr/best_overall/valid/secstr": base,
        "lora/bl_disorder/best_overall/valid/disorder": base,
        "accgrad_lora/all_r4/best_overall/valid/secstr": better,
        "accgrad_lora/all_r4/best_overall/valid/disorder": better,
    }


def test_export_full_results_typst_contains_headers_and_bold_best():
    table = export_full_results_typst(_make_eval_config(), _make_metric_jsons())
    assert "#table(" in table
    assert "Task" in table
    assert "Single-task LoRA r4" in table
    assert "MT r4" in table
    assert "0.850 +- 0.040" in table
    assert "#strong[0.850 +- 0.040]" in table


def test_export_delta_summary_typst_contains_summary_columns():
    summary = generate_evaluation_summary(_make_eval_config(), _make_metric_jsons())
    table = export_delta_summary_typst(summary)
    assert "Mean delta (%)" in table
    assert "Wins" in table
    assert "Losses" in table
    assert "Neutral" in table
    assert "MT r4" in table


def test_export_full_results_typst_with_variant_override():
    """Tables should work when an approach has a per-approach variant override."""
    eval_config = {
        "id": "test_variant_override",
        "split": "valid",
        "variant": "best_overall",
        "datasets": [
            {"task": "secstr", "metric": "accuracy"},
        ],
        "baseline": {
            "label": "Single-task LoRA r4",
            "runs": {
                "secstr": {"section": "lora", "id": "bl_secstr"},
            },
        },
        "approaches": [
            {
                "label": "Heads-only",
                "variant": "best_task",
                "run": {"section": "heads_only", "id": "ho_run"},
            },
        ],
    }
    metric_jsons = {
        "lora/bl_secstr/best_overall/valid/secstr": {
            "metrics": {
                "accuracy": {"value": 0.8, "std": 0.02, "ci95": 0.04},
            },
            "bootstrap_values": {
                "accuracy": [0.79 + 0.01 * i for i in range(10)],
            },
        },
        "heads_only/ho_run/best_task/valid/secstr": {
            "metrics": {
                "accuracy": {"value": 0.82, "std": 0.02, "ci95": 0.04},
            },
            "bootstrap_values": {
                "accuracy": [0.81 + 0.01 * i for i in range(10)],
            },
        },
    }
    table = export_full_results_typst(eval_config, metric_jsons)
    assert "#table(" in table
    assert "Heads-only" in table
    assert "0.820 +- 0.040" in table
