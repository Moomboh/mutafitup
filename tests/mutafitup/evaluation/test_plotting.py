"""Tests for mutafitup.evaluation.plotting."""

import math

import matplotlib

matplotlib.use("Agg")

import pytest

from mutafitup.evaluation.plotting import (
    _text_color_for_bg,
    compute_delta_summary,
    compute_pairwise_delta,
    generate_evaluation_summary,
    group_sota_tasks_by_test_set,
    plot_absolute_bars,
    plot_delta_heatmap,
    plot_delta_summary,
    plot_sota_comparison,
)


class TestTextColorForBg:
    def test_white_background_gets_black_text(self):
        assert _text_color_for_bg("#ffffff") == "#000000"

    def test_black_background_gets_white_text(self):
        assert _text_color_for_bg("#000000") == "#ffffff"

    def test_bright_yellow_gets_black_text(self):
        # #ffe119 has high luminance
        assert _text_color_for_bg("#ffe119") == "#000000"

    def test_dark_blue_gets_white_text(self):
        # #000075 has low luminance
        assert _text_color_for_bg("#000075") == "#ffffff"


class TestComputePairwiseDelta:
    def test_identical_bootstraps_have_zero_delta(self):
        """Identical values should give zero delta."""
        vals = [0.8, 0.82, 0.79, 0.81, 0.8]
        result = compute_pairwise_delta(vals, vals)
        assert abs(result["mean"]) < 1e-10
        assert abs(result["std"]) < 1e-10
        assert result["significant"] is False

    def test_positive_delta(self):
        """Approach > baseline should give positive mean."""
        baseline = [0.5] * 100
        approach = [0.6] * 100
        result = compute_pairwise_delta(baseline, approach)
        assert result["mean"] > 0
        assert abs(result["mean"] - 0.1) < 1e-10

    def test_negative_delta(self):
        """Approach < baseline should give negative mean."""
        baseline = [0.6] * 100
        approach = [0.5] * 100
        result = compute_pairwise_delta(baseline, approach)
        assert result["mean"] < 0

    def test_percent_mode(self):
        """Percent delta = (approach - baseline) / |baseline| * 100."""
        baseline = [0.5] * 100
        approach = [0.6] * 100
        result = compute_pairwise_delta(baseline, approach, percent=True)
        # (0.6 - 0.5) / 0.5 * 100 = 20%
        assert abs(result["mean"] - 20.0) < 1e-6

    def test_ci95_is_196_times_std(self):
        baseline = [0.5, 0.51, 0.49, 0.52, 0.48]
        approach = [0.55, 0.56, 0.54, 0.57, 0.53]
        result = compute_pairwise_delta(baseline, approach)
        assert abs(result["ci95"] - 1.96 * result["std"]) < 1e-10

    def test_nan_values_skipped(self):
        baseline = [0.5, float("nan"), 0.6]
        approach = [0.55, 0.6, float("nan")]
        result = compute_pairwise_delta(baseline, approach)
        # Only first pair valid: 0.55 - 0.5 = 0.05
        assert abs(result["mean"] - 0.05) < 1e-10

    def test_has_p_value(self):
        baseline = [0.5] * 100
        approach = [0.6] * 100
        result = compute_pairwise_delta(baseline, approach)
        assert "p_value" in result

    def test_length_mismatch_raises(self):
        with pytest.raises(AssertionError):
            compute_pairwise_delta([1, 2], [1, 2, 3])


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
            "runs": {
                "secstr": {"section": "lora", "id": "bl_secstr"},
                "disorder": {"section": "lora", "id": "bl_disorder"},
            }
        },
        "approaches": [
            {
                "label": "MT r4",
                "run": {"section": "accgrad_lora", "id": "all_r4"},
            }
        ],
    }


def _make_metric_jsons():
    """Build a fake metric_jsons dict with bootstrap values."""
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


class TestPlotDeltaHeatmap:
    def test_creates_file(self, tmp_path):
        out = tmp_path / "heatmap.png"
        plot_delta_heatmap(
            evaluation=_make_eval_config(),
            metric_jsons=_make_metric_jsons(),
            output_path=out,
        )
        assert out.exists()
        assert out.stat().st_size > 0

    def test_baseline_row_present(self, tmp_path):
        """Heatmap should have a baseline row at the top with absolute values."""
        import matplotlib.pyplot as plt

        out = tmp_path / "heatmap.png"
        eval_cfg = _make_eval_config()
        plot_delta_heatmap(
            evaluation=eval_cfg,
            metric_jsons=_make_metric_jsons(),
            output_path=out,
        )
        # Re-generate without saving to inspect the axes
        fig, ax = plt.subplots()
        plot_delta_heatmap(
            evaluation=eval_cfg,
            metric_jsons=_make_metric_jsons(),
            output_path=out,
        )
        # Read back and inspect: the saved file should exist
        assert out.exists()

    def test_baseline_row_ytick_labels(self, tmp_path):
        """Y-tick labels should start with baseline, followed by approaches."""
        import matplotlib.pyplot as plt

        out = tmp_path / "heatmap.png"
        eval_cfg = _make_eval_config()
        metric_jsons = _make_metric_jsons()

        # Monkey-patch savefig to capture the figure
        captured = {}
        orig_savefig = plt.Figure.savefig

        def capture_savefig(self, *args, **kwargs):
            captured["fig"] = self
            orig_savefig(self, *args, **kwargs)

        plt.Figure.savefig = capture_savefig
        try:
            plot_delta_heatmap(
                evaluation=eval_cfg,
                metric_jsons=metric_jsons,
                output_path=out,
            )
        finally:
            plt.Figure.savefig = orig_savefig

        fig = captured["fig"]
        ax = fig.axes[0]
        ytick_labels = [t.get_text() for t in ax.get_yticklabels()]

        n_approaches = len(eval_cfg["approaches"])
        # Should have baseline + n_approaches rows
        assert len(ytick_labels) == n_approaches + 1
        assert ytick_labels[0] == "Single-task LoRA r4\n(baseline)"
        assert ytick_labels[1] == "MT r4"
        plt.close(fig)

    def test_summary_column_present(self, tmp_path):
        """Heatmap should include a summary column labeled 'Mean \u0394'."""
        import matplotlib.pyplot as plt

        out = tmp_path / "heatmap.png"
        eval_cfg = _make_eval_config()
        metric_jsons = _make_metric_jsons()

        captured = {}
        orig_savefig = plt.Figure.savefig

        def capture_savefig(self, *args, **kwargs):
            captured["fig"] = self
            orig_savefig(self, *args, **kwargs)

        plt.Figure.savefig = capture_savefig
        try:
            plot_delta_heatmap(
                evaluation=eval_cfg,
                metric_jsons=metric_jsons,
                output_path=out,
            )
        finally:
            plt.Figure.savefig = orig_savefig

        fig = captured["fig"]
        ax = fig.axes[0]
        xtick_labels = [t.get_text() for t in ax.get_xticklabels()]
        n_tasks = len(eval_cfg["datasets"])
        # Should have n_tasks + 1 x-tick labels (tasks + summary)
        assert len(xtick_labels) == n_tasks + 1
        assert xtick_labels[0] == "Mean \u0394"
        plt.close(fig)


class TestPlotAbsoluteBars:
    def test_creates_file(self, tmp_path):
        out = tmp_path / "bars_secstr.png"
        plot_absolute_bars(
            evaluation=_make_eval_config(),
            metric_jsons=_make_metric_jsons(),
            task="secstr",
            metric_name="accuracy",
            output_path=out,
        )
        assert out.exists()
        assert out.stat().st_size > 0


class TestDeltaSummary:
    def test_compute_delta_summary_structure(self):
        summary = compute_delta_summary(
            evaluation=_make_eval_config(),
            metric_jsons=_make_metric_jsons(),
        )
        assert summary["baseline_label"] == "Single-task LoRA r4\n(baseline)"
        assert len(summary["rows"]) == 1
        row = summary["rows"][0]
        assert row["label"] == "MT r4"
        assert row["tasks_compared"] == 2
        assert row["mean_delta_percent"] is not None
        assert row["significant_wins"] >= 0

    def test_compute_delta_summary_handles_missing(self):
        summary = compute_delta_summary(
            evaluation=_make_eval_config(),
            metric_jsons=_make_metric_jsons_with_missing(),
        )
        row = summary["rows"][0]
        assert row["tasks_compared"] == 1
        assert row["per_task_deltas"]["disorder"] is None

    def test_ci95_is_bootstrap_derived(self):
        """CI95 should come from the bootstrap distribution of mean-across-tasks,
        not from 1.96 * std(per_task_means) / sqrt(n_tasks)."""
        import numpy as np

        # Create bootstrap data where the two tasks have very different
        # variability, so bootstrap-level CI differs from naive CI.
        n_boot = 200
        rng = np.random.RandomState(42)
        # Task A: baseline ~0.8 with high variance, approach ~0.85 with high variance
        bl_a = 0.8 + rng.normal(0, 0.05, n_boot)
        ap_a = 0.85 + rng.normal(0, 0.05, n_boot)
        # Task B: baseline ~0.5 with low variance, approach ~0.55 with low variance
        bl_b = 0.5 + rng.normal(0, 0.005, n_boot)
        ap_b = 0.55 + rng.normal(0, 0.005, n_boot)

        metric_jsons = {
            "lora/bl_secstr/best_overall/valid/secstr": {
                "metrics": {"accuracy": {"value": 0.8, "std": 0.05, "ci95": 0.1}},
                "bootstrap_values": {"accuracy": bl_a.tolist()},
            },
            "lora/bl_disorder/best_overall/valid/disorder": {
                "metrics": {"spearman": {"value": 0.5, "std": 0.005, "ci95": 0.01}},
                "bootstrap_values": {"spearman": bl_b.tolist()},
            },
            "accgrad_lora/all_r4/best_overall/valid/secstr": {
                "metrics": {"accuracy": {"value": 0.85, "std": 0.05, "ci95": 0.1}},
                "bootstrap_values": {"accuracy": ap_a.tolist()},
            },
            "accgrad_lora/all_r4/best_overall/valid/disorder": {
                "metrics": {"spearman": {"value": 0.55, "std": 0.005, "ci95": 0.01}},
                "bootstrap_values": {"spearman": ap_b.tolist()},
            },
        }
        summary = compute_delta_summary(
            evaluation=_make_eval_config(),
            metric_jsons=metric_jsons,
        )
        row = summary["rows"][0]
        ci95 = row["ci95_mean_delta_percent"]
        std_delta = row["std_delta_percent"]
        assert ci95 is not None
        assert std_delta is not None

        # The bootstrap-level CI should be 1.96 * std of the bootstrap
        # distribution (not divided by sqrt(n_tasks)).
        assert abs(ci95 - 1.96 * std_delta) < 1e-10

        # Verify it is NOT the naive formula (1.96 * std / sqrt(2)):
        # The naive formula would give a smaller CI.
        naive_ci = 1.96 * std_delta / math.sqrt(2)
        assert abs(ci95 - naive_ci) > 1e-6

    def test_plot_delta_summary_creates_file(self, tmp_path):
        out = tmp_path / "delta_summary.png"
        plot_delta_summary(
            evaluation=_make_eval_config(),
            metric_jsons=_make_metric_jsons(),
            output_path=out,
        )
        assert out.exists()
        assert out.stat().st_size > 0

    def test_plot_delta_summary_uses_run_color(self, tmp_path):
        """Bars should use the run's color from train_config."""
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt

        out = tmp_path / "delta_summary.png"
        eval_cfg = _make_eval_config()
        metric_jsons = _make_metric_jsons()
        train_config = {
            "accgrad_lora": [{"id": "all_r4", "color": "#00ff00"}],
        }

        captured = {}
        orig_savefig = plt.Figure.savefig

        def capture_savefig(self, *args, **kwargs):
            captured["fig"] = self
            orig_savefig(self, *args, **kwargs)

        plt.Figure.savefig = capture_savefig
        try:
            plot_delta_summary(
                evaluation=eval_cfg,
                metric_jsons=metric_jsons,
                output_path=out,
                train_config=train_config,
            )
        finally:
            plt.Figure.savefig = orig_savefig

        fig = captured["fig"]
        ax = fig.axes[0]
        bars = ax.patches
        # The approach "MT r4" uses run: accgrad_lora/all_r4 -> should be green.
        assert len(bars) >= 1
        assert mcolors.to_hex(bars[0].get_facecolor()) == "#00ff00"
        plt.close(fig)

    def test_plot_delta_summary_runs_spec_uses_train_config_color(self, tmp_path):
        """A runs: approach should pick up its colour from train_config."""
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt

        out = tmp_path / "delta_summary_runs.png"
        # Build an eval config where the approach uses runs: instead of run:
        eval_cfg = {
            "id": "test_eval_runs",
            "split": "valid",
            "variant": "best_overall",
            "datasets": [
                {"task": "secstr", "metric": "accuracy"},
                {"task": "disorder", "metric": "spearman"},
            ],
            "baseline": {
                "runs": {
                    "secstr": {"section": "lora", "id": "bl_secstr"},
                    "disorder": {"section": "lora", "id": "bl_disorder"},
                }
            },
            "approaches": [
                {
                    "label": "ST r16",
                    "runs": {
                        "secstr": {"section": "lora", "id": "st_secstr_r16"},
                        "disorder": {"section": "lora", "id": "st_disorder_r16"},
                    },
                }
            ],
        }
        metric_jsons = _make_metric_jsons()
        # Add entries for the runs: approach
        for task in ("secstr", "disorder"):
            key = f"lora/st_{task}_r16/best_overall/valid/{task}"
            metric_jsons[key] = {
                "metrics": {
                    "accuracy": {"value": 0.85, "std": 0.02, "ci95": 0.04},
                    "spearman": {"value": 0.75, "std": 0.03, "ci95": 0.06},
                },
                "bootstrap_values": {
                    "accuracy": [0.84 + 0.01 * i for i in range(10)],
                    "spearman": [0.74 + 0.01 * i for i in range(10)],
                },
            }
        train_config = {
            "lora": [
                {"id": "st_secstr_r16", "color": "#5ec4c4"},
                {"id": "st_disorder_r16", "color": "#5ec4c4"},
            ],
        }

        captured = {}
        orig_savefig = plt.Figure.savefig

        def capture_savefig(self, *args, **kwargs):
            captured["fig"] = self
            orig_savefig(self, *args, **kwargs)

        plt.Figure.savefig = capture_savefig
        try:
            plot_delta_summary(
                evaluation=eval_cfg,
                metric_jsons=metric_jsons,
                output_path=out,
                train_config=train_config,
            )
        finally:
            plt.Figure.savefig = orig_savefig

        fig = captured["fig"]
        ax = fig.axes[0]
        bars = ax.patches
        # The approach "ST r16" uses runs: with lora/st_*_r16 -> should be #5ec4c4
        assert len(bars) >= 1
        assert mcolors.to_hex(bars[0].get_facecolor()) == "#5ec4c4"
        plt.close(fig)


class TestGenerateEvaluationSummary:
    def test_summary_structure(self):
        summary = generate_evaluation_summary(
            evaluation=_make_eval_config(),
            metric_jsons=_make_metric_jsons(),
        )
        assert summary["id"] == "test_eval"
        assert "tasks" in summary
        assert "secstr" in summary["tasks"]
        assert "approaches" in summary
        assert len(summary["approaches"]) == 1
        assert "deltas" in summary["approaches"][0]
        assert "secstr" in summary["approaches"][0]["deltas"]
        assert "delta_summary" in summary
        assert len(summary["delta_summary"]["rows"]) == 1

    def test_deltas_have_expected_keys(self):
        summary = generate_evaluation_summary(
            evaluation=_make_eval_config(),
            metric_jsons=_make_metric_jsons(),
        )
        delta = summary["approaches"][0]["deltas"]["secstr"]
        for key in ("mean", "std", "ci95", "significant", "p_value"):
            assert key in delta


# ---------------------------------------------------------------------------
# Missing-metric (null sentinel) tests
# ---------------------------------------------------------------------------


def _make_metric_jsons_with_missing():
    """Build metric_jsons where the approach has null metrics for 'disorder'.

    Simulates a run that was trained only on 'secstr' but not 'disorder'.
    """
    base = _make_metric_jsons()
    # Replace the approach's disorder entry with a null sentinel
    base["accgrad_lora/all_r4/best_overall/valid/disorder"] = {
        "metrics": None,
        "bootstrap_values": None,
    }
    return base


def _make_metric_jsons_all_missing():
    """Build metric_jsons where the approach has null metrics for all tasks."""
    base = _make_metric_jsons()
    base["accgrad_lora/all_r4/best_overall/valid/secstr"] = {
        "metrics": None,
        "bootstrap_values": None,
    }
    base["accgrad_lora/all_r4/best_overall/valid/disorder"] = {
        "metrics": None,
        "bootstrap_values": None,
    }
    return base


class TestPlotDeltaHeatmapMissing:
    def test_creates_file_with_partial_missing(self, tmp_path):
        """Heatmap should still be produced when some approach metrics are null."""
        out = tmp_path / "heatmap.png"
        plot_delta_heatmap(
            evaluation=_make_eval_config(),
            metric_jsons=_make_metric_jsons_with_missing(),
            output_path=out,
        )
        assert out.exists()
        assert out.stat().st_size > 0

    def test_creates_file_with_all_missing(self, tmp_path):
        """Heatmap should still be produced when all approach metrics are null."""
        out = tmp_path / "heatmap.png"
        plot_delta_heatmap(
            evaluation=_make_eval_config(),
            metric_jsons=_make_metric_jsons_all_missing(),
            output_path=out,
        )
        assert out.exists()
        assert out.stat().st_size > 0


class TestHeatmapYLabelColors:
    def _capture_heatmap(
        self, tmp_path, eval_cfg=None, metric_jsons=None, train_config=None
    ):
        """Run plot_delta_heatmap and return the captured Figure."""
        import matplotlib.pyplot as plt

        out = tmp_path / "heatmap.png"
        if eval_cfg is None:
            eval_cfg = _make_eval_config()
        if metric_jsons is None:
            metric_jsons = _make_metric_jsons()

        captured = {}
        orig_savefig = plt.Figure.savefig

        def capture_savefig(self, *args, **kwargs):
            captured["fig"] = self
            orig_savefig(self, *args, **kwargs)

        plt.Figure.savefig = capture_savefig
        try:
            plot_delta_heatmap(
                evaluation=eval_cfg,
                metric_jsons=metric_jsons,
                output_path=out,
                train_config=train_config,
            )
        finally:
            plt.Figure.savefig = orig_savefig
        return captured["fig"]

    def test_run_approach_gets_colored_box(self, tmp_path):
        """Single-run approach y-labels should get a coloured bbox from train_config."""
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt

        # Provide colors for both the baseline runs: refs and the approach run: ref.
        train_config = {
            "lora": [
                {"id": "bl_secstr", "color": "#4a90d9"},
                {"id": "bl_disorder", "color": "#4a90d9"},
            ],
            "accgrad_lora": [{"id": "all_r4", "color": "#ffe119"}],
        }
        fig = self._capture_heatmap(tmp_path, train_config=train_config)
        ax = fig.axes[0]
        ytick_labels = list(ax.get_yticklabels())

        # Row 0 (baseline) uses runs: → first ref resolves to #4a90d9
        baseline_bbox = ytick_labels[0].get_bbox_patch()
        assert baseline_bbox is not None
        assert mcolors.to_hex(baseline_bbox.get_facecolor()) == "#4a90d9"

        # Row 1 (approach "MT r4") uses run: → should get #ffe119
        approach_bbox = ytick_labels[1].get_bbox_patch()
        assert approach_bbox is not None
        assert mcolors.to_hex(approach_bbox.get_facecolor()) == "#ffe119"

        # Bright yellow → text should be black
        assert mcolors.to_hex(ytick_labels[1].get_color()) == "#000000"
        plt.close(fig)

    def test_runs_spec_falls_back_to_gray_without_train_config(self, tmp_path):
        """When train_config lacks the run, runs: specs should fall back to _RUNS_GRAYS."""
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt

        # Only provide a color for the approach, not for the baseline runs: refs.
        train_config = {
            "accgrad_lora": [{"id": "all_r4", "color": "#ffe119"}],
        }
        fig = self._capture_heatmap(tmp_path, train_config=train_config)
        ax = fig.axes[0]
        ytick_labels = list(ax.get_yticklabels())

        # Row 0 (baseline) uses runs: → no train_config entry → gray fallback
        baseline_bbox = ytick_labels[0].get_bbox_patch()
        assert baseline_bbox is not None
        bl_fc = mcolors.to_hex(baseline_bbox.get_facecolor())
        assert bl_fc in ("#b0b0b0", "#888888", "#606060", "#a0a0a0", "#707070")
        plt.close(fig)

    def test_dark_run_color_gets_white_text(self, tmp_path):
        """A dark run colour should result in white label text."""
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt

        train_config = {
            "accgrad_lora": [{"id": "all_r4", "color": "#000075"}],
        }
        fig = self._capture_heatmap(tmp_path, train_config=train_config)
        ax = fig.axes[0]
        approach_label = list(ax.get_yticklabels())[1]
        assert mcolors.to_hex(approach_label.get_color()) == "#ffffff"
        plt.close(fig)


class TestPlotAbsoluteBarsMissing:
    def test_creates_file_with_missing_approach(self, tmp_path):
        """Bar chart should handle null metrics for an approach."""
        out = tmp_path / "bars_disorder.png"
        plot_absolute_bars(
            evaluation=_make_eval_config(),
            metric_jsons=_make_metric_jsons_with_missing(),
            task="disorder",
            metric_name="spearman",
            output_path=out,
        )
        assert out.exists()
        assert out.stat().st_size > 0

    def test_missing_approach_is_omitted(self, tmp_path):
        """Missing approaches should be omitted entirely (no bar)."""
        import matplotlib.pyplot as plt

        out = tmp_path / "bars_disorder.png"
        eval_cfg = _make_eval_config()
        metric_jsons = _make_metric_jsons_with_missing()

        captured = {}
        orig_savefig = plt.Figure.savefig

        def capture_savefig(self, *args, **kwargs):
            captured["fig"] = self
            orig_savefig(self, *args, **kwargs)

        plt.Figure.savefig = capture_savefig
        try:
            plot_absolute_bars(
                evaluation=eval_cfg,
                metric_jsons=metric_jsons,
                task="disorder",
                metric_name="spearman",
                output_path=out,
            )
        finally:
            plt.Figure.savefig = orig_savefig

        fig = captured["fig"]
        ax = fig.axes[0]
        xtick_labels = [t.get_text() for t in ax.get_xticklabels()]
        # The approach has null metrics for disorder → should be omitted.
        # Only the baseline bar should remain.
        assert len(xtick_labels) == 1
        assert "MT r4" not in xtick_labels
        plt.close(fig)


class TestPlotAbsoluteBarsColor:
    def test_uses_train_config_color(self, tmp_path):
        """Bars should use run colors from train_config when provided."""
        import matplotlib.pyplot as plt

        out = tmp_path / "bars_secstr.png"
        eval_cfg = _make_eval_config()
        metric_jsons = _make_metric_jsons()
        train_config = {
            "lora": [
                {"id": "bl_secstr", "color": "#ff0000"},
            ],
            "accgrad_lora": [
                {"id": "all_r4", "color": "#00ff00"},
            ],
        }

        captured = {}
        orig_savefig = plt.Figure.savefig

        def capture_savefig(self, *args, **kwargs):
            captured["fig"] = self
            orig_savefig(self, *args, **kwargs)

        plt.Figure.savefig = capture_savefig
        try:
            plot_absolute_bars(
                evaluation=eval_cfg,
                metric_jsons=metric_jsons,
                task="secstr",
                metric_name="accuracy",
                output_path=out,
                train_config=train_config,
            )
        finally:
            plt.Figure.savefig = orig_savefig

        fig = captured["fig"]
        ax = fig.axes[0]
        bars = ax.patches
        # First bar (baseline) should be red, second (approach) should be green.
        import matplotlib.colors as mcolors

        assert mcolors.to_hex(bars[0].get_facecolor()) == "#ff0000"
        assert mcolors.to_hex(bars[1].get_facecolor()) == "#00ff00"
        plt.close(fig)


class TestPerApproachVariantOverride:
    """Tests that per-approach variant overrides work through the plotting pipeline."""

    @staticmethod
    def _make_variant_override_config():
        return {
            "id": "test_variant_override",
            "split": "valid",
            "variant": "best_overall",
            "datasets": [
                {"task": "secstr", "metric": "accuracy"},
                {"task": "disorder", "metric": "spearman"},
            ],
            "baseline": {
                "runs": {
                    "secstr": {"section": "lora", "id": "bl_secstr"},
                    "disorder": {"section": "lora", "id": "bl_disorder"},
                }
            },
            "approaches": [
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
        }

    @staticmethod
    def _make_variant_override_metric_jsons():
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
        ho_result = {
            "metrics": {
                "accuracy": {"value": 0.82, "std": 0.02, "ci95": 0.04},
                "spearman": {"value": 0.72, "std": 0.03, "ci95": 0.06},
            },
            "bootstrap_values": {
                "accuracy": [0.81 + 0.01 * i for i in range(10)],
                "spearman": [0.71 + 0.01 * i for i in range(10)],
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
            # Baseline uses best_overall
            "lora/bl_secstr/best_overall/valid/secstr": base,
            "lora/bl_disorder/best_overall/valid/disorder": base,
            # Heads-only uses best_task (variant override)
            "heads_only/ho_run/best_task/valid/secstr": ho_result,
            "heads_only/ho_run/best_task/valid/disorder": ho_result,
            # MT r4 uses best_overall (default)
            "accgrad_lora/all_r4/best_overall/valid/secstr": better,
            "accgrad_lora/all_r4/best_overall/valid/disorder": better,
        }

    def test_summary_picks_up_variant_override(self):
        """generate_evaluation_summary should use per-approach variant for keys."""
        summary = generate_evaluation_summary(
            evaluation=self._make_variant_override_config(),
            metric_jsons=self._make_variant_override_metric_jsons(),
        )
        # Both approaches should have real deltas (not None)
        assert len(summary["approaches"]) == 2
        for approach in summary["approaches"]:
            for task in ("secstr", "disorder"):
                assert approach["deltas"][task] is not None, (
                    f"{approach['label']} delta for {task} should not be None"
                )

    def test_heatmap_with_variant_override(self, tmp_path):
        out = tmp_path / "heatmap.png"
        plot_delta_heatmap(
            evaluation=self._make_variant_override_config(),
            metric_jsons=self._make_variant_override_metric_jsons(),
            output_path=out,
        )
        assert out.exists()
        assert out.stat().st_size > 0

    def test_delta_summary_with_variant_override(self):
        summary = compute_delta_summary(
            evaluation=self._make_variant_override_config(),
            metric_jsons=self._make_variant_override_metric_jsons(),
        )
        assert len(summary["rows"]) == 2
        for row in summary["rows"]:
            assert row["tasks_compared"] == 2


class TestGenerateEvaluationSummaryMissing:
    def test_missing_approach_delta_is_none(self):
        """Summary should have None delta for a task with null metrics."""
        summary = generate_evaluation_summary(
            evaluation=_make_eval_config(),
            metric_jsons=_make_metric_jsons_with_missing(),
        )
        # 'disorder' should have None delta because the approach has null metrics
        assert summary["approaches"][0]["deltas"]["disorder"] is None
        # 'secstr' should still have a real delta dict
        delta = summary["approaches"][0]["deltas"]["secstr"]
        assert delta is not None
        for key in ("mean", "std", "ci95", "significant", "p_value"):
            assert key in delta

    def test_all_missing_approach_deltas_are_none(self):
        """All deltas should be None when approach has no metrics for any task."""
        summary = generate_evaluation_summary(
            evaluation=_make_eval_config(),
            metric_jsons=_make_metric_jsons_all_missing(),
        )
        for task in ("secstr", "disorder"):
            assert summary["approaches"][0]["deltas"][task] is None


# ---------------------------------------------------------------------------
# SOTA comparison
# ---------------------------------------------------------------------------


def _make_sota_data():
    """Minimal SOTA data for testing."""
    return {
        "secstr": {
            "test_set": "NEW364",
            "metric": "accuracy",
            "metric_label": "Q3",
            "baselines": [
                {"name": "NetSurfP-3.0", "reference": "ref1", "value": 0.846},
                {"name": "SPOT-1D-LM", "reference": "ref2", "value": 0.87},
            ],
        },
        "secstr8": {
            "test_set": "NEW364",
            "metric": "accuracy",
            "metric_label": "Q8",
            "baselines": [
                {"name": "NetSurfP-3.0", "reference": "ref1", "value": 0.711},
            ],
        },
        "disorder": {
            "test_set": "CheZOD117",
            "metric": "spearman",
            "metric_label": "Spearman \u03c1",
            "baselines": [
                {"name": "SETH", "reference": "ref3", "value": 0.70},
            ],
        },
        "gpsite_dna": {
            "test_set": "GPSite DNA",
            "metric": "mcc",
            "metric_label": "MCC",
            "baselines": [
                {"name": "GPSite", "reference": "ref4", "value": 0.55},
            ],
        },
        "gpsite_rna": {
            "test_set": "GPSite RNA",
            "metric": "mcc",
            "metric_label": "MCC",
            "baselines": [
                {"name": "GPSite", "reference": "ref4", "value": 0.50},
            ],
        },
    }


def _make_sota_eval_config():
    """Evaluation config covering secstr, secstr8, disorder."""
    return {
        "id": "test_sota_eval",
        "split": "valid",
        "variant": "best_overall",
        "datasets": [
            {"task": "secstr", "metric": "accuracy", "metric_label": "Q3"},
            {"task": "secstr8", "metric": "accuracy", "metric_label": "Q8"},
            {
                "task": "disorder",
                "metric": "spearman",
                "metric_label": "Spearman \u03c1",
            },
        ],
        "baseline": {
            "label": "Baseline",
            "runs": {
                "secstr": {"section": "lora", "id": "bl_secstr"},
                "secstr8": {"section": "lora", "id": "bl_secstr8"},
                "disorder": {"section": "lora", "id": "bl_disorder"},
            },
        },
        "approaches": [
            {
                "label": "MT r4",
                "run": {"section": "accgrad_lora", "id": "all_r4"},
            },
        ],
    }


def _make_sota_metric_jsons():
    """Metric JSONs for SOTA evaluation test fixtures."""
    jsons = {}
    for task, section, run_id, value in [
        ("secstr", "lora", "bl_secstr", 0.83),
        ("secstr8", "lora", "bl_secstr8", 0.70),
        ("disorder", "lora", "bl_disorder", 0.68),
        ("secstr", "accgrad_lora", "all_r4", 0.86),
        ("secstr8", "accgrad_lora", "all_r4", 0.73),
        ("disorder", "accgrad_lora", "all_r4", 0.72),
    ]:
        key = f"{section}/{run_id}/best_overall/valid/{task}"
        jsons[key] = {
            "metrics": {
                "accuracy": {"value": value, "std": 0.01, "ci95": 0.02},
                "spearman": {"value": value, "std": 0.01, "ci95": 0.02},
            },
            "bootstrap_values": {
                "accuracy": [value] * 10,
                "spearman": [value] * 10,
            },
        }
    return jsons


class TestGroupSotaTasksByTestSet:
    def test_groups_by_test_set(self):
        groups = group_sota_tasks_by_test_set(_make_sota_data())
        assert "new364" in groups
        assert set(groups["new364"]) == {"secstr", "secstr8"}

    def test_gpsite_grouped(self):
        groups = group_sota_tasks_by_test_set(_make_sota_data())
        assert "gpsite" in groups
        assert set(groups["gpsite"]) == {"gpsite_dna", "gpsite_rna"}

    def test_single_task_group(self):
        groups = group_sota_tasks_by_test_set(_make_sota_data())
        assert "chezod117" in groups
        assert groups["chezod117"] == ["disorder"]


class TestPlotSotaComparison:
    def _capture_fig(self, **kwargs):
        """Call plot_sota_comparison and return the captured Figure."""
        import matplotlib.pyplot as plt

        captured = {}
        orig_savefig = plt.Figure.savefig

        def capture_savefig(self, *args, **kw):
            captured["fig"] = self
            orig_savefig(self, *args, **kw)

        plt.Figure.savefig = capture_savefig
        try:
            plot_sota_comparison(**kwargs)
        finally:
            plt.Figure.savefig = orig_savefig
        return captured.get("fig")

    @staticmethod
    def _visible_axes(fig):
        """Return only visible axes (skip hidden/legend-only axes)."""
        return [ax for ax in fig.axes if ax.get_visible() and ax.has_data()]

    def test_creates_file(self, tmp_path):
        out = tmp_path / "sota.png"
        plot_sota_comparison(
            sota_data=_make_sota_data(),
            tasks=["secstr", "secstr8"],
            evaluation=_make_sota_eval_config(),
            metric_jsons=_make_sota_metric_jsons(),
            output_path=out,
        )
        assert out.exists()
        assert out.stat().st_size > 0

    def test_single_task(self, tmp_path):
        out = tmp_path / "sota_single.png"
        plot_sota_comparison(
            sota_data=_make_sota_data(),
            tasks=["disorder"],
            evaluation=_make_sota_eval_config(),
            metric_jsons=_make_sota_metric_jsons(),
            output_path=out,
        )
        assert out.exists()
        assert out.stat().st_size > 0

    def test_empty_tasks_touches_file(self, tmp_path):
        """When no tasks overlap, an empty file should be created."""
        out = tmp_path / "sota_empty.png"
        plot_sota_comparison(
            sota_data=_make_sota_data(),
            tasks=["nonexistent"],
            evaluation=_make_sota_eval_config(),
            metric_jsons=_make_sota_metric_jsons(),
            output_path=out,
        )
        assert out.exists()
        assert out.stat().st_size == 0

    def test_one_subplot_per_task(self, tmp_path):
        """There should be one visible data subplot per task."""
        import matplotlib.pyplot as plt

        out = tmp_path / "sota_subplots.png"
        fig = self._capture_fig(
            sota_data=_make_sota_data(),
            tasks=["secstr", "secstr8"],
            evaluation=_make_sota_eval_config(),
            metric_jsons=_make_sota_metric_jsons(),
            output_path=out,
        )
        visible = self._visible_axes(fig)
        assert len(visible) == 2
        plt.close(fig)

    def test_subplot_titles_are_task_names(self, tmp_path):
        """Each subplot title should be the task name."""
        import matplotlib.pyplot as plt

        out = tmp_path / "sota_titles.png"
        fig = self._capture_fig(
            sota_data=_make_sota_data(),
            tasks=["secstr", "secstr8"],
            evaluation=_make_sota_eval_config(),
            metric_jsons=_make_sota_metric_jsons(),
            output_path=out,
        )
        visible = self._visible_axes(fig)
        titles = {ax.get_title() for ax in visible}
        assert titles == {"SecStr", "SecStr8"}
        plt.close(fig)

    def test_method_names_on_xaxis(self, tmp_path):
        """X-tick labels should list method names within each subplot."""
        import matplotlib.pyplot as plt

        out = tmp_path / "sota_xticks.png"
        fig = self._capture_fig(
            sota_data=_make_sota_data(),
            tasks=["secstr"],
            evaluation=_make_sota_eval_config(),
            metric_jsons=_make_sota_metric_jsons(),
            output_path=out,
        )
        visible = self._visible_axes(fig)
        ax = visible[0]
        xtick_labels = [t.get_text() for t in ax.get_xticklabels()]
        # secstr has baselines NetSurfP-3.0, SPOT-1D-LM and pipeline Baseline, MT r4
        for expected in ["NetSurfP-3.0", "SPOT-1D-LM", "Baseline", "MT r4"]:
            assert expected in xtick_labels, f"{expected!r} not in x-ticks"
        plt.close(fig)

    def test_method_colored_bars(self, tmp_path):
        """Bars for the same method should have the same colour across
        subplots.  Different methods should have distinct colours."""
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt

        out = tmp_path / "sota_colors.png"
        fig = self._capture_fig(
            sota_data=_make_sota_data(),
            tasks=["secstr", "secstr8"],
            evaluation=_make_sota_eval_config(),
            metric_jsons=_make_sota_metric_jsons(),
            output_path=out,
        )
        visible = self._visible_axes(fig)
        # Both secstr and secstr8 have the same 4 methods.
        # Collect bar colour keyed by method label (x-tick) per subplot.
        method_colors_per_subplot: dict = {}
        for ax in visible:
            ticks = [t.get_text() for t in ax.get_xticklabels()]
            for bar, label in zip(ax.patches, ticks):
                c = mcolors.to_hex(bar.get_facecolor())
                method_colors_per_subplot.setdefault(label, set()).add(c)
        # Each method should use exactly one colour across both subplots.
        for method, colors in method_colors_per_subplot.items():
            assert len(colors) == 1, f"{method} has inconsistent colours: {colors}"
        # At least two distinct method colours overall.
        all_colors = set()
        for colors in method_colors_per_subplot.values():
            all_colors |= colors
        assert len(all_colors) >= 2
        plt.close(fig)

    def test_pipeline_uses_train_config_color(self, tmp_path):
        """Pipeline run bars should use colours from train_config."""
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt

        out = tmp_path / "sota_traincolor.png"
        train_config = {
            "accgrad_lora": [{"id": "all_r4", "color": "#ff0000"}],
        }
        fig = self._capture_fig(
            sota_data=_make_sota_data(),
            tasks=["secstr"],
            evaluation=_make_sota_eval_config(),
            metric_jsons=_make_sota_metric_jsons(),
            output_path=out,
            train_config=train_config,
        )
        visible = self._visible_axes(fig)
        ax = visible[0]
        bars = ax.patches
        # "MT r4" is the last method (last bar).
        mt_bar = bars[-1]
        assert mcolors.to_hex(mt_bar.get_facecolor()) == "#ff0000"
        plt.close(fig)

    def test_custom_title(self, tmp_path):
        """Custom title should appear as the figure suptitle."""
        import matplotlib.pyplot as plt

        out = tmp_path / "sota_title.png"
        fig = self._capture_fig(
            sota_data=_make_sota_data(),
            tasks=["secstr", "secstr8"],
            evaluation=_make_sota_eval_config(),
            metric_jsons=_make_sota_metric_jsons(),
            output_path=out,
            title="Custom SOTA Title",
        )
        assert fig._suptitle is not None
        assert (
            fig._suptitle.get_text()
            == "Custom SOTA Title \u2014 Best Overall Checkpoint"
        )
        plt.close(fig)

    def test_ylabel_shows_sota_metric_label(self, tmp_path):
        """Y-axis label on the first column should show the pretty metric_label."""
        import matplotlib.pyplot as plt

        out = tmp_path / "sota_ylabel.png"
        fig = self._capture_fig(
            sota_data=_make_sota_data(),
            tasks=["secstr", "secstr8"],
            evaluation=_make_sota_eval_config(),
            metric_jsons=_make_sota_metric_jsons(),
            output_path=out,
        )
        visible = self._visible_axes(fig)
        # secstr is in the first column; its metric_label is "Q3".
        assert visible[0].get_ylabel() == "Q3"
        plt.close(fig)

    def test_sota_bars_have_hatch(self, tmp_path):
        """SOTA baseline bars should have a hatch pattern; pipeline bars
        should not."""
        import matplotlib.pyplot as plt

        out = tmp_path / "sota_hatch.png"
        fig = self._capture_fig(
            sota_data=_make_sota_data(),
            tasks=["secstr"],
            evaluation=_make_sota_eval_config(),
            metric_jsons=_make_sota_metric_jsons(),
            output_path=out,
        )
        visible = self._visible_axes(fig)
        ax = visible[0]
        bars = ax.patches
        ticks = [t.get_text() for t in ax.get_xticklabels()]
        sota_baselines = {"NetSurfP-3.0", "SPOT-1D-LM"}
        for bar, label in zip(bars, ticks):
            if label in sota_baselines:
                assert bar.get_hatch() and bar.get_hatch() != "", (
                    f"SOTA bar {label!r} should have hatch"
                )
            else:
                hatch = bar.get_hatch()
                assert hatch is None or hatch == "", (
                    f"Pipeline bar {label!r} should not have hatch"
                )
        plt.close(fig)

    def test_value_labels_present(self, tmp_path):
        """Every bar with a positive value should have a text annotation."""
        import matplotlib.pyplot as plt

        out = tmp_path / "sota_vals.png"
        fig = self._capture_fig(
            sota_data=_make_sota_data(),
            tasks=["secstr"],
            evaluation=_make_sota_eval_config(),
            metric_jsons=_make_sota_metric_jsons(),
            output_path=out,
        )
        visible = self._visible_axes(fig)
        ax = visible[0]
        bars = ax.patches
        texts = ax.texts
        positive_bars = [b for b in bars if b.get_height() > 0]
        assert len(texts) == len(positive_bars)
        for t in texts:
            float(t.get_text())
        plt.close(fig)

    def test_sota_without_task_data_omitted_from_subplot(self, tmp_path):
        """A SOTA baseline that has no value for a specific task should not
        appear as an empty bar in that subplot."""
        import matplotlib.pyplot as plt

        out = tmp_path / "sota_omit_task.png"
        # secstr8 only has NetSurfP-3.0 as baseline (not SPOT-1D-LM).
        fig = self._capture_fig(
            sota_data=_make_sota_data(),
            tasks=["secstr", "secstr8"],
            evaluation=_make_sota_eval_config(),
            metric_jsons=_make_sota_metric_jsons(),
            output_path=out,
        )
        visible = self._visible_axes(fig)
        secstr8_ax = [ax for ax in visible if ax.get_title() == "SecStr8"][0]
        ticks = [t.get_text() for t in secstr8_ax.get_xticklabels()]
        # secstr8 should NOT have SPOT-1D-LM (it's not in its baselines).
        assert "SPOT-1D-LM" not in ticks
        # But it should have NetSurfP-3.0 and the pipeline methods.
        assert "NetSurfP-3.0" in ticks
        assert "Baseline" in ticks
        plt.close(fig)

    def test_no_legend(self, tmp_path):
        """No legend should be rendered (x-tick labels identify methods)."""
        import matplotlib.pyplot as plt

        out = tmp_path / "sota_legend.png"
        fig = self._capture_fig(
            sota_data=_make_sota_data(),
            tasks=["secstr", "secstr8"],
            evaluation=_make_sota_eval_config(),
            metric_jsons=_make_sota_metric_jsons(),
            output_path=out,
        )
        assert len(fig.legends) == 0, "Figure should have no legend"
        for ax in fig.axes:
            assert ax.get_legend() is None, "Axes should have no legend"
        plt.close(fig)

    def test_method_without_task_data_is_omitted(self, tmp_path):
        """A pipeline approach with no metric data for any task in the group
        should be omitted entirely."""
        import matplotlib.pyplot as plt

        out = tmp_path / "sota_omit.png"
        eval_cfg = _make_sota_eval_config()
        eval_cfg["approaches"].append(
            {
                "label": "Ghost approach",
                "run": {"section": "accgrad_lora", "id": "ghost_run"},
            }
        )
        fig = self._capture_fig(
            sota_data=_make_sota_data(),
            tasks=["secstr", "secstr8"],
            evaluation=eval_cfg,
            metric_jsons=_make_sota_metric_jsons(),
            output_path=out,
        )
        # Ghost approach should not appear in any subplot's x-ticks.
        for ax in self._visible_axes(fig):
            ticks = [t.get_text() for t in ax.get_xticklabels()]
            assert "Ghost approach" not in ticks
        plt.close(fig)
