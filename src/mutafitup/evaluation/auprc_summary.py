"""AUPRC evaluation summary and bar chart generation.

Produces a summary JSON and a simple grouped bar chart comparing AUPRC
values across approaches for GPSite tasks.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib
import matplotlib.pyplot as plt

from mutafitup.task_display_names import task_display_name
import numpy as np

matplotlib.use("Agg")


def generate_auprc_summary(
    evaluation: Dict[str, Any],
    metrics_dir: str = "results/auprc_metrics",
) -> Dict[str, Any]:
    """Build a summary dict from AUPRC metric JSONs for an evaluation.

    Parameters
    ----------
    evaluation : dict
        An ``auprc_evaluations`` config entry.
    metrics_dir : str
        Base directory containing AUPRC metric JSONs.

    Returns
    -------
    dict
        Summary with structure::

            {
                "evaluation_id": "...",
                "split": "...",
                "tasks": ["gpsite_dna", ...],
                "baseline": {"label": "...", "results": {"gpsite_dna": {...}, ...}},
                "approaches": [
                    {"label": "...", "results": {"gpsite_dna": {...}, ...}},
                    ...
                ]
            }
    """
    split = evaluation["split"]
    default_variant = evaluation["variant"]
    tasks = evaluation["tasks"]

    def _spec_variant(spec: Dict[str, Any]) -> str:
        return spec.get("variant", default_variant)

    def _resolve_run_ref(spec: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
        if "run" in spec:
            ref = spec["run"]
            return {t: ref for t in tasks}
        return spec["runs"]

    def _load_results(spec: Dict[str, Any]) -> Dict[str, Optional[Dict[str, Any]]]:
        variant = _spec_variant(spec)
        run_map = _resolve_run_ref(spec)
        results: Dict[str, Optional[Dict[str, Any]]] = {}

        for task in tasks:
            ref = run_map[task]
            path = (
                Path(metrics_dir)
                / ref["section"]
                / ref["id"]
                / variant
                / split
                / f"{task}.json"
            )

            if path.exists():
                with open(path) as f:
                    data = json.load(f)
                if data.get("metrics") is not None:
                    results[task] = data["metrics"]["auprc"]
                else:
                    results[task] = None
            else:
                results[task] = None

        return results

    baseline = evaluation["baseline"]
    baseline_label = baseline.get("label", "Baseline")
    baseline_results = _load_results(baseline)

    approach_summaries = []
    for approach in evaluation["approaches"]:
        label = approach.get("label", "Approach")
        results = _load_results(approach)
        approach_summaries.append({"label": label, "results": results})

    return {
        "evaluation_id": evaluation["id"],
        "split": split,
        "tasks": tasks,
        "baseline": {"label": baseline_label, "results": baseline_results},
        "approaches": approach_summaries,
    }


def plot_auprc_bar_chart(
    summary: Dict[str, Any],
    output_path: Union[str, Path],
    figsize: tuple = (14, 6),
    dpi: int = 150,
) -> None:
    """Generate a grouped bar chart of AUPRC values from a summary dict.

    Parameters
    ----------
    summary : dict
        Output of :func:`generate_auprc_summary`.
    output_path : str or Path
        Where to save the PNG.
    figsize : tuple
        Figure size (width, height) in inches.
    dpi : int
        Output resolution.
    """
    tasks = summary["tasks"]
    n_tasks = len(tasks)

    # Collect all series: baseline + approaches
    all_series = [summary["baseline"]] + summary["approaches"]
    n_series = len(all_series)

    bar_width = 0.8 / n_series
    x = np.arange(n_tasks)

    fig, ax = plt.subplots(figsize=figsize)

    for i, series in enumerate(all_series):
        values = []
        errors = []
        for task in tasks:
            result = series["results"].get(task)
            if result is not None:
                values.append(result["value"])
                errors.append(result.get("ci95", 0.0))
            else:
                values.append(0.0)
                errors.append(0.0)

        offset = (i - n_series / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset,
            values,
            bar_width,
            yerr=errors,
            label=series["label"],
            capsize=2,
            alpha=0.85,
        )

    task_labels = [task_display_name(t) for t in tasks]
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels, rotation=45, ha="right")
    ax.set_ylabel("AUPRC")
    ax.set_title(f"AUPRC by Task — {summary['evaluation_id']} ({summary['split']})")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
