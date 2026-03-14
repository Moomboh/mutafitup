"""Evaluation plotting: delta heatmaps, absolute bar charts, and SOTA comparison.

Delta computation is done on-demand from per-run metric JSONs that
contain ``bootstrap_values`` arrays (paired comparison using the same
bootstrap indices).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle
from scipy import stats

from mutafitup.task_display_names import task_display_name

# Variant ID -> human-readable display name for plot titles.
_VARIANT_DISPLAY_NAMES: Dict[str, str] = {
    "best_overall": "Best Overall Checkpoint",
    "best_task": "Best Per-Task Checkpoint",
    "best_loss_overall": "Best Overall Loss Checkpoint",
    "best_loss_task": "Best Per-Task Loss Checkpoint",
}


def _evaluation_title(evaluation: Dict[str, Any]) -> str:
    """Build the plot title from evaluation config.

    Format: ``"{title} \u2014 {variant_display}"`` where the variant
    display name is looked up from :data:`_VARIANT_DISPLAY_NAMES`.
    Falls back to the raw variant string when not found.
    """
    base = evaluation.get("title", evaluation.get("id", "unnamed"))
    variant = evaluation.get("variant", "")
    variant_display = _VARIANT_DISPLAY_NAMES.get(variant, variant)
    if variant_display:
        return f"{base} \u2014 {variant_display}"
    return base


# Diverging colormap: red (negative) -> pale yellow (zero) -> blue (positive)
_DELTA_CMAP = LinearSegmentedColormap.from_list(
    "RdPaleYlBu",
    [(0.8, 0.15, 0.15), (1.0, 1.0, 0.75), (0.15, 0.35, 0.8)],
)

# Distinct grays for ``runs:``-based specs (per-task run maps) so they
# are visually distinguishable in the heatmap y-label boxes.
_RUNS_GRAYS = ["#b0b0b0", "#888888", "#606060", "#a0a0a0", "#707070"]

# Vibrant colours for SOTA literature baselines.  Hatching distinguishes
# them from pipeline bars, so overlap with pipeline run colours is fine.
_SOTA_BASELINE_COLORS = [
    "#e6194b",  # red
    "#3cb44b",  # green
    "#4363d8",  # blue
    "#f58231",  # orange
    "#911eb4",  # purple
    "#42d4f4",  # cyan
    "#f032e6",  # magenta
    "#bfef45",  # lime
]


def _text_color_for_bg(hex_color: str) -> str:
    """Return ``'#000000'`` or ``'#ffffff'`` for readable text on *hex_color*."""
    r, g, b = (int(hex_color[i : i + 2], 16) / 255.0 for i in (1, 3, 5))
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#000000" if luminance > 0.5 else "#ffffff"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_metric_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a per-run metric JSON file."""
    with open(path) as f:
        return json.load(f)


def _collect_absolute_values(
    evaluation: Dict[str, Any],
    metric_jsons: Dict[str, Dict[str, Any]],
) -> Tuple[List[str], List[str], np.ndarray, np.ndarray]:
    datasets = evaluation["datasets"]
    approaches = evaluation["approaches"]
    split = evaluation["split"]
    default_variant = evaluation["variant"]
    baseline = evaluation["baseline"]

    task_names = [ds["task"] for ds in datasets]
    task_metrics = {ds["task"]: ds["metric"] for ds in datasets}
    labels = [baseline.get("label", "Single-task LoRA r4\n(baseline)")] + [
        approach["label"] for approach in approaches
    ]

    values = np.full((len(labels), len(task_names)), np.nan, dtype=float)
    ci95s = np.full((len(labels), len(task_names)), np.nan, dtype=float)

    baseline_variant = _spec_variant(baseline, default_variant)
    baseline_runs = _resolve_run_ref_local(baseline, datasets)
    for j, task in enumerate(task_names):
        metric_name = task_metrics[task]
        ref = baseline_runs[task]
        key = f"{ref['section']}/{ref['id']}/{baseline_variant}/{split}/{task}"
        metric_container = metric_jsons.get(key, {}).get("metrics")
        if metric_container is None:
            continue
        metric_entry = metric_container.get(metric_name, {})
        values[0, j] = metric_entry.get("value", np.nan)
        ci95s[0, j] = metric_entry.get("ci95", np.nan)

    for i, approach in enumerate(approaches, start=1):
        approach_variant = _spec_variant(approach, default_variant)
        approach_runs = _resolve_run_ref_local(approach, datasets)
        for j, task in enumerate(task_names):
            metric_name = task_metrics[task]
            ref = approach_runs[task]
            key = f"{ref['section']}/{ref['id']}/{approach_variant}/{split}/{task}"
            metric_container = metric_jsons.get(key, {}).get("metrics")
            if metric_container is None:
                continue
            metric_entry = metric_container.get(metric_name, {})
            values[i, j] = metric_entry.get("value", np.nan)
            ci95s[i, j] = metric_entry.get("ci95", np.nan)

    return labels, task_names, values, ci95s


def compute_delta_summary(
    evaluation: Dict[str, Any],
    metric_jsons: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute aggregate delta statistics across tasks for each approach.

    Per-task deltas and significance are computed via
    :func:`compute_pairwise_delta`.  Aggregate statistics
    (``mean_delta_percent``, ``std_delta_percent``,
    ``ci95_mean_delta_percent``) are derived from a bootstrap-level
    computation: for each bootstrap replicate *b*, the per-task relative
    deltas at index *b* are averaged across tasks (NaN-aware), yielding a
    bootstrap distribution of the "mean relative improvement" statistic.
    Mean, std, and CI are then taken from this distribution.

    ``median_delta_percent`` is the median of the per-task mean deltas
    (not bootstrap-derived) and serves as an indicator of task-level
    skewness.
    """
    datasets = evaluation["datasets"]
    approaches = evaluation["approaches"]
    split = evaluation["split"]
    default_variant = evaluation["variant"]
    baseline = evaluation["baseline"]

    task_names = [ds["task"] for ds in datasets]
    task_metrics = {ds["task"]: ds["metric"] for ds in datasets}
    baseline_variant = _spec_variant(baseline, default_variant)
    baseline_runs = _resolve_run_ref_local(baseline, datasets)

    rows = []
    for approach in approaches:
        approach_variant = _spec_variant(approach, default_variant)
        approach_runs = _resolve_run_ref_local(approach, datasets)
        per_task: Dict[str, Any] = {}
        task_deltas: List[float] = []
        significant_wins = 0
        significant_losses = 0
        nonsignificant = 0

        # Collect raw bootstrap arrays per task for bootstrap-level aggregation.
        task_boot_pairs: List[Tuple[List[float], List[float]]] = []

        for task in task_names:
            metric_name = task_metrics[task]
            baseline_ref = baseline_runs[task]
            approach_ref = approach_runs[task]

            baseline_key = f"{baseline_ref['section']}/{baseline_ref['id']}/{baseline_variant}/{split}/{task}"
            approach_key = f"{approach_ref['section']}/{approach_ref['id']}/{approach_variant}/{split}/{task}"
            baseline_json = metric_jsons.get(baseline_key, {})
            approach_json = metric_jsons.get(approach_key, {})

            if (
                baseline_json.get("metrics") is None
                or approach_json.get("metrics") is None
            ):
                per_task[task] = None
                continue

            baseline_boot = (baseline_json.get("bootstrap_values") or {}).get(
                metric_name, []
            )
            approach_boot = (approach_json.get("bootstrap_values") or {}).get(
                metric_name, []
            )
            if baseline_boot and approach_boot:
                delta = compute_pairwise_delta(
                    baseline_boot, approach_boot, percent=True
                )
                per_task[task] = delta
                task_deltas.append(delta["mean"])
                task_boot_pairs.append((baseline_boot, approach_boot))
                if delta["significant"]:
                    if delta["mean"] > 0:
                        significant_wins += 1
                    elif delta["mean"] < 0:
                        significant_losses += 1
                else:
                    nonsignificant += 1
            else:
                per_task[task] = None

        # ---- Bootstrap-level aggregate statistics ----
        # For each bootstrap replicate, compute the per-task relative delta
        # and average across tasks.  This gives a proper bootstrap
        # distribution of the "mean relative improvement" statistic, from
        # which we derive mean, std, and CI.
        if task_boot_pairs:
            n_boot = len(task_boot_pairs[0][0])
            # (n_tasks, n_boot) matrix; NaN where a replicate is invalid.
            delta_matrix = np.full((len(task_boot_pairs), n_boot), np.nan, dtype=float)
            for j, (b_boot, a_boot) in enumerate(task_boot_pairs):
                for b in range(min(len(b_boot), len(a_boot))):
                    bv = b_boot[b]
                    av = a_boot[b]
                    if not math.isnan(bv) and not math.isnan(av) and abs(bv) >= 1e-12:
                        delta_matrix[j, b] = (av - bv) / abs(bv) * 100.0

            # Per-replicate mean across tasks (NaN-aware).
            with np.errstate(all="ignore"):
                mean_per_boot = np.nanmean(delta_matrix, axis=0)

            # Drop replicates where no task had a valid delta.
            valid_mask = ~np.isnan(mean_per_boot)
            valid_means = mean_per_boot[valid_mask]

            if valid_means.size > 0:
                mean_delta = float(valid_means.mean())
                std_delta = float(valid_means.std(ddof=0))
                ci95_mean = float(1.96 * std_delta)
            else:
                mean_delta = math.nan
                std_delta = math.nan
                ci95_mean = math.nan

            # Median of per-task mean deltas (not bootstrap-derived).
            median_delta = float(np.median(task_deltas))
        else:
            mean_delta = math.nan
            median_delta = math.nan
            std_delta = math.nan
            ci95_mean = math.nan

        rows.append(
            {
                "label": approach["label"],
                "tasks_compared": len(task_deltas),
                "mean_delta_percent": None if math.isnan(mean_delta) else mean_delta,
                "median_delta_percent": None
                if math.isnan(median_delta)
                else median_delta,
                "std_delta_percent": None if math.isnan(std_delta) else std_delta,
                "ci95_mean_delta_percent": None if math.isnan(ci95_mean) else ci95_mean,
                "significant_wins": significant_wins,
                "significant_losses": significant_losses,
                "nonsignificant": nonsignificant,
                "per_task_deltas": per_task,
            }
        )

    rows.sort(
        key=lambda row: (
            float("-inf")
            if row["mean_delta_percent"] is None
            else -row["mean_delta_percent"],
            row["label"],
        )
    )

    return {
        "baseline_label": baseline.get("label", "Single-task LoRA r4\n(baseline)"),
        "tasks": task_names,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# On-demand pairwise delta computation
# ---------------------------------------------------------------------------


def compute_pairwise_delta(
    baseline_bootstrap: List[float],
    approach_bootstrap: List[float],
    percent: bool = False,
) -> Dict[str, Any]:
    """Compute paired delta statistics from bootstrap value arrays.

    Parameters
    ----------
    baseline_bootstrap : list of float
        Bootstrap metric values for the baseline run.
    approach_bootstrap : list of float
        Bootstrap metric values for the approach run (same length, paired).
    percent : bool
        If True, express deltas as percentage relative to baseline.

    Returns
    -------
    dict with keys: mean, std, ci95, significant, p_value
    """
    assert len(baseline_bootstrap) == len(approach_bootstrap), (
        f"Bootstrap arrays must have same length, got "
        f"{len(baseline_bootstrap)} vs {len(approach_bootstrap)}"
    )

    deltas = []
    for b, a in zip(baseline_bootstrap, approach_bootstrap):
        if math.isnan(b) or math.isnan(a):
            continue
        if percent:
            if abs(b) < 1e-12:
                continue
            deltas.append((a - b) / abs(b) * 100.0)
        else:
            deltas.append(a - b)

    if not deltas:
        return {
            "mean": 0.0,
            "std": 0.0,
            "ci95": 0.0,
            "significant": False,
            "p_value": 1.0,
        }

    arr = np.array(deltas)
    mean = float(arr.mean())
    std = float(arr.std(ddof=0))
    ci95 = 1.96 * std
    significant = abs(mean) > ci95

    # Paired t-test
    if std > 0:
        t_stat = mean / (std / math.sqrt(len(deltas)))
        p_value = float(2.0 * stats.t.sf(abs(t_stat), df=len(deltas) - 1))
    else:
        p_value = 1.0 if abs(mean) < 1e-12 else 0.0

    return {
        "mean": mean,
        "std": std,
        "ci95": ci95,
        "significant": significant,
        "p_value": p_value,
    }


# ---------------------------------------------------------------------------
# Delta heatmap
# ---------------------------------------------------------------------------


def plot_delta_heatmap(
    evaluation: Dict[str, Any],
    metric_jsons: Dict[str, Dict[str, Any]],
    output_path: Union[str, Path],
    figsize: Optional[Tuple[float, float]] = None,
    train_config: Optional[Dict[str, Any]] = None,
) -> None:
    """Render and save a delta heatmap for an evaluation.

    The first row shows the baseline with absolute metric values on a
    gray background.  Subsequent rows show approach deltas (%) relative
    to the baseline, colour-mapped with a diverging red-blue scheme
    (red = regression, pale yellow = no change, blue = improvement).

    Parameters
    ----------
    evaluation : dict
        The evaluation config entry.
    metric_jsons : dict
        Mapping ``"{section}/{run_id}/{variant}/{split}/{task}"`` to the
        loaded JSON dict.
    output_path : str or Path
        Where to save the PNG.
    figsize : tuple, optional
        Figure size.
    train_config : dict, optional
        The ``config["train"]`` section.  When provided, y-axis labels
        get a coloured background box derived from the referenced
        training run's ``color`` field.
    """
    datasets = evaluation["datasets"]
    approaches = evaluation["approaches"]
    split = evaluation["split"]
    default_variant = evaluation["variant"]
    baseline = evaluation["baseline"]

    task_names = [ds["task"] for ds in datasets]
    task_metrics = {ds["task"]: ds["metric"] for ds in datasets}
    task_labels = {ds["task"]: ds.get("metric_label", ds["metric"]) for ds in datasets}
    approach_labels = [a["label"] for a in approaches]

    n_approaches = len(approaches)
    n_tasks = len(task_names)
    n_cols = n_tasks + 1  # task columns + summary column
    n_rows = n_approaches + 1  # baseline row + approach rows

    if figsize is None:
        figsize = (max(5.0, n_cols * 1.25 + 1.7), max(2.5, n_rows * 0.67 + 1.7))

    # ---- Compute bootstrap-level aggregate deltas per approach ----
    delta_summary = compute_delta_summary(evaluation, metric_jsons)
    summary_by_label: Dict[str, Dict[str, Any]] = {
        row["label"]: row for row in delta_summary["rows"]
    }

    # Resolve baseline run refs
    baseline_variant = _spec_variant(baseline, default_variant)
    baseline_runs = _resolve_run_ref_local(baseline, datasets)

    # ---- Collect baseline absolute values ----
    baseline_values = np.zeros(n_tasks)
    baseline_ci95s = np.zeros(n_tasks)
    baseline_missing = np.zeros(n_tasks, dtype=bool)

    for j, task in enumerate(task_names):
        metric_name = task_metrics[task]
        ref = baseline_runs[task]
        key = f"{ref['section']}/{ref['id']}/{baseline_variant}/{split}/{task}"
        bj = metric_jsons.get(key, {})
        bm_container = bj.get("metrics")
        if bm_container is None:
            baseline_missing[j] = True
        else:
            bm = bm_container.get(metric_name, {})
            baseline_values[j] = bm.get("value", 0.0)
            baseline_ci95s[j] = bm.get("ci95", 0.0)

    # ---- Build delta matrix for approach rows ----
    delta_means = np.zeros((n_approaches, n_tasks))
    delta_ci95s = np.zeros((n_approaches, n_tasks))
    significant = np.zeros((n_approaches, n_tasks), dtype=bool)
    approach_missing = np.zeros((n_approaches, n_tasks), dtype=bool)

    for i, approach in enumerate(approaches):
        approach_variant = _spec_variant(approach, default_variant)
        approach_runs = _resolve_run_ref_local(approach, datasets)
        for j, task in enumerate(task_names):
            metric_name = task_metrics[task]

            baseline_ref = baseline_runs[task]
            approach_ref = approach_runs[task]

            baseline_key = f"{baseline_ref['section']}/{baseline_ref['id']}/{baseline_variant}/{split}/{task}"
            approach_key = f"{approach_ref['section']}/{approach_ref['id']}/{approach_variant}/{split}/{task}"

            baseline_json = metric_jsons.get(baseline_key, {})
            approach_json = metric_jsons.get(approach_key, {})

            # Check for sentinel null metrics (run not trained on this task)
            if (
                baseline_json.get("metrics") is None
                or approach_json.get("metrics") is None
            ):
                approach_missing[i, j] = True
                continue

            baseline_boot = (baseline_json.get("bootstrap_values") or {}).get(
                metric_name, []
            )
            approach_boot = (approach_json.get("bootstrap_values") or {}).get(
                metric_name, []
            )

            if baseline_boot and approach_boot:
                delta = compute_pairwise_delta(
                    baseline_boot, approach_boot, percent=True
                )
                delta_means[i, j] = delta["mean"]
                delta_ci95s[i, j] = delta["ci95"]
                significant[i, j] = delta["significant"]
            else:
                approach_missing[i, j] = True

    # ---- Plot ----
    # Layout: column 0 = "Mean Δ" summary, columns 1..n_tasks = per-task.
    summary_col = 0  # x-position of the summary column
    task_x_offset = 1  # per-task columns start here

    fig, ax = plt.subplots(figsize=figsize)

    # Mask missing approach cells so they don't affect the colour range
    valid_deltas = delta_means[~approach_missing]
    if valid_deltas.size > 0:
        abs_max = max(abs(valid_deltas.min()), abs(valid_deltas.max()), 1e-6)
    else:
        abs_max = 1e-6
    norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
    cmap = _DELTA_CMAP

    # Draw the delta heatmap for approach rows (rows 1..n_rows-1).
    # extent covers only the per-task columns (starting at x=0.5);
    # the summary column at x=0 is rendered separately.
    im = ax.imshow(
        delta_means,
        cmap=cmap,
        norm=norm,
        aspect="auto",
        extent=(
            task_x_offset - 0.5,
            task_x_offset + n_tasks - 0.5,
            n_rows - 0.5,
            0.5,
        ),
    )

    # Draw gray background rectangles for the baseline row (row 0),
    # including the summary column.
    for j in range(n_cols):
        ax.add_patch(
            Rectangle(
                (j - 0.5, -0.5),
                1,
                1,
                facecolor="#d9d9d9",
                edgecolor="white",
                linewidth=1,
            )
        )

    # Overlay gray rectangles on missing approach cells
    for i in range(n_approaches):
        for j in range(n_tasks):
            if approach_missing[i, j]:
                ax.add_patch(
                    Rectangle(
                        (task_x_offset + j - 0.5, i + 0.5),
                        1,
                        1,
                        facecolor="#d9d9d9",
                        edgecolor="white",
                        linewidth=1,
                    )
                )

    # ---- Summary column (approach rows) — at x=0 ----
    for i, approach in enumerate(approaches):
        label = approach["label"]
        srow = summary_by_label.get(label)
        if srow is None or srow["mean_delta_percent"] is None:
            # No valid summary — gray cell
            ax.add_patch(
                Rectangle(
                    (summary_col - 0.5, i + 0.5),
                    1,
                    1,
                    facecolor="#d9d9d9",
                    edgecolor="white",
                    linewidth=1,
                )
            )
            ax.text(
                summary_col,
                i + 1,
                "N/A",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="normal",
                color="#888888",
            )
        else:
            s_mean = srow["mean_delta_percent"]
            s_ci95 = srow["ci95_mean_delta_percent"] or 0.0
            s_sig = abs(s_mean) > s_ci95
            # Color-mapped rectangle
            ax.add_patch(
                Rectangle(
                    (summary_col - 0.5, i + 0.5),
                    1,
                    1,
                    facecolor=cmap(norm(s_mean)),
                    edgecolor="white",
                    linewidth=1,
                )
            )
            text = f"{s_mean:+.1f}%\n\u00b1{s_ci95:.1f}"
            weight = "bold" if s_sig else "normal"
            alpha = 1.0 if s_sig else 0.5
            ax.text(
                summary_col,
                i + 1,
                text,
                ha="center",
                va="center",
                fontsize=10,
                fontweight=weight,
                alpha=alpha,
            )

    # Separator lines
    ax.axhline(y=0.5, color="black", linewidth=1.5)
    ax.axvline(x=task_x_offset - 0.5, color="black", linewidth=1.5)

    # ---- Axis labels ----
    row_labels = [
        baseline.get("label", "Single-task LoRA r4\n(baseline)")
    ] + approach_labels
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(
        ["Mean \u0394"] + [task_display_name(t) for t in task_names],
        rotation=45,
        ha="right",
        fontsize=13,
    )
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=13)
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)

    # ---- Coloured boxes behind y-axis labels ----
    # For ``run:`` specs (single multi-task run) use the run's colour;
    # for ``runs:`` specs (per-task run maps) try the first run's colour
    # from train_config, falling back to distinct grays.
    specs = [baseline] + list(approaches)
    runs_gray_idx = 0
    row_colors: List[str] = []
    for spec in specs:
        if "run" in spec:
            row_colors.append(_resolve_run_color(spec["run"], train_config, "#cccccc"))
        else:
            first_ref = next(iter(spec["runs"].values()))
            color = _resolve_run_color(first_ref, train_config, "")
            if color:
                row_colors.append(color)
            else:
                row_colors.append(_RUNS_GRAYS[runs_gray_idx % len(_RUNS_GRAYS)])
                runs_gray_idx += 1

    for tick_label, bg_color in zip(ax.get_yticklabels(), row_colors):
        tick_label.set_bbox(
            dict(facecolor=bg_color, edgecolor="none", boxstyle="round,pad=0.3")
        )
        tick_label.set_color(_text_color_for_bg(bg_color))

    # ---- Annotate baseline cells (row 0) with absolute values ----
    for j in range(n_tasks):
        label = task_labels[task_names[j]]
        x = task_x_offset + j
        if baseline_missing[j]:
            text = f"{label}\nN/A"
            ax.text(
                x,
                0,
                text,
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="#888888",
            )
        else:
            val = baseline_values[j]
            ci = baseline_ci95s[j]
            text = f"{label}\n{val:.3f}\n\u00b1{ci:.3f}"
            ax.text(
                x,
                0,
                text,
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
            )

    # ---- Annotate approach cells (rows 1..n_rows-1) with deltas ----
    for i in range(n_approaches):
        for j in range(n_tasks):
            x = task_x_offset + j
            if approach_missing[i, j]:
                ax.text(
                    x,
                    i + 1,
                    "N/A",
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="normal",
                    color="#888888",
                )
            else:
                mean = delta_means[i, j]
                ci = delta_ci95s[i, j]
                is_sig = significant[i, j]
                text = f"{mean:+.1f}%\n\u00b1{ci:.1f}"
                weight = "bold" if is_sig else "normal"
                alpha = 1.0 if is_sig else 0.5
                ax.text(
                    x,
                    i + 1,
                    text,
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight=weight,
                    alpha=alpha,
                )

    fig.colorbar(im, ax=ax, label="% delta vs baseline", shrink=0.8)
    ax.set_title(_evaluation_title(evaluation), fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Absolute bar charts
# ---------------------------------------------------------------------------


def _resolve_run_color(
    ref: Dict[str, str],
    train_config: Optional[Dict[str, Any]],
    fallback: str = "#999999",
) -> str:
    """Look up a training run's color from the train config.

    Parameters
    ----------
    ref : dict
        Run reference with ``"section"`` and ``"id"`` keys.
    train_config : dict or None
        The ``config["train"]`` mapping (section → list of run configs).
    fallback : str
        Hex color to use when the lookup fails.
    """
    if train_config is None:
        return fallback
    section = ref.get("section", "")
    run_id = ref.get("id", "")
    for run in train_config.get(section, []):
        if run.get("id") == run_id:
            return run.get("color", fallback)
    return fallback


def plot_absolute_bars(
    evaluation: Dict[str, Any],
    metric_jsons: Dict[str, Dict[str, Any]],
    task: str,
    metric_name: str,
    output_path: Union[str, Path],
    figsize: Optional[Tuple[float, float]] = None,
    train_config: Optional[Dict[str, Any]] = None,
) -> None:
    """Render absolute metric bar chart for a single task.

    Shows the baseline and all approaches with 95% CI error bars.
    Approaches whose metrics are missing (not trained on the task)
    are omitted entirely.

    Parameters
    ----------
    evaluation : dict
        The evaluation config entry.
    metric_jsons : dict
        Same key format as :func:`plot_delta_heatmap`.
    task : str
        Which task to plot.
    metric_name : str
        Which metric to plot.
    output_path : str or Path
        Where to save the PNG.
    figsize : tuple, optional
    train_config : dict, optional
        The ``config["train"]`` section.  When provided, each bar is
        colored with the referenced training run's ``color`` field.
    """
    split = evaluation["split"]
    default_variant = evaluation["variant"]
    datasets = evaluation["datasets"]
    baseline = evaluation["baseline"]
    approaches = evaluation["approaches"]

    baseline_variant = _spec_variant(baseline, default_variant)
    baseline_runs = _resolve_run_ref_local(baseline, datasets)

    # Collect values — skip entries with missing metrics entirely.
    labels: List[str] = []
    values: List[float] = []
    errors: List[float] = []
    colors: List[str] = []

    baseline_ref = baseline_runs[task]
    baseline_key = f"{baseline_ref['section']}/{baseline_ref['id']}/{baseline_variant}/{split}/{task}"
    baseline_json = metric_jsons.get(baseline_key, {})
    baseline_m_container = baseline_json.get("metrics")
    if baseline_m_container is not None:
        baseline_m = baseline_m_container.get(metric_name, {})
        labels.append(baseline.get("label", "Single-task LoRA r4\n(baseline)"))
        values.append(baseline_m.get("value", 0.0))
        errors.append(baseline_m.get("ci95", 0.0))
        colors.append(_resolve_run_color(baseline_ref, train_config))

    for i, approach in enumerate(approaches):
        approach_variant = _spec_variant(approach, default_variant)
        approach_runs = _resolve_run_ref_local(approach, datasets)
        ref = approach_runs[task]
        key = f"{ref['section']}/{ref['id']}/{approach_variant}/{split}/{task}"
        aj = metric_jsons.get(key, {})
        am_container = aj.get("metrics")
        if am_container is None:
            continue  # Omit approaches not trained on this task.
        am = am_container.get(metric_name, {})
        labels.append(approach["label"])
        values.append(am.get("value", 0.0))
        errors.append(am.get("ci95", 0.0))
        colors.append(_resolve_run_color(ref, train_config, fallback=f"C{i}"))

    if figsize is None:
        figsize = (max(4, len(labels) * 1.2), 4)

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(labels))
    bars = ax.bar(x, values, yerr=errors, capsize=4, color=colors, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    # Look up metric_label from datasets config; fall back to metric_name.
    metric_label = metric_name
    for ds in datasets:
        if ds["task"] == task:
            metric_label = ds.get("metric_label", metric_name)
            break
    ax.set_ylabel(metric_label)
    variant = evaluation.get("variant", "")
    variant_display = _VARIANT_DISPLAY_NAMES.get(variant, variant)
    bar_title = f"{task_display_name(task)} \u2014 {metric_label}"
    if variant_display:
        bar_title = f"{bar_title} \u2014 {variant_display}"
    ax.set_title(bar_title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_delta_summary(
    evaluation: Dict[str, Any],
    metric_jsons: Dict[str, Dict[str, Any]],
    output_path: Union[str, Path],
    figsize: Optional[Tuple[float, float]] = None,
    train_config: Optional[Dict[str, Any]] = None,
) -> None:
    """Render a horizontal bar chart of mean delta vs baseline per approach.

    Each bar is colored by the training run's color (from *train_config*)
    using the same logic as the heatmap y-axis label boxes:
    ``run:`` specs get the run's color, ``runs:`` specs get distinct grays.

    Parameters
    ----------
    evaluation : dict
        The evaluation config entry.
    metric_jsons : dict
        Same key format as :func:`plot_delta_heatmap`.
    output_path : str or Path
        Where to save the PNG.
    figsize : tuple, optional
    train_config : dict, optional
        The ``config["train"]`` section (section -> list of run configs).
    """
    delta_summary = compute_delta_summary(evaluation, metric_jsons)
    rows = delta_summary["rows"]

    approaches = evaluation["approaches"]

    # Build label -> approach spec mapping for color resolution.
    approach_by_label = {a["label"]: a for a in approaches}

    # Filter out approaches with ``delta_summary: false``.
    excluded_labels = {
        a["label"] for a in approaches if a.get("delta_summary") is False
    }
    rows = [row for row in rows if row["label"] not in excluded_labels]

    if not rows:
        # All rows excluded or no data — write an empty figure placeholder.
        Path(output_path).touch()
        return

    labels = [row["label"] for row in rows]
    means = [
        row["mean_delta_percent"] if row["mean_delta_percent"] is not None else math.nan
        for row in rows
    ]
    errors = [
        row["ci95_mean_delta_percent"]
        if row["ci95_mean_delta_percent"] is not None
        else 0.0
        for row in rows
    ]

    # Resolve bar colors from train_config (same logic as heatmap y-labels).
    runs_gray_idx = 0
    colors: List[str] = []
    for label in labels:
        spec = approach_by_label.get(label)
        if spec is not None and "run" in spec:
            colors.append(_resolve_run_color(spec["run"], train_config, "#cccccc"))
        elif spec is not None and "runs" in spec:
            first_ref = next(iter(spec["runs"].values()))
            color = _resolve_run_color(first_ref, train_config, "")
            if color:
                colors.append(color)
            else:
                colors.append(_RUNS_GRAYS[runs_gray_idx % len(_RUNS_GRAYS)])
                runs_gray_idx += 1
        else:
            colors.append(_RUNS_GRAYS[runs_gray_idx % len(_RUNS_GRAYS)])
            runs_gray_idx += 1

    if figsize is None:
        figsize = (9, max(2.5, 0.45 * len(labels) + 1.2))

    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(len(labels))

    ax.barh(y, means, xerr=errors, color=colors, alpha=0.85, capsize=3)
    ax.axvline(0.0, color="black", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Mean delta vs baseline (%)")
    ax.set_title(_evaluation_title(evaluation))
    ax.grid(axis="x", alpha=0.25)

    finite_means = np.array([m for m in means if not math.isnan(m)], dtype=float)
    finite_errors = np.array(
        [e for m, e in zip(means, errors) if not math.isnan(m)], dtype=float
    )
    if finite_means.size > 0:
        extent = np.max(np.abs(finite_means) + finite_errors)
        extent = max(extent, 1.0)
        ax.set_xlim(-extent * 1.15, extent * 1.15)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# SOTA comparison
# ---------------------------------------------------------------------------


def group_sota_tasks_by_test_set(
    sota_data: Dict[str, Any],
) -> Dict[str, List[str]]:
    """Group SOTA task names by their ``test_set`` field.

    Returns a dict mapping a normalised group key (e.g. ``"new364"``,
    ``"gpsite"``) to the list of task names that share that test set.

    GPSite tasks are all grouped under one ``"gpsite"`` key regardless
    of their individual ``test_set`` value (e.g. "GPSite DNA",
    "GPSite RNA").
    """
    groups: Dict[str, List[str]] = {}
    for task_name, task_info in sota_data.items():
        test_set: str = task_info.get("test_set", task_name)
        if test_set.lower().startswith("gpsite"):
            key = "gpsite"
        else:
            # Normalise: lowercase, replace "/" and spaces with "_"
            key = test_set.lower().replace("/", "_").replace(" ", "_")
        groups.setdefault(key, []).append(task_name)
    return groups


def plot_sota_comparison(
    sota_data: Dict[str, Any],
    tasks: List[str],
    evaluation: Dict[str, Any],
    metric_jsons: Dict[str, Dict[str, Any]],
    output_path: Union[str, Path],
    train_config: Optional[Dict[str, Any]] = None,
    title: Optional[str] = None,
    max_cols: int = 4,
) -> None:
    """Plot a SOTA comparison as a grid of subplots (one per task).

    Each subplot shows SOTA baselines and pipeline runs as bars on the
    x-axis for a single task.  Methods that have no data for a specific
    task are omitted from that subplot.  SOTA baseline bars use hatching
    to visually distinguish them from pipeline runs.

    Parameters
    ----------
    sota_data : dict
        The ``tasks`` section of ``sota_metrics.yml``.
    tasks : list[str]
        Subset of task names to include (must all be keys in *sota_data*).
    evaluation : dict
        The evaluation config entry.
    metric_jsons : dict
        Same key format as :func:`plot_delta_heatmap`.
    output_path : str or Path
        Where to save the PNG.
    train_config : dict, optional
        The ``config["train"]`` section.
    title : str, optional
        Figure title.  Falls back to the test set name of the first task.
    max_cols : int
        Maximum number of subplot columns (default 4).
    """
    split = evaluation["split"]
    default_variant = evaluation["variant"]
    datasets = evaluation["datasets"]
    baseline_spec = evaluation["baseline"]
    approaches = evaluation["approaches"]

    baseline_variant = _spec_variant(baseline_spec, default_variant)

    eval_task_set = {ds["task"] for ds in datasets}
    tasks = [t for t in tasks if t in eval_task_set and t in sota_data]
    if not tasks:
        Path(output_path).touch()
        return

    n_tasks = len(tasks)

    # --- Collect ordered method names (SOTA baselines, then pipeline) ---
    all_baseline_names: List[str] = []
    for task in tasks:
        for b in sota_data[task].get("baselines", []):
            if b["name"] not in all_baseline_names:
                all_baseline_names.append(b["name"])

    pipeline_entries: List[Dict[str, Any]] = [
        {
            "label": baseline_spec.get("label", "Baseline"),
            "spec": baseline_spec,
            "variant": baseline_variant,
        }
    ]
    for approach in approaches:
        pipeline_entries.append(
            {
                "label": approach.get("label", "?"),
                "spec": approach,
                "variant": _spec_variant(approach, default_variant),
            }
        )

    all_method_names: List[str] = list(all_baseline_names)
    for entry in pipeline_entries:
        all_method_names.append(entry["label"])
    sota_name_set = set(all_baseline_names)

    # --- Build data matrix: method -> task -> (value, ci) ---
    method_data: Dict[str, Dict[str, Tuple[float, float]]] = {
        m: {} for m in all_method_names
    }
    for task in tasks:
        task_sota = sota_data[task]
        metric_name = task_sota["metric"]
        for b in task_sota.get("baselines", []):
            method_data[b["name"]][task] = (b["value"], 0.0)
        for entry in pipeline_entries:
            spec = entry["spec"]
            variant = entry["variant"]
            runs_map = _resolve_run_ref_local(spec, datasets)
            if task not in runs_map:
                continue
            ref = runs_map[task]
            key = f"{ref['section']}/{ref['id']}/{variant}/{split}/{task}"
            mj = metric_jsons.get(key, {})
            m_container = mj.get("metrics")
            if m_container is None:
                continue
            m = m_container.get(metric_name, {})
            val = m.get("value")
            if val is None:
                continue
            method_data[entry["label"]][task] = (val, m.get("ci95", 0.0))

    # --- Filter out pipeline methods with no data for any task ---
    pipeline_entries = [
        entry
        for entry in pipeline_entries
        if any(task in method_data[entry["label"]] for task in tasks)
    ]
    all_method_names = list(all_baseline_names)
    for entry in pipeline_entries:
        all_method_names.append(entry["label"])
    sota_name_set = set(all_baseline_names)

    if not all_method_names:
        Path(output_path).touch()
        return

    # --- Method colours ---
    method_colors: Dict[str, str] = {}
    for i, name in enumerate(all_baseline_names):
        method_colors[name] = _SOTA_BASELINE_COLORS[i % len(_SOTA_BASELINE_COLORS)]
    for entry in pipeline_entries:
        spec = entry["spec"]
        if "run" in spec:
            method_colors[entry["label"]] = _resolve_run_color(
                spec["run"], train_config
            )
        elif "runs" in spec:
            first_ref = next(iter(spec["runs"].values()))
            method_colors[entry["label"]] = _resolve_run_color(first_ref, train_config)
        else:
            method_colors[entry["label"]] = "#999999"

    # --- Subplot grid layout ---
    n_cols = min(max_cols, n_tasks)
    n_rows = math.ceil(n_tasks / n_cols)
    total_cells = n_rows * n_cols
    # Empty cells at the end can hold the legend.
    empty_cells = total_cells - n_tasks

    max_methods = max(
        len([m for m in all_method_names if task in method_data[m]]) for task in tasks
    )
    subplot_w = max(1.8, 0.28 * max_methods + 0.6)
    subplot_h = 3.8
    fig_width = subplot_w * n_cols
    fig_height = subplot_h * n_rows
    fig, axes_2d = plt.subplots(
        n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False
    )

    # Shared y-limits across all subplots for easy comparison.
    global_max = 0.0
    for method in all_method_names:
        for task in tasks:
            entry = method_data[method].get(task)
            if entry is not None:
                global_max = max(global_max, entry[0] + entry[1])
    y_top = global_max * 1.30  # headroom for rotated value labels

    for t_idx, task in enumerate(tasks):
        row, col = divmod(t_idx, n_cols)
        ax = axes_2d[row][col]

        # Per-task method list: only methods with data for this task.
        task_methods = [m for m in all_method_names if task in method_data[m]]
        n_task_methods = len(task_methods)
        if n_task_methods == 0:
            ax.set_visible(False)
            continue

        x = np.arange(n_task_methods)
        bar_width = 1.0

        for bar_idx, method in enumerate(task_methods):
            val, ci = method_data[method][task]
            is_sota = method in sota_name_set
            yerr = 0.0 if is_sota else ci
            hatch = "///" if is_sota else None

            ax.bar(
                bar_idx,
                val,
                bar_width,
                yerr=yerr if yerr > 0 else None,
                capsize=3 if yerr > 0 else 0,
                color=method_colors[method],
                alpha=0.85,
                edgecolor="white",
                linewidth=0.5,
                hatch=hatch,
            )

            # Value label above bar, rotated 90 degrees.
            label_y = val + yerr + 0.02 if yerr > 0 else val + 0.02
            ax.text(
                bar_idx,
                label_y,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=6,
                rotation=90,
                color="#333333",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(task_methods, fontsize=7, rotation=60, ha="right")
        ax.set_title(task_display_name(task), fontsize=10)
        ax.set_ylim(0, y_top)
        ax.grid(axis="y", alpha=0.2)
        task_metric_label = sota_data[task].get(
            "metric_label", sota_data[task]["metric"]
        )
        if col == 0:
            ax.set_ylabel(task_metric_label, fontsize=9)

    # --- Hide unused subplot cells ---
    for cell_idx in range(n_tasks, total_cells):
        row, col = divmod(cell_idx, n_cols)
        axes_2d[row][col].set_visible(False)

    resolved_title: str = (
        title
        if title is not None
        else str(sota_data[tasks[0]].get("test_set", "SOTA Comparison"))
    )
    variant = evaluation.get("variant", "")
    variant_display = _VARIANT_DISPLAY_NAMES.get(variant, variant)
    if variant_display:
        resolved_title = f"{resolved_title} \u2014 {variant_display}"
    fig.suptitle(resolved_title, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_run_ref_local(
    spec: Dict[str, Any],
    datasets: List[Dict[str, str]],
) -> Dict[str, Dict[str, str]]:
    """Same logic as resolve._resolve_run_ref but local to this module."""
    if "run" in spec:
        ref = spec["run"]
        return {ds["task"]: ref for ds in datasets}
    elif "runs" in spec:
        return spec["runs"]
    else:
        raise ValueError(
            f"Run spec must have either 'run' or 'runs' key, got: {sorted(spec.keys())}"
        )


def _spec_variant(spec: Dict[str, Any], default_variant: str) -> str:
    """Return the variant for a run spec, falling back to *default_variant*.

    If *spec* contains a ``"variant"`` key, that value overrides the
    evaluation-level *default_variant*.
    """
    return spec.get("variant", default_variant)


def generate_evaluation_summary(
    evaluation: Dict[str, Any],
    metric_jsons: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Generate a summary JSON for an evaluation.

    Returns a dict suitable for writing to
    ``results/evaluation/{eval_id}/summary_{split}.json``.
    """
    datasets = evaluation["datasets"]
    approaches = evaluation["approaches"]
    split = evaluation["split"]
    default_variant = evaluation["variant"]
    baseline = evaluation["baseline"]

    task_names = [ds["task"] for ds in datasets]
    task_metrics = {ds["task"]: ds["metric"] for ds in datasets}

    baseline_variant = _spec_variant(baseline, default_variant)
    baseline_runs = _resolve_run_ref_local(baseline, datasets)

    summary: Dict[str, Any] = {
        "id": evaluation.get("id", "unnamed"),
        "split": split,
        "variant": default_variant,
        "tasks": {},
        "approaches": [],
    }

    # Baseline metrics
    for task in task_names:
        metric_name = task_metrics[task]
        ref = baseline_runs[task]
        key = f"{ref['section']}/{ref['id']}/{baseline_variant}/{split}/{task}"
        bj = metric_jsons.get(key, {})
        bm_container = bj.get("metrics")
        if bm_container is None:
            summary["tasks"][task] = {
                "metric": metric_name,
                "baseline": None,
            }
        else:
            bm = bm_container.get(metric_name, {})
            summary["tasks"][task] = {
                "metric": metric_name,
                "baseline": bm,
            }

    # Approach deltas
    for approach in approaches:
        approach_variant = _spec_variant(approach, default_variant)
        approach_runs = _resolve_run_ref_local(approach, datasets)
        approach_entry = {
            "label": approach["label"],
            "deltas": {},
        }
        for task in task_names:
            metric_name = task_metrics[task]
            baseline_ref = baseline_runs[task]
            approach_ref = approach_runs[task]

            baseline_key = f"{baseline_ref['section']}/{baseline_ref['id']}/{baseline_variant}/{split}/{task}"
            approach_key = f"{approach_ref['section']}/{approach_ref['id']}/{approach_variant}/{split}/{task}"

            baseline_json = metric_jsons.get(baseline_key, {})
            approach_json = metric_jsons.get(approach_key, {})

            # Check for sentinel null metrics (run not trained on this task)
            if (
                baseline_json.get("metrics") is None
                or approach_json.get("metrics") is None
            ):
                approach_entry["deltas"][task] = None
                continue

            baseline_boot = (baseline_json.get("bootstrap_values") or {}).get(
                metric_name, []
            )
            approach_boot = (approach_json.get("bootstrap_values") or {}).get(
                metric_name, []
            )

            if baseline_boot and approach_boot:
                delta = compute_pairwise_delta(
                    baseline_boot, approach_boot, percent=True
                )
                approach_entry["deltas"][task] = delta
            else:
                approach_entry["deltas"][task] = None

        summary["approaches"].append(approach_entry)

    summary["delta_summary"] = compute_delta_summary(evaluation, metric_jsons)

    return summary
