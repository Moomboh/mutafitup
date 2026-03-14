import json
import math
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from mutafitup.task_display_names import task_display_name


def _load_history(history_path: str):
    with open(history_path) as f:
        return json.load(f)


def _get_task_names(history, task_names: Optional[Sequence[str]]):
    if task_names is not None:
        return list(task_names)
    if not history:
        return []
    tasks = history[0].get("tasks", {})
    return list(tasks.keys())


def _create_subplot_grid(
    n_tasks: int,
    max_cols: int = 4,
    subplot_width: float = 4.5,
    subplot_height: float = 3.5,
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """Create a subplot grid for *n_tasks* panels.

    Returns ``(fig, axes)`` where *axes* is a flat list of length
    *n_tasks*.
    """
    n_cols = min(max_cols, n_tasks)
    n_rows = math.ceil(n_tasks / n_cols)

    fig, axes_2d = plt.subplots(
        n_rows,
        n_cols,
        figsize=(subplot_width * n_cols, subplot_height * n_rows),
        squeeze=False,
        layout="constrained",
    )

    axes = [axes_2d[r][c] for r in range(n_rows) for c in range(n_cols)]

    # Hide unused cells.
    for idx in range(n_tasks, n_rows * n_cols):
        axes[idx].set_visible(False)

    return fig, axes[:n_tasks]


def _get_early_stopping_status(history: List[dict]) -> Optional[str]:
    if not history:
        return None
    last_entry = history[-1]
    es_info = last_entry.get("early_stopping")
    if es_info is None:
        return None

    if not es_info.get("triggered", False):
        return "completed"

    patience = es_info.get("patience", 0)
    counters = es_info.get("counters", {})

    if counters and patience > 0:
        all_exhausted = all(c >= patience for c in counters.values())
        if all_exhausted:
            return "early stopped"

    return "completed"


def _build_title(run_id: Optional[str], history: List[dict]) -> Optional[str]:
    if not run_id:
        return None
    es_status = _get_early_stopping_status(history)
    if es_status is None:
        return run_id
    return f"{run_id} ({es_status})"


def plot_losses(
    history_path: str,
    output_path: Optional[str] = None,
    run_id: Optional[str] = None,
    task_names: Optional[Sequence[str]] = None,
):
    history = _load_history(history_path)
    if not history:
        return
    if "tasks" not in history[0]:
        return

    names = _get_task_names(history, task_names)
    if not names:
        return

    validation_steps = list(range(1, len(history) + 1))

    # Check if this is an AlignLoRA run with KL loss data
    has_kl_loss = any("align_lora_kl_loss" in entry for entry in history)
    kl_losses: Optional[List[float]] = None
    if has_kl_loss:
        kl_losses = [entry.get("align_lora_kl_loss", 0.0) for entry in history]

    fig, axes = _create_subplot_grid(len(names))

    title = _build_title(run_id, history)
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")

    max_cols = min(4, len(names))
    for t_idx, name in enumerate(names):
        _grid_row, grid_col = divmod(t_idx, max_cols)
        ax = axes[t_idx]
        train_loss = [entry["tasks"][name]["train_loss"] for entry in history]
        eval_loss = [entry["tasks"][name]["eval_loss"] for entry in history]
        line_train = ax.plot(
            validation_steps, train_loss, label="Train Loss", marker="o"
        )
        line_eval = ax.plot(
            validation_steps, eval_loss, label="Validation Loss", marker="s"
        )
        ax.set_ylabel("Loss")
        ax.set_xlabel("Validation Step")
        ax.set_title(f"{task_display_name(name)} Loss")
        ax.grid(True, alpha=0.3)

        lines = line_train + line_eval

        if kl_losses is not None:
            ax2 = ax.twinx()
            line_kl = ax2.plot(
                validation_steps,
                kl_losses,
                label="Align LoRA KL Loss",
                marker="^",
                linestyle="--",
                color="#d62728",
                alpha=0.8,
            )
            lines += line_kl
            # Show KL y-label on rightmost column of each row.
            if grid_col == max_cols - 1 or t_idx == len(names) - 1:
                ax2.set_ylabel("Align LoRA KL Loss", color="#d62728")
            ax2.tick_params(axis="y", labelcolor="#d62728")

        labels = [line.get_label() for line in lines]
        ax.legend(lines, labels)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved loss plot to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_metrics(
    history_path: str,
    output_path: Optional[str] = None,
    run_id: Optional[str] = None,
    task_names: Optional[Sequence[str]] = None,
):
    history = _load_history(history_path)
    if not history:
        return
    if "tasks" not in history[0]:
        return

    names = _get_task_names(history, task_names)
    if not names:
        return

    validation_steps = list(range(1, len(history) + 1))

    all_metrics = {}
    for name in names:
        task_metric_names = []
        for entry in history:
            task = entry["tasks"].get(name, {})
            task_metrics = task.get("metrics")
            if task_metrics:
                task_metric_names = list(task_metrics.keys())
                break
        if not task_metric_names:
            first_task = history[0]["tasks"].get(name, {})
            default_name = first_task.get("metric_name") or "metric"
            task_metric_names = [default_name]

        metric_data = {}
        for metric_name in task_metric_names:
            vals = []
            for entry in history:
                task = entry["tasks"].get(name, {})
                task_metrics = task.get("metrics")
                if task_metrics and metric_name in task_metrics:
                    vals.append(task_metrics[metric_name].get("value", 0.0))
                else:
                    if (
                        metric_name == task.get("metric_name")
                        or metric_name == "metric"
                    ):
                        vals.append(task.get("metric", 0.0))
                    else:
                        vals.append(0.0)

            metric_data[metric_name] = {"values": np.array(vals)}

        all_metrics[name] = metric_data

    fig, axes = _create_subplot_grid(len(names))

    title = _build_title(run_id, history)
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")

    for t_idx, name in enumerate(names):
        ax = axes[t_idx]
        for metric_name, data in all_metrics[name].items():
            ax.plot(validation_steps, data["values"], label=metric_name, marker="o")
        ax.set_ylabel("Metric")
        ax.set_xlabel("Validation Step")
        ax.set_title(f"{task_display_name(name)} Metrics")
        ax.legend()
        ax.grid(True, alpha=0.3)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved metrics plot to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_learning_rate(
    history_path: str,
    output_path: Optional[str] = None,
    run_id: Optional[str] = None,
):
    with open(history_path) as f:
        history = json.load(f)

    validation_steps = list(range(1, len(history) + 1))
    learning_rates = [entry["learning_rate"] for entry in history]

    fig, ax = plt.subplots(figsize=(8.0, 5.0), layout="constrained")

    title = _build_title(run_id, history)
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")

    ax.plot(validation_steps, learning_rates, marker="o")
    ax.set_ylabel("Learning Rate")
    ax.set_xlabel("Validation Step")
    ax.set_title("Learning Rate Schedule")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved learning rate plot to {output_path}")
    else:
        plt.show()

    plt.close()
