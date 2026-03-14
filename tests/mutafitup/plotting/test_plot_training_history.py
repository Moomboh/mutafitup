import json
from pathlib import Path

from mutafitup.plotting import plot_learning_rate, plot_losses, plot_metrics


def _write_history(path: Path, history):
    path.write_text(json.dumps(history))


def _single_task_history():
    return [
        {
            "epoch": 1,
            "learning_rate": 1e-4,
            "tasks": {
                "secstr": {
                    "train_epoch": 1.0,
                    "train_loss": 1.0,
                    "eval_loss": 1.2,
                    "metric": 0.7,
                    "metric_name": "accuracy",
                    "metrics": {
                        "accuracy": {"value": 0.7},
                        "f1_macro": {"value": 0.6},
                    },
                }
            },
        },
        {
            "epoch": 2,
            "learning_rate": 5e-5,
            "tasks": {
                "secstr": {
                    "train_epoch": 2.0,
                    "train_loss": 0.8,
                    "eval_loss": 1.0,
                    "metric": 0.75,
                    "metric_name": "accuracy",
                    "metrics": {
                        "accuracy": {"value": 0.75},
                        "f1_macro": {"value": 0.65},
                    },
                }
            },
        },
    ]


def _multi_task_history():
    return [
        {
            "epoch": 1,
            "learning_rate": 1e-4,
            "tasks": {
                "secstr": {
                    "train_epoch": 1.0,
                    "train_loss": 1.0,
                    "eval_loss": 1.2,
                    "metric": 0.7,
                    "metric_name": "accuracy",
                    "metrics": {
                        "accuracy": {"value": 0.7},
                        "f1_macro": {"value": 0.6},
                    },
                },
                "disorder": {
                    "train_epoch": 1.0,
                    "train_loss": 0.9,
                    "eval_loss": 1.1,
                    "metric": 0.5,
                    "metric_name": "spearman",
                    "metrics": {
                        "spearman": {"value": 0.5},
                        "pearson": {"value": 0.45},
                    },
                },
            },
        },
        {
            "epoch": 2,
            "learning_rate": 5e-5,
            "tasks": {
                "secstr": {
                    "train_epoch": 2.0,
                    "train_loss": 0.8,
                    "eval_loss": 1.0,
                    "metric": 0.75,
                    "metric_name": "accuracy",
                    "metrics": {
                        "accuracy": {"value": 0.75},
                        "f1_macro": {"value": 0.65},
                    },
                },
                "disorder": {
                    "train_epoch": 2.0,
                    "train_loss": 0.7,
                    "eval_loss": 0.9,
                    "metric": 0.55,
                    "metric_name": "spearman",
                    "metrics": {
                        "spearman": {"value": 0.55},
                        "pearson": {"value": 0.5},
                    },
                },
            },
        },
    ]


def test_plot_training_history_single_task(tmp_path):
    history_file = tmp_path / "history_single.json"
    _write_history(history_file, _single_task_history())

    losses_path = tmp_path / "losses.png"
    metrics_path = tmp_path / "metrics.png"
    lr_path = tmp_path / "learning_rate.png"

    plot_losses(str(history_file), str(losses_path), "checkpoint", ["secstr"])
    plot_metrics(str(history_file), str(metrics_path), "checkpoint", ["secstr"])
    plot_learning_rate(str(history_file), str(lr_path), "checkpoint")

    assert losses_path.is_file()
    assert metrics_path.is_file()
    assert lr_path.is_file()


def test_plot_training_history_multi_task(tmp_path):
    history_file = tmp_path / "history_multi.json"
    _write_history(history_file, _multi_task_history())

    losses_path = tmp_path / "losses.png"
    metrics_path = tmp_path / "metrics.png"

    plot_losses(
        str(history_file), str(losses_path), "checkpoint", ["secstr", "disorder"]
    )
    plot_metrics(
        str(history_file), str(metrics_path), "checkpoint", ["secstr", "disorder"]
    )

    assert losses_path.is_file()
    assert metrics_path.is_file()


def _align_lora_history():
    """Multi-task history with align_lora_kl_loss at the top level."""
    base = _multi_task_history()
    base[0]["align_lora_kl_loss"] = 0.05
    base[1]["align_lora_kl_loss"] = 0.03
    return base


def test_plot_losses_with_align_lora_kl(tmp_path):
    history_file = tmp_path / "history_align_lora.json"
    _write_history(history_file, _align_lora_history())

    losses_path = tmp_path / "losses.png"

    plot_losses(
        str(history_file),
        str(losses_path),
        "align_lora/run1",
        ["secstr", "disorder"],
    )

    assert losses_path.is_file()


def test_plot_losses_without_kl_unchanged(tmp_path):
    """Non-AlignLoRA history should produce plots without a secondary axis."""
    history_file = tmp_path / "history_no_kl.json"
    _write_history(history_file, _multi_task_history())

    losses_path = tmp_path / "losses.png"

    plot_losses(
        str(history_file),
        str(losses_path),
        "accgrad_lora/run1",
        ["secstr", "disorder"],
    )

    assert losses_path.is_file()


def _six_task_history():
    """History with 6 tasks to test grid wrapping (>4 cols)."""
    task_template = {
        "train_epoch": 1.0,
        "train_loss": 1.0,
        "eval_loss": 1.2,
        "metric": 0.7,
        "metric_name": "accuracy",
        "metrics": {"accuracy": {"value": 0.7}},
    }
    task_names = ["secstr", "secstr8", "rsa", "disorder", "meltome", "subloc"]
    entry1 = {
        "epoch": 1,
        "learning_rate": 1e-4,
        "tasks": {t: {**task_template} for t in task_names},
    }
    entry2 = {
        "epoch": 2,
        "learning_rate": 5e-5,
        "tasks": {
            t: {
                **task_template,
                "train_epoch": 2.0,
                "train_loss": 0.8,
                "eval_loss": 1.0,
                "metric": 0.75,
                "metrics": {"accuracy": {"value": 0.75}},
            }
            for t in task_names
        },
    }
    return entry1, entry2, task_names


def test_grid_wraps_with_more_than_4_tasks(tmp_path):
    """Six tasks should produce a 2-row x 4-col grid (second row has 2)."""
    import matplotlib.pyplot as plt

    entry1, entry2, task_names = _six_task_history()
    history_file = tmp_path / "history_6.json"
    _write_history(history_file, [entry1, entry2])

    losses_path = tmp_path / "losses.png"
    metrics_path = tmp_path / "metrics.png"

    plot_losses(str(history_file), str(losses_path), "run", task_names)
    plot_metrics(str(history_file), str(metrics_path), "run", task_names)

    assert losses_path.is_file()
    assert metrics_path.is_file()


def test_grid_layout_dimensions(tmp_path):
    """Subplot grid should have one axis per task."""
    import matplotlib.pyplot as plt
    from mutafitup.plotting.plot_training_history import _create_subplot_grid

    fig, axes = _create_subplot_grid(5)

    assert len(axes) == 5
    # 5 tasks → 2 rows x 4 cols grid; 3 cells unused/hidden
    all_axes = fig.get_axes()
    visible = [ax for ax in all_axes if ax.get_visible()]
    assert len(visible) == 5
    plt.close(fig)
