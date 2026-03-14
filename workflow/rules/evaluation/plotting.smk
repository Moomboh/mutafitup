import yaml as _yaml

from mutafitup.evaluation.plotting import group_sota_tasks_by_test_set
from mutafitup.evaluation.resolve import resolve_evaluation_inputs

# ---------------------------------------------------------------------------
# Load SOTA metrics at parse time so we can derive test-set groups for the
# ``plot_sota_comparison`` rule wildcard.
# ---------------------------------------------------------------------------

with open("resources/sota_metrics.yml") as _f:
    _SOTA_DATA = _yaml.safe_load(_f)["tasks"]

_SOTA_GROUPS = group_sota_tasks_by_test_set(_SOTA_DATA)

# Human-readable titles for each test-set group.
_SOTA_GROUP_TITLES = {}
for _grp, _tasks in _SOTA_GROUPS.items():
    if _grp == "gpsite":
        # All binding-site tasks are grouped under one key; use a shared title.
        _SOTA_GROUP_TITLES[_grp] = "Binding-site family"
    else:
        # Use the test_set value from the first task in the group.
        _SOTA_GROUP_TITLES[_grp] = _SOTA_DATA[_tasks[0]].get("test_set", _grp)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _get_evaluation_by_id(eval_id):
    """Look up an evaluation config entry by ID."""
    for ev in config.get("evaluations", []):
        if ev["id"] == eval_id:
            return ev
    raise ValueError(f"Unknown evaluation ID: {eval_id!r}")


def _get_evaluation_metric_inputs(wildcards):
    """Return all metric JSON inputs needed for an evaluation."""
    ev = _get_evaluation_by_id(wildcards.eval_id)
    return resolve_evaluation_inputs(ev, train_config=config.get("train", {}))


def _get_evaluation_bar_outputs(wildcards):
    """Return per-task bar chart output paths for an evaluation."""
    ev = _get_evaluation_by_id(wildcards.eval_id)
    return [
        f"results/evaluation/{ev['id']}/bars_{ds['task']}_{ev['split']}.png"
        for ds in ev["datasets"]
    ]


def _get_evaluation_task_metric(wildcards):
    """Return the configured metric name for one evaluation task."""
    ev = _get_evaluation_by_id(wildcards.eval_id)
    for ds in ev["datasets"]:
        if ds["task"] == wildcards.task:
            return ds["metric"]
    raise ValueError(
        f"Task {wildcards.task!r} not found in evaluation {wildcards.eval_id!r}"
    )


def _get_sota_groups_for_evaluation(eval_id):
    """Return test-set group keys that overlap with an evaluation's tasks."""
    ev = _get_evaluation_by_id(eval_id)
    eval_tasks = {ds["task"] for ds in ev["datasets"]}
    groups = []
    for grp, tasks in _SOTA_GROUPS.items():
        if eval_tasks & set(tasks):
            groups.append(grp)
    return groups


def _get_sota_metric_inputs(wildcards):
    """Return standard metric JSON paths for the SOTA comparison.

    Always uses ``results/metrics/`` (never ``results/auprc_metrics/``)
    because SOTA baselines report standard metrics (accuracy, mcc,
    spearman) and the pipeline's standard metric JSONs contain these.
    """
    ev = _get_evaluation_by_id(wildcards.eval_id)
    paths = resolve_evaluation_inputs(ev, train_config=config.get("train", {}))
    return [
        p.replace("results/auprc_metrics/", "results/metrics/") for p in paths
    ]


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------


rule plot_evaluation:
    """Generate evaluation summary plots and JSON for a comparison."""
    input:
        metric_jsons=_get_evaluation_metric_inputs,
    output:
        heatmap="results/evaluation/{eval_id}/heatmap_{split}.png",
        delta_summary_plot="results/evaluation/{eval_id}/delta_summary_{split}.png",
        summary="results/evaluation/{eval_id}/summary_{split}.json",
    params:
        evaluation=lambda wc: _get_evaluation_by_id(wc.eval_id),
        train_config=config.get("train", {}),
    log:
        f"{LOG_PREFIX}/evaluation/plotting/{{eval_id}}_{{split}}.log",
    script:
        "../../scripts/evaluation/plot_evaluation.py"


rule plot_evaluation_bar:
    """Generate one per-task bar chart for an evaluation."""
    input:
        metric_jsons=_get_evaluation_metric_inputs,
    output:
        plot="results/evaluation/{eval_id}/bars_{task}_{split}.png",
    params:
        evaluation=lambda wc: _get_evaluation_by_id(wc.eval_id),
        metric_name=_get_evaluation_task_metric,
        train_config=config.get("train", {}),
    log:
        f"{LOG_PREFIX}/evaluation/bar_plotting/{{eval_id}}_{{task}}_{{split}}.log",
    script:
        "../../scripts/evaluation/plot_evaluation_bar.py"


rule plot_sota_comparison:
    """Generate a SOTA comparison plot for one test-set group within an evaluation."""
    input:
        sota_db="resources/sota_metrics.yml",
        metric_jsons=_get_sota_metric_inputs,
    output:
        plot="results/evaluation/{eval_id}/sota_{test_set_group}_{split}.png",
    params:
        evaluation=lambda wc: _get_evaluation_by_id(wc.eval_id),
        train_config=config.get("train", {}),
        title=lambda wc: _SOTA_GROUP_TITLES.get(wc.test_set_group),
    log:
        f"{LOG_PREFIX}/evaluation/sota_plotting/{{eval_id}}_{{test_set_group}}_{{split}}.log",
    script:
        "../../scripts/evaluation/plot_sota.py"
