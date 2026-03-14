"""Config resolution: evaluation definitions -> required file paths.

Takes an evaluation config entry and resolves it into the concrete list of
per-run metric JSON paths that must exist as Snakemake rule inputs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple


def _spec_variant(spec: Dict[str, Any], default_variant: str) -> str:
    """Return the variant for a run spec, falling back to *default_variant*.

    If *spec* contains a ``"variant"`` key, that value is used; otherwise
    the evaluation-level *default_variant* is returned.
    """
    return spec.get("variant", default_variant)


def _resolve_run_ref(
    run_spec: Dict[str, Any],
    datasets: List[Dict[str, str]],
) -> Dict[str, Dict[str, str]]:
    """Normalise a baseline or approach run spec to per-task mapping.

    Accepts either:
      - ``{"run": {"section": ..., "id": ...}}`` (single run for all tasks)
      - ``{"runs": {"task": {"section": ..., "id": ...}, ...}}``

    Returns ``{task_name: {"section": ..., "id": ...}}`` for every task
    in *datasets*.
    """
    if "run" in run_spec:
        ref = run_spec["run"]
        return {ds["task"]: ref for ds in datasets}
    elif "runs" in run_spec:
        return run_spec["runs"]
    else:
        raise ValueError(
            "Run spec must have either 'run' or 'runs' key, "
            f"got: {sorted(run_spec.keys())}"
        )


def _get_run_tasks(
    train_config: Dict[str, list],
    section: str,
    run_id: str,
) -> Optional[Set[str]]:
    """Look up the set of task names a training run was configured for.

    Parameters
    ----------
    train_config : dict
        The ``config["train"]`` dict mapping section names to lists of
        run dicts (each with ``"id"`` and ``"tasks"`` keys).
    section : str
        Training section (e.g. ``"lora"``, ``"accgrad_lora"``).
    run_id : str
        Training run ID.

    Returns
    -------
    set of str or None
        The set of task name strings the run was trained on, or ``None``
        if the run could not be found.
    """
    for run in train_config.get(section, []):
        if run["id"] == run_id:
            return set(run.get("tasks", []))
    return None


def resolve_evaluation_inputs(
    evaluation: Dict[str, Any],
    train_config: Optional[Dict[str, list]] = None,
) -> List[str]:
    """Return the list of metric JSON paths required for an evaluation.

    Parameters
    ----------
    evaluation : dict
        A single evaluation config entry with keys: ``id``, ``split``,
        ``variant``, ``datasets``, ``baseline``, ``approaches``.
    train_config : dict, optional
        The ``config["train"]`` dict.  When provided, AUPRC metric paths
        are only emitted for runs that were actually trained on the
        corresponding task.  This prevents Snakemake from scheduling
        ``predict_proba`` for runs that have no binary classification
        tasks (e.g. a structural-only model asked about GPSite tasks).
        When omitted the function behaves as before (all AUPRC paths
        are emitted unconditionally).

    Returns
    -------
    list of str
        Paths like ``results/metrics/{section}/{id}/{variant}/{split}/{task}.json``
        or ``results/auprc_metrics/...`` for datasets whose configured metric
        is ``"auprc"``.
    """
    split = evaluation["split"]
    default_variant = evaluation["variant"]
    datasets = evaluation["datasets"]

    # Build a task -> metric lookup so _add_path can choose the right
    # results directory.
    task_metrics = {ds["task"]: ds.get("metric", "") for ds in datasets}

    paths: List[str] = []
    seen: set = set()

    def _add_path(section: str, run_id: str, task: str, variant: str) -> None:
        # When train_config is available, skip paths entirely for runs
        # that were not trained on this task.  This avoids scheduling
        # unnecessary Snakemake jobs (e.g. predict_proba for a
        # structural-only model asked about GPSite tasks).  The
        # downstream plotting code handles missing metric_jsons keys
        # gracefully (renders N/A cells).
        if train_config is not None:
            run_tasks = _get_run_tasks(train_config, section, run_id)
            if run_tasks is not None and task not in run_tasks:
                return

        if task_metrics.get(task) == "auprc":
            prefix = "results/auprc_metrics"
        else:
            prefix = "results/metrics"
        p = f"{prefix}/{section}/{run_id}/{variant}/{split}/{task}.json"
        if p not in seen:
            seen.add(p)
            paths.append(p)

    # Baseline
    baseline = evaluation["baseline"]
    baseline_variant = _spec_variant(baseline, default_variant)
    baseline_runs = _resolve_run_ref(baseline, datasets)
    for ds in datasets:
        task = ds["task"]
        ref = baseline_runs[task]
        _add_path(ref["section"], ref["id"], task, baseline_variant)

    # Approaches
    for approach in evaluation["approaches"]:
        approach_variant = _spec_variant(approach, default_variant)
        approach_runs = _resolve_run_ref(approach, datasets)
        for ds in datasets:
            task = ds["task"]
            ref = approach_runs[task]
            _add_path(ref["section"], ref["id"], task, approach_variant)

    return paths


def resolve_bootstrap_inputs(
    config: Dict[str, Any],
) -> List[Tuple[str, str, str]]:
    """Return (subset_type, dataset, split) tuples for all needed bootstraps.

    Examines all evaluations in the config and determines which dataset
    splits need bootstrap indices.

    Parameters
    ----------
    config : dict
        The full Snakemake config dict. Must have ``evaluations`` and
        ``standardized_datasets`` keys.

    Returns
    -------
    list of (subset_type, dataset, split) tuples
    """
    seen = set()
    result = []

    std_datasets = config.get("standardized_datasets", {})

    for evaluation in config.get("evaluations", []):
        split = evaluation["split"]
        for ds in evaluation["datasets"]:
            task = ds["task"]
            # Find the subset_type for this task
            subset_type = None
            for st, datasets_map in std_datasets.items():
                if task in datasets_map:
                    subset_type = st
                    break
            if subset_type is None:
                raise ValueError(f"Task {task!r} not found in standardized_datasets")
            key = (subset_type, task, split)
            if key not in seen:
                seen.add(key)
                result.append(key)

    return result


def resolve_all_metric_inputs(
    config: Dict[str, Any],
) -> List[Dict[str, str]]:
    """Return all (section, run_id, variant, split, task) combos needing metrics.

    Parameters
    ----------
    config : dict
        Full Snakemake config.

    Returns
    -------
    list of dicts with keys: section, run_id, variant, split, task
    """
    train_config = config.get("train", {})
    seen = set()
    result = []

    for evaluation in config.get("evaluations", []):
        split = evaluation["split"]
        default_variant = evaluation["variant"]
        datasets = evaluation["datasets"]

        # Collect all run refs (baseline + approaches)
        all_specs = [evaluation["baseline"]] + evaluation["approaches"]

        for spec in all_specs:
            variant = _spec_variant(spec, default_variant)
            run_map = _resolve_run_ref(spec, datasets)
            for ds in datasets:
                task = ds["task"]
                ref = run_map[task]

                # Skip runs not trained on this task.
                if train_config:
                    run_tasks = _get_run_tasks(train_config, ref["section"], ref["id"])
                    if run_tasks is not None and task not in run_tasks:
                        continue

                key = (ref["section"], ref["id"], variant, split, task)
                if key not in seen:
                    seen.add(key)
                    result.append(
                        {
                            "section": ref["section"],
                            "run_id": ref["id"],
                            "variant": variant,
                            "split": split,
                            "task": task,
                        }
                    )

    return result
