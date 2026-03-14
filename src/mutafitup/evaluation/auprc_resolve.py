"""Config resolution for AUPRC evaluations.

Completely independent from the main evaluation resolver
(:mod:`mutafitup.evaluation.resolve`).  Takes ``auprc_evaluations``
config entries and resolves them into concrete file paths for the
AUPRC-specific pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


def _spec_variant(spec: Dict[str, Any], default_variant: str) -> str:
    """Return the variant for a run spec, falling back to *default_variant*."""
    return spec.get("variant", default_variant)


def _resolve_run_ref(
    run_spec: Dict[str, Any],
    tasks: List[str],
) -> Dict[str, Dict[str, str]]:
    """Normalise a run spec to per-task mapping.

    Accepts either:
      - ``{"run": {"section": ..., "id": ...}}`` (single run for all tasks)
      - ``{"runs": {"task": {"section": ..., "id": ...}, ...}}``
    """
    if "run" in run_spec:
        ref = run_spec["run"]
        return {task: ref for task in tasks}
    elif "runs" in run_spec:
        return run_spec["runs"]
    else:
        raise ValueError(
            "Run spec must have either 'run' or 'runs' key, "
            f"got: {sorted(run_spec.keys())}"
        )


def resolve_auprc_evaluation_inputs(
    evaluation: Dict[str, Any],
) -> List[str]:
    """Return AUPRC metric JSON paths required for an AUPRC evaluation.

    Parameters
    ----------
    evaluation : dict
        A single ``auprc_evaluations`` config entry with keys: ``id``,
        ``split``, ``variant``, ``tasks``, ``baseline``, ``approaches``.

    Returns
    -------
    list of str
        Paths like ``results/auprc_metrics/{section}/{id}/{variant}/{split}/{task}.json``.
    """
    split = evaluation["split"]
    default_variant = evaluation["variant"]
    tasks = evaluation["tasks"]

    paths: List[str] = []
    seen: set = set()

    def _add_path(section: str, run_id: str, task: str, variant: str) -> None:
        p = f"results/auprc_metrics/{section}/{run_id}/{variant}/{split}/{task}.json"
        if p not in seen:
            seen.add(p)
            paths.append(p)

    # Baseline
    baseline = evaluation["baseline"]
    baseline_variant = _spec_variant(baseline, default_variant)
    baseline_runs = _resolve_run_ref(baseline, tasks)
    for task in tasks:
        ref = baseline_runs[task]
        _add_path(ref["section"], ref["id"], task, baseline_variant)

    # Approaches
    for approach in evaluation["approaches"]:
        approach_variant = _spec_variant(approach, default_variant)
        approach_runs = _resolve_run_ref(approach, tasks)
        for task in tasks:
            ref = approach_runs[task]
            _add_path(ref["section"], ref["id"], task, approach_variant)

    return paths


def resolve_auprc_prob_prediction_inputs(
    config: Dict[str, Any],
) -> List[Dict[str, str]]:
    """Return all (section, run_id, variant, split, task) combos needing prob predictions.

    Parameters
    ----------
    config : dict
        Full Snakemake config (must have ``auprc_evaluations``).

    Returns
    -------
    list of dicts with keys: section, run_id, variant, split, task
    """
    seen: set = set()
    result: List[Dict[str, str]] = []

    for evaluation in config.get("auprc_evaluations", []):
        split = evaluation["split"]
        default_variant = evaluation["variant"]
        tasks = evaluation["tasks"]

        all_specs = [evaluation["baseline"]] + evaluation["approaches"]

        for spec in all_specs:
            variant = _spec_variant(spec, default_variant)
            run_map = _resolve_run_ref(spec, tasks)
            for task in tasks:
                ref = run_map[task]
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


def resolve_auprc_bootstrap_inputs(
    config: Dict[str, Any],
) -> List[Tuple[str, str, str]]:
    """Return (subset_type, dataset, split) tuples for AUPRC bootstraps.

    AUPRC tasks are always ``per_residue_classification``.

    Parameters
    ----------
    config : dict
        Full Snakemake config.

    Returns
    -------
    list of (subset_type, dataset, split) tuples
    """
    seen: set = set()
    result: List[Tuple[str, str, str]] = []

    for evaluation in config.get("auprc_evaluations", []):
        split = evaluation["split"]
        for task in evaluation["tasks"]:
            key = ("per_residue_classification", task, split)
            if key not in seen:
                seen.add(key)
                result.append(key)

    return result
