rule auprc_metrics:
    """Compute bootstrapped AUPRC for a single run/variant/split/task."""
    input:
        prob_predictions_dir="results/prob_predictions/{section}/{run}/{variant}/{split}",
        bootstraps=lambda wc: "results/bootstraps/{}/{}/{}.parquet".format(
            "per_residue_classification", wc.task, wc.split
        ),
    output:
        metrics="results/auprc_metrics/{section}/{run}/{variant}/{split}/{task}.json",
    params:
        prob_predictions_path=lambda wc: "results/prob_predictions/{}/{}/{}/{}/{}.jsonl".format(
            wc.section, wc.run, wc.variant, wc.split, wc.task
        ),
        section=lambda wc: wc.section,
        run_id=lambda wc: wc.run,
        variant=lambda wc: wc.variant,
        split=lambda wc: wc.split,
        task=lambda wc: wc.task,
    log:
        f"{LOG_PREFIX}/evaluation/auprc_metrics/{{section}}_{{run}}_{{variant}}_{{split}}_{{task}}.log",
    script:
        "../../scripts/evaluation/auprc_metrics.py"


def _get_auprc_summary_inputs(wildcards):
    """Return all AUPRC metric JSON inputs for an auprc_evaluation entry."""
    eval_id = wildcards.eval_id
    evaluation = None
    for ev in config.get("auprc_evaluations", []):
        if ev["id"] == eval_id:
            evaluation = ev
            break
    if evaluation is None:
        raise ValueError(f"Unknown auprc_evaluation id: {eval_id!r}")

    from mutafitup.evaluation.auprc_resolve import resolve_auprc_evaluation_inputs
    metric_paths = resolve_auprc_evaluation_inputs(evaluation)

    return {
        "metric_jsons": metric_paths,
    }


def _get_auprc_evaluation(wildcards):
    """Look up the auprc_evaluation config entry for the given eval_id."""
    for ev in config.get("auprc_evaluations", []):
        if ev["id"] == wildcards.eval_id:
            return ev
    raise ValueError(f"Unknown auprc_evaluation id: {wildcards.eval_id!r}")


rule auprc_summary:
    """Generate AUPRC summary JSON and bar chart for an evaluation."""
    input:
        unpack(_get_auprc_summary_inputs),
    output:
        summary="results/auprc_evaluation/{eval_id}/summary_{split}.json",
        bar_chart="results/auprc_evaluation/{eval_id}/bar_chart_{split}.png",
    wildcard_constraints:
        eval_id="[^/]+",
        split="[^/]+",
    params:
        evaluation=lambda wc: _get_auprc_evaluation(wc),
    log:
        f"{LOG_PREFIX}/evaluation/auprc_summary/{{eval_id}}_{{split}}.log",
    script:
        "../../scripts/evaluation/auprc_summary.py"
