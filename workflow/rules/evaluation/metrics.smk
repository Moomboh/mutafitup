def _get_metric_names_for_task(task_name):
    """Look up which metrics to compute for a task based on its subset_type."""
    subset_type = resolve_dataset_type(task_name)
    return config.get("metrics", {}).get(subset_type, [])


def _get_num_classes_for_task(task_name):
    """Look up num_labels for a classification task (returns 2 as default)."""
    meta = resolve_dataset_meta(task_name)
    return meta.get("num_labels", 2)


rule metrics:
    """Compute bootstrapped metrics for a single run/variant/split/task."""
    input:
        predictions_dir="results/predictions/{section}/{run}/{variant}/{split}",
        bootstraps=lambda wc: "results/bootstraps/{}/{}/{}.parquet".format(
            resolve_dataset_type(wc.task), wc.task, wc.split
        ),
    output:
        metrics="results/metrics/{section}/{run}/{variant}/{split}/{task}.json",
    params:
        predictions_path=lambda wc: "results/predictions/{}/{}/{}/{}/{}.jsonl".format(
            wc.section, wc.run, wc.variant, wc.split, wc.task
        ),
        metric_names=lambda wc: _get_metric_names_for_task(wc.task),
        subset_type=lambda wc: resolve_dataset_type(wc.task),
        num_classes=lambda wc: _get_num_classes_for_task(wc.task),
        section=lambda wc: wc.section,
        run_id=lambda wc: wc.run,
        variant=lambda wc: wc.variant,
        split=lambda wc: wc.split,
        task=lambda wc: wc.task,
    log:
        f"{LOG_PREFIX}/evaluation/metrics/{{section}}_{{run}}_{{variant}}_{{split}}_{{task}}.log",
    script:
        "../../scripts/evaluation/metrics.py"
