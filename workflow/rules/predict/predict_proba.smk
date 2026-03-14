def get_predict_proba_inputs(wildcards):
    """Build inputs for probability prediction (binary per-residue tasks only)."""
    section = wildcards.section
    run_id = wildcards.run
    variant = wildcards.variant
    run = get_train_run(section, run_id)
    tasks = expand_tasks(run["tasks"])

    # Filter to binary per-residue classification tasks only
    proba_tasks = [
        t for t in tasks
        if t["subset_type"] == "per_residue_classification"
        and t.get("num_labels", 2) == 2
    ]

    if not proba_tasks:
        raise ValueError(
            f"No binary per-residue classification tasks found for "
            f"{section}/{run_id} — predict_proba has nothing to do."
        )

    inputs = {
    }

    inputs["model_dir"] = _model_path_for_variant(section, run_id, variant)

    for task in proba_tasks:
        name = task["name"]
        dataset = task["dataset"]
        base = f"results/datasets_resplit/per_residue_classification/{dataset}"
        inputs[f"{name}_train"] = f"{base}/train.parquet"
        inputs[f"{name}_valid"] = f"{base}/valid.parquet"
        inputs[f"{name}_test"] = f"{base}/test.parquet"

    return inputs


def _get_proba_tasks(section, run_id):
    """Return expanded task dicts for binary per-residue classification tasks only."""
    run = get_train_run(section, run_id)
    tasks = expand_tasks(run["tasks"])
    return [
        t for t in tasks
        if t["subset_type"] == "per_residue_classification"
        and t.get("num_labels", 2) == 2
    ]


rule predict_proba:
    """Generate probability predictions for binary per-residue classification tasks (AUPRC)."""
    input:
        unpack(get_predict_proba_inputs),
    output:
        predictions_dir=directory("results/prob_predictions/{section}/{run}/{variant}/{split}"),
    wildcard_constraints:
        section="[^/]+",
        run="[^/]+",
        variant="[^/]+",
        split="[^/]+",
    params:
        split=lambda wc: wc.split,
        variant=lambda wc: wc.variant,
        tasks=lambda wc: _get_proba_tasks(wc.section, wc.run),
        batch=lambda wc: get_train_run(wc.section, wc.run)["batch"],
        output_dir=lambda wc: f"results/prob_predictions/{wc.section}/{wc.run}/{wc.variant}/{wc.split}",
    conda:
        "../../envs/finetune/train.yml"
    log:
        f"{LOG_PREFIX}/predict_proba/{{section}}_{{run}}_{{variant}}_{{split}}.log"
    script:
        "../../scripts/predict/predict_proba.py"
