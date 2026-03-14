def _model_path_for_variant(section, run_id, variant):
    """Return the model directory path for the given variant."""
    base = f"results/train/{section}/{run_id}"
    if variant == "final":
        return f"{base}/model"
    elif variant == "best_overall":
        return f"{base}/best_overall_model"
    elif variant == "best_task":
        # best_task loads per-task models; we depend on the directory
        return f"{base}/best_task_models"
    elif variant == "best_loss_overall":
        return f"{base}/best_loss_overall_model"
    elif variant == "best_loss_task":
        # best_loss_task loads per-task models; we depend on the directory
        return f"{base}/best_loss_task_models"
    else:
        raise ValueError(f"Unknown variant: {variant}")


def get_predict_inputs(wildcards):
    section = wildcards.section
    run_id = wildcards.run
    variant = wildcards.variant
    run = get_train_run(section, run_id)
    tasks = expand_tasks(run["tasks"])

    inputs = {
    }

    inputs["model_dir"] = _model_path_for_variant(section, run_id, variant)

    for task in tasks:
        name = task["name"]
        subset_type = task["subset_type"]
        dataset = task["dataset"]

        base = f"results/datasets_resplit/{subset_type}/{dataset}"
        inputs[f"{name}_train"] = f"{base}/train.parquet"
        inputs[f"{name}_valid"] = f"{base}/valid.parquet"
        inputs[f"{name}_test"] = f"{base}/test.parquet"

    return inputs


rule predict:
    input:
        unpack(get_predict_inputs),
    output:
        predictions_dir=directory("results/predictions/{section}/{run}/{variant}/{split}"),
    wildcard_constraints:
        section="[^/]+",
        run="[^/]+",
        variant="[^/]+",
        split="[^/]+",
    params:
        split=lambda wc: wc.split,
        variant=lambda wc: wc.variant,
        tasks=lambda wc: expand_tasks(get_train_run(wc.section, wc.run)["tasks"]),
        batch=lambda wc: get_train_run(wc.section, wc.run)["batch"],
        output_dir=lambda wc: f"results/predictions/{wc.section}/{wc.run}/{wc.variant}/{wc.split}",
    conda:
        "../../envs/finetune/train.yml"
    log:
        f"{LOG_PREFIX}/predict/{{section}}_{{run}}_{{variant}}_{{split}}.log"
    script:
        "../../scripts/predict/predict.py"


# ---------------------------------------------------------------------------
# ONNX validation (Node.js predictions vs PyTorch predictions)
# ---------------------------------------------------------------------------

rule validate_onnx_node:
    """Validate ONNX model predictions (via mutafitup-node) against PyTorch predictions."""
    input:
        export_metadata="results/onnx_export/{section}/{run}/{variant}/export_metadata.json",
        predictions_dir="results/predictions/{section}/{run}/{variant}/{split}",
    output:
        report="results/onnx_validate/{section}/{run}/{variant}/{split}/report.json",
    wildcard_constraints:
        section="[^/]+",
        run="[^/]+",
        variant="[^/]+",
        split="[^/]+",
    conda:
        "../../envs/finetune/validate_onnx_node.yml"
    log:
        f"{LOG_PREFIX}/onnx_validate/{{section}}_{{run}}_{{variant}}_{{split}}.log"
    params:
        export_dir=lambda wc: f"results/onnx_export/{wc.section}/{wc.run}/{wc.variant}",
    shell:
        "bash workflow/scripts/finetune/validate_onnx_node.sh {params.export_dir} {input.predictions_dir} {output.report} 2>&1 | tee {log}"
