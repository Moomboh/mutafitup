def _get_train_runs(section):
    train_cfg = config.get("train", {})
    return train_cfg.get(section, [])


def _get_train_run_local(section, run_id):
    for run in _get_train_runs(section):
        if run["id"] == run_id:
            return run
    raise ValueError(f"Unknown train.{section} run id: {run_id}")


def _get_parquet_inputs(section, wildcards):
    run = _get_train_run_local(section, wildcards.run)
    task_names = run["tasks"]
    tasks = expand_tasks(task_names)

    inputs = {
    }

    for task in tasks:
        name = task["name"]
        subset_type = task["subset_type"]
        dataset = task["dataset"]

        base = f"results/datasets_resplit/{subset_type}/{dataset}"
        inputs[f"{name}_train"] = f"{base}/train.parquet"
        inputs[f"{name}_valid"] = f"{base}/valid.parquet"

    return inputs


def get_heads_only_parquet_inputs(wildcards):
    return _get_parquet_inputs("heads_only", wildcards)


def get_lora_parquet_inputs(wildcards):
    return _get_parquet_inputs("lora", wildcards)


def get_accgrad_lora_parquet_inputs(wildcards):
    return _get_parquet_inputs("accgrad_lora", wildcards)


def get_align_lora_parquet_inputs(wildcards):
    return _get_parquet_inputs("align_lora", wildcards)


def _build_primary_metrics(task_names):
    """Build primary_metrics dict from task name list."""
    tasks = expand_tasks(task_names)
    return {t["name"]: t["primary_metric"] for t in tasks}


def _build_early_stopping_metrics(task_names):
    """Build early_stopping_metrics dict from standardized_datasets config.

    For each task, looks up optional early_stopping_metric from the dataset
    metadata, falling back to primary_metric.
    """
    result = {}
    for name in task_names:
        meta = resolve_dataset_meta(name)
        es_metric = meta.get("early_stopping_metric")
        if es_metric:
            result[name] = es_metric
    return result


rule finetune_heads_only:
    input:
        unpack(get_heads_only_parquet_inputs),
    output:
        tokenizer=directory("results/train/heads_only/{run}/tokenizer"),
        model=directory("results/train/heads_only/{run}/model"),
        best_overall_model=directory("results/train/heads_only/{run}/best_overall_model"),
        best_task_models=directory("results/train/heads_only/{run}/best_task_models"),
        best_loss_overall_model=directory("results/train/heads_only/{run}/best_loss_overall_model"),
        best_loss_task_models=directory("results/train/heads_only/{run}/best_loss_task_models"),
        history="results/train/heads_only/{run}/history.json",
        best_checkpoints="results/train/heads_only/{run}/best_checkpoints.json",
    params:
        checkpoint=lambda wc: _get_train_run_local("heads_only", wc.run)["checkpoint"],
        max_epochs=lambda wc: _get_train_run_local("heads_only", wc.run).get("max_epochs"),
        max_validations=lambda wc: _get_train_run_local("heads_only", wc.run).get(
            "max_validations"
        ),
        batch=lambda wc: _get_train_run_local("heads_only", wc.run)["batch"],
        lr=lambda wc: _get_train_run_local("heads_only", wc.run)["lr"],
        dropout=lambda wc: _get_train_run_local("heads_only", wc.run)["dropout"],
        warmup_ratio=lambda wc: _get_train_run_local("heads_only", wc.run)["warmup_ratio"],
        auto_mixed_precision=lambda wc: _get_train_run_local("heads_only", wc.run).get(
            "auto_mixed_precision", False
        ),
        validate_every_n_train_batches=config.get("train", {}).get(
            "validate_every_n_train_batches", 0
        ),
        embedding_cache_dir=config.get("train", {}).get("embedding_cache_dir", None),
        early_stopping_patience=lambda wc: _get_train_run_local("heads_only", wc.run).get(
            "early_stopping_patience", 0
        ),
        early_stopping_metrics=lambda wc: _build_early_stopping_metrics(
            _get_train_run_local("heads_only", wc.run)["tasks"]
        ),
        primary_metrics=lambda wc: _build_primary_metrics(
            _get_train_run_local("heads_only", wc.run)["tasks"]
        ),
        training_checkpoint_dir=lambda wc: f"{config.get('train', {}).get('training_checkpoint_dir')}/heads_only/{wc.run}"
        if config.get("train", {}).get("training_checkpoint_dir")
        else None,
        checkpoint_every_n_validations=config.get("train", {}).get(
            "checkpoint_every_n_validations", 1
        ),
        tasks=lambda wc: expand_tasks(_get_train_run_local("heads_only", wc.run)["tasks"]),
        uncertainty_weighting=lambda wc: _get_train_run_local("heads_only", wc.run).get("uncertainty_weighting", False),
    conda:
        "../../envs/finetune/train.yml"
    log:
        f"{LOG_PREFIX}/train/heads_only/{{run}}.log"
    script:
        "../../scripts/finetune/finetune_heads_only.py"


rule finetune_lora:
    input:
        unpack(get_lora_parquet_inputs),
    output:
        tokenizer=directory("results/train/lora/{run}/tokenizer"),
        model=directory("results/train/lora/{run}/model"),
        best_overall_model=directory("results/train/lora/{run}/best_overall_model"),
        best_task_models=directory("results/train/lora/{run}/best_task_models"),
        best_loss_overall_model=directory("results/train/lora/{run}/best_loss_overall_model"),
        best_loss_task_models=directory("results/train/lora/{run}/best_loss_task_models"),
        history="results/train/lora/{run}/history.json",
        best_checkpoints="results/train/lora/{run}/best_checkpoints.json",
    params:
        checkpoint=lambda wc: _get_train_run_local("lora", wc.run)["checkpoint"],
        max_epochs=lambda wc: _get_train_run_local("lora", wc.run).get("max_epochs"),
        max_validations=lambda wc: _get_train_run_local("lora", wc.run).get(
            "max_validations"
        ),
        batch=lambda wc: _get_train_run_local("lora", wc.run)["batch"],
        lr=lambda wc: _get_train_run_local("lora", wc.run)["lr"],
        dropout=lambda wc: _get_train_run_local("lora", wc.run)["dropout"],
        warmup_ratio=lambda wc: _get_train_run_local("lora", wc.run)["warmup_ratio"],
        auto_mixed_precision=lambda wc: _get_train_run_local("lora", wc.run).get(
            "auto_mixed_precision", False
        ),
        lora_rank=lambda wc: _get_train_run_local("lora", wc.run).get("lora_rank", None),
        lora_alpha=lambda wc: _get_train_run_local("lora", wc.run).get(
            "lora_alpha", None
        ),
        validate_every_n_train_batches=config.get("train", {}).get(
            "validate_every_n_train_batches", 0
        ),
        early_stopping_patience=lambda wc: _get_train_run_local("lora", wc.run).get(
            "early_stopping_patience", 0
        ),
        early_stopping_metrics=lambda wc: _build_early_stopping_metrics(
            _get_train_run_local("lora", wc.run)["tasks"]
        ),
        primary_metrics=lambda wc: _build_primary_metrics(
            _get_train_run_local("lora", wc.run)["tasks"]
        ),
        training_checkpoint_dir=lambda wc: f"{config.get('train', {}).get('training_checkpoint_dir')}/lora/{wc.run}"
        if config.get("train", {}).get("training_checkpoint_dir")
        else None,
        checkpoint_every_n_validations=config.get("train", {}).get(
            "checkpoint_every_n_validations", 1
        ),
        tasks=lambda wc: expand_tasks(_get_train_run_local("lora", wc.run)["tasks"]),
        uncertainty_weighting=lambda wc: _get_train_run_local("lora", wc.run).get("uncertainty_weighting", False),
    conda:
        "../../envs/finetune/train.yml"
    log:
        f"{LOG_PREFIX}/train/lora/{{run}}.log"
    script:
        "../../scripts/finetune/finetune_lora.py"


rule finetune_accgrad_lora:
    input:
        unpack(get_accgrad_lora_parquet_inputs),
    output:
        tokenizer=directory("results/train/accgrad_lora/{run}/tokenizer"),
        model=directory("results/train/accgrad_lora/{run}/model"),
        best_overall_model=directory("results/train/accgrad_lora/{run}/best_overall_model"),
        best_task_models=directory("results/train/accgrad_lora/{run}/best_task_models"),
        best_loss_overall_model=directory("results/train/accgrad_lora/{run}/best_loss_overall_model"),
        best_loss_task_models=directory("results/train/accgrad_lora/{run}/best_loss_task_models"),
        history="results/train/accgrad_lora/{run}/history.json",
        best_checkpoints="results/train/accgrad_lora/{run}/best_checkpoints.json",
    params:
        checkpoint=lambda wc: _get_train_run_local("accgrad_lora", wc.run)["checkpoint"],
        max_epochs=lambda wc: _get_train_run_local("accgrad_lora", wc.run).get(
            "max_epochs"
        ),
        max_validations=lambda wc: _get_train_run_local("accgrad_lora", wc.run).get(
            "max_validations"
        ),
        batch=lambda wc: _get_train_run_local("accgrad_lora", wc.run)["batch"],
        lr=lambda wc: _get_train_run_local("accgrad_lora", wc.run)["lr"],
        dropout=lambda wc: _get_train_run_local("accgrad_lora", wc.run)["dropout"],
        warmup_ratio=lambda wc: _get_train_run_local("accgrad_lora", wc.run)[
            "warmup_ratio"
        ],
        auto_mixed_precision=lambda wc: _get_train_run_local("accgrad_lora", wc.run).get(
            "auto_mixed_precision", False
        ),
        lora_rank=lambda wc: _get_train_run_local("accgrad_lora", wc.run).get(
            "lora_rank", None
        ),
        lora_alpha=lambda wc: _get_train_run_local("accgrad_lora", wc.run).get(
            "lora_alpha", None
        ),
        validate_every_n_train_batches=config.get("train", {}).get(
            "validate_every_n_train_batches", 0
        ),
        early_stopping_patience=lambda wc: _get_train_run_local(
            "accgrad_lora", wc.run
        ).get("early_stopping_patience", 0),
        early_stopping_metrics=lambda wc: _build_early_stopping_metrics(
            _get_train_run_local("accgrad_lora", wc.run)["tasks"]
        ),
        primary_metrics=lambda wc: _build_primary_metrics(
            _get_train_run_local("accgrad_lora", wc.run)["tasks"]
        ),
        training_checkpoint_dir=lambda wc: f"{config.get('train', {}).get('training_checkpoint_dir')}/accgrad_lora/{wc.run}"
        if config.get("train", {}).get("training_checkpoint_dir")
        else None,
        checkpoint_every_n_validations=config.get("train", {}).get(
            "checkpoint_every_n_validations", 1
        ),
        tasks=lambda wc: expand_tasks(_get_train_run_local("accgrad_lora", wc.run)["tasks"]),
        uncertainty_weighting=lambda wc: _get_train_run_local("accgrad_lora", wc.run).get("uncertainty_weighting", False),
    conda:
        "../../envs/finetune/train.yml"
    log:
        f"{LOG_PREFIX}/train/accgrad_lora/{{run}}.log"
    script:
        "../../scripts/finetune/finetune_accgrad_lora.py"


rule finetune_align_lora:
    input:
        unpack(get_align_lora_parquet_inputs),
    output:
        tokenizer=directory("results/train/align_lora/{run}/tokenizer"),
        model=directory("results/train/align_lora/{run}/model"),
        best_overall_model=directory("results/train/align_lora/{run}/best_overall_model"),
        best_task_models=directory("results/train/align_lora/{run}/best_task_models"),
        best_loss_overall_model=directory("results/train/align_lora/{run}/best_loss_overall_model"),
        best_loss_task_models=directory("results/train/align_lora/{run}/best_loss_task_models"),
        history="results/train/align_lora/{run}/history.json",
        best_checkpoints="results/train/align_lora/{run}/best_checkpoints.json",
    params:
        checkpoint=lambda wc: _get_train_run_local("align_lora", wc.run)["checkpoint"],
        max_epochs=lambda wc: _get_train_run_local("align_lora", wc.run).get(
            "max_epochs"
        ),
        max_validations=lambda wc: _get_train_run_local("align_lora", wc.run).get(
            "max_validations"
        ),
        batch=lambda wc: _get_train_run_local("align_lora", wc.run)["batch"],
        lr=lambda wc: _get_train_run_local("align_lora", wc.run)["lr"],
        dropout=lambda wc: _get_train_run_local("align_lora", wc.run)["dropout"],
        warmup_ratio=lambda wc: _get_train_run_local("align_lora", wc.run)[
            "warmup_ratio"
        ],
        auto_mixed_precision=lambda wc: _get_train_run_local("align_lora", wc.run).get(
            "auto_mixed_precision", False
        ),
        lora_rank=lambda wc: _get_train_run_local("align_lora", wc.run).get(
            "lora_rank", None
        ),
        lora_alpha=lambda wc: _get_train_run_local("align_lora", wc.run).get(
            "lora_alpha", None
        ),
        align_lora_kl_lambda=lambda wc: _get_train_run_local("align_lora", wc.run).get(
            "align_lora_kl_lambda", 0.01
        ),
        gradient_checkpointing=lambda wc: _get_train_run_local("align_lora", wc.run).get(
            "gradient_checkpointing", False
        ),
        validate_every_n_train_batches=config.get("train", {}).get(
            "validate_every_n_train_batches", 0
        ),
        early_stopping_patience=lambda wc: _get_train_run_local("align_lora", wc.run).get(
            "early_stopping_patience", 0
        ),
        early_stopping_metrics=lambda wc: _build_early_stopping_metrics(
            _get_train_run_local("align_lora", wc.run)["tasks"]
        ),
        primary_metrics=lambda wc: _build_primary_metrics(
            _get_train_run_local("align_lora", wc.run)["tasks"]
        ),
        training_checkpoint_dir=lambda wc: f"{config.get('train', {}).get('training_checkpoint_dir')}/align_lora/{wc.run}"
        if config.get("train", {}).get("training_checkpoint_dir")
        else None,
        checkpoint_every_n_validations=config.get("train", {}).get(
            "checkpoint_every_n_validations", 1
        ),
        tasks=lambda wc: expand_tasks(_get_train_run_local("align_lora", wc.run)["tasks"]),
        uncertainty_weighting=lambda wc: _get_train_run_local("align_lora", wc.run).get("uncertainty_weighting", False),
    conda:
        "../../envs/finetune/train.yml"
    log:
        f"{LOG_PREFIX}/train/align_lora/{{run}}.log"
    script:
        "../../scripts/finetune/finetune_align_lora.py"


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def _onnx_model_path_for_variant(section, run_id, variant):
    """Return the model directory path for the given variant."""
    base = f"results/train/{section}/{run_id}"
    if variant == "final":
        return f"{base}/model"
    elif variant == "best_overall":
        return f"{base}/best_overall_model"
    else:
        raise ValueError(f"Unknown ONNX export variant: {variant}")


def get_onnx_export_inputs(wildcards):
    section = wildcards.section
    run_id = wildcards.run
    variant = wildcards.variant

    inputs = {
    }
    inputs["model_dir"] = _onnx_model_path_for_variant(section, run_id, variant)
    return inputs


rule export_onnx:
    """Export a trained MultitaskModel to ONNX with merged LoRA weights and tokenizer."""
    input:
        unpack(get_onnx_export_inputs),
    output:
        model="results/onnx_export/{section}/{run}/{variant}/model.onnx",
        metadata="results/onnx_export/{section}/{run}/{variant}/export_metadata.json",
        tokenizer="results/onnx_export/{section}/{run}/{variant}/tokenizer/tokenizer.json",
    wildcard_constraints:
        section="[^/]+",
        run="[^/]+",
        variant="[^/]+",
    conda:
        "../../envs/finetune/export_onnx.yml"
    log:
        f"{LOG_PREFIX}/onnx_export/{{section}}_{{run}}_{{variant}}.log"
    script:
        "../../scripts/finetune/export_onnx.py"


# ---------------------------------------------------------------------------
# Normalization statistics for regression tasks
# ---------------------------------------------------------------------------

def _get_normalization_stats_parquet_inputs(wildcards):
    """Build input dict with training parquets for all tasks in the ONNX export run."""
    section = wildcards.section
    run_id = wildcards.run
    run = _get_train_run_local(section, run_id)
    tasks = expand_tasks(run["tasks"])

    inputs = {
    }

    for task in tasks:
        name = task["name"]
        subset_type = task["subset_type"]
        dataset = task["dataset"]
        base = f"results/datasets_resplit/{subset_type}/{dataset}"
        inputs[f"{name}_train"] = f"{base}/train.parquet"

    return inputs


rule generate_model_manifest:
    """Generate a model manifest (models.json) listing all ONNX exports.

    The Dioxus inference app fetches this manifest on startup to populate
    its model selector dropdown.
    """
    input:
        export_metadata=[
            f"results/onnx_export/{e['section']}/{e['id']}/{e['variant']}/export_metadata.json"
            for e in config.get("onnx_export", [])
        ],
    output:
        manifest="results/onnx_export/models.json",
    params:
        onnx_export_entries=config.get("onnx_export", []),
        onnx_export_base="results/onnx_export",
    log:
        f"{LOG_PREFIX}/generate_model_manifest.log"
    script:
        "../../scripts/finetune/generate_model_manifest.py"


rule compute_normalization_stats:
    """Compute per-task normalization statistics from training parquets.

    Only regression tasks produce entries. The resulting JSON is co-located
    with the ONNX export so each model directory is self-contained.
    """
    input:
        unpack(_get_normalization_stats_parquet_inputs),
        export_metadata="results/onnx_export/{section}/{run}/{variant}/export_metadata.json",
    output:
        normalization_stats="results/onnx_export/{section}/{run}/{variant}/normalization_stats.json",
    wildcard_constraints:
        section="[^/]+",
        run="[^/]+",
        variant="[^/]+",
    params:
        tasks=lambda wc: expand_tasks(
            _get_train_run_local(wc.section, wc.run)["tasks"]
        ),
        datasets_resplit_dir="results/datasets_resplit",
    conda:
        "../../envs/finetune/normalization_stats.yml"
    log:
        f"{LOG_PREFIX}/normalization_stats/{{section}}_{{run}}_{{variant}}.log"
    script:
        "../../scripts/finetune/compute_normalization_stats.py"
