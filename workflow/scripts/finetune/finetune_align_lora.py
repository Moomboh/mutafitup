import json

from snakemake.script import snakemake

from mutafitup.datasets.per_protein_classification_dataset import (
    PerProteinClassificationDataset,
)
from mutafitup.datasets.per_protein_regression_dataset import (
    PerProteinRegressionDataset,
)
from mutafitup.datasets.per_residue_classification_dataset import (
    PerResidueClassificationDataset,
)
from mutafitup.datasets.per_residue_regression_dataset import (
    PerResidueRegressionDataset,
)
from mutafitup.models import build_backbone_and_tokenizer
from mutafitup.models.multitask_heads import (
    PerProteinClassificationHead,
    PerProteinLatePoolClassificationHead,
    PerProteinLatePoolRegressionHead,
    PerProteinRegressionHead,
    PerResidueClassificationHead,
    PerResidueRegressionHead,
)
from mutafitup.models.multitask_model import (
    HeadConfig,
    MultitaskBackbone,
    MultitaskModel,
)
from mutafitup.train.train_multitask_model import train_multitask_align_lora
from wfutils.logging import get_logger, log_snakemake_info


logger = get_logger(__name__)
log_snakemake_info(logger)


def build_multitask_model(
    backbone: MultitaskBackbone,
    dropout: float,
    tasks,
) -> MultitaskModel:
    hidden_size = backbone.hidden_size  # type: ignore[attr-defined]

    heads = {}

    for task in tasks:
        name = task["name"]
        subset_type = task["subset_type"]

        if subset_type == "per_residue_classification":
            num_labels = task.get("num_labels")
            if num_labels is None:
                raise ValueError(
                    f"num_labels is required for classification task '{name}'"
                )
            heads[name] = HeadConfig(
                head=PerResidueClassificationHead(
                    dropout, hidden_size, hidden_size, num_labels
                ),
                problem_type="classification",
                level="per_residue",
            )
        elif subset_type == "per_residue_regression":
            heads[name] = HeadConfig(
                head=PerResidueRegressionHead(dropout, hidden_size, hidden_size),
                problem_type="regression",
                level="per_residue",
            )
        elif subset_type == "per_protein_classification":
            num_labels = task.get("num_labels")
            if num_labels is None:
                raise ValueError(
                    f"num_labels is required for classification task '{name}'"
                )
            head_variant = task.get("head_variant", "default")
            if head_variant == "late_pool":
                head_cls = PerProteinLatePoolClassificationHead
            else:
                head_cls = PerProteinClassificationHead
            heads[name] = HeadConfig(
                head=head_cls(
                    dropout, hidden_size, hidden_size, num_labels
                ),
                problem_type="classification",
                level="per_protein",
            )
        elif subset_type == "per_protein_regression":
            head_variant = task.get("head_variant", "default")
            if head_variant == "late_pool":
                head_cls = PerProteinLatePoolRegressionHead
            else:
                head_cls = PerProteinRegressionHead
            heads[name] = HeadConfig(
                head=head_cls(dropout, hidden_size, hidden_size),
                problem_type="regression",
                level="per_protein",
            )
        else:
            raise ValueError(f"Unsupported subset type {subset_type} for task {name}")

    return MultitaskModel(backbone=backbone, heads=heads)


checkpoint = snakemake.params["checkpoint"]
batch = snakemake.params["batch"]
max_epochs = snakemake.params["max_epochs"]
max_validations = snakemake.params["max_validations"]
lr = snakemake.params["lr"]
warmup_ratio = snakemake.params["warmup_ratio"]
dropout = snakemake.params["dropout"]
lora_rank = snakemake.params["lora_rank"]
lora_alpha = snakemake.params["lora_alpha"]
tasks = snakemake.params["tasks"]
validate_every_n_train_batches = snakemake.params["validate_every_n_train_batches"]
early_stopping_patience = snakemake.params["early_stopping_patience"]
early_stopping_metrics = snakemake.params["early_stopping_metrics"]
primary_metrics = snakemake.params["primary_metrics"]
training_checkpoint_dir = snakemake.params["training_checkpoint_dir"]
checkpoint_every_n_validations = snakemake.params["checkpoint_every_n_validations"]
uncertainty_weighting = snakemake.params.get("uncertainty_weighting", False)
align_lora_kl_lambda = snakemake.params["align_lora_kl_lambda"]
gradient_checkpointing = snakemake.params["gradient_checkpointing"]

logger.info("Start training (MultitaskModel align_lora)")
auto_mixed_precision = snakemake.params["auto_mixed_precision"]

backbone, tokenizer = build_backbone_and_tokenizer(
    checkpoint,
    lora_rank,
    lora_alpha,
)
model = build_multitask_model(backbone, dropout, tasks)

task_datasets = {}

for task in tasks:
    name = task["name"]
    subset_type = task["subset_type"]

    train_key = f"{name}_train"
    valid_key = f"{name}_valid"

    if subset_type == "per_residue_classification":
        task_datasets[name] = PerResidueClassificationDataset(
            name=name,
            train_parquet=snakemake.input[train_key],
            valid_parquet=snakemake.input[valid_key],
            label_column="label",
        )
    elif subset_type == "per_residue_regression":
        task_datasets[name] = PerResidueRegressionDataset(
            name=name,
            train_parquet=snakemake.input[train_key],
            valid_parquet=snakemake.input[valid_key],
            label_column="score",
            unresolved_marker=999.0,
        )
    elif subset_type == "per_protein_classification":
        task_datasets[name] = PerProteinClassificationDataset(
            name=name,
            train_parquet=snakemake.input[train_key],
            valid_parquet=snakemake.input[valid_key],
            label_column="label_numeric",
        )
    elif subset_type == "per_protein_regression":
        task_datasets[name] = PerProteinRegressionDataset(
            name=name,
            train_parquet=snakemake.input[train_key],
            valid_parquet=snakemake.input[valid_key],
            label_column="score",
            unresolved_marker=999.0,
        )
    else:
        raise ValueError(f"Unsupported subset type {subset_type} for task {name}")


model, history, best_checkpoints = train_multitask_align_lora(
    model=model,
    tokenizer=tokenizer,
    checkpoint=checkpoint,
    task_datasets=task_datasets,
    batch=batch,
    max_epochs=max_epochs,
    max_validations=max_validations,
    lr=lr,
    warmup_ratio=warmup_ratio,
    logger=logger,
    validate_every_n_train_batches=validate_every_n_train_batches,
    early_stopping_patience=early_stopping_patience,
    early_stopping_metrics=early_stopping_metrics,
    primary_metrics=primary_metrics,
    training_checkpoint_dir=training_checkpoint_dir,
    checkpoint_every_n_validations=checkpoint_every_n_validations,
    align_lora_kl_lambda=align_lora_kl_lambda,
    gradient_checkpointing=gradient_checkpointing,
    best_overall_model_dir=snakemake.output["best_overall_model"],
    best_task_models_dir=snakemake.output["best_task_models"],
    best_loss_overall_model_dir=snakemake.output["best_loss_overall_model"],
    best_loss_task_models_dir=snakemake.output["best_loss_task_models"],
    auto_mixed_precision=auto_mixed_precision,
    uncertainty_weighting=uncertainty_weighting,
)

logger.info("Training complete (MultitaskModel align_lora)")

tokenizer.save_pretrained(snakemake.output["tokenizer"])
logger.info(f"Tokenizer saved to {snakemake.output['tokenizer']}")

num_params = model.save_trainable_weights(
    save_directory=snakemake.output["model"],
    base_checkpoint=checkpoint,
)
logger.info(f"Model saved to {snakemake.output['model']}")
logger.info(f"Model size: {num_params / 1e6:.2f}M parameters")

with open(snakemake.output["history"], "w") as f:
    f.write(json.dumps(history, indent=2))

logger.info(f"History saved to {snakemake.output['history']}")

with open(snakemake.output["best_checkpoints"], "w") as f:
    f.write(json.dumps(best_checkpoints, indent=2))

logger.info(f"Best checkpoints info saved to {snakemake.output['best_checkpoints']}")

for name, info in best_checkpoints["tasks"].items():
    logger.info(
        f"Best checkpoint for task {name}: epoch {info['epoch']} "
        f"with metric {info['metric']:.4f}"
    )

best_overall = best_checkpoints["best_overall"]
logger.info(
    f"Best overall checkpoint (by relative improvement): epoch {best_overall['epoch']}"
)

for name, info in best_checkpoints["best_loss_tasks"].items():
    logger.info(
        f"Best loss checkpoint for task {name}: epoch {info['epoch']} "
        f"with loss {info['loss']:.4f}"
    )

best_loss_overall = best_checkpoints["best_loss_overall"]
logger.info(
    f"Best overall loss checkpoint (by relative improvement): "
    f"epoch {best_loss_overall['epoch']}"
)
