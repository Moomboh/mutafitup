import os

from snakemake.script import snakemake

from mutafitup.datasets import (
    PerProteinClassificationDataset,
    PerProteinRegressionDataset,
    PerResidueClassificationDataset,
    PerResidueRegressionDataset,
)
from mutafitup.device import get_device
from mutafitup.models.multitask_model import MultitaskModel
from mutafitup.predict import predict, save_predictions_jsonl
from wfutils.logging import get_logger, log_snakemake_info


logger = get_logger(__name__)
log_snakemake_info(logger)


variant = snakemake.params["variant"]
split = snakemake.params["split"]
tasks = snakemake.params["tasks"]
batch_size = snakemake.params["batch"]
output_dir = snakemake.params["output_dir"]

device = get_device()


def load_model_and_tokenizer(model_path):
    """Load a MultitaskModel and its tokenizer from a model.pt file."""
    logger.info(f"Loading model from {model_path}")
    model = MultitaskModel.load_from_file(model_path, device=device)
    model.eval()

    saved_data = __import__("torch").load(
        model_path, map_location="cpu", weights_only=False
    )
    base_checkpoint = saved_data["config"]["base_checkpoint"]

    backbone_class_path = saved_data["config"]["backbone_class"]
    module_name, attr_name = backbone_class_path.rsplit(".", 1)
    backbone_module = __import__(module_name, fromlist=[attr_name])
    backbone_cls = getattr(backbone_module, attr_name)
    _, tokenizer = backbone_cls.from_pretrained(
        checkpoint=base_checkpoint,
        lora_config=None,
    )

    logger.info(f"Model loaded, base_checkpoint={base_checkpoint}")
    return model, tokenizer, base_checkpoint


def build_task_datasets(tasks_config):
    """Build task datasets from the snakemake task config."""
    datasets = {}
    for task in tasks_config:
        name = task["name"]
        subset_type = task["subset_type"]

        train_key = f"{name}_train"
        valid_key = f"{name}_valid"
        test_key = f"{name}_test"
        test_parquet = snakemake.input.get(test_key, None)

        if subset_type == "per_residue_classification":
            datasets[name] = PerResidueClassificationDataset(
                name=name,
                train_parquet=snakemake.input[train_key],
                valid_parquet=snakemake.input[valid_key],
                label_column="label",
                test_parquet=test_parquet,
            )
        elif subset_type == "per_residue_regression":
            datasets[name] = PerResidueRegressionDataset(
                name=name,
                train_parquet=snakemake.input[train_key],
                valid_parquet=snakemake.input[valid_key],
                label_column="score",
                unresolved_marker=999.0,
                test_parquet=test_parquet,
            )
        elif subset_type == "per_protein_classification":
            datasets[name] = PerProteinClassificationDataset(
                name=name,
                train_parquet=snakemake.input[train_key],
                valid_parquet=snakemake.input[valid_key],
                label_column="label_numeric",
                test_parquet=test_parquet,
            )
        elif subset_type == "per_protein_regression":
            datasets[name] = PerProteinRegressionDataset(
                name=name,
                train_parquet=snakemake.input[train_key],
                valid_parquet=snakemake.input[valid_key],
                label_column="score",
                unresolved_marker=999.0,
                test_parquet=test_parquet,
            )
        else:
            raise ValueError(f"Unsupported subset type {subset_type} for task {name}")

    return datasets


task_datasets = build_task_datasets(tasks)

logger.info(f"Running predictions on split={split}, variant={variant}")

if variant in ("final", "best_overall", "best_loss_overall"):
    # Single model for all tasks
    model_dir = snakemake.input["model_dir"]
    model_path = os.path.join(model_dir, "model.pt")
    model, tokenizer, base_checkpoint = load_model_and_tokenizer(model_path)

    predictions = predict(
        model=model,
        tokenizer=tokenizer,
        checkpoint=base_checkpoint,
        task_datasets=task_datasets,
        split=split,
        batch_size=batch_size,
        logger=logger,
    )

elif variant in ("best_task", "best_loss_task"):
    # Per-task models: load each task's best model separately
    model_dir = snakemake.input["model_dir"]
    predictions = {}

    for task in tasks:
        name = task["name"]
        task_model_path = os.path.join(model_dir, name, "model.pt")

        if not os.path.exists(task_model_path):
            logger.warning(
                f"No best_task model found for {name} at {task_model_path}, skipping"
            )
            continue

        model, tokenizer, base_checkpoint = load_model_and_tokenizer(task_model_path)

        # Predict only this task
        single_task_datasets = {name: task_datasets[name]}
        task_preds = predict(
            model=model,
            tokenizer=tokenizer,
            checkpoint=base_checkpoint,
            task_datasets=single_task_datasets,
            split=split,
            batch_size=batch_size,
            logger=logger,
        )
        predictions[name] = task_preds[name]

else:
    raise ValueError(f"Unknown variant: {variant}")

written = save_predictions_jsonl(
    predictions=predictions,
    output_dir=output_dir,
    logger=logger,
)

logger.info("Prediction complete")
for task_name, path in written.items():
    logger.info(f"  {task_name}: {path}")
