import json
import logging
import os
from typing import Dict, List, Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from mutafitup.datasets import BaseMultitaskDataset
from mutafitup.device import get_device
from mutafitup.models.multitask_model import MultitaskForwardArgs, MultitaskModel


def predict(
    model: MultitaskModel,
    tokenizer: PreTrainedTokenizerBase,
    checkpoint: str,
    task_datasets: Dict[str, BaseMultitaskDataset],
    split: str,
    batch_size: int = 4,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, List[dict]]:
    """Run inference on a multitask model and return per-sample predictions.

    Parameters
    ----------
    model : MultitaskModel
        The loaded multitask model (already on the correct device).
    tokenizer : PreTrainedTokenizerBase
        Tokenizer matching the model's backbone.
    checkpoint : str
        The base checkpoint identifier (needed for dataloader construction).
    task_datasets : dict
        Mapping from task name to a ``BaseMultitaskDataset`` instance.
    split : str
        Which split to predict on (e.g. ``"valid"`` or ``"test"``).
    batch_size : int
        Batch size for inference.
    logger : Logger, optional
        Logger for progress messages.

    Returns
    -------
    dict
        Mapping from task name to a list of dicts, each with keys
        ``id``, ``prediction``, ``target``, ``sequence``.
    """
    device = next(model.parameters()).device

    task_names = list(task_datasets.keys())
    results: Dict[str, List[dict]] = {}

    model.eval()

    with torch.no_grad():
        for name in task_names:
            if logger is not None:
                logger.info(f"Predicting task={name} split={split}")

            dataset = task_datasets[name]
            config = model.head_configs[name]
            problem_type = config.problem_type
            level = config.level
            ignore_index = config.ignore_index

            # Get the dataframe for sequences and ids
            if split == "train":
                df = dataset.train_df
            elif split == "valid":
                df = dataset.valid_df
            elif split == "test":
                if dataset.test_df is None:
                    raise ValueError(f"No test_parquet was provided for task {name!r}.")
                df = dataset.test_df
            else:
                raise ValueError(
                    f"Unsupported split {split!r}. "
                    f"Supported splits: 'train', 'valid', 'test'."
                )

            sequences = list(df["sequence"])

            # Determine id column: use dataframe index
            ids = list(df.index)

            loader = dataset.get_dataloader(
                split=split,
                backbone=model.backbone,
                tokenizer=tokenizer,
                checkpoint=checkpoint,
                batch_size=batch_size,
            )

            task_results: List[dict] = []
            sample_idx = 0

            for batch_data in loader:
                batch_data = {k: v.to(device) for k, v in batch_data.items()}

                if problem_type == "regression":
                    batch_data["labels"] = batch_data["labels"].float()

                args = MultitaskForwardArgs(
                    attention_mask=batch_data["attention_mask"],
                    input_ids=batch_data["input_ids"],
                )
                _, logits = model(
                    task=name,
                    args=args,
                    labels=batch_data["labels"],
                )

                labels = batch_data["labels"]
                attention_mask = batch_data["attention_mask"]
                current_batch_size = labels.size(0)

                for i in range(current_batch_size):
                    seq = sequences[sample_idx]
                    sample_id = ids[sample_idx]

                    if level == "per_residue":
                        attn = attention_mask[i]
                        sample_labels = labels[i]
                        mask = (sample_labels != ignore_index) & (attn == 1)

                        if problem_type == "classification":
                            sample_preds = torch.argmax(logits[i], dim=-1)
                            pred_values = sample_preds[mask].cpu().tolist()
                        else:
                            sample_preds = logits[i].view(-1)
                            pred_values = sample_preds[mask].cpu().tolist()

                        target_values = sample_labels[mask].cpu().tolist()

                    elif level == "per_protein":
                        if problem_type == "classification":
                            pred_values = int(
                                torch.argmax(logits[i], dim=-1).cpu().item()
                            )
                        else:
                            pred_values = float(logits[i].view(-1).cpu().item())

                        target_values = labels[i].cpu().item()
                        if problem_type == "classification":
                            target_values = int(target_values)
                        else:
                            target_values = float(target_values)

                    else:
                        raise ValueError(f"Unsupported level: {level}")

                    task_results.append(
                        {
                            "id": sample_id,
                            "prediction": pred_values,
                            "target": target_values,
                            "sequence": seq,
                        }
                    )

                    sample_idx += 1

            results[name] = task_results

            if logger is not None:
                logger.info(
                    f"Task {name}: {len(task_results)} predictions on split={split}"
                )

    return results


def save_predictions_jsonl(
    predictions: Dict[str, List[dict]],
    output_dir: str,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, str]:
    """Write predictions to JSONL files.

    Writes one file per task at ``{output_dir}/{task}.jsonl``.

    Parameters
    ----------
    predictions : dict
        Output of :func:`predict`.
    output_dir : str
        Output directory (e.g. ``results/predictions/{run_id}/{split}``).
    logger : Logger, optional

    Returns
    -------
    dict
        Mapping from task name to the path of the written JSONL file.
    """
    os.makedirs(output_dir, exist_ok=True)
    written: Dict[str, str] = {}

    for task_name, rows in predictions.items():
        out_path = os.path.join(output_dir, f"{task_name}.jsonl")

        with open(out_path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

        written[task_name] = out_path

        if logger is not None:
            logger.info(f"Wrote {len(rows)} predictions to {out_path}")

    return written
