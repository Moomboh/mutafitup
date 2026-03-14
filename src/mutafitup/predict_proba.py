"""Probability prediction for binary per-residue classification tasks.

Like :mod:`mutafitup.predict` but stores ``softmax(logits, dim=-1)[:, 1]``
(the positive-class probability per residue) instead of ``argmax`` class
indices.  This is required for computing AUPRC.

Output format (JSONL per task)::

    {"id": "...", "probability": [0.12, 0.95, ...], "target": [0, 1, ...]}
"""

import json
import logging
import os
from typing import Dict, List, Optional

import torch
from transformers import PreTrainedTokenizerBase

from mutafitup.datasets import BaseMultitaskDataset
from mutafitup.models.multitask_model import MultitaskForwardArgs, MultitaskModel


def predict_proba(
    model: MultitaskModel,
    tokenizer: PreTrainedTokenizerBase,
    checkpoint: str,
    task_datasets: Dict[str, BaseMultitaskDataset],
    split: str,
    batch_size: int = 4,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, List[dict]]:
    """Run inference and return per-residue positive-class probabilities.

    Only supports **per-residue binary classification** tasks
    (``num_labels == 2``).

    Parameters
    ----------
    model : MultitaskModel
        The loaded multitask model (already on the correct device).
    tokenizer : PreTrainedTokenizerBase
        Tokenizer matching the model's backbone.
    checkpoint : str
        The base checkpoint identifier.
    task_datasets : dict
        Mapping from task name to a ``BaseMultitaskDataset`` instance.
    split : str
        Which split to predict on (e.g. ``"valid"`` or ``"test"``).
    batch_size : int
        Batch size for inference.
    logger : Logger, optional

    Returns
    -------
    dict
        Mapping from task name to a list of dicts, each with keys
        ``id``, ``probability`` (list of floats), ``target`` (list of ints).
    """
    device = next(model.parameters()).device

    task_names = list(task_datasets.keys())
    results: Dict[str, List[dict]] = {}

    model.eval()

    with torch.no_grad():
        for name in task_names:
            if logger is not None:
                logger.info(f"Predicting probabilities task={name} split={split}")

            dataset = task_datasets[name]
            config = model.head_configs[name]
            problem_type = config.problem_type
            level = config.level
            num_labels = config.head.output.out_features
            ignore_index = config.ignore_index

            if level != "per_residue":
                raise ValueError(
                    f"predict_proba only supports per_residue tasks, "
                    f"got level={level!r} for task {name!r}"
                )
            if problem_type != "classification":
                raise ValueError(
                    f"predict_proba only supports classification tasks, "
                    f"got problem_type={problem_type!r} for task {name!r}"
                )
            if num_labels != 2:
                raise ValueError(
                    f"predict_proba only supports binary classification "
                    f"(num_labels=2), got num_labels={num_labels} for task {name!r}"
                )

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

                # logits shape: [B, L, 2]
                probs = torch.softmax(logits, dim=-1)  # [B, L, 2]

                for i in range(current_batch_size):
                    sample_id = ids[sample_idx]

                    attn = attention_mask[i]
                    sample_labels = labels[i]
                    mask = (sample_labels != ignore_index) & (attn == 1)

                    # P(positive class) per residue
                    prob_values = probs[i, :, 1][mask].cpu().tolist()
                    target_values = sample_labels[mask].cpu().tolist()

                    task_results.append(
                        {
                            "id": sample_id,
                            "probability": prob_values,
                            "target": target_values,
                        }
                    )

                    sample_idx += 1

            results[name] = task_results

            if logger is not None:
                logger.info(
                    f"Task {name}: {len(task_results)} probability predictions "
                    f"on split={split}"
                )

    return results


def save_prob_predictions_jsonl(
    predictions: Dict[str, List[dict]],
    output_dir: str,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, str]:
    """Write probability predictions to JSONL files.

    Writes one file per task at ``{output_dir}/{task}.jsonl``.

    Parameters
    ----------
    predictions : dict
        Output of :func:`predict_proba`.
    output_dir : str
        Output directory.
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
            logger.info(f"Wrote {len(rows)} probability predictions to {out_path}")

    return written
