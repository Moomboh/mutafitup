"""Training module for aligned LoRA-based multitask model training.

This approach uses Low-Rank Adaptation (LoRA) with gradient accumulation across
tasks and adds a KL divergence regularization term to align LoRA representations
across tasks.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from transformers import PreTrainedTokenizerBase

from mutafitup.datasets import BaseMultitaskDataset
from mutafitup.models.multitask_model import MultitaskModel

from .base_trainer import MultitaskTrainer
from .strategy import AlignLoRAStrategy


def train_multitask_align_lora(
    model: MultitaskModel,
    tokenizer: PreTrainedTokenizerBase,
    checkpoint: str,
    task_datasets: Dict[str, BaseMultitaskDataset],
    batch: int = 4,
    max_epochs: Optional[int] = None,
    max_validations: Optional[int] = None,
    lr: float = 3e-4,
    warmup_ratio: float = 0.0,
    validate_every_n_train_batches: int = 0,
    seed: int = 42,
    logger: Optional[logging.Logger] = None,
    early_stopping_patience: int = 0,
    early_stopping_metrics: Optional[Dict[str, str]] = None,
    primary_metrics: Optional[Dict[str, str]] = None,
    training_checkpoint_dir: Optional[str] = None,
    checkpoint_every_n_validations: int = 1,
    align_lora_kl_lambda: float = 0.01,
    gradient_checkpointing: bool = False,
    best_overall_model_dir: Optional[str] = None,
    best_task_models_dir: Optional[str] = None,
    best_loss_overall_model_dir: Optional[str] = None,
    best_loss_task_models_dir: Optional[str] = None,
    auto_mixed_precision: bool = False,
) -> Tuple[MultitaskModel, List[Dict], Dict[str, Any]]:
    """Train a multitask model using LoRA with aligned representations.

    This function trains the model using Low-Rank Adaptation (LoRA) with
    gradient accumulation across tasks. Additionally, it adds a KL divergence
    regularization term that encourages the LoRA representations to be
    similar across tasks, promoting shared feature learning.

    Args:
        model: The MultitaskModel to train (should have LoRA config applied).
        tokenizer: Tokenizer for preprocessing sequences.
        checkpoint: Model checkpoint name for preprocessing.
        task_datasets: Dictionary mapping task names to their datasets.
        batch: Batch size for training.
        max_epochs: Maximum number of epochs to train (mutually exclusive with max_validations).
        max_validations: Maximum number of validation steps (mutually exclusive with max_epochs).
        lr: Learning rate.
        warmup_ratio: Ratio of total steps for learning rate warmup.
        validate_every_n_train_batches: Validate every N training batches (0 = validate per epoch).
        seed: Random seed for reproducibility.
        logger: Logger for training progress.
        early_stopping_patience: Number of validations without improvement before stopping (0 = disabled).
        early_stopping_metrics: Custom metrics for early stopping per task.
        primary_metrics: Dict mapping task name to primary metric name string.
        training_checkpoint_dir: Directory for saving training checkpoints.
        checkpoint_every_n_validations: Save checkpoint every N validations.
        align_lora_kl_lambda: Weight for the KL divergence alignment loss.
        gradient_checkpointing: Enable gradient checkpointing to reduce GPU memory
            usage at the cost of extra computation (~30-35% slower).
        best_overall_model_dir: Directory for saving the best overall model weights.
        best_task_models_dir: Directory for saving per-task best model weights.
        best_loss_overall_model_dir: Directory for saving the best overall loss model weights.
        best_loss_task_models_dir: Directory for saving per-task best loss model weights.
        auto_mixed_precision: Use automatic mixed precision (AMP) with GradScaler.

    Returns:
        Tuple of (trained_model, history, best_checkpoints).
    """
    strategy = AlignLoRAStrategy(
        align_lora_kl_lambda=align_lora_kl_lambda,
        gradient_checkpointing=gradient_checkpointing,
    )
    trainer = MultitaskTrainer(
        strategy=strategy,
        model=model,
        tokenizer=tokenizer,
        checkpoint=checkpoint,
        task_datasets=task_datasets,
        batch=batch,
        max_epochs=max_epochs,
        max_validations=max_validations,
        lr=lr,
        warmup_ratio=warmup_ratio,
        validate_every_n_train_batches=validate_every_n_train_batches,
        seed=seed,
        logger=logger,
        embedding_cache_dir=None,  # Cannot use embedding cache
        early_stopping_patience=early_stopping_patience,
        early_stopping_metrics=early_stopping_metrics,
        primary_metrics=primary_metrics,
        training_checkpoint_dir=training_checkpoint_dir,
        checkpoint_every_n_validations=checkpoint_every_n_validations,
        best_overall_model_dir=best_overall_model_dir,
        best_task_models_dir=best_task_models_dir,
        best_loss_overall_model_dir=best_loss_overall_model_dir,
        best_loss_task_models_dir=best_loss_task_models_dir,
        auto_mixed_precision=auto_mixed_precision,
    )
    return trainer.train()
