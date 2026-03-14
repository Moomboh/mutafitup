"""Multitask model training package.

This package provides training utilities for multitask models with
different training strategies:

- `train_multitask_heads_only`: Train only task-specific heads with frozen backbone.
- `train_multitask_lora`: Train with LoRA fine-tuning.
- `train_multitask_accgrad_lora`: Train with LoRA and gradient accumulation.
- `train_multitask_align_lora`: Train with LoRA, gradient accumulation, and KL alignment.

Example usage:
    >>> from mutafitup.train import train_multitask_heads_only
    >>> model, history, best = train_multitask_heads_only(
    ...     model=model,
    ...     tokenizer=tokenizer,
    ...     checkpoint=checkpoint,
    ...     task_datasets=datasets,
    ...     max_epochs=10,
    ... )

For advanced usage, you can use the MultitaskTrainer class directly with custom strategies:
    >>> from mutafitup.train import MultitaskTrainer
    >>> from mutafitup.train.strategy import HeadsOnlyStrategy
    >>> strategy = HeadsOnlyStrategy()
    >>> trainer = MultitaskTrainer(strategy=strategy, model=model, ...)
    >>> model, history, best = trainer.train()
"""

from .train_multitask_heads_only import train_multitask_heads_only
from .train_multitask_lora import train_multitask_lora
from .train_multitask_accgrad_lora import train_multitask_accgrad_lora
from .train_multitask_align_lora import train_multitask_align_lora
from .base_trainer import MultitaskTrainer
from .strategy import (
    TrainingStrategy,
    HeadsOnlyStrategy,
    LoRAStrategy,
    AccGradLoRAStrategy,
    AlignLoRAStrategy,
)

__all__ = [
    # Public training functions
    "train_multitask_heads_only",
    "train_multitask_lora",
    "train_multitask_accgrad_lora",
    "train_multitask_align_lora",
    # Trainer class
    "MultitaskTrainer",
    # Strategy protocol and implementations
    "TrainingStrategy",
    "HeadsOnlyStrategy",
    "LoRAStrategy",
    "AccGradLoRAStrategy",
    "AlignLoRAStrategy",
]
