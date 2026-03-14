"""Backward-compatible module for test monkeypatching support.

This module exists primarily to support tests that use monkeypatching patterns like:
    monkeypatch.setattr(train_module.torch.optim, "AdamW", DummyOptimizer)
    monkeypatch.setattr(train_module, "get_linear_schedule_with_warmup", ...)
    monkeypatch.setattr(train_module, "_save_training_checkpoint", ...)

For new code, import directly from the specific modules:
    from mutafitup.train import train_multitask_heads_only
    from mutafitup.train import MultitaskTrainer
    from mutafitup.train.strategy import HeadsOnlyStrategy
"""

# Import torch at module level for test monkeypatching compatibility
# Tests use: monkeypatch.setattr(train_module.torch.optim, "AdamW", DummyOptimizer)
import torch

# Import get_linear_schedule_with_warmup for test monkeypatching compatibility
# Tests use: monkeypatch.setattr(train_module, "get_linear_schedule_with_warmup", ...)
from transformers import get_linear_schedule_with_warmup

# Re-export public training functions
from .train_multitask_heads_only import train_multitask_heads_only
from .train_multitask_lora import train_multitask_lora
from .train_multitask_accgrad_lora import train_multitask_accgrad_lora
from .train_multitask_align_lora import train_multitask_align_lora

# Re-export checkpoint functions for test monkeypatching compatibility
# Tests use: monkeypatch.setattr(train_module, "_save_training_checkpoint", ...)
from .checkpointing import (
    save_training_checkpoint as _save_training_checkpoint,
    load_training_checkpoint as _load_training_checkpoint,
)

# Re-export trainer and strategies
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
    # Trainer and strategies
    "MultitaskTrainer",
    "TrainingStrategy",
    "HeadsOnlyStrategy",
    "LoRAStrategy",
    "AccGradLoRAStrategy",
    "AlignLoRAStrategy",
    # For test monkeypatching
    "torch",
    "get_linear_schedule_with_warmup",
    "_save_training_checkpoint",
    "_load_training_checkpoint",
]
