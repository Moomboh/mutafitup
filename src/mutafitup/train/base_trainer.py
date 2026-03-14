"""Multitask trainer using composition with training strategies.

This module provides the MultitaskTrainer class that handles the core training loop,
using TrainingStrategy objects to customize behavior for different training approaches.
"""

import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase
from transformers import (
    get_linear_schedule_with_warmup as _default_get_linear_schedule_with_warmup,
)

from mutafitup.datasets import BaseMultitaskDataset
from mutafitup.device import get_device
from mutafitup.metrics import (
    compute_classification_metrics,
    compute_regression_metrics,
)
from mutafitup.models.multitask_model import (
    MultitaskForwardArgs,
    MultitaskModel,
)
from mutafitup.random import set_seeds

from .checkpointing import load_training_checkpoint as _load_training_checkpoint_impl
from .checkpointing import save_training_checkpoint as _save_training_checkpoint_impl
from .dataloaders import build_task_dataloaders
from .strategy import TrainingStrategy
from .utils import (
    attach_generators_to_train_loaders,
    compute_params_hash,
    format_time,
    init_train_sampler_generators,
)


def get_optimizer_class() -> type:
    """Get the optimizer class, supporting test monkeypatching via train_multitask_model module."""
    # Import at runtime to allow monkeypatching in tests
    # Tests do: monkeypatch.setattr(train_module.torch.optim, "AdamW", DummyOptimizer)
    from . import train_multitask_model as train_module

    return train_module.torch.optim.AdamW


def get_scheduler_factory() -> Callable:
    """Get the scheduler factory, supporting test monkeypatching via train_multitask_model module."""
    # Import at runtime to allow monkeypatching in tests
    # Tests do: monkeypatch.setattr(train_module, "get_linear_schedule_with_warmup", ...)
    from . import train_multitask_model as train_module

    return train_module.get_linear_schedule_with_warmup


def get_save_checkpoint_fn() -> Callable:
    """Get the checkpoint save function, supporting test monkeypatching."""
    # Import at runtime to allow monkeypatching in tests
    # Tests do: monkeypatch.setattr(train_module, "_save_training_checkpoint", ...)
    from . import train_multitask_model as train_module

    return train_module._save_training_checkpoint


def get_load_checkpoint_fn() -> Callable:
    """Get the checkpoint load function, supporting test monkeypatching."""
    from . import train_multitask_model as train_module

    return train_module._load_training_checkpoint


INITIAL_CHECKPOINT_MARKER = ".initial_checkpoint"


def _get_gpu_info() -> Dict[str, Any]:
    """Collect GPU/device information for history logging.

    Returns a dict with device identity, memory stats, and version info.
    Supports CUDA, MPS (Apple Silicon), and CPU.
    """
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        to_mb = 1 / (1024**2)

        info: Dict[str, Any] = {
            "device": f"cuda:{idx}",
            "device_name": props.name,
            "total_memory_mb": round(props.total_memory * to_mb),
            "allocated_memory_mb": round(torch.cuda.memory_allocated(idx) * to_mb),
            "reserved_memory_mb": round(torch.cuda.memory_reserved(idx) * to_mb),
            "peak_memory_mb": round(torch.cuda.max_memory_allocated(idx) * to_mb),
            "compute_capability": f"{props.major}.{props.minor}",
        }

        if torch.version.cuda is not None:
            info["cuda_version"] = torch.version.cuda

        if torch.backends.cudnn.is_available():
            info["cudnn_version"] = torch.backends.cudnn.version()

        return info

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        to_mb = 1 / (1024**2)
        return {
            "device": "mps",
            "device_name": "mps",
            "recommended_max_memory_mb": round(
                torch.mps.recommended_max_memory() * to_mb
            ),
            "allocated_memory_mb": round(torch.mps.current_allocated_memory() * to_mb),
            "driver_allocated_memory_mb": round(
                torch.mps.driver_allocated_memory() * to_mb
            ),
        }

    return {"device": "cpu", "device_name": "cpu"}


class MultitaskTrainer:
    """Trainer for multitask models using composition with training strategies.

    This class handles the core training loop logic, delegating approach-specific
    behavior to a TrainingStrategy object. Different training approaches (heads-only,
    LoRA, accumulated gradient, aligned LoRA) are implemented as strategy objects.

    Args:
        strategy: The training strategy that customizes behavior.
        model: The MultitaskModel to train.
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
        embedding_cache_dir: Directory for caching embeddings (speeds up training).
        early_stopping_patience: Number of validations without improvement before stopping (0 = disabled).
        early_stopping_metrics: Custom metrics for early stopping per task.
        primary_metrics: Dict mapping task name to primary metric name string (e.g. "accuracy", "spearman").
            Used for best-model tracking, early stopping fallback, and history logging.
            If not provided, falls back to "accuracy" for classification and "spearman" for regression.
        training_checkpoint_dir: Directory for saving training checkpoints.
        checkpoint_every_n_validations: Save checkpoint every N validations.
        auto_mixed_precision: Use automatic mixed precision (AMP) with GradScaler.
    """

    def __init__(
        self,
        strategy: TrainingStrategy,
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
        embedding_cache_dir: Optional[str] = None,
        early_stopping_patience: int = 0,
        early_stopping_metrics: Optional[Dict[str, str]] = None,
        primary_metrics: Optional[Dict[str, str]] = None,
        training_checkpoint_dir: Optional[str] = None,
        checkpoint_every_n_validations: int = 1,
        best_overall_model_dir: Optional[str] = None,
        best_task_models_dir: Optional[str] = None,
        best_loss_overall_model_dir: Optional[str] = None,
        best_loss_task_models_dir: Optional[str] = None,
        auto_mixed_precision: bool = False,
    ):
        if max_epochs is None and max_validations is None:
            raise ValueError("Either max_epochs or max_validations must be specified")

        self.strategy = strategy
        self.model = model
        self.tokenizer = tokenizer
        self.checkpoint = checkpoint
        self.task_datasets = task_datasets
        self.batch = batch
        self.max_epochs = max_epochs
        self.max_validations = max_validations
        self.lr = lr
        self.warmup_ratio = warmup_ratio
        self.validate_every_n_train_batches = validate_every_n_train_batches
        self.seed = seed
        self.logger = logger
        self.embedding_cache_dir = embedding_cache_dir
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_metrics = early_stopping_metrics or {}
        self.primary_metrics = primary_metrics or {}
        self.training_checkpoint_dir = training_checkpoint_dir
        self.checkpoint_every_n_validations = checkpoint_every_n_validations
        self.best_overall_model_dir = best_overall_model_dir
        self.best_task_models_dir = best_task_models_dir
        self.best_loss_overall_model_dir = best_loss_overall_model_dir
        self.best_loss_task_models_dir = best_loss_task_models_dir
        self.auto_mixed_precision = auto_mixed_precision

        # Validate task datasets match model heads
        if set(task_datasets.keys()) != set(model.head_configs.keys()):
            raise ValueError(
                "Tasks in dataset and model do not match. "
                f"Dataset: {set(task_datasets.keys())}, "
                f"Model: {set(model.head_configs.keys())}"
            )

        self.task_names = list(task_datasets.keys())
        self.device = get_device()

    def train(self) -> Tuple[MultitaskModel, List[Dict], Dict[str, Any]]:
        """Execute the training loop and return the trained model, history, and best checkpoints."""
        set_seeds(self.seed)

        self.model = self.model.to(self.device)

        # Setup automatic mixed precision
        self._autocast_device_type = self.device.type
        self.scaler = torch.amp.GradScaler(enabled=self.auto_mixed_precision)

        # Determine if we can use embedding cache
        use_cache = (
            self.strategy.use_embedding_cache
            and self.embedding_cache_dir is not None
            and self.model.backbone.lora_config is None
        )

        # Build dataloaders
        self.train_loaders, self.valid_loaders = build_task_dataloaders(
            model=self.model,
            tokenizer=self.tokenizer,
            checkpoint=self.checkpoint,
            task_datasets=self.task_datasets,
            task_names=self.task_names,
            batch=self.batch,
            use_embedding_cache=use_cache,
            embedding_cache_dir=self.embedding_cache_dir,
            device=self.device,
            logger=self.logger,
        )
        self._use_embedding_cache = use_cache

        # Setup optimizer using module-level factory (supports testing)
        optimizer_class = get_optimizer_class()
        self.optimizer = optimizer_class(self.model.parameters(), lr=self.lr)

        # Calculate total steps
        self.num_batches_per_epoch = max(len(dl) for dl in self.train_loaders.values())
        accumulate = self.strategy.accumulate_across_tasks
        combined_backward = self.strategy.accumulate_requires_combined_backward

        if self.max_validations is not None:
            if self.validate_every_n_train_batches > 0:
                total_batches = (
                    self.max_validations * self.validate_every_n_train_batches
                )
            else:
                total_batches = self.max_validations * self.num_batches_per_epoch
            if accumulate:
                self.total_steps = total_batches
            else:
                self.total_steps = total_batches * len(self.task_datasets)
            self.loop_epochs = (total_batches // self.num_batches_per_epoch) + 1
        else:
            # max_epochs is guaranteed to be not None here (checked in __init__)
            assert self.max_epochs is not None
            self.loop_epochs = self.max_epochs
            if accumulate:
                self.total_steps = self.max_epochs * self.num_batches_per_epoch
            else:
                self.total_steps = (
                    self.max_epochs
                    * self.num_batches_per_epoch
                    * len(self.task_datasets)
                )

        num_warmup_steps = int(self.total_steps * self.warmup_ratio)
        scheduler_factory = get_scheduler_factory()
        self.scheduler = scheduler_factory(
            self.optimizer, num_warmup_steps, self.total_steps
        )

        # Build checkpointing params
        checkpointing_params = {
            "checkpoint": self.checkpoint,
            "task_names": sorted(self.task_names),
            "batch": self.batch,
            "max_epochs": self.max_epochs,
            "max_validations": self.max_validations,
            "lr": self.lr,
            "warmup_ratio": self.warmup_ratio,
            "validate_every_n_train_batches": self.validate_every_n_train_batches,
            "seed": self.seed,
            "accumulate_across_tasks": accumulate,
            "early_stopping_patience": self.early_stopping_patience,
            "early_stopping_metrics": self.early_stopping_metrics,
            "head_configs": {
                n: str(self.model.head_configs[n]) for n in self.task_names
            },
            **self.strategy.get_extra_checkpointing_params(),
        }
        self.params_hash = compute_params_hash(checkpointing_params)
        self.checkpointing_params = checkpointing_params

        # Initialize sampler generators for reproducibility
        self.train_sampler_generators = init_train_sampler_generators(
            self.task_names, self.seed
        )
        attach_generators_to_train_loaders(
            self.train_loaders, self.train_sampler_generators
        )

        # Initialize tracking variables
        self.history: List[Dict] = []
        self.task_metrics: Dict[str, List[float]] = {n: [] for n in self.task_names}
        self.best_metrics: Dict[str, float] = {
            n: float("-inf") for n in self.task_names
        }
        self.best_epochs: Dict[str, int] = {n: 0 for n in self.task_names}
        self.cumulative_batches: Dict[str, int] = {n: 0 for n in self.task_names}

        # Get head configs
        self.ignore_index = {
            n: self.model.head_configs[n].ignore_index for n in self.task_names
        }
        self.problem_type = {
            n: self.model.head_configs[n].problem_type for n in self.task_names
        }
        self.levels = {n: self.model.head_configs[n].level for n in self.task_names}

        self.global_step = 0
        self.training_start_time = time.time()
        self.elapsed_time_offset = 0.0
        self.cumulative_validation_time = 0.0
        self.validation_count = 0
        self.start_epoch = 0
        self._prev_cumulative_training_time = 0.0
        self._prev_global_step = 0

        # Best-so-far model tracking (relative improvement sum)
        self.best_so_far_overall_validation_step: int = 0
        self.best_so_far_overall_epoch: int = 0
        self.best_so_far_overall_global_step: int = 0
        self.best_so_far_overall_baseline_metrics: Dict[str, float] = {
            n: 1e-6 for n in self.task_names
        }
        self.best_so_far_overall_metrics: Dict[str, float] = {
            n: 0.0 for n in self.task_names
        }
        self.best_so_far_task_validation_steps: Dict[str, int] = {
            n: 0 for n in self.task_names
        }
        self.best_so_far_task_epochs: Dict[str, int] = {n: 0 for n in self.task_names}
        self.best_so_far_task_global_steps: Dict[str, int] = {
            n: 0 for n in self.task_names
        }

        # Best-so-far loss tracking (per-task)
        self.task_val_losses: Dict[str, List[float]] = {n: [] for n in self.task_names}
        self.best_val_losses: Dict[str, float] = {
            n: float("inf") for n in self.task_names
        }
        self.best_loss_epochs: Dict[str, int] = {n: 0 for n in self.task_names}
        self.best_so_far_loss_task_validation_steps: Dict[str, int] = {
            n: 0 for n in self.task_names
        }
        self.best_so_far_loss_task_epochs: Dict[str, int] = {
            n: 0 for n in self.task_names
        }
        self.best_so_far_loss_task_global_steps: Dict[str, int] = {
            n: 0 for n in self.task_names
        }

        # Best-so-far loss tracking (overall, relative improvement sum on losses)
        self.best_so_far_loss_overall_validation_step: int = 0
        self.best_so_far_loss_overall_epoch: int = 0
        self.best_so_far_loss_overall_global_step: int = 0
        self.best_so_far_loss_overall_baseline_losses: Dict[str, float] = {}
        self.best_so_far_loss_overall_losses: Dict[str, float] = {
            n: 0.0 for n in self.task_names
        }

        # Tracks which best-model snapshot keys have been saved to the
        # snapshot cache directory on disk.  Categories use the format:
        # "overall", "loss_overall", "task:{name}", "loss_task:{name}".
        # The actual weight blobs are stored as individual .pt files in
        # the snapshot cache dir rather than being held in memory.
        self._best_model_snapshot_keys: Set[str] = set()

        # Resume state tracking
        resume_step_within_epoch = 0
        self.resuming_mid_epoch = False
        resumed_checkpoint_data: Optional[Dict[str, Any]] = None
        resumed_train_sampler_iterator_start_states: Optional[
            Dict[str, torch.Tensor]
        ] = None
        resumed_train_sampler_iterator_start_cumulative_batches: Optional[
            Dict[str, int]
        ] = None
        resumed_train_total_train_loss: Optional[Dict[str, float]] = None
        resumed_train_total_train_batches: Optional[Dict[str, int]] = None

        self.steps_per_epoch = (
            self.num_batches_per_epoch
            if accumulate
            else self.num_batches_per_epoch * len(self.task_names)
        )
        self.train_sampler_iterator_start_states: Dict[str, torch.Tensor] = {}
        self.train_sampler_iterator_start_cumulative_batches: Dict[str, int] = {}

        # Early stopping state
        self.early_stopping_enabled = self.early_stopping_patience > 0
        self.early_stopping_best: Dict[str, float] = {
            n: float("-inf") for n in self.task_names
        }
        self.early_stopping_best_loss: Dict[str, float] = {
            n: float("inf") for n in self.task_names
        }
        self.early_stopping_counter: Dict[str, int] = {n: 0 for n in self.task_names}
        self.early_stopping_triggered = False
        self.stop_reason: Optional[str] = None

        # Checkpoint path for history entries
        self.training_checkpoint_path = None
        if self.training_checkpoint_dir is not None:
            self.training_checkpoint_path = os.path.join(
                self.training_checkpoint_dir, "training_checkpoint.pt"
            )

            # Try to resume from checkpoint
            load_checkpoint_fn = get_load_checkpoint_fn()
            resumed_data = load_checkpoint_fn(
                self.training_checkpoint_dir,
                self.params_hash,
                self.model,
                self.optimizer,
                self.scheduler,
                self.logger,
            )
            if resumed_data is not None:
                resumed_checkpoint_data = resumed_data
                self.global_step = resumed_data["global_step"]
                self.start_epoch = self.global_step // self.steps_per_epoch
                resume_step_within_epoch = self.global_step % self.steps_per_epoch
                self.resuming_mid_epoch = resume_step_within_epoch != 0
                self.validation_count = resumed_data["validation_count"]
                self.history = resumed_data["history"]
                self.task_metrics = resumed_data["task_metrics"]
                self.best_metrics = resumed_data["best_metrics"]
                self.best_epochs = resumed_data["best_epochs"]
                self.cumulative_batches = resumed_data["cumulative_batches"]
                self.early_stopping_best = resumed_data["early_stopping_best"]
                self.early_stopping_best_loss = resumed_data.get(
                    "early_stopping_best_loss",
                    {n: float("inf") for n in self.task_names},
                )
                self.early_stopping_counter = resumed_data["early_stopping_counter"]
                self.elapsed_time_offset = resumed_data["elapsed_time"]
                self.cumulative_validation_time = resumed_data[
                    "cumulative_validation_time"
                ]
                self._prev_cumulative_training_time = (
                    self.elapsed_time_offset - self.cumulative_validation_time
                )
                self._prev_global_step = self.global_step

                # Restore best-so-far state
                best_so_far_ckpt = resumed_data.get("best_so_far")
                if best_so_far_ckpt is not None:
                    overall = best_so_far_ckpt.get("overall", {})
                    self.best_so_far_overall_validation_step = overall.get(
                        "validation_step", 0
                    )
                    self.best_so_far_overall_epoch = overall.get("epoch", 0)
                    self.best_so_far_overall_global_step = overall.get("global_step", 0)
                    self.best_so_far_overall_baseline_metrics = overall.get(
                        "baseline_metrics",
                        {n: 1e-6 for n in self.task_names},
                    )
                    self.best_so_far_overall_metrics = overall.get(
                        "metrics",
                        {n: 0.0 for n in self.task_names},
                    )
                    tasks_ckpt = best_so_far_ckpt.get("tasks", {})
                    for name in self.task_names:
                        task_data = tasks_ckpt.get(name, {})
                        self.best_so_far_task_validation_steps[name] = task_data.get(
                            "validation_step", 0
                        )
                        self.best_so_far_task_epochs[name] = task_data.get("epoch", 0)
                        self.best_so_far_task_global_steps[name] = task_data.get(
                            "global_step", 0
                        )

                    # Restore per-task loss best-so-far
                    loss_tasks_ckpt = best_so_far_ckpt.get("loss_tasks", {})
                    for name in self.task_names:
                        loss_task_data = loss_tasks_ckpt.get(name, {})
                        self.best_so_far_loss_task_validation_steps[name] = (
                            loss_task_data.get("validation_step", 0)
                        )
                        self.best_so_far_loss_task_epochs[name] = loss_task_data.get(
                            "epoch", 0
                        )
                        self.best_so_far_loss_task_global_steps[name] = (
                            loss_task_data.get("global_step", 0)
                        )

                    # Restore overall loss best-so-far
                    loss_overall = best_so_far_ckpt.get("loss_overall", {})
                    self.best_so_far_loss_overall_validation_step = loss_overall.get(
                        "validation_step", 0
                    )
                    self.best_so_far_loss_overall_epoch = loss_overall.get("epoch", 0)
                    self.best_so_far_loss_overall_global_step = loss_overall.get(
                        "global_step", 0
                    )
                    self.best_so_far_loss_overall_baseline_losses = loss_overall.get(
                        "baseline_losses", {}
                    )
                    self.best_so_far_loss_overall_losses = loss_overall.get(
                        "losses",
                        {n: 0.0 for n in self.task_names},
                    )

                    # Restore model weight snapshots and recreate any
                    # missing best-model output directories (e.g. after
                    # Snakemake --rerun-incomplete deleted them).
                    snapshot_keys = best_so_far_ckpt.get("model_snapshot_keys")
                    legacy_snapshots = best_so_far_ckpt.get("model_snapshots")
                    if snapshot_keys is not None:
                        # New format: keys reference files in the cache dir
                        self._best_model_snapshot_keys = set(snapshot_keys)
                        self._restore_missing_best_model_dirs_from_snapshots(
                            self._best_model_snapshot_keys
                        )
                    elif legacy_snapshots:
                        # Old format: full weight blobs stored in checkpoint
                        self._restore_missing_best_model_dirs_from_legacy_snapshots(
                            legacy_snapshots
                        )
                        # Remove the heavy blobs from the checkpoint dict so
                        # they can be garbage-collected while training continues.
                        best_so_far_ckpt.pop("model_snapshots", None)
                        del legacy_snapshots

                # Restore loss tracking lists
                resumed_task_val_losses = resumed_data.get("task_val_losses")
                if resumed_task_val_losses is not None:
                    self.task_val_losses = resumed_task_val_losses
                resumed_best_val_losses = resumed_data.get("best_val_losses")
                if resumed_best_val_losses is not None:
                    self.best_val_losses = resumed_best_val_losses
                resumed_best_loss_epochs = resumed_data.get("best_loss_epochs")
                if resumed_best_loss_epochs is not None:
                    self.best_loss_epochs = resumed_best_loss_epochs

                resumed_train_sampler_states = resumed_data.get(
                    "train_sampler_generator_states"
                )
                resumed_train_sampler_iterator_start_states = resumed_data.get(
                    "train_sampler_iterator_start_states"
                )
                resumed_train_sampler_iterator_start_cumulative_batches = (
                    resumed_data.get("train_sampler_iterator_start_cumulative_batches")
                )
                resumed_train_total_train_loss = resumed_data.get(
                    "train_total_train_loss"
                )
                resumed_train_total_train_batches = resumed_data.get(
                    "train_total_train_batches"
                )
                if resumed_train_sampler_states is not None:
                    for name, state in resumed_train_sampler_states.items():
                        if name in self.train_sampler_generators and state is not None:
                            self.train_sampler_generators[name].set_state(state)
                    attach_generators_to_train_loaders(
                        self.train_loaders, self.train_sampler_generators
                    )

                if self.history:
                    checkpointing = self.history[-1].setdefault("checkpointing", {})
                    resumes = checkpointing.setdefault("resumed", [])
                    resumes.append(
                        {
                            "checkpoint_path": self.training_checkpoint_path,
                            "params_hash": self.params_hash,
                            "at": {
                                "epoch": resumed_data["epoch"],
                                "epoch_index": self.start_epoch,
                                "step_within_epoch": resume_step_within_epoch,
                                "global_step": self.global_step,
                                "validation_count": self.validation_count,
                            },
                            "gpu": _get_gpu_info(),
                        }
                    )

        # Initialize per-epoch tracking variables before the loop so that
        # _run_validation() can access them even when the loop is skipped
        # (e.g. when resuming from a checkpoint at the end of training).
        self.total_train_loss = {n: 0.0 for n in self.task_names}
        self.batches = {n: 0 for n in self.task_names}

        # If we have checkpoint data for these, apply it
        if resumed_train_total_train_loss is not None:
            for name, val in resumed_train_total_train_loss.items():
                if name in self.total_train_loss:
                    self.total_train_loss[name] = float(val)
        if resumed_train_total_train_batches is not None:
            for name, val in resumed_train_total_train_batches.items():
                if name in self.batches:
                    self.batches[name] = int(val)

        # Setup strategy-specific state
        self.strategy.setup(self.model, self.task_names)

        # Restore strategy-specific state from checkpoint (must be after setup()
        # so that setup() initializes state before we override with checkpoint values)
        if resumed_checkpoint_data is not None:
            self.strategy.restore_from_checkpoint(resumed_checkpoint_data)

        # Log GPU/device info at the start of training
        if self.logger is not None:
            self.logger.info("Training device: %s", _get_gpu_info())

        # Reset CUDA peak memory stats so the first validation captures
        # only the peak from the initial training steps.
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Main training loop
        assert self.loop_epochs is not None  # Set in conditional above
        epoch = self.start_epoch  # Initialize for use after loop
        for epoch in range(self.start_epoch, self.loop_epochs):
            self.model.train()
            self.strategy.on_epoch_start()

            self.total_train_loss = {n: 0.0 for n in self.task_names}
            self.batches = {n: 0 for n in self.task_names}
            iters: Dict[str, Any] = {}
            self.train_sampler_iterator_start_states = {}
            self.train_sampler_iterator_start_cumulative_batches = {}

            # Handle mid-epoch resume
            if epoch == self.start_epoch and self.resuming_mid_epoch:
                if resumed_train_total_train_loss is not None:
                    for name, val in resumed_train_total_train_loss.items():
                        if name in self.total_train_loss:
                            self.total_train_loss[name] = float(val)

                if resumed_train_total_train_batches is not None:
                    for name, val in resumed_train_total_train_batches.items():
                        if name in self.batches:
                            self.batches[name] = int(val)

                if (
                    resumed_train_sampler_iterator_start_states is None
                    and self.logger is not None
                ):
                    self.logger.warning(
                        "Resuming mid-epoch but checkpoint is missing per-task "
                        "iterator start states; train order may differ from an "
                        "uninterrupted run."
                    )

                if (
                    resumed_train_sampler_iterator_start_states is not None
                    and resumed_train_sampler_iterator_start_cumulative_batches is None
                    and self.logger is not None
                ):
                    self.logger.warning(
                        "Resuming mid-epoch but checkpoint is missing per-task "
                        "iterator start cumulative batch counters; train order may "
                        "differ when a task wraps."
                    )

            if (
                epoch == self.start_epoch
                and self.resuming_mid_epoch
                and resumed_train_sampler_iterator_start_states is not None
            ):
                for name, state in resumed_train_sampler_iterator_start_states.items():
                    if name in self.train_sampler_generators and state is not None:
                        self.train_sampler_generators[name].set_state(state)
                attach_generators_to_train_loaders(
                    self.train_loaders, self.train_sampler_generators
                )

            for name in self.task_names:
                self.train_sampler_iterator_start_states[name] = (
                    self.train_sampler_generators[name].get_state()
                )

                if (
                    epoch == self.start_epoch
                    and self.resuming_mid_epoch
                    and resumed_train_sampler_iterator_start_cumulative_batches
                    is not None
                ):
                    self.train_sampler_iterator_start_cumulative_batches[name] = int(
                        resumed_train_sampler_iterator_start_cumulative_batches.get(
                            name, self.cumulative_batches[name]
                        )
                    )
                else:
                    self.train_sampler_iterator_start_cumulative_batches[name] = int(
                        self.cumulative_batches[name]
                    )

                iters[name] = iter(self.train_loaders[name])

            # Skip batches if resuming mid-epoch
            if epoch == self.start_epoch and self.resuming_mid_epoch:
                for name in self.task_names:
                    dl_len = len(self.train_loaders[name])
                    if dl_len <= 0:
                        raise ValueError(
                            f"Train DataLoader for task {name} has length {dl_len}"
                        )

                    if (
                        resumed_train_sampler_iterator_start_cumulative_batches
                        is not None
                    ):
                        start_cum = int(
                            self.train_sampler_iterator_start_cumulative_batches[name]
                        )
                        consumed_since_start = (
                            int(self.cumulative_batches[name]) - start_cum
                        )
                        if consumed_since_start < 0:
                            consumed_since_start = 0
                        offset = consumed_since_start % dl_len
                        skip_count = (
                            dl_len
                            if (consumed_since_start > 0 and offset == 0)
                            else offset
                        )
                    else:
                        skip_count = int(self.cumulative_batches[name]) % dl_len

                    for _ in range(skip_count):
                        next(iters[name])

            # Calculate resume indices
            if epoch == self.start_epoch:
                if accumulate:
                    resume_outer_idx = resume_step_within_epoch
                    resume_task_idx = 0
                else:
                    tasks_per_outer = len(self.task_names)
                    resume_outer_idx = resume_step_within_epoch // tasks_per_outer
                    resume_task_idx = resume_step_within_epoch % tasks_per_outer
            else:
                resume_outer_idx = 0
                resume_task_idx = 0

            # Batch iteration loop
            for outer_idx in range(self.num_batches_per_epoch):
                if self.early_stopping_triggered:
                    break
                if epoch == self.start_epoch and outer_idx < resume_outer_idx:
                    continue

                self.strategy.on_step_start()

                if accumulate:
                    self.optimizer.zero_grad()

                task_losses: List[torch.Tensor] = []

                for task_idx, name in enumerate(self.task_names):
                    if (
                        epoch == self.start_epoch
                        and outer_idx == resume_outer_idx
                        and task_idx < resume_task_idx
                    ):
                        continue

                    try:
                        batch_data = next(iters[name])
                    except StopIteration:
                        self.train_sampler_iterator_start_states[name] = (
                            self.train_sampler_generators[name].get_state()
                        )
                        self.train_sampler_iterator_start_cumulative_batches[name] = (
                            int(self.cumulative_batches[name])
                        )
                        iters[name] = iter(self.train_loaders[name])
                        batch_data = next(iters[name])

                    batch_data = {k: v.to(self.device) for k, v in batch_data.items()}

                    if self.problem_type[name] == "regression":
                        batch_data["labels"] = batch_data["labels"].float()

                    if not accumulate:
                        self.optimizer.zero_grad()

                    self.strategy.on_task_start(name)

                    with torch.amp.autocast(
                        device_type=self._autocast_device_type,
                        enabled=self.auto_mixed_precision,
                    ):
                        if self._use_embedding_cache:
                            loss, logits = self.model.forward_from_embeddings(
                                task=name,
                                embeddings=batch_data["embeddings"],
                                attention_mask=batch_data["attention_mask"],
                                labels=batch_data["labels"],
                            )
                        else:
                            args = MultitaskForwardArgs(
                                attention_mask=batch_data["attention_mask"],
                                input_ids=batch_data["input_ids"],
                            )
                            loss, logits = self.model(
                                task=name,
                                args=args,
                                labels=batch_data["labels"],
                            )

                    self.strategy.on_task_end(name)

                    if loss is None:
                        raise ValueError("Loss is None during training")

                    loss = self.strategy.process_loss(loss, name, len(self.task_names))

                    if accumulate:
                        if combined_backward:
                            # Keep live tensor for combined backward (e.g. AlignLoRA)
                            task_losses.append(loss)
                        else:
                            # Per-task backward: free graph immediately
                            task_losses.append(loss.detach())
                            self.scaler.scale(loss).backward()
                    else:
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.scheduler.step()
                        self.global_step += 1

                    self.total_train_loss[name] += loss.item()
                    self.batches[name] += 1
                    self.cumulative_batches[name] += 1

                    if not accumulate:
                        if self.validate_every_n_train_batches > 0 and (
                            self.global_step % self.validate_every_n_train_batches == 0
                        ):
                            self._run_validation(epoch + 1)
                            if self.early_stopping_triggered:
                                break

                if self.early_stopping_triggered:
                    break

                if accumulate:
                    if combined_backward:
                        # Combined backward for strategies that need it (e.g. AlignLoRA KL)
                        combined_loss = self.strategy.combine_losses(
                            task_losses, self.device
                        )
                        self.scaler.scale(combined_loss).backward()
                    else:
                        # Gradients already accumulated from per-task backwards.
                        # Still call combine_losses for bookkeeping (e.g. logging).
                        self.strategy.combine_losses(task_losses, self.device)

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    self.global_step += 1

                    if self.validate_every_n_train_batches > 0 and (
                        self.global_step % self.validate_every_n_train_batches == 0
                    ):
                        self._run_validation(epoch + 1)

            if self.early_stopping_triggered:
                self._log_stop_reason(epoch + 1)
                break

            if self.validate_every_n_train_batches == 0:
                self._run_validation(epoch + 1)
                if self.early_stopping_triggered:
                    self._log_stop_reason(epoch + 1)
                    break

        # Final validation if needed
        if (
            not self.early_stopping_triggered
            and self.validate_every_n_train_batches > 0
        ):
            if (
                self.global_step % self.validate_every_n_train_batches != 0
                or not self.history
            ):
                current_epoch = epoch + 1
                if self.logger is not None:
                    self.logger.info(
                        f"Final validation at global_step {self.global_step} "
                        f"(epoch {current_epoch}, every "
                        f"{self.validate_every_n_train_batches} steps)"
                    )
                self._run_validation(current_epoch)

        # Cleanup
        self.strategy.teardown()

        # Compute best checkpoints using relative improvement sum
        eps = 1e-6
        baseline_metrics = {name: eps for name in self.task_names}
        best_rel_improve_idx = 0
        for i in range(len(self.history)):
            total_rel_improvement = 0.0
            current_metrics: Dict[str, float] = {}
            for name in self.task_names:
                metric = max(float(self.task_metrics[name][i]), eps)
                current_metrics[name] = metric
                baseline = max(baseline_metrics[name], eps)
                total_rel_improvement += (metric - baseline) / baseline
            if total_rel_improvement >= 0.0:
                baseline_metrics = dict(current_metrics)
                best_rel_improve_idx = i

        # Compute best loss checkpoint using relative improvement sum on losses
        # For losses (lower is better): sum of (baseline - current) / baseline
        best_loss_rel_improve_idx = 0
        loss_baseline: Dict[str, float] = {}
        for i in range(len(self.history)):
            current_losses_at_i: Dict[str, float] = {}
            for name in self.task_names:
                current_losses_at_i[name] = float(self.task_val_losses[name][i])
            if not loss_baseline:
                # First validation: always accept, set baseline
                loss_baseline = dict(current_losses_at_i)
                best_loss_rel_improve_idx = i
            else:
                total_loss_rel_imp = 0.0
                for name in self.task_names:
                    bl = loss_baseline[name]
                    if bl > 0:
                        total_loss_rel_imp += (bl - current_losses_at_i[name]) / bl
                    else:
                        if current_losses_at_i[name] <= 0:
                            pass
                        else:
                            total_loss_rel_imp -= 1.0
                if total_loss_rel_imp >= 0.0:
                    loss_baseline = dict(current_losses_at_i)
                    best_loss_rel_improve_idx = i

        best_checkpoints = {
            "tasks": {
                name: {
                    "epoch": self.best_epochs[name],
                    "metric": float(self.best_metrics[name]),
                }
                for name in self.task_names
            },
            "best_overall": {
                "epoch": self.history[best_rel_improve_idx]["epoch"],
                "metrics": {
                    name: float(self.task_metrics[name][best_rel_improve_idx])
                    for name in self.task_names
                },
            },
            "best_loss_tasks": {
                name: {
                    "epoch": self.best_loss_epochs[name],
                    "loss": float(self.best_val_losses[name]),
                }
                for name in self.task_names
            },
            "best_loss_overall": {
                "epoch": self.history[best_loss_rel_improve_idx]["epoch"],
                "losses": {
                    name: float(self.task_val_losses[name][best_loss_rel_improve_idx])
                    for name in self.task_names
                },
            },
        }

        return self.model, self.history, best_checkpoints

    def _capture_model_snapshot(self) -> Dict[str, Any]:
        """Capture the current trainable weights and config as a serializable blob.

        The returned dict has the same structure as a ``model.pt`` file produced
        by :meth:`MultitaskModel.save_to_file`, i.e.
        ``{"state_dict": ..., "config": ...}``.
        """
        state_dict = {
            name: param.detach().cpu()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        config = self.model._build_serialization_config(base_checkpoint=self.checkpoint)
        return {"state_dict": state_dict, "config": config}

    @property
    def _snapshot_cache_dir(self) -> Optional[str]:
        """Directory for storing best-model snapshot files on disk."""
        if self.training_checkpoint_dir is None:
            return None
        return os.path.join(self.training_checkpoint_dir, "model_snapshots")

    @staticmethod
    def _snapshot_key_to_filename(key: str) -> str:
        """Convert a snapshot key to a filesystem-safe filename.

        Replaces ``":"`` with ``"__"`` and appends ``.pt``.
        E.g. ``"task:secstr"`` becomes ``"task__secstr.pt"``.
        """
        return key.replace(":", "__") + ".pt"

    def _save_snapshot_to_cache(self, key: str) -> None:
        """Capture current trainable weights and save to the snapshot cache dir.

        The snapshot is written to disk immediately and not held in memory.
        A ``params_hash`` field is included so that stale cache files from a
        previous run with different parameters can be detected on resume.
        """
        cache_dir = self._snapshot_cache_dir
        if cache_dir is None:
            return
        os.makedirs(cache_dir, exist_ok=True)

        snapshot = self._capture_model_snapshot()
        snapshot["params_hash"] = self.params_hash

        filename = self._snapshot_key_to_filename(key)
        path = os.path.join(cache_dir, filename)
        temp_path = path + ".tmp"
        torch.save(snapshot, temp_path)
        os.replace(temp_path, path)

        self._best_model_snapshot_keys.add(key)

    def _load_snapshot_from_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """Load a snapshot blob from the cache directory.

        Returns ``None`` if the file does not exist or the ``params_hash``
        does not match the current run.
        """
        cache_dir = self._snapshot_cache_dir
        if cache_dir is None:
            return None
        filename = self._snapshot_key_to_filename(key)
        path = os.path.join(cache_dir, filename)
        if not os.path.exists(path):
            return None

        snapshot = torch.load(path, weights_only=False)
        stored_hash = snapshot.pop("params_hash", None)
        if stored_hash is not None and stored_hash != self.params_hash:
            if self.logger is not None:
                self.logger.warning(
                    f"Snapshot cache file '{path}' has mismatched params_hash "
                    f"(expected {self.params_hash}, got {stored_hash}). Ignoring."
                )
            return None
        return snapshot

    @staticmethod
    def _restore_snapshot_to_dir(save_directory: str, snapshot: Dict[str, Any]) -> None:
        """Write a model snapshot blob to *save_directory* as ``model.pt``."""
        os.makedirs(save_directory, exist_ok=True)
        path = os.path.join(save_directory, "model.pt")
        torch.save(snapshot, path)

    def _restore_missing_best_model_dirs_from_snapshots(
        self, snapshot_keys: Set[str]
    ) -> None:
        """Restore best-model directories that are missing on disk.

        After a Snakemake ``--rerun-incomplete`` (or similar), the output
        directories may have been deleted while the training checkpoint still
        exists.  This method recreates any missing directories by loading the
        snapshot from the cache directory on demand, writing it to the output
        directory, and immediately releasing it from memory.
        """
        dir_key_pairs: List[Tuple[str, str]] = []

        if self.best_overall_model_dir is not None:
            dir_key_pairs.append((self.best_overall_model_dir, "overall"))
        if self.best_loss_overall_model_dir is not None:
            dir_key_pairs.append((self.best_loss_overall_model_dir, "loss_overall"))

        for name in self.task_names:
            if self.best_task_models_dir is not None:
                task_dir = os.path.join(self.best_task_models_dir, name)
                dir_key_pairs.append((task_dir, f"task:{name}"))
            if self.best_loss_task_models_dir is not None:
                task_dir = os.path.join(self.best_loss_task_models_dir, name)
                dir_key_pairs.append((task_dir, f"loss_task:{name}"))

        for directory, key in dir_key_pairs:
            if not os.path.exists(directory):
                if key not in snapshot_keys:
                    continue
                snapshot = self._load_snapshot_from_cache(key)
                if snapshot is not None:
                    self._restore_snapshot_to_dir(directory, snapshot)
                    del snapshot  # free memory immediately
                    if self.logger is not None:
                        self.logger.info(
                            f"Restored missing best model directory "
                            f"'{directory}' from cached snapshot"
                        )
                else:
                    if self.logger is not None:
                        self.logger.warning(
                            f"Best model directory '{directory}' is missing "
                            f"and no valid snapshot found in cache"
                        )

    def _restore_missing_best_model_dirs_from_legacy_snapshots(
        self, snapshots: Dict[str, Optional[Dict[str, Any]]]
    ) -> None:
        """Backward-compatible restore from in-memory snapshot blobs.

        Old training checkpoints stored full model weight blobs inside the
        checkpoint file under ``best_so_far["model_snapshots"]``.  This method
        handles that legacy format: it restores any missing output directories,
        writes each snapshot to the new cache directory for future use, and
        populates ``self._best_model_snapshot_keys``.

        The caller should discard the *snapshots* dict after this call to free
        memory.
        """
        dir_key_pairs: List[Tuple[str, str]] = []

        if self.best_overall_model_dir is not None:
            dir_key_pairs.append((self.best_overall_model_dir, "overall"))
        if self.best_loss_overall_model_dir is not None:
            dir_key_pairs.append((self.best_loss_overall_model_dir, "loss_overall"))

        for name in self.task_names:
            if self.best_task_models_dir is not None:
                task_dir = os.path.join(self.best_task_models_dir, name)
                dir_key_pairs.append((task_dir, f"task:{name}"))
            if self.best_loss_task_models_dir is not None:
                task_dir = os.path.join(self.best_loss_task_models_dir, name)
                dir_key_pairs.append((task_dir, f"loss_task:{name}"))

        cache_dir = self._snapshot_cache_dir

        for directory, key in dir_key_pairs:
            snapshot = snapshots.get(key)
            if snapshot is None:
                continue

            # Migrate: write the legacy blob to the new cache directory
            if cache_dir is not None:
                os.makedirs(cache_dir, exist_ok=True)
                snapshot_with_hash = dict(snapshot)
                snapshot_with_hash["params_hash"] = self.params_hash
                filename = self._snapshot_key_to_filename(key)
                path = os.path.join(cache_dir, filename)
                temp_path = path + ".tmp"
                torch.save(snapshot_with_hash, temp_path)
                os.replace(temp_path, path)
                self._best_model_snapshot_keys.add(key)

            # Restore missing output directory
            if not os.path.exists(directory):
                self._restore_snapshot_to_dir(directory, snapshot)
                if self.logger is not None:
                    self.logger.info(
                        f"Restored missing best model directory "
                        f"'{directory}' from legacy checkpoint snapshot"
                    )

    def _save_model_checkpoint(
        self,
        save_directory: str,
        is_initial: bool,
        snapshot_key: Optional[str] = None,
    ) -> None:
        """Save model weights to a directory, managing the initial checkpoint marker.

        On the first validation, all checkpoint directories are saved with an
        ``.initial_checkpoint`` marker file so that Snakemake ``directory()``
        outputs always exist.  When a subsequent validation triggers a genuine
        improvement the checkpoint is overwritten and the marker is removed.

        Args:
            save_directory: Directory to save model weights to.
            is_initial: If True, write ``.initial_checkpoint`` marker.
                If False, remove the marker if it exists.
            snapshot_key: If provided, capture the current trainable weights
                and save them to the snapshot cache directory on disk.
                This allows missing output directories to be restored
                after a resume.
        """
        self.model.save_trainable_weights(
            save_directory=save_directory,
            base_checkpoint=self.checkpoint,
        )
        marker_path = os.path.join(save_directory, INITIAL_CHECKPOINT_MARKER)
        if is_initial:
            with open(marker_path, "w") as f:
                f.write(
                    "This checkpoint was saved at the first validation "
                    "and may not represent an improved model.\n"
                )
        else:
            if os.path.exists(marker_path):
                os.remove(marker_path)

        if snapshot_key is not None:
            self._save_snapshot_to_cache(snapshot_key)

    def _run_validation(self, current_epoch: int) -> bool:
        """Run validation and update metrics. Returns True if early stopping triggered."""
        validation_start_time = time.time()
        self.model.eval()

        gpu_info = _get_gpu_info()

        task_logs: Dict[str, Dict] = {}

        with torch.no_grad():
            for name in self.task_names:
                val_loss = 0.0
                val_batches = 0

                preds_list: List[torch.Tensor] = []
                targets_list: List[torch.Tensor] = []

                forward_start = time.time()
                for batch_data in self.valid_loaders[name]:
                    batch_data = {k: v.to(self.device) for k, v in batch_data.items()}

                    if self.problem_type[name] == "regression":
                        batch_data["labels"] = batch_data["labels"].float()

                    with torch.amp.autocast(
                        device_type=self._autocast_device_type,
                        enabled=self.auto_mixed_precision,
                    ):
                        if self._use_embedding_cache:
                            loss, logits = self.model.forward_from_embeddings(
                                task=name,
                                embeddings=batch_data["embeddings"],
                                attention_mask=batch_data["attention_mask"],
                                labels=batch_data["labels"],
                            )
                        else:
                            args = MultitaskForwardArgs(
                                attention_mask=batch_data["attention_mask"],
                                input_ids=batch_data["input_ids"],
                            )
                            loss, logits = self.model(
                                task=name,
                                args=args,
                                labels=batch_data["labels"],
                            )

                    if loss is None:
                        raise ValueError("Loss is None during validation")

                    val_loss += loss.item()
                    val_batches += 1
                    labels = batch_data["labels"]

                    if self.problem_type[name] == "classification":
                        preds = torch.argmax(logits, dim=-1)
                        mask = labels != self.ignore_index[name]
                        preds_list.append(preds[mask])
                        targets_list.append(labels[mask])
                    else:
                        preds = logits.view(-1)
                        labels = labels.view(-1)

                        if self.levels[name] == "per_residue":
                            attn = batch_data["attention_mask"].view(-1)
                            mask = (labels != self.ignore_index[name]) & (attn == 1)
                        else:
                            mask = labels != self.ignore_index[name]

                        preds_list.append(preds[mask])
                        targets_list.append(labels[mask])
                forward_time = time.time() - forward_start

                avg_train_loss = (
                    self.total_train_loss[name] / self.batches[name]
                    if self.batches[name]
                    else 0.0
                )
                avg_val_loss = val_loss / val_batches if val_batches else 0.0

                metrics: Dict[str, Dict[str, float]] = {}
                metric = 0.0
                metrics_time = 0.0
                num_samples = 0
                metric_name = self.primary_metrics.get(name, "")

                if preds_list and targets_list:
                    preds = torch.cat(preds_list)
                    targets = torch.cat(targets_list)
                    num_samples = preds.numel()
                    metrics_start = time.time()
                    if self.problem_type[name] == "classification":
                        head_cfg = self.model.head_configs[name].head
                        num_classes = getattr(
                            getattr(head_cfg, "output", None), "out_features", None
                        )
                        if num_classes is None:
                            max_val = torch.max(torch.cat([preds, targets])).item()
                            num_classes = int(max_val) + 1
                        if num_classes < 2:
                            num_classes = 2
                        metrics = compute_classification_metrics(
                            preds,
                            targets,
                            num_classes=num_classes,
                            device=self.device,
                            num_bootstraps=0,
                        )
                        if not metric_name:
                            metric_name = "accuracy"
                    else:
                        preds = preds.to(dtype=torch.float32)
                        targets = targets.to(dtype=torch.float32)
                        metrics = compute_regression_metrics(
                            preds,
                            targets,
                            device=self.device,
                            ndcg_k=10,
                            num_bootstraps=0,
                        )
                        if not metric_name:
                            metric_name = "spearman"
                    metrics_time = time.time() - metrics_start

                    if metric_name in metrics:
                        metric = metrics[metric_name]["value"]

                self.task_metrics[name].append(metric)

                # Track per-task best-so-far and save model
                # validation_count has not been incremented yet at this point
                is_first_validation = self.validation_count == 0
                metric_improved = metric > self.best_metrics[name]
                if metric_improved:
                    self.best_metrics[name] = metric
                    self.best_epochs[name] = current_epoch
                    self.best_so_far_task_validation_steps[name] = (
                        self.validation_count + 1
                    )
                    self.best_so_far_task_epochs[name] = current_epoch
                    self.best_so_far_task_global_steps[name] = self.global_step

                    if self.best_task_models_dir is not None:
                        task_dir = os.path.join(self.best_task_models_dir, name)
                        self._save_model_checkpoint(
                            task_dir,
                            is_initial=is_first_validation,
                            snapshot_key=f"task:{name}",
                        )
                        if self.logger is not None:
                            self.logger.info(
                                f"Saved best-so-far model for task {name} "
                                f"(metric={metric:.4f}) at validation "
                                f"{self.validation_count + 1}"
                            )
                elif is_first_validation and self.best_task_models_dir is not None:
                    # Always save on first validation so directory exists
                    task_dir = os.path.join(self.best_task_models_dir, name)
                    self._save_model_checkpoint(
                        task_dir,
                        is_initial=True,
                        snapshot_key=f"task:{name}",
                    )
                    if self.logger is not None:
                        self.logger.info(
                            f"Saved initial model for task {name} "
                            f"(no metric improvement) at validation "
                            f"{self.validation_count + 1}"
                        )

                # Track per-task best-so-far by validation loss (lower is better)
                self.task_val_losses[name].append(avg_val_loss)
                loss_task_improved = avg_val_loss < self.best_val_losses[name]
                if loss_task_improved:
                    self.best_val_losses[name] = avg_val_loss
                    self.best_loss_epochs[name] = current_epoch
                    self.best_so_far_loss_task_validation_steps[name] = (
                        self.validation_count + 1
                    )
                    self.best_so_far_loss_task_epochs[name] = current_epoch
                    self.best_so_far_loss_task_global_steps[name] = self.global_step

                    if self.best_loss_task_models_dir is not None:
                        task_dir = os.path.join(self.best_loss_task_models_dir, name)
                        self._save_model_checkpoint(
                            task_dir,
                            is_initial=is_first_validation,
                            snapshot_key=f"loss_task:{name}",
                        )
                        if self.logger is not None:
                            self.logger.info(
                                f"Saved best-so-far loss model for task {name} "
                                f"(val_loss={avg_val_loss:.4f}) at validation "
                                f"{self.validation_count + 1}"
                            )
                elif is_first_validation and self.best_loss_task_models_dir is not None:
                    # Always save on first validation so directory exists
                    task_dir = os.path.join(self.best_loss_task_models_dir, name)
                    self._save_model_checkpoint(
                        task_dir,
                        is_initial=True,
                        snapshot_key=f"loss_task:{name}",
                    )
                    if self.logger is not None:
                        self.logger.info(
                            f"Saved initial loss model for task {name} "
                            f"(no loss improvement) at validation "
                            f"{self.validation_count + 1}"
                        )

                train_epoch = self.cumulative_batches[name] / len(
                    self.train_loaders[name]
                )
                task_logs[name] = {
                    "train_epoch": train_epoch,
                    "train_loss": avg_train_loss,
                    "eval_loss": avg_val_loss,
                    "metric": metric,
                    "metric_name": metric_name,
                    "metrics": metrics,
                    "forward_time": forward_time,
                    "metrics_time": metrics_time,
                    "num_samples": num_samples,
                }

        # Early stopping check
        if self.early_stopping_enabled:
            for name in self.task_names:
                es_metric_name = self.early_stopping_metrics.get(
                    name, self.primary_metrics.get(name, task_logs[name]["metric_name"])
                )
                task_metrics_dict = task_logs[name].get("metrics", {})
                if es_metric_name in task_metrics_dict:
                    es_metric_value = task_metrics_dict[es_metric_name]["value"]
                else:
                    es_metric_value = task_logs[name]["metric"]

                es_loss_value = float(task_logs[name]["eval_loss"])

                metric_improved = es_metric_value > self.early_stopping_best[name]
                loss_improved = es_loss_value < self.early_stopping_best_loss[name]

                if metric_improved:
                    self.early_stopping_best[name] = es_metric_value
                if loss_improved:
                    self.early_stopping_best_loss[name] = es_loss_value

                if metric_improved or loss_improved:
                    self.early_stopping_counter[name] = 0
                else:
                    self.early_stopping_counter[name] += 1

            all_exhausted = all(
                self.early_stopping_counter[n] >= self.early_stopping_patience
                for n in self.task_names
            )
            if all_exhausted:
                self.early_stopping_triggered = True
                self.stop_reason = "early_stopping"

        validation_time = time.time() - validation_start_time
        self.cumulative_validation_time += validation_time
        self.validation_count += 1

        if (
            self.max_validations is not None
            and self.validation_count >= self.max_validations
        ):
            self.early_stopping_triggered = True
            if self.stop_reason is None:
                self.stop_reason = "max_validations"

        total_elapsed = (
            time.time() - self.training_start_time
        ) + self.elapsed_time_offset
        training_time = total_elapsed - self.cumulative_validation_time

        if self.global_step > 0:
            time_per_step = training_time / self.global_step
        else:
            time_per_step = 0.0

        # Interval training time since the last validation
        training_time_since_last_validation = (
            training_time - self._prev_cumulative_training_time
        )
        training_batches_since_last_validation = (
            self.global_step - self._prev_global_step
        )
        training_time_per_batch = (
            training_time_since_last_validation / training_batches_since_last_validation
            if training_batches_since_last_validation > 0
            else 0.0
        )
        self._prev_cumulative_training_time = training_time
        self._prev_global_step = self.global_step

        if self.validation_count > 0:
            avg_validation_time = (
                self.cumulative_validation_time / self.validation_count
            )
        else:
            avg_validation_time = 0.0

        current_lr = self.scheduler.get_last_lr()[0]
        epoch_float = (
            self.global_step / self.steps_per_epoch
            if self.steps_per_epoch > 0
            else float(current_epoch)
        )
        history_entry: Dict[str, Any] = {
            "epoch": current_epoch,
            "epoch_float": epoch_float,
            "global_step": self.global_step,
            "learning_rate": current_lr,
            "validation_step_timestamp": validation_start_time,
            "gpu": gpu_info,
            "tasks": task_logs,
            "timing": {
                "elapsed_total": total_elapsed,
                "elapsed_training": training_time,
                "elapsed_validation": self.cumulative_validation_time,
                "time_per_step": time_per_step,
                "validation_time": validation_time,
                "validation_forward_time": sum(
                    task_logs[n]["forward_time"] for n in self.task_names
                ),
                "validation_metrics_time": sum(
                    task_logs[n]["metrics_time"] for n in self.task_names
                ),
                "training_time_since_last_validation": training_time_since_last_validation,
                "training_batches_since_last_validation": training_batches_since_last_validation,
                "training_time_per_batch": training_time_per_batch,
            },
        }

        # Add any extra data from strategy
        history_entry.update(self.strategy.get_extra_history_data())

        if self.early_stopping_enabled:
            history_entry["early_stopping"] = {
                "patience": self.early_stopping_patience,
                "counters": dict(self.early_stopping_counter),
                "best_values": dict(self.early_stopping_best),
                "best_loss_values": dict(self.early_stopping_best_loss),
                "triggered": self.early_stopping_triggered,
            }
        if self.stop_reason is not None:
            history_entry["stop_reason"] = self.stop_reason
        self.history.append(history_entry)

        # Reset CUDA peak memory stats so the next validation captures
        # only the peak from the intervening training steps.
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Compute best-so-far overall using relative improvement sum
        eps = 1e-6
        total_rel_improvement = 0.0
        current_overall_metrics: Dict[str, float] = {}
        for name in self.task_names:
            metric = max(float(self.task_metrics[name][-1]), eps)
            current_overall_metrics[name] = metric
            baseline = max(self.best_so_far_overall_baseline_metrics[name], eps)
            total_rel_improvement += (metric - baseline) / baseline

        # validation_count has already been incremented at this point
        is_first_validation_overall = self.validation_count == 1
        overall_improved = total_rel_improvement >= 0.0
        if overall_improved:
            self.best_so_far_overall_baseline_metrics = dict(current_overall_metrics)
            self.best_so_far_overall_validation_step = self.validation_count
            self.best_so_far_overall_epoch = current_epoch
            self.best_so_far_overall_global_step = self.global_step
            self.best_so_far_overall_metrics = dict(current_overall_metrics)

            if self.best_overall_model_dir is not None:
                self._save_model_checkpoint(
                    self.best_overall_model_dir,
                    is_initial=is_first_validation_overall,
                    snapshot_key="overall",
                )
                if self.logger is not None:
                    self.logger.info(
                        f"Saved best-so-far overall model "
                        f"(rel_improve_sum={total_rel_improvement:.6f}) "
                        f"at validation {self.validation_count}"
                    )

        # Compute best-so-far overall by validation loss using relative improvement sum
        # For losses (lower is better): sum of (baseline - current) / baseline
        # Empty baseline_losses dict means first validation (always triggers checkpoint)
        current_losses: Dict[str, float] = {}
        for name in self.task_names:
            current_losses[name] = float(task_logs[name]["eval_loss"])

        if not self.best_so_far_loss_overall_baseline_losses:
            # First validation: always save and use current losses as baseline
            loss_overall_improved = True
            self.best_so_far_loss_overall_baseline_losses = dict(current_losses)
            self.best_so_far_loss_overall_validation_step = self.validation_count
            self.best_so_far_loss_overall_epoch = current_epoch
            self.best_so_far_loss_overall_global_step = self.global_step
            self.best_so_far_loss_overall_losses = dict(current_losses)

            if self.best_loss_overall_model_dir is not None:
                self._save_model_checkpoint(
                    self.best_loss_overall_model_dir,
                    is_initial=True,
                    snapshot_key="loss_overall",
                )
                if self.logger is not None:
                    self.logger.info(
                        f"Saved best-so-far overall loss model "
                        f"(first validation) "
                        f"at validation {self.validation_count}"
                    )
        else:
            total_loss_rel_improvement = 0.0
            for name in self.task_names:
                baseline_loss = self.best_so_far_loss_overall_baseline_losses[name]
                if baseline_loss > 0:
                    total_loss_rel_improvement += (
                        baseline_loss - current_losses[name]
                    ) / baseline_loss
                else:
                    # baseline is 0; if current is also 0 no change, otherwise worse
                    if current_losses[name] <= 0:
                        pass  # no change to rel improvement
                    else:
                        total_loss_rel_improvement -= 1.0

            loss_overall_improved = total_loss_rel_improvement >= 0.0
            if loss_overall_improved:
                self.best_so_far_loss_overall_baseline_losses = dict(current_losses)
                self.best_so_far_loss_overall_validation_step = self.validation_count
                self.best_so_far_loss_overall_epoch = current_epoch
                self.best_so_far_loss_overall_global_step = self.global_step
                self.best_so_far_loss_overall_losses = dict(current_losses)

                if self.best_loss_overall_model_dir is not None:
                    self._save_model_checkpoint(
                        self.best_loss_overall_model_dir,
                        is_initial=False,
                        snapshot_key="loss_overall",
                    )
                    if self.logger is not None:
                        self.logger.info(
                            f"Saved best-so-far overall loss model "
                            f"(loss_rel_improve_sum="
                            f"{total_loss_rel_improvement:.6f}) "
                            f"at validation {self.validation_count}"
                        )

        # Add best_so_far block to history entry
        history_entry["best_so_far"] = {
            "overall": {
                "validation_step": self.best_so_far_overall_validation_step,
                "epoch": self.best_so_far_overall_epoch,
                "global_step": self.best_so_far_overall_global_step,
                "baseline_metrics": dict(self.best_so_far_overall_baseline_metrics),
                "metrics": dict(self.best_so_far_overall_metrics),
                "improved": overall_improved,
            },
            "tasks": {
                name: {
                    "metric": float(self.best_metrics[name]),
                    "validation_step": self.best_so_far_task_validation_steps[name],
                    "epoch": self.best_so_far_task_epochs[name],
                    "global_step": self.best_so_far_task_global_steps[name],
                    "improved": task_logs[name]["metric"] >= self.best_metrics[name]
                    and self.best_so_far_task_validation_steps[name]
                    == self.validation_count,
                }
                for name in self.task_names
            },
            "loss_overall": {
                "validation_step": self.best_so_far_loss_overall_validation_step,
                "epoch": self.best_so_far_loss_overall_epoch,
                "global_step": self.best_so_far_loss_overall_global_step,
                "baseline_losses": dict(self.best_so_far_loss_overall_baseline_losses),
                "losses": dict(self.best_so_far_loss_overall_losses),
                "improved": loss_overall_improved,
            },
            "loss_tasks": {
                name: {
                    "loss": float(self.best_val_losses[name]),
                    "validation_step": self.best_so_far_loss_task_validation_steps[
                        name
                    ],
                    "epoch": self.best_so_far_loss_task_epochs[name],
                    "global_step": self.best_so_far_loss_task_global_steps[name],
                    "improved": task_logs[name]["eval_loss"]
                    <= self.best_val_losses[name]
                    and self.best_so_far_loss_task_validation_steps[name]
                    == self.validation_count,
                }
                for name in self.task_names
            },
        }

        # Log progress
        if self.logger is not None:
            if self.total_steps > 0:
                progress_pct = 100.0 * self.global_step / self.total_steps
            else:
                progress_pct = 0.0

            if self.max_validations is not None:
                progress_str = (
                    f"Validation {self.validation_count}/{self.max_validations} "
                    f"(Epoch {current_epoch})"
                )
            else:
                progress_str = f"Epoch {current_epoch}/{self.max_epochs}"

            self.logger.info(f"--- {progress_str} | {progress_pct:.1f}% ---")

            if self.global_step > 0:
                remaining_steps = self.total_steps - self.global_step
                remaining_validations = (
                    remaining_steps / (self.global_step / self.validation_count)
                    if self.validation_count > 0
                    else 0
                )
                eta_training = remaining_steps * time_per_step
                eta_validation = remaining_validations * avg_validation_time
                eta = eta_training + eta_validation
                self.logger.info(
                    f"  Elapsed: {format_time(total_elapsed)} | "
                    f"ETA: {format_time(eta)} | "
                    f"{format_time(time_per_step)}/step | "
                    f"{format_time(validation_time)} val"
                )
            else:
                self.logger.info(f"  Elapsed: {format_time(total_elapsed)}")

            self.logger.info(f"  LR: {current_lr:.2e}")

            for name in self.task_names:
                t = task_logs[name]
                self.logger.info(
                    f"  {name}: train_loss={t['train_loss']:.4f} | "
                    f"val_loss={t['eval_loss']:.4f} | "
                    f"epochs={t['train_epoch']:.2f}"
                )
                metrics_dict = t.get("metrics", {})
                if metrics_dict:
                    indent = " " * (len(name) + 4)
                    metric_strs = [
                        f"{m_name}={m_vals['value']:.4f}"
                        for m_name, m_vals in metrics_dict.items()
                    ]
                    self.logger.info(f"{indent}{' | '.join(metric_strs)}")

        # Save checkpoint if configured
        if (
            self.training_checkpoint_dir is not None
            and self.checkpoint_every_n_validations > 0
            and self.validation_count % self.checkpoint_every_n_validations == 0
        ):
            align_loss, align_steps = self.strategy.get_checkpoint_data()

            history_entry.setdefault("checkpointing", {})["saved"] = {
                "checkpoint_path": self.training_checkpoint_path,
                "params_hash": self.params_hash,
                "at": {
                    "epoch": current_epoch,
                    "global_step": self.global_step,
                    "validation_count": self.validation_count,
                },
            }
            save_checkpoint_fn = get_save_checkpoint_fn()
            save_checkpoint_fn(
                checkpoint_dir=self.training_checkpoint_dir,
                params_hash=self.params_hash,
                checkpointing_params=self.checkpointing_params,
                train_sampler_generator_states={
                    n: g.get_state() for n, g in self.train_sampler_generators.items()
                },
                train_sampler_iterator_start_states=dict(
                    self.train_sampler_iterator_start_states
                ),
                train_sampler_iterator_start_cumulative_batches=dict(
                    self.train_sampler_iterator_start_cumulative_batches
                ),
                train_total_train_loss=dict(self.total_train_loss),
                train_total_train_batches=dict(self.batches),
                train_total_align_lora_kl_loss=align_loss,
                train_total_align_lora_kl_steps=align_steps,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=current_epoch,
                global_step=self.global_step,
                validation_count=self.validation_count,
                history=self.history,
                task_metrics=self.task_metrics,
                best_metrics=self.best_metrics,
                best_epochs=self.best_epochs,
                cumulative_batches=self.cumulative_batches,
                early_stopping_best=self.early_stopping_best,
                early_stopping_best_loss=self.early_stopping_best_loss,
                early_stopping_counter=self.early_stopping_counter,
                elapsed_time=total_elapsed,
                cumulative_validation_time=self.cumulative_validation_time,
                logger=self.logger,
                best_so_far=self._build_best_so_far_checkpoint_state(),
                task_val_losses=self.task_val_losses,
                best_val_losses=self.best_val_losses,
                best_loss_epochs=self.best_loss_epochs,
            )

        self.model.train()
        return self.early_stopping_triggered

    def _log_stop_reason(self, current_epoch: int) -> None:
        """Log the reason training was stopped."""
        if self.logger is None:
            return
        if self.stop_reason == "early_stopping":
            self.logger.info(
                f"Early stopping triggered at epoch {current_epoch}, "
                f"global_step {self.global_step}, "
                f"validation_step {self.validation_count} "
                f"(patience={self.early_stopping_patience})"
            )
        elif self.stop_reason == "max_validations":
            self.logger.info(
                f"Max validations reached ({self.max_validations}) "
                f"at epoch {current_epoch}, "
                f"global_step {self.global_step}, "
                f"validation_step {self.validation_count}"
            )
        else:
            self.logger.info(
                f"Training stopped at epoch {current_epoch}, "
                f"global_step {self.global_step}, "
                f"validation_step {self.validation_count}"
            )

    def _build_best_so_far_checkpoint_state(self) -> Dict[str, Any]:
        """Build the best-so-far state dict for checkpointing."""
        return {
            "overall": {
                "validation_step": self.best_so_far_overall_validation_step,
                "epoch": self.best_so_far_overall_epoch,
                "global_step": self.best_so_far_overall_global_step,
                "baseline_metrics": dict(self.best_so_far_overall_baseline_metrics),
                "metrics": dict(self.best_so_far_overall_metrics),
            },
            "tasks": {
                name: {
                    "validation_step": self.best_so_far_task_validation_steps[name],
                    "epoch": self.best_so_far_task_epochs[name],
                    "global_step": self.best_so_far_task_global_steps[name],
                }
                for name in self.task_names
            },
            "loss_overall": {
                "validation_step": self.best_so_far_loss_overall_validation_step,
                "epoch": self.best_so_far_loss_overall_epoch,
                "global_step": self.best_so_far_loss_overall_global_step,
                "baseline_losses": dict(self.best_so_far_loss_overall_baseline_losses),
                "losses": dict(self.best_so_far_loss_overall_losses),
            },
            "loss_tasks": {
                name: {
                    "validation_step": self.best_so_far_loss_task_validation_steps[
                        name
                    ],
                    "epoch": self.best_so_far_loss_task_epochs[name],
                    "global_step": self.best_so_far_loss_task_global_steps[name],
                }
                for name in self.task_names
            },
            "model_snapshot_keys": sorted(self._best_model_snapshot_keys),
        }


# Backward compatibility alias
TrainerBase = MultitaskTrainer
