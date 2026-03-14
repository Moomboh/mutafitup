import logging
import os
import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from mutafitup.models.multitask_model import MultitaskModel


def save_training_checkpoint(
    checkpoint_dir: str,
    params_hash: str,
    checkpointing_params: Dict[str, Any],
    train_sampler_generator_states: Optional[Dict[str, torch.Tensor]],
    train_sampler_iterator_start_states: Optional[Dict[str, torch.Tensor]],
    train_sampler_iterator_start_cumulative_batches: Optional[Dict[str, int]],
    train_total_train_loss: Optional[Dict[str, float]],
    train_total_train_batches: Optional[Dict[str, int]],
    train_total_align_lora_kl_loss: Optional[float],
    train_total_align_lora_kl_steps: Optional[int],
    model: MultitaskModel,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    global_step: int,
    validation_count: int,
    history: List[Dict],
    task_metrics: Dict[str, List[float]],
    best_metrics: Dict[str, float],
    best_epochs: Dict[str, int],
    cumulative_batches: Dict[str, int],
    early_stopping_best: Dict[str, float],
    early_stopping_best_loss: Dict[str, float],
    early_stopping_counter: Dict[str, int],
    elapsed_time: float,
    cumulative_validation_time: float,
    logger: Optional[logging.Logger] = None,
    best_so_far: Optional[Dict[str, Any]] = None,
    task_val_losses: Optional[Dict[str, List[float]]] = None,
    best_val_losses: Optional[Dict[str, float]] = None,
    best_loss_epochs: Optional[Dict[str, int]] = None,
) -> None:
    """Save a training checkpoint to disk."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "training_checkpoint.pt")

    checkpoint_data = {
        "params_hash": params_hash,
        "checkpointing_params": checkpointing_params,
        "train_sampler_generator_states": train_sampler_generator_states,
        "train_sampler_iterator_start_states": train_sampler_iterator_start_states,
        "train_sampler_iterator_start_cumulative_batches": train_sampler_iterator_start_cumulative_batches,
        "train_total_train_loss": train_total_train_loss,
        "train_total_train_batches": train_total_train_batches,
        "train_total_align_lora_kl_loss": train_total_align_lora_kl_loss,
        "train_total_align_lora_kl_steps": train_total_align_lora_kl_steps,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "validation_count": validation_count,
        "history": history,
        "task_metrics": task_metrics,
        "best_metrics": best_metrics,
        "best_epochs": best_epochs,
        "cumulative_batches": cumulative_batches,
        "early_stopping_best": early_stopping_best,
        "early_stopping_best_loss": early_stopping_best_loss,
        "early_stopping_counter": early_stopping_counter,
        "elapsed_time": elapsed_time,
        "cumulative_validation_time": cumulative_validation_time,
        "rng_state_torch": torch.get_rng_state(),
        "rng_state_torch_cuda": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None,
        "rng_state_numpy": np.random.get_state(),
        "rng_state_random": random.getstate(),
        "best_so_far": best_so_far,
        "task_val_losses": task_val_losses,
        "best_val_losses": best_val_losses,
        "best_loss_epochs": best_loss_epochs,
    }

    temp_path = checkpoint_path + ".tmp"
    torch.save(checkpoint_data, temp_path)
    os.replace(temp_path, checkpoint_path)

    if logger is not None:
        logger.info(f"Saved training checkpoint at epoch {epoch}, step {global_step}")


def load_training_checkpoint(
    checkpoint_dir: str,
    params_hash: str,
    model: MultitaskModel,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    logger: Optional[logging.Logger] = None,
) -> Optional[Dict[str, Any]]:
    """Load a training checkpoint from disk if it exists and matches the params hash."""
    checkpoint_path = os.path.join(checkpoint_dir, "training_checkpoint.pt")

    if not os.path.exists(checkpoint_path):
        return None

    checkpoint_data = torch.load(checkpoint_path, weights_only=False)

    if checkpoint_data.get("params_hash") != params_hash:
        if logger is not None:
            logger.warning(
                f"Checkpoint params hash mismatch. "
                f"Expected {params_hash}, got {checkpoint_data.get('params_hash')}. "
                f"Starting fresh training."
            )
        return None

    model.load_state_dict(checkpoint_data["model_state_dict"])
    optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint_data["scheduler_state_dict"])

    torch.set_rng_state(checkpoint_data["rng_state_torch"])
    if (
        checkpoint_data["rng_state_torch_cuda"] is not None
        and torch.cuda.is_available()
    ):
        torch.cuda.set_rng_state_all(checkpoint_data["rng_state_torch_cuda"])
    np.random.set_state(checkpoint_data["rng_state_numpy"])
    random.setstate(checkpoint_data["rng_state_random"])

    if logger is not None:
        logger.info(
            f"Resumed from checkpoint at epoch {checkpoint_data['epoch']}, "
            f"step {checkpoint_data['global_step']}"
        )

    return checkpoint_data


# Backward compatibility aliases (prefixed with underscore for internal use)
_save_training_checkpoint = save_training_checkpoint
_load_training_checkpoint = load_training_checkpoint
