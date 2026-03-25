"""Training strategy protocol and implementations.

This module defines the TrainingStrategy protocol that encapsulates
approach-specific training behavior, and provides concrete implementations
for different training approaches.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple

import torch

from mutafitup.models.multitask_model import MultitaskModel
from mutafitup.models.uncertainty_loss_weights import UncertaintyLossWeights

from .utils import AlignLoraContext


def _sum_losses(task_losses: List[torch.Tensor], device: torch.device) -> torch.Tensor:
    """Sum task losses, returning a zero tensor if empty."""
    if not task_losses:
        return torch.tensor(0.0, device=device)
    result = task_losses[0]
    for loss in task_losses[1:]:
        result = result + loss
    return result


def _setup_uncertainty_weights(
    model: MultitaskModel, task_names: List[str]
) -> Optional[UncertaintyLossWeights]:
    """Create and attach UncertaintyLossWeights to the model."""
    problem_types = {
        name: model.head_configs[name].problem_type for name in task_names
    }
    weights = UncertaintyLossWeights(
        task_names=task_names, problem_types=problem_types
    )
    # Move to same device as model
    device = next(model.parameters()).device
    weights = weights.to(device)
    model.uncertainty_loss_weights = weights
    return weights


def _apply_uncertainty_weighting(
    ulw: Optional[UncertaintyLossWeights],
    loss: torch.Tensor,
    task_name: str,
) -> torch.Tensor:
    """Apply uncertainty weighting if active, otherwise return loss unchanged."""
    if ulw is not None:
        return ulw(loss, task_name)
    return loss


def _uncertainty_history_data(
    ulw: Optional[UncertaintyLossWeights],
) -> Dict[str, Any]:
    """Return current uncertainty log-var values for history logging."""
    if ulw is None:
        return {}
    data: Dict[str, Any] = {}
    for name, param in ulw.log_vars.items():
        log_var = param.item()
        data[f"uncertainty_log_var_{name}"] = log_var
        # Effective weight: exp(-s) for classification, 0.5*exp(-s) for regression
        if ulw.problem_types[name] == "regression":
            data[f"uncertainty_weight_{name}"] = 0.5 * torch.exp(-param).item()
        else:
            data[f"uncertainty_weight_{name}"] = torch.exp(-param).item()
    return data


class TrainingStrategy(Protocol):
    """Protocol defining approach-specific training behavior.

    Implementations control:
    - Whether embedding caching can be used
    - Whether gradients are accumulated across tasks
    - How losses are processed per-task and combined
    - What extra data appears in history/checkpoints
    """

    @property
    def use_embedding_cache(self) -> bool:
        """Whether this strategy can use embedding caching."""
        ...

    @property
    def accumulate_across_tasks(self) -> bool:
        """Whether to accumulate gradients across all tasks before stepping."""
        ...

    @property
    def accumulate_requires_combined_backward(self) -> bool:
        """Whether the accumulate path needs a single combined backward.

        When True, all task losses are kept as live tensors and backward is called
        once on the combined loss (needed e.g. for AlignLoRA KL gradient flow).
        When False, each task's loss is backpropagated immediately and gradients
        accumulate additively (more memory-efficient).
        Only relevant when accumulate_across_tasks=True.
        """
        ...

    def setup(self, model: MultitaskModel, task_names: List[str]) -> None:
        """Called once before training starts."""
        ...

    def teardown(self) -> None:
        """Called once after training ends."""
        ...

    def on_epoch_start(self) -> None:
        """Called at the start of each epoch."""
        ...

    def on_step_start(self) -> None:
        """Called before each training step (outer batch iteration)."""
        ...

    def on_task_start(self, task_name: str) -> None:
        """Called before forward pass for a specific task."""
        ...

    def on_task_end(self, task_name: str) -> None:
        """Called after forward pass for a specific task."""
        ...

    def process_loss(
        self, loss: torch.Tensor, task_name: str, num_tasks: int
    ) -> torch.Tensor:
        """Process the loss for a task. Called for each task's loss."""
        ...

    def combine_losses(
        self, task_losses: List[torch.Tensor], device: torch.device
    ) -> torch.Tensor:
        """Combine task losses into final loss for backward pass.

        Only called when accumulate_across_tasks=True.
        """
        ...

    def get_extra_checkpointing_params(self) -> Dict[str, Any]:
        """Return extra parameters to include in checkpoint hash."""
        ...

    def get_extra_history_data(self) -> Dict[str, Any]:
        """Return extra data to add to validation history entries."""
        ...

    def get_checkpoint_data(self) -> Tuple[Optional[float], Optional[int]]:
        """Return (align_lora_kl_loss, align_lora_kl_steps) for checkpointing."""
        ...

    def restore_from_checkpoint(self, checkpoint_data: Dict[str, Any]) -> None:
        """Restore strategy-specific state from a checkpoint.

        Called after loading a training checkpoint so that internal counters
        (e.g. AlignLoRA KL totals) are consistent with the checkpoint state.
        """
        ...


@dataclass
class HeadsOnlyStrategy:
    """Strategy for heads-only training.

    Trains only task-specific heads with frozen backbone.
    Can use embedding caching for faster training.
    Does not accumulate gradients across tasks.
    """

    uncertainty_weighting: bool = False
    use_embedding_cache: bool = True
    accumulate_across_tasks: bool = False
    accumulate_requires_combined_backward: bool = False

    _ulw: Optional[UncertaintyLossWeights] = field(
        default=None, init=False, repr=False
    )

    def setup(self, model: MultitaskModel, task_names: List[str]) -> None:
        if self.uncertainty_weighting:
            self._ulw = _setup_uncertainty_weights(model, task_names)

    def teardown(self) -> None:
        pass

    def on_epoch_start(self) -> None:
        pass

    def on_step_start(self) -> None:
        pass

    def on_task_start(self, task_name: str) -> None:
        pass

    def on_task_end(self, task_name: str) -> None:
        pass

    def process_loss(
        self, loss: torch.Tensor, task_name: str, num_tasks: int
    ) -> torch.Tensor:
        return _apply_uncertainty_weighting(self._ulw, loss, task_name)

    def combine_losses(
        self, task_losses: List[torch.Tensor], device: torch.device
    ) -> torch.Tensor:
        # Not used since accumulate_across_tasks=False
        if not task_losses:
            return torch.tensor(0.0, device=device)
        result = task_losses[0]
        for loss in task_losses[1:]:
            result = result + loss
        return result

    def get_extra_checkpointing_params(self) -> Dict[str, Any]:
        if self.uncertainty_weighting:
            return {"uncertainty_weighting": True}
        return {}

    def get_extra_history_data(self) -> Dict[str, Any]:
        return _uncertainty_history_data(self._ulw)

    def get_checkpoint_data(self) -> Tuple[Optional[float], Optional[int]]:
        return None, None

    def restore_from_checkpoint(self, checkpoint_data: Dict[str, Any]) -> None:
        pass


@dataclass
class LoRAStrategy:
    """Strategy for LoRA fine-tuning.

    Fine-tunes the backbone using LoRA adapters.
    Cannot use embedding caching since backbone is being trained.
    Does not accumulate gradients across tasks.
    """

    uncertainty_weighting: bool = False
    use_embedding_cache: bool = False
    accumulate_across_tasks: bool = False
    accumulate_requires_combined_backward: bool = False

    _ulw: Optional[UncertaintyLossWeights] = field(
        default=None, init=False, repr=False
    )

    def setup(self, model: MultitaskModel, task_names: List[str]) -> None:
        if self.uncertainty_weighting:
            self._ulw = _setup_uncertainty_weights(model, task_names)

    def teardown(self) -> None:
        pass

    def on_epoch_start(self) -> None:
        pass

    def on_step_start(self) -> None:
        pass

    def on_task_start(self, task_name: str) -> None:
        pass

    def on_task_end(self, task_name: str) -> None:
        pass

    def process_loss(
        self, loss: torch.Tensor, task_name: str, num_tasks: int
    ) -> torch.Tensor:
        return _apply_uncertainty_weighting(self._ulw, loss, task_name)

    def combine_losses(
        self, task_losses: List[torch.Tensor], device: torch.device
    ) -> torch.Tensor:
        # Not used since accumulate_across_tasks=False
        return _sum_losses(task_losses, device)

    def get_extra_checkpointing_params(self) -> Dict[str, Any]:
        if self.uncertainty_weighting:
            return {"uncertainty_weighting": True}
        return {}

    def get_extra_history_data(self) -> Dict[str, Any]:
        return _uncertainty_history_data(self._ulw)

    def get_checkpoint_data(self) -> Tuple[Optional[float], Optional[int]]:
        return None, None

    def restore_from_checkpoint(self, checkpoint_data: Dict[str, Any]) -> None:
        pass


@dataclass
class AccGradLoRAStrategy:
    """Strategy for accumulated gradient LoRA training.

    Accumulates gradients from all tasks before optimizer step.
    Each task's loss is backpropagated immediately for memory efficiency.
    Divides each task's loss by number of tasks for proper scaling.
    """

    uncertainty_weighting: bool = False
    use_embedding_cache: bool = False
    accumulate_across_tasks: bool = True
    accumulate_requires_combined_backward: bool = False

    _ulw: Optional[UncertaintyLossWeights] = field(
        default=None, init=False, repr=False
    )

    def setup(self, model: MultitaskModel, task_names: List[str]) -> None:
        if self.uncertainty_weighting:
            self._ulw = _setup_uncertainty_weights(model, task_names)

    def teardown(self) -> None:
        pass

    def on_epoch_start(self) -> None:
        pass

    def on_step_start(self) -> None:
        pass

    def on_task_start(self, task_name: str) -> None:
        pass

    def on_task_end(self, task_name: str) -> None:
        pass

    def process_loss(
        self, loss: torch.Tensor, task_name: str, num_tasks: int
    ) -> torch.Tensor:
        loss = _apply_uncertainty_weighting(self._ulw, loss, task_name)
        return loss / num_tasks

    def combine_losses(
        self, task_losses: List[torch.Tensor], device: torch.device
    ) -> torch.Tensor:
        return _sum_losses(task_losses, device)

    def get_extra_checkpointing_params(self) -> Dict[str, Any]:
        if self.uncertainty_weighting:
            return {"uncertainty_weighting": True}
        return {}

    def get_extra_history_data(self) -> Dict[str, Any]:
        return _uncertainty_history_data(self._ulw)

    def get_checkpoint_data(self) -> Tuple[Optional[float], Optional[int]]:
        return None, None

    def restore_from_checkpoint(self, checkpoint_data: Dict[str, Any]) -> None:
        pass


@dataclass
class AlignLoRAStrategy:
    """Strategy for aligned LoRA training with KL divergence regularization.

    Extends accumulated gradient training with a KL divergence term
    that encourages LoRA representations to be similar across tasks.
    """

    uncertainty_weighting: bool = False
    align_lora_kl_lambda: float = 0.01
    gradient_checkpointing: bool = False
    use_embedding_cache: bool = False
    accumulate_across_tasks: bool = True
    accumulate_requires_combined_backward: bool = True

    # Internal state
    _ulw: Optional[UncertaintyLossWeights] = field(
        default=None, init=False, repr=False
    )
    _context: Optional[AlignLoraContext] = field(default=None, init=False, repr=False)
    _task_names: List[str] = field(default_factory=list, init=False, repr=False)
    _total_kl_loss: float = field(default=0.0, init=False, repr=False)
    _total_kl_steps: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.align_lora_kl_lambda < 0:
            raise ValueError("align_lora_kl_lambda must be >= 0")

    def setup(self, model: MultitaskModel, task_names: List[str]) -> None:
        if len(task_names) < 2:
            raise ValueError("align_lora_kl requires at least two tasks")
        self._task_names = task_names
        if self.uncertainty_weighting:
            self._ulw = _setup_uncertainty_weights(model, task_names)
        self._context = AlignLoraContext(model, task_names, align_lora_kl=True)
        self._total_kl_loss = 0.0
        self._total_kl_steps = 0
        if self.gradient_checkpointing:
            model.enable_gradient_checkpointing()

    def teardown(self) -> None:
        if self._context is not None:
            self._context.close()
            self._context = None

    def on_epoch_start(self) -> None:
        self._total_kl_loss = 0.0
        self._total_kl_steps = 0

    def on_step_start(self) -> None:
        if self._context is not None:
            self._context.begin_step()

    def on_task_start(self, task_name: str) -> None:
        if self._context is not None:
            self._context.set_current_task(task_name)

    def on_task_end(self, task_name: str) -> None:
        if self._context is not None:
            self._context.set_current_task(None)

    def process_loss(
        self, loss: torch.Tensor, task_name: str, num_tasks: int
    ) -> torch.Tensor:
        loss = _apply_uncertainty_weighting(self._ulw, loss, task_name)
        return loss / num_tasks

    def combine_losses(
        self, task_losses: List[torch.Tensor], device: torch.device
    ) -> torch.Tensor:
        combined = _sum_losses(task_losses, device)

        if self._context is not None:
            kl = self._context.compute_kl(self._task_names, self.align_lora_kl_lambda)
            self._total_kl_loss += float(kl.detach().cpu())
            self._total_kl_steps += 1
            combined = combined + kl

        return combined

    def get_extra_checkpointing_params(self) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "align_lora_kl": True,
            "align_lora_kl_lambda": self.align_lora_kl_lambda,
        }
        if self.uncertainty_weighting:
            params["uncertainty_weighting"] = True
        return params

    def get_extra_history_data(self) -> Dict[str, Any]:
        avg_kl = (
            self._total_kl_loss / self._total_kl_steps
            if self._total_kl_steps > 0
            else 0.0
        )
        data: Dict[str, Any] = {"align_lora_kl_loss": avg_kl}
        data.update(_uncertainty_history_data(self._ulw))
        return data

    def get_checkpoint_data(self) -> Tuple[Optional[float], Optional[int]]:
        return self._total_kl_loss, self._total_kl_steps

    def restore_from_checkpoint(self, checkpoint_data: Dict[str, Any]) -> None:
        """Restore KL loss tracking state from a checkpoint."""
        kl_loss = checkpoint_data.get("train_total_align_lora_kl_loss")
        kl_steps = checkpoint_data.get("train_total_align_lora_kl_steps")
        if kl_loss is not None:
            self._total_kl_loss = float(kl_loss)
        if kl_steps is not None:
            self._total_kl_steps = int(kl_steps)
