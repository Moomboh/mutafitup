import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple

import torch

from mutafitup.device import get_device


def find_lora_a_default_modules(
    backbone: torch.nn.Module,
    name_substring: str = "lora_A.default",
) -> Dict[str, torch.nn.Module]:
    """Find all modules in the backbone whose name contains the given substring."""
    return {
        name: module
        for name, module in backbone.named_modules()
        if name_substring in name
    }


def compute_align_lora_kl_from_outputs(
    outputs_by_task: Dict[str, Dict[str, torch.Tensor]],
    task_names: List[str],
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute symmetric KL divergence between task outputs for LoRA alignment."""
    device: Optional[torch.device] = get_device()

    module_names: List[str] = []

    for t in task_names:
        module_names.extend(list(outputs_by_task.get(t, {}).keys()))

    module_names = sorted(set(module_names))

    if not module_names:
        return torch.tensor(0.0, device=device)

    total: Optional[torch.Tensor] = None

    count = 0

    for module_name in module_names:
        stats: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for t in task_names:
            out = outputs_by_task.get(t, {}).get(module_name)

            if out is None or not isinstance(out, torch.Tensor) or out.dim() == 0:
                continue

            flat = out.float().reshape(-1, out.shape[-1])
            mu = flat.mean(dim=0)
            var = flat.var(dim=0, unbiased=False)
            var = torch.clamp(var, min=float(eps))
            stats.append((mu, var))

        if len(stats) < 2:
            continue

        pair_total: Optional[torch.Tensor] = None
        pair_count = 0

        for i in range(len(stats)):
            mu_i, var_i = stats[i]
            for j in range(i + 1, len(stats)):
                mu_j, var_j = stats[j]
                diff2 = (mu_i - mu_j).pow(2)
                kl_ij = 0.5 * torch.sum(
                    torch.log(var_j / var_i) + (var_i + diff2) / var_j - 1.0
                )
                kl_ji = 0.5 * torch.sum(
                    torch.log(var_i / var_j) + (var_j + diff2) / var_i - 1.0
                )
                sym = 0.5 * (kl_ij + kl_ji)

                # Clamp to prevent degenerate variance ratios from
                # producing inf/nan that would poison the combined loss.
                sym = torch.clamp(sym, max=1e4)
                pair_total = sym if pair_total is None else (pair_total + sym)
                pair_count += 1

        if pair_total is None or pair_count == 0:
            continue

        mod_val = pair_total / pair_count
        total = mod_val if total is None else (total + mod_val)
        count += 1

    if total is None or count == 0:
        return torch.tensor(0.0, device=device)

    return total / count


def compute_params_hash(params: Dict[str, Any]) -> str:
    """Compute a hash of the training parameters for checkpoint validation."""
    params_json = json.dumps(params, sort_keys=True)
    return hashlib.sha256(params_json.encode()).hexdigest()[:16]


def compute_per_task_seed(seed: int, task_name: str) -> int:
    """Compute a deterministic seed for a specific task."""
    h = hashlib.sha256(f"{seed}:{task_name}".encode()).hexdigest()
    return int(h[:16], 16) % (2**63 - 1)


def init_train_sampler_generators(
    task_names: List[str], seed: int
) -> Dict[str, torch.Generator]:
    """Initialize random generators for each task's data sampler."""
    gens: Dict[str, torch.Generator] = {}
    for name in task_names:
        g = torch.Generator(device="cpu")
        g.manual_seed(compute_per_task_seed(seed, name))
        gens[name] = g
    return gens


def attach_generators_to_train_loaders(
    train_loaders: Dict[str, "torch.utils.data.DataLoader"],
    train_sampler_generators: Dict[str, torch.Generator],
) -> None:
    """Attach random generators to the train loaders' samplers."""
    for name, dl in train_loaders.items():
        sampler = getattr(dl, "sampler", None)
        if sampler is not None and hasattr(sampler, "generator"):
            sampler.generator = train_sampler_generators[name]


def format_time(seconds: float) -> str:
    """Format a time duration in seconds as a human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m{secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h{mins}m"


class AlignLoraContext:
    """Context manager for tracking LoRA module outputs across tasks for KL alignment."""

    def __init__(
        self,
        model: "MultitaskModel",
        task_names: List[str],
        align_lora_kl: bool,
    ) -> None:
        self.align_lora_kl = align_lora_kl
        self.task_names = task_names
        self.outputs_by_task: Dict[str, Dict[str, torch.Tensor]] = {}
        self.current_task: Optional[str] = None
        self.handles: List[Any] = []

        if not align_lora_kl:
            return

        modules = find_lora_a_default_modules(model.backbone)
        if not modules:
            raise ValueError(
                "align_lora_kl=True but no modules matched 'lora_A.default' in backbone"
            )

        for module_name, module in modules.items():
            self.handles.append(
                module.register_forward_hook(self._make_hook(module_name))
            )

    def _make_hook(self, module_name: str):
        def _hook(_module, _inputs, output):
            if self.current_task is None:
                return

            out = output
            if isinstance(out, (tuple, list)) and out:
                out = out[0]
            if not isinstance(out, torch.Tensor):
                return

            # Upcast to float32 to prevent numerical issues in KL computation
            self.outputs_by_task.setdefault(self.current_task, {})[module_name] = (
                out.float()
            )

        return _hook

    def begin_step(self) -> None:
        """Clear outputs at the beginning of a training step."""
        if not self.align_lora_kl:
            return
        self.outputs_by_task.clear()
        self.current_task = None

    def set_current_task(self, task_name: Optional[str]) -> None:
        """Set the current task being processed."""
        if not self.align_lora_kl:
            return
        self.current_task = task_name

    def compute_kl(
        self, task_names: List[str], align_lora_kl_lambda: float
    ) -> torch.Tensor:
        """Compute the weighted KL divergence loss."""
        if not self.align_lora_kl:
            device = get_device()
            return torch.tensor(0.0, device=device)
        kl = compute_align_lora_kl_from_outputs(self.outputs_by_task, task_names)
        return kl * float(align_lora_kl_lambda)

    def close(self) -> None:
        """Remove all registered hooks."""
        for handle in self.handles:
            try:
                handle.remove()
            except Exception:
                pass


# Backward compatibility aliases (prefixed with underscore for internal use)
_find_lora_a_default_modules = find_lora_a_default_modules
_compute_align_lora_kl_from_outputs = compute_align_lora_kl_from_outputs
_compute_params_hash = compute_params_hash
_compute_per_task_seed = compute_per_task_seed
_init_train_sampler_generators = init_train_sampler_generators
_attach_generators_to_train_loaders = attach_generators_to_train_loaders
_format_time = format_time
_AlignLoraContext = AlignLoraContext
