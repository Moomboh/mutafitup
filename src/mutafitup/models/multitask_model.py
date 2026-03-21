import importlib
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
from peft import LoraConfig
from torch import nn


@dataclass
class HeadConfig:
    head: nn.Module
    problem_type: Literal["classification", "regression"]
    level: Literal["per_residue", "per_protein"]
    ignore_index: int = -100


@dataclass
class MultitaskForwardArgs:
    attention_mask: torch.Tensor
    input_ids: Optional[torch.Tensor] = None
    position_ids: Optional[torch.Tensor] = None
    head_mask: Optional[torch.Tensor] = None
    inputs_embeds: Optional[torch.Tensor] = None
    output_attentions: Optional[bool] = None
    output_hidden_states: Optional[bool] = None


class MultitaskBackbone(nn.Module, ABC):
    _lora_config: Optional[LoraConfig] = None

    @abstractmethod
    def forward(
        self,
        args: MultitaskForwardArgs,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def preprocess_sequences(self, sequences: List[str], checkpoint: str) -> List[str]:
        return sequences

    def apply_lora_config(self, lora_config: Optional[LoraConfig]) -> None:
        self._inject_lora_config(lora_config)
        self._lora_config = lora_config

    @property
    def lora_config(self) -> Optional[LoraConfig]:
        return self._lora_config

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing if the backbone supports it.

        Concrete subclasses (e.g. ``EsmcBackbone``) may override this to
        monkey-patch their internal model.  The default implementation is a
        no-op so that strategies can call it unconditionally.
        """
        pass

    @abstractmethod
    def _inject_lora_config(self, lora_config: Optional[LoraConfig]) -> None:
        raise NotImplementedError


class MultitaskModel(nn.Module):
    def __init__(
        self,
        backbone: MultitaskBackbone,
        heads: Dict[str, HeadConfig],
    ):
        super().__init__()
        self.backbone = backbone
        self.head_configs = heads
        self.heads = nn.ModuleDict({name: cfg.head for name, cfg in heads.items()})

    def enable_gradient_checkpointing(self) -> None:
        """Delegate gradient checkpointing to the backbone."""
        self.backbone.enable_gradient_checkpointing()

    def forward(
        self,
        task: str,
        args: MultitaskForwardArgs,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        if task not in self.heads:
            raise ValueError(f"Unknown task: {task}")

        config = self.head_configs[task]

        hidden_states = self.backbone.forward(args)

        logits = self.heads[task](hidden_states, args.attention_mask)
        loss = None

        if labels is not None:
            if config.problem_type == "classification":
                loss = self._classification_loss(
                    config,
                    logits,
                    labels,
                    args.attention_mask,
                )
            elif config.problem_type == "regression":
                loss = self._regression_loss(
                    config,
                    logits,
                    labels,
                    args.attention_mask,
                )
            else:
                raise ValueError(f"Unsupported problem_type: {config.problem_type}")

        return loss, logits

    def forward_from_embeddings(
        self,
        task: str,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        if task not in self.heads:
            raise ValueError(f"Unknown task: {task}")

        config = self.head_configs[task]

        logits = self.heads[task](embeddings, attention_mask)
        loss = None

        if labels is not None:
            if config.problem_type == "classification":
                loss = self._classification_loss(
                    config,
                    logits,
                    labels,
                    attention_mask,
                )
            elif config.problem_type == "regression":
                loss = self._regression_loss(
                    config,
                    logits,
                    labels,
                    attention_mask,
                )
            else:
                raise ValueError(f"Unsupported problem_type: {config.problem_type}")

        return loss, logits

    def _classification_loss(
        self,
        config: HeadConfig,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        criterion = nn.CrossEntropyLoss()

        if config.level == "per_residue":
            if logits.dim() != 3:
                raise ValueError("Per-residue classification expects logits [B, L, C]")

            num_labels = logits.size(-1)
            flat_logits = logits.view(-1, num_labels).float()
            flat_labels = labels.view(-1)
            active = attention_mask.view(-1) == 1
            ignore = torch.full_like(flat_labels, config.ignore_index)
            active_labels = torch.where(active, flat_labels, ignore)
            mask = active_labels != config.ignore_index

            if not torch.any(mask):
                return logits.new_tensor(0.0)

            return criterion(flat_logits[mask], active_labels[mask])

        if config.level == "per_protein":
            if logits.dim() != 2:
                raise ValueError("Per-protein classification expects logits [B, C]")

            return criterion(logits.float(), labels)

        raise ValueError(f"Unsupported level: {config.level}")

    def _regression_loss(
        self,
        config: HeadConfig,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        criterion = nn.MSELoss()

        if config.level == "per_residue":
            if logits.dim() != 3:
                raise ValueError("Per-residue regression expects logits [B, L]")

            flat_logits = logits.view(-1).float()
            flat_labels = labels.view(-1).float()
            mask = (flat_labels != config.ignore_index) & (attention_mask.view(-1) == 1)
            if not torch.any(mask):
                return logits.new_tensor(0.0)
            return criterion(flat_logits[mask], flat_labels[mask])

        if config.level == "per_protein":
            if logits.dim() != 2:
                raise ValueError("Per-protein regression expects logits [B]")

            flat_logits = logits.view(-1).float()
            flat_labels = labels.view(-1).float()
            mask = flat_labels != config.ignore_index

            if not torch.any(mask):
                return logits.new_tensor(0.0)

            return criterion(flat_logits[mask], flat_labels[mask])

        raise ValueError(f"Unsupported level: {config.level}")

    def save_trainable_weights(self, save_directory: str, base_checkpoint: str) -> int:
        os.makedirs(save_directory, exist_ok=True)
        path = os.path.join(save_directory, "model.pt")
        return type(self).save_to_file(self, path=path, base_checkpoint=base_checkpoint)

    @classmethod
    def save_to_file(
        cls,
        model: "MultitaskModel",
        path: Union[str, os.PathLike],
        base_checkpoint: str,
    ) -> int:
        state_dict = {
            name: param.detach().cpu()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        config = model._build_serialization_config(base_checkpoint=base_checkpoint)

        torch.save({"state_dict": state_dict, "config": config}, os.fspath(path))

        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    @classmethod
    def load_from_file(
        cls,
        path: Union[str, os.PathLike],
        device: Optional[torch.device] = None,
    ) -> "MultitaskModel":
        data = torch.load(
            os.fspath(path), map_location=device or "cpu", weights_only=False
        )

        config: Dict[str, Any] = data["config"]
        state_dict: Dict[str, torch.Tensor] = data["state_dict"]

        base_checkpoint: str = config["base_checkpoint"]
        backbone_class_path: str = config["backbone_class"]
        backbone_cls = cls._import_string(backbone_class_path)

        lora_config: Optional[LoraConfig] = config.get("lora_config")

        backbone, _ = backbone_cls.from_pretrained(  # type: ignore[attr-defined]
            checkpoint=base_checkpoint,
            lora_config=None,
        )

        backbone.apply_lora_config(lora_config)

        backbone_hidden_size: Optional[int] = config.get("backbone_hidden_size")
        tasks_config: Dict[str, Dict[str, Any]] = config.get("tasks", {})

        heads: Dict[str, HeadConfig] = {}

        for name, tcfg in tasks_config.items():
            head_cls = cls._import_string(tcfg["head_class"])

            if backbone_hidden_size is None:
                raise ValueError("backbone_hidden_size missing from config")

            dropout_prop = tcfg.get("dropout_prop", 0.0)
            input_hidden_size = tcfg.get("input_hidden_size", backbone_hidden_size)
            head_hidden_size = tcfg.get("head_hidden_size", backbone_hidden_size)

            if tcfg["problem_type"] == "classification":
                num_outputs: int = tcfg["num_outputs"]
                head = head_cls(
                    dropout_prop,
                    input_hidden_size,
                    head_hidden_size,
                    num_outputs,
                )
            elif tcfg["problem_type"] == "regression":
                head = head_cls(
                    dropout_prop,
                    input_hidden_size,
                    head_hidden_size,
                )
            else:
                raise ValueError(
                    f"Unsupported problem_type in config: {tcfg['problem_type']}"
                )

            heads[name] = HeadConfig(
                head=head,
                problem_type=tcfg["problem_type"],
                level=tcfg["level"],
                ignore_index=tcfg.get("ignore_index", -100),
            )

        model = cls(backbone=backbone, heads=heads)
        model.load_state_dict(state_dict, strict=False)

        if device is not None:
            model.to(device)

        return model

    def _build_serialization_config(self, base_checkpoint: str) -> Dict[str, Any]:
        backbone_class = type(self.backbone)
        tasks: Dict[str, Dict[str, Any]] = {}

        for name, cfg in self.head_configs.items():
            head = cfg.head
            head_class = type(head)
            num_outputs = self._infer_head_num_outputs(head)

            dropout = getattr(head, "dropout", None)
            dropout_prop = getattr(dropout, "p", 0.0)

            linear = getattr(head, "linear", None)
            input_hidden_size = None
            head_hidden_size = None
            if isinstance(linear, nn.Linear):
                input_hidden_size = linear.in_features
                head_hidden_size = linear.out_features

            tasks[name] = {
                "problem_type": cfg.problem_type,
                "level": cfg.level,
                "ignore_index": cfg.ignore_index,
                "head_class": f"{head_class.__module__}.{head_class.__qualname__}",
                "num_outputs": num_outputs,
                "dropout_prop": dropout_prop,
                "input_hidden_size": input_hidden_size,
                "head_hidden_size": head_hidden_size,
            }

        return {
            "base_checkpoint": base_checkpoint,
            "backbone_class": f"{backbone_class.__module__}.{backbone_class.__qualname__}",
            "backbone_hidden_size": getattr(self.backbone, "hidden_size", None),
            "lora_config": getattr(self.backbone, "lora_config", None),
            "tasks": tasks,
        }

    @staticmethod
    def _import_string(path: str):
        module_name, attr_name = path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, attr_name)

    @staticmethod
    def _infer_head_num_outputs(head: nn.Module) -> int:
        output = getattr(head, "output", None)
        if isinstance(output, nn.Linear):
            return output.out_features
        raise ValueError(f"Cannot infer num_outputs for head type {type(head)}")
