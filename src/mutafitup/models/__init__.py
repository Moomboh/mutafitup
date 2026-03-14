from typing import Optional, Tuple

from peft import LoraConfig
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from mutafitup.models.multitask_backbones import EsmBackbone, EsmcBackbone, T5Backbone
from mutafitup.models.multitask_model import MultitaskBackbone


def build_backbone_and_tokenizer(
    checkpoint: str,
    lora_rank: Optional[int],
    lora_alpha: Optional[int],
) -> Tuple[MultitaskBackbone, PreTrainedTokenizerBase]:
    if (lora_rank is None) != (lora_alpha is None):
        raise ValueError("lora_rank and lora_alpha must be both None or both not None")

    lora_config = None
    if lora_rank is not None and lora_alpha is not None:
        if "esmc" in checkpoint:
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=["layernorm_qkv.1", "out_proj"],
            )
        elif "esm" in checkpoint:
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=["query", "key", "value", "dense"],
            )
        elif "T5" in checkpoint or "t5" in checkpoint or "ankh" in checkpoint:
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=["q", "k", "v", "o"],
            )
        else:
            raise ValueError(f"Unsupported checkpoint for LoRA: {checkpoint}")

    if "esmc" in checkpoint:
        backbone, tokenizer = EsmcBackbone.from_pretrained(
            checkpoint=checkpoint,
            lora_config=lora_config,
        )
    elif "esm" in checkpoint:
        backbone, tokenizer = EsmBackbone.from_pretrained(
            checkpoint=checkpoint,
            lora_config=lora_config,
        )
    elif "T5" in checkpoint or "t5" in checkpoint or "ankh" in checkpoint:
        backbone, tokenizer = T5Backbone.from_pretrained(
            checkpoint=checkpoint,
            lora_config=lora_config,
        )
    else:
        raise ValueError(f"Unsupported checkpoint: {checkpoint}")

    return backbone, tokenizer


__all__ = ["build_backbone_and_tokenizer"]
