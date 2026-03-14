from dataclasses import fields
from typing import List, Optional, Tuple, cast

import torch
from peft import LoraConfig, inject_adapter_in_model
from transformers import AutoTokenizer
from transformers.models.esm.modeling_esm import EsmModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from mutafitup.models.multitask_model import MultitaskBackbone, MultitaskForwardArgs


class EsmBackbone(MultitaskBackbone):
    def __init__(self, model: EsmModel):
        super().__init__()
        self.model = model
        self.hidden_size = model.config.hidden_size

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str,
        lora_config: Optional[LoraConfig] = None,
    ) -> Tuple["EsmBackbone", PreTrainedTokenizerBase]:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = EsmModel.from_pretrained(checkpoint)

        backbone = cls(model)
        backbone.apply_lora_config(lora_config)

        return backbone, tokenizer

    def forward(
        self,
        args: MultitaskForwardArgs,
    ) -> torch.Tensor:
        # Avoid dataclasses.asdict() which deep-copies tensor fields and
        # fails during ONNX tracing (FakeTensor doesn't support .data_ptr).
        kwargs = {
            f.name: getattr(args, f.name)
            for f in fields(args)
            if getattr(args, f.name) is not None
        }
        outputs = self.model(**kwargs)

        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state

        raise ValueError(f"Unsupported model output type: {type(outputs)}")

    def preprocess_sequences(self, sequences: List[str], checkpoint: str) -> List[dict]:
        return sequences

    def _inject_lora_config(self, lora_config: Optional[LoraConfig]) -> None:
        if lora_config is None:
            return
        self.model = cast(EsmModel, inject_adapter_in_model(lora_config, self.model))
