from dataclasses import fields
from typing import List, Optional, Tuple, cast

import torch
from peft import LoraConfig, inject_adapter_in_model
from transformers import AutoTokenizer, T5EncoderModel, T5Tokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from mutafitup.models.multitask_model import (
    MultitaskBackbone,
    MultitaskForwardArgs,
)


class T5Backbone(MultitaskBackbone):
    def __init__(self, model: T5EncoderModel):
        super().__init__()
        self.model = model
        self.hidden_size = model.config.d_model

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str,
        lora_config: Optional[LoraConfig] = None,
    ) -> Tuple["T5Backbone", PreTrainedTokenizerBase]:
        model: Optional[T5EncoderModel] = None
        tokenizer: Optional[PreTrainedTokenizerBase] = None

        if "ankh" in checkpoint:
            model = T5EncoderModel.from_pretrained(checkpoint)
            tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        elif "prot_t5" in checkpoint:
            model = T5EncoderModel.from_pretrained(checkpoint)
            tokenizer = T5Tokenizer.from_pretrained(checkpoint)
        elif "ProstT5" in checkpoint:
            model = T5EncoderModel.from_pretrained(checkpoint)
            tokenizer = T5Tokenizer.from_pretrained(checkpoint)
        if model is None or tokenizer is None:
            raise ValueError(f"Unsupported checkpoint {checkpoint}")

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

        raise ValueError(f"Unsupported model output type {type(outputs)}")

    def enable_gradient_checkpointing(self) -> None:
        # use_reentrant=False is required because:
        # 1. We inject LoRA via peft's inject_adapter_in_model() (not PEFT's
        #    higher-level API), so _hf_peft_config_loaded is False and HF
        #    won't automatically call enable_input_require_grads().
        # 2. With use_reentrant=True (HF's default), checkpoint() requires at
        #    least one input to have requires_grad=True, which isn't the case
        #    for frozen model inputs — gradients silently become None.
        # 3. use_reentrant=False handles non-leaf tensors and hooks correctly,
        #    consistent with the ESMC backbone's checkpointing approach.
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    def preprocess_sequences(self, sequences: List[str], checkpoint: str) -> List[str]:
        trans = str.maketrans({"O": "X", "B": "X", "U": "X", "Z": "X", "J": "X"})

        processed = [s.translate(trans) for s in sequences]

        if "Rostlab" in checkpoint:
            processed = [" ".join(s) for s in processed]
        if "ProstT5" in checkpoint:
            processed = [f"<AA2fold> {s}" for s in processed]

        return processed

    def _inject_lora_config(self, lora_config: Optional[LoraConfig]) -> None:
        if lora_config is None:
            return
        self.model = cast(
            T5EncoderModel, inject_adapter_in_model(lora_config, self.model)
        )
