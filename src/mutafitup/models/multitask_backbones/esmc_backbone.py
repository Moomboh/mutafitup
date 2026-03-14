import types
from typing import List, Optional, Tuple, cast

import torch
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from esm.models.esmc import ESMC, ESMCOutput
from esm.tokenization import get_esmc_model_tokenizers
from peft import LoraConfig, inject_adapter_in_model
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from mutafitup.models.multitask_model import MultitaskBackbone, MultitaskForwardArgs


class EsmcBackbone(MultitaskBackbone):
    def __init__(self, model: ESMC):
        super().__init__()
        self.model = model
        self.hidden_size = model.embed.embedding_dim

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str,
        lora_config: Optional[LoraConfig] = None,
    ) -> Tuple["EsmcBackbone", PreTrainedTokenizerBase]:
        model = ESMC.from_pretrained(checkpoint)
        model = model.float()
        tokenizer = get_esmc_model_tokenizers()
        # The ESM-C tokenizer declares model_input_names as ["sequence_tokens", ...]
        # but __call__() returns "input_ids". Override to match the standard HF
        # convention so that tokenizer.pad() works with the rest of the pipeline.
        tokenizer.model_input_names = ["input_ids", "attention_mask"]

        backbone = cls(model)
        backbone.apply_lora_config(lora_config)

        return backbone, tokenizer

    def forward(
        self,
        args: MultitaskForwardArgs,
    ) -> torch.Tensor:
        output = self.model.forward(sequence_tokens=args.input_ids)
        return output.embeddings

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing for the transformer blocks.

        Monkey-patches ``TransformerStack.forward`` to wrap each transformer
        block in ``torch.utils.checkpoint.checkpoint(use_reentrant=False)``
        and skips collecting per-layer hidden states (which would otherwise
        keep all intermediate activations alive and defeat checkpointing).

        ``ESMC.forward`` is also patched so that it no longer pads / stacks
        the (now-empty) hidden-state list.
        """
        transformer = self.model.transformer

        # --- patch TransformerStack.forward -----------------------------------
        def _checkpointed_transformer_forward(
            self_transformer,
            x: torch.Tensor,
            sequence_id: torch.Tensor | None = None,
            affine=None,
            affine_mask=None,
            chain_id: torch.Tensor | None = None,
        ):
            *batch_dims, _ = x.shape
            if chain_id is None:
                chain_id = torch.ones(
                    size=batch_dims, dtype=torch.int64, device=x.device
                )
            for block in self_transformer.blocks:
                x = torch_checkpoint(
                    block,
                    x,
                    sequence_id,
                    affine,
                    affine_mask,
                    chain_id,
                    use_reentrant=False,
                )
            # Return empty hiddens list — EsmcBackbone.forward() discards
            # hidden_states anyway, and not collecting them is essential for
            # gradient checkpointing to actually save memory.
            return self_transformer.norm(x), x, []

        transformer.forward = types.MethodType(
            _checkpointed_transformer_forward, transformer
        )

        # --- patch ESMC.forward -----------------------------------------------
        # The original ESMC.forward pads each hidden state and then calls
        # torch.stack(hiddens), which would fail on an empty list.  Since
        # EsmcBackbone never uses hidden_states we simply skip that work.
        try:
            from flash_attn.bert_padding import pad_input, unpad_input  # type: ignore
        except ImportError:
            pad_input = None  # type: ignore[assignment]
            unpad_input = None  # type: ignore[assignment]

        def _esmc_forward_no_hiddens(
            self_esmc,
            sequence_tokens: torch.Tensor | None = None,
            sequence_id: torch.Tensor | None = None,
        ) -> ESMCOutput:
            if sequence_id is None:
                sequence_id = sequence_tokens != self_esmc.tokenizer.pad_token_id

            x = self_esmc.embed(sequence_tokens)
            B, L = x.shape[:2]

            if self_esmc._use_flash_attn:
                assert sequence_id.dtype == torch.bool, (
                    "sequence_id must be a boolean mask if Flash Attention is used"
                )
                assert sequence_id.shape == (B, L)
                assert unpad_input is not None
                x, indices, *_ = unpad_input(x, sequence_id)  # type: ignore[misc]
            else:
                indices = None

            x, _, _hiddens = self_esmc.transformer(x, sequence_id=sequence_id)

            if self_esmc._use_flash_attn:
                assert indices is not None
                assert pad_input is not None
                x = pad_input(x, indices, B, L)
                # Skip hiddens padding — the list is empty.

            sequence_logits = self_esmc.sequence_head(x)
            return ESMCOutput(
                sequence_logits=sequence_logits,
                embeddings=x,
                hidden_states=None,
            )

        self.model.forward = types.MethodType(  # type: ignore[method-assign]
            _esmc_forward_no_hiddens, self.model
        )

    def preprocess_sequences(self, sequences: List[str], checkpoint: str) -> List[str]:
        return sequences

    def _inject_lora_config(self, lora_config: Optional[LoraConfig]) -> None:
        if lora_config is None:
            return
        self.model = cast(ESMC, inject_adapter_in_model(lora_config, self.model))
