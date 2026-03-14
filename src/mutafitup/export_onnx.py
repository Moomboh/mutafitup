"""Export a trained MultitaskModel to ONNX format.

Loads a trained model checkpoint, merges LoRA adapter weights (if any)
into the base model, wraps the backbone + all heads into a single
``nn.Module``, and exports it via ``torch.onnx.export`` with dynamic
batch and sequence-length axes.

The export directory contains:

* ``model.onnx`` -- the ONNX graph with all weights embedded in a
  single file (inputs: ``input_ids``, ``attention_mask``; outputs:
  ``{task}_logits`` per head).  External data files produced by
  ``torch.onnx.export`` are merged back automatically so that
  onnxruntime-web can load the model from a URL.
* ``tokenizer/`` -- HuggingFace tokenizer saved via ``save_pretrained``
* ``export_metadata.json`` -- task schema, preprocessing instructions,
  and named input/output descriptions for the JS consumer
"""

import json
import logging
import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnx
import torch
from torch import nn
from transformers import PreTrainedTokenizerBase

from mutafitup.models.multitask_model import (
    MultitaskForwardArgs,
    MultitaskModel,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LoRA merge helpers
# ---------------------------------------------------------------------------


def _merge_lora_weights(model: MultitaskModel) -> None:
    """Merge LoRA adapter weights into the base layers in-place.

    After merging, each peft ``LoraLayer`` wrapper is replaced by its
    underlying ``nn.Linear`` so the exported graph contains only standard
    operations.
    """
    try:
        from peft.tuners.lora import LoraLayer
    except ImportError:
        logger.info("peft not installed -- skipping LoRA merge")
        return

    # First pass: merge weights in-place
    merged_count = 0
    for module in model.modules():
        if isinstance(module, LoraLayer):
            module.merge()
            merged_count += 1

    if merged_count == 0:
        return

    # Second pass: replace LoRA wrappers with plain base layers
    _replace_lora_wrappers(model)
    logger.info("Merged and unwrapped %d LoRA layers", merged_count)


def _replace_lora_wrappers(module: nn.Module) -> None:
    """Recursively replace ``LoraLayer`` wrappers with their base layers."""
    from peft.tuners.lora import LoraLayer

    for name, child in list(module.named_children()):
        if isinstance(child, LoraLayer) and hasattr(child, "get_base_layer"):
            setattr(module, name, child.get_base_layer())
        else:
            _replace_lora_wrappers(child)


# ---------------------------------------------------------------------------
# ONNX export wrapper
# ---------------------------------------------------------------------------


class _OnnxMultitaskWrapper(nn.Module):
    """Thin wrapper that presents backbone + all heads as a single module.

    Accepts ``(input_ids, attention_mask)`` and returns a tuple of logit
    tensors, one per task in a fixed, deterministic order.
    """

    def __init__(
        self,
        backbone: nn.Module,
        heads: nn.ModuleDict,
        task_order: List[str],
        head_levels: Dict[str, str],
    ):
        super().__init__()
        self.backbone = backbone
        self.heads = heads
        self.task_order = task_order
        self.head_levels = head_levels

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        args = MultitaskForwardArgs(
            attention_mask=attention_mask,
            input_ids=input_ids,
        )
        hidden_states = self.backbone(args)

        outputs = []
        for name in self.task_order:
            outputs.append(self.heads[name](hidden_states))
        return tuple(outputs)


# ---------------------------------------------------------------------------
# Slow → fast tokenizer conversion
# ---------------------------------------------------------------------------


def _build_fast_tokenizer(
    slow_tokenizer: "PreTrainedTokenizerBase",
) -> "tokenizers.Tokenizer":
    """Build a ``tokenizers`` (Rust) fast tokenizer from a HuggingFace slow tokenizer.

    ``transformers.convert_slow_tokenizer`` doesn't support ``EsmTokenizer``
    and fails on ProtT5's SentencePiece model, so we build the fast tokenizer
    directly from the vocabulary.

    Two modes are detected automatically:

    - **Metaspace** (SentencePiece-style): vocabulary contains ``▁``-prefixed
      tokens (e.g. ProtT5). Input is expected to be space-separated.
    - **Character-level**: no ``▁`` prefix (e.g. ESM-2). Each character is
      tokenized independently.
    """
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors

    vocab = slow_tokenizer.get_vocab()

    # Detect whether the vocabulary uses SentencePiece ▁ prefixing
    regular_tokens = [
        t for t in vocab if not t.startswith("<") and not t.startswith("▁")
    ]
    sp_tokens = [t for t in vocab if t.startswith("▁")]
    uses_metaspace = len(sp_tokens) > len(regular_tokens)

    fast = Tokenizer(models.WordLevel(vocab=vocab, unk_token=slow_tokenizer.unk_token))

    if uses_metaspace:
        fast.pre_tokenizer = pre_tokenizers.Metaspace()
        fast.decoder = decoders.Metaspace()
    else:
        # Character-level: split every character into its own token
        fast.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")

    # Build post-processor from special tokens
    eos_token = slow_tokenizer.eos_token
    eos_id = slow_tokenizer.eos_token_id
    cls_token = getattr(slow_tokenizer, "cls_token", None)
    cls_id = getattr(slow_tokenizer, "cls_token_id", None)

    if cls_token is not None and cls_id is not None:
        # CLS ... EOS (e.g. ESM-2)
        fast.post_processor = processors.TemplateProcessing(
            single=f"{cls_token} $A {eos_token}",
            pair=f"{cls_token} $A {eos_token} $B:1 {eos_token}:1",
            special_tokens=[(cls_token, cls_id), (eos_token, eos_id)],
        )
    elif eos_token is not None and eos_id is not None:
        # ... EOS (e.g. T5)
        fast.post_processor = processors.TemplateProcessing(
            single=f"$A {eos_token}",
            pair=f"$A {eos_token} $B:1 {eos_token}:1",
            special_tokens=[(eos_token, eos_id)],
        )

    return fast


# ---------------------------------------------------------------------------
# Preprocessing metadata
# ---------------------------------------------------------------------------


def _build_preprocessing_meta(
    backbone_class: str,
    base_checkpoint: str,
) -> Dict[str, Any]:
    """Build preprocessing instructions for the JS tokenizer consumer.

    The returned dict describes string transformations that must be applied
    to a raw protein sequence *before* tokenization, mirroring each
    backbone's ``preprocess_sequences()`` logic.
    """
    meta: Dict[str, Any] = {
        "space_separate": False,
        "prefix": None,
        "char_replacements": {},
    }

    if "T5Backbone" in backbone_class:
        # T5-family models replace rare amino acids
        meta["char_replacements"] = {
            "O": "X",
            "B": "X",
            "U": "X",
            "Z": "X",
            "J": "X",
        }
        if "Rostlab" in base_checkpoint or "prot_t5" in base_checkpoint:
            meta["space_separate"] = True
        if "ProstT5" in base_checkpoint:
            meta["space_separate"] = True
            meta["prefix"] = "<AA2fold>"

    return meta


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def export_to_onnx(
    model_path: str,
    output_dir: str,
    tokenizer: PreTrainedTokenizerBase,
    base_checkpoint: str,
    opset_version: int = 18,
    validate: bool = True,
    logger: Optional[logging.Logger] = None,
) -> str:
    """Export a trained MultitaskModel to ONNX.

    Parameters
    ----------
    model_path
        Path to the ``model.pt`` file (any variant: final, best_overall, etc.)
    output_dir
        Directory where ``model.onnx``, ``tokenizer/``, and
        ``export_metadata.json`` will be written.
    tokenizer
        The HuggingFace tokenizer matching the model's backbone.
    base_checkpoint
        HuggingFace checkpoint identifier (e.g. ``"facebook/esm2_t6_8M_UR50D"``).
    opset_version
        ONNX opset version (default 14).
    validate
        If True, load the exported model with onnxruntime and verify outputs
        match PyTorch for a dummy input.
    logger
        Logger for progress messages.

    Returns
    -------
    str
        Path to the exported ``model.onnx`` file.
    """
    _log = logger or logging.getLogger(__name__)

    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    _log.info("Loading model from %s", model_path)
    model = MultitaskModel.load_from_file(model_path, device=torch.device("cpu"))
    model.eval()

    # ------------------------------------------------------------------
    # 2. Merge LoRA weights (if any)
    # ------------------------------------------------------------------
    if model.backbone.lora_config is not None:
        _log.info("Merging LoRA adapter weights into base model")
        _merge_lora_weights(model)
    else:
        _log.info("No LoRA config -- skipping merge")

    # ------------------------------------------------------------------
    # 3. Determine task order and build wrapper
    # ------------------------------------------------------------------
    task_order = sorted(model.head_configs.keys())
    head_levels = {name: cfg.level for name, cfg in model.head_configs.items()}

    wrapper = _OnnxMultitaskWrapper(
        backbone=model.backbone,
        heads=model.heads,
        task_order=task_order,
        head_levels=head_levels,
    )
    wrapper.eval()

    # ------------------------------------------------------------------
    # 4. Create dummy inputs
    # ------------------------------------------------------------------
    dummy_seq_len = 10
    dummy_batch = 2
    dummy_input_ids = torch.ones(dummy_batch, dummy_seq_len, dtype=torch.long)
    dummy_attention_mask = torch.ones(dummy_batch, dummy_seq_len, dtype=torch.long)

    # ------------------------------------------------------------------
    # 5. Build dynamic axes and output names
    # ------------------------------------------------------------------
    input_names = ["input_ids", "attention_mask"]
    output_names = [f"{name}_logits" for name in task_order]

    dynamic_axes: Dict[str, Dict[int, str]] = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
    }

    for name in task_order:
        level = head_levels[name]
        out_name = f"{name}_logits"
        if level == "per_residue":
            dynamic_axes[out_name] = {0: "batch_size", 1: "sequence_length"}
        else:  # per_protein — mean pooling removes sequence dim
            dynamic_axes[out_name] = {0: "batch_size"}

    # ------------------------------------------------------------------
    # 6. Export to ONNX
    # ------------------------------------------------------------------
    onnx_path = os.path.join(output_dir, "model.onnx")
    _log.info("Exporting ONNX to %s (opset %d)", onnx_path, opset_version)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_input_ids, dummy_attention_mask),
            onnx_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
        )

    _log.info("ONNX model written to %s", onnx_path)

    # ------------------------------------------------------------------
    # 6b. Merge external data into single file
    # ------------------------------------------------------------------
    # torch.onnx.export auto-splits weights into a separate .data file
    # when the model exceeds the ~2 GB protobuf limit. ort-web
    # (onnxruntime-web) does not support external data files, so we
    # merge everything back into a single .onnx file.
    external_data_path = onnx_path + ".data"
    if os.path.exists(external_data_path):
        external_size = os.path.getsize(external_data_path)
        # protobuf has a hard ~2 GB serialization limit
        PROTOBUF_LIMIT = 2 * 1024 * 1024 * 1024

        if external_size < PROTOBUF_LIMIT:
            _log.info("Merging external data back into %s", onnx_path)
            onnx_model = onnx.load(onnx_path, load_external_data=True)
            onnx.save_model(onnx_model, onnx_path, save_as_external_data=False)
            os.remove(external_data_path)
            _log.info("Removed %s", external_data_path)
        else:
            _log.warning(
                "External data is %.1f GB (exceeds 2 GB protobuf limit), "
                "keeping external data file. This model will not work with ort-web.",
                external_size / (1024**3),
            )

    # ------------------------------------------------------------------
    # 7. Save tokenizer
    # ------------------------------------------------------------------
    # The Rust tokenizers crate (used by the Dioxus app) only reads the
    # fast-tokenizer format (tokenizer.json). Slow tokenizers like
    # BertTokenizer (ESM-2) and T5Tokenizer (ProtT5) don't produce this
    # file via save_pretrained(), so we convert them explicitly.
    tokenizer_dir = os.path.join(output_dir, "tokenizer")
    _log.info("Saving tokenizer to %s", tokenizer_dir)
    tokenizer.save_pretrained(tokenizer_dir)

    fast_tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
    if not os.path.exists(fast_tokenizer_path):
        _log.info(
            "No tokenizer.json found (slow tokenizer); "
            "building fast tokenizer from vocabulary"
        )
        fast_tokenizer = _build_fast_tokenizer(tokenizer)
        fast_tokenizer.save(fast_tokenizer_path)
        _log.info("Saved fast tokenizer to %s", fast_tokenizer_path)

    # ------------------------------------------------------------------
    # 8. Write export metadata
    # ------------------------------------------------------------------
    data = torch.load(model_path, map_location="cpu", weights_only=False)
    model_config = data["config"]
    backbone_class = model_config["backbone_class"]

    tasks_meta: Dict[str, Dict[str, Any]] = {}
    for name in task_order:
        cfg = model.head_configs[name]
        num_outputs = MultitaskModel._infer_head_num_outputs(cfg.head)
        tasks_meta[name] = {
            "problem_type": cfg.problem_type,
            "level": cfg.level,
            "num_outputs": num_outputs,
            "output_name": f"{name}_logits",
        }

    # Determine the number of leading special tokens (e.g. CLS) that
    # precede the amino acid positions in the tokenizer output.
    # ESM/ESM-C: CLS $A EOS → cls_token_offset = 1
    # T5:        $A EOS     → cls_token_offset = 0
    cls_token = getattr(tokenizer, "cls_token", None)
    cls_id = getattr(tokenizer, "cls_token_id", None)
    cls_token_offset = 1 if (cls_token is not None and cls_id is not None) else 0

    metadata: Dict[str, Any] = {
        "base_checkpoint": base_checkpoint,
        "backbone_class": backbone_class,
        "preprocessing": _build_preprocessing_meta(backbone_class, base_checkpoint),
        "cls_token_offset": cls_token_offset,
        "tasks": tasks_meta,
        "inputs": input_names,
        "outputs": output_names,
        "opset_version": opset_version,
    }

    # Add license metadata for ESM-C backbone derivatives (Cambrian Open License).
    if "esmc" in base_checkpoint.lower() or "esmc" in backbone_class.lower():
        metadata["license"] = "cambrian-open"
        metadata["license_url"] = (
            "https://www.evolutionaryscale.ai/policies/cambrian-open-license-agreement"
        )
        metadata["license_notice"] = (
            "The ESMC 300M Model is licensed under the "
            "EvolutionaryScale Cambrian Open License Agreement."
        )
        metadata["attribution"] = "Built with ESM"

    meta_path = os.path.join(output_dir, "export_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    _log.info("Metadata written to %s", meta_path)

    # ------------------------------------------------------------------
    # 9. Validate with onnxruntime (optional)
    # ------------------------------------------------------------------
    if validate:
        _validate_onnx_export(
            onnx_path=onnx_path,
            wrapper=wrapper,
            dummy_input_ids=dummy_input_ids,
            dummy_attention_mask=dummy_attention_mask,
            output_names=output_names,
            logger=_log,
        )

    return onnx_path


def _validate_onnx_export(
    onnx_path: str,
    wrapper: nn.Module,
    dummy_input_ids: torch.Tensor,
    dummy_attention_mask: torch.Tensor,
    output_names: List[str],
    logger: logging.Logger,
) -> None:
    """Load the ONNX model with onnxruntime and compare outputs to PyTorch."""
    try:
        import onnxruntime as ort
    except ImportError:
        logger.warning("onnxruntime not installed -- skipping ONNX validation")
        return

    logger.info("Validating ONNX export with onnxruntime")

    session = ort.InferenceSession(onnx_path)

    ort_inputs = {
        "input_ids": dummy_input_ids.numpy().astype(np.int64),
        "attention_mask": dummy_attention_mask.numpy().astype(np.int64),
    }
    ort_outputs = session.run(output_names, ort_inputs)

    with torch.no_grad():
        pt_outputs = wrapper(dummy_input_ids, dummy_attention_mask)

    for i, name in enumerate(output_names):
        pt_np = pt_outputs[i].numpy()
        ort_np = ort_outputs[i]
        if not np.allclose(pt_np, ort_np, atol=1e-5, rtol=1e-4):
            max_diff = float(np.max(np.abs(pt_np - ort_np)))
            logger.warning("Validation mismatch for %s: max_diff=%.6f", name, max_diff)
        else:
            logger.info("Validation OK for %s", name)

    logger.info("ONNX validation complete")
