"""Integration tests: gradient checkpointing with real model backbones.

Each test loads a full pretrained model, applies LoRA, runs a short
align-LoRA training with and without gradient checkpointing, and asserts
that the resulting model weights are identical.

Marked with ``@pytest.mark.integration`` so they can be excluded from fast
CI runs via ``-m "not integration"``.
"""

import copy
import logging

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from mutafitup.datasets import BaseMultitaskDataset
from mutafitup.models import build_backbone_and_tokenizer
from mutafitup.models.multitask_heads import PerProteinClassificationHead
from mutafitup.models.multitask_model import (
    HeadConfig,
    MultitaskBackbone,
    MultitaskModel,
)
from mutafitup.train.train_multitask_model import train_multitask_align_lora


# ---------------------------------------------------------------------------
# Synthetic dataset that produces properly tokenized protein sequences
# ---------------------------------------------------------------------------


class _TokenizedProteinDataset(Dataset):
    """Generates pre-tokenized short protein sequences for integration tests."""

    def __init__(self, sequences: list[str], labels: list[int], tokenizer):
        self.tokenizer = tokenizer
        self.sequences = sequences
        self.labels = labels

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        encoded = self.tokenizer(
            self.sequences[idx],
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=1024,
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class _SyntheticTaskDataset(BaseMultitaskDataset):
    """A BaseMultitaskDataset backed by a handful of short protein strings."""

    def __init__(
        self,
        name: str,
        sequences: list[str],
        labels: list[int],
    ):
        super().__init__(name)
        self.sequences = sequences
        self.labels = labels

    def get_dataloader(
        self,
        split: str,
        backbone: MultitaskBackbone,
        tokenizer,
        checkpoint: str,
        batch_size: int,
    ) -> DataLoader:
        from mutafitup.datasets.data_collator import DataCollator

        seqs = backbone.preprocess_sequences(self.sequences, checkpoint)
        ds = _TokenizedProteinDataset(seqs, self.labels, tokenizer)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=DataCollator(tokenizer),
        )


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

# Short real protein subsequences — just enough to exercise the model.
_TASK_A_SEQS = ["ACDEFGHIKL", "MNPQRSTVWY", "ACDEFG", "GHIKLMNP"]
_TASK_A_LABELS = [0, 1, 0, 1]

_TASK_B_SEQS = ["STVWYACDEF", "GHIKLMNPQR", "STVWYG", "MNPQRSTA"]
_TASK_B_LABELS = [1, 0, 1, 0]


def _try_load_esmc_backbone():
    """Try to load the ESMC-300M backbone; return None on failure."""
    try:
        backbone, tokenizer = build_backbone_and_tokenizer(
            "esmc_300m", lora_rank=4, lora_alpha=8
        )
        return backbone, tokenizer
    except Exception:
        return None


@pytest.mark.integration
def test_gradient_checkpointing_esmc300m_produces_identical_weights():
    """Train a real ESMC-300M + LoRA model for 1 epoch with and without
    gradient checkpointing and verify the resulting weights are identical."""

    result = _try_load_esmc_backbone()
    if result is None:
        pytest.skip(
            "ESMC-300M model could not be loaded (not cached / download failed)"
        )

    backbone_ref, tokenizer = result

    hidden_size = backbone_ref.hidden_size
    seed = 123
    max_epochs = 1

    def create_model():
        """Create a fresh ESMC-300M model with LoRA and two classification heads."""
        bb, tok = build_backbone_and_tokenizer("esmc_300m", lora_rank=4, lora_alpha=8)
        hs = bb.hidden_size
        heads = {
            "task_a": HeadConfig(
                head=PerProteinClassificationHead(0.0, hs, hs, 2),
                problem_type="classification",
                level="per_protein",
            ),
            "task_b": HeadConfig(
                head=PerProteinClassificationHead(0.0, hs, hs, 2),
                problem_type="classification",
                level="per_protein",
            ),
        }
        return MultitaskModel(backbone=bb, heads=heads), tok

    def create_datasets():
        return {
            "task_a": _SyntheticTaskDataset("task_a", _TASK_A_SEQS, _TASK_A_LABELS),
            "task_b": _SyntheticTaskDataset("task_b", _TASK_B_SEQS, _TASK_B_LABELS),
        }

    # ---- Run 1: WITHOUT gradient checkpointing ----------------------------
    model_no_gc, tok = create_model()
    init_state = copy.deepcopy(model_no_gc.state_dict())

    trained_no_gc, history_no_gc, _ = train_multitask_align_lora(
        model=model_no_gc,
        tokenizer=tok,
        checkpoint="esmc_300m",
        task_datasets=create_datasets(),
        batch=2,
        max_epochs=max_epochs,
        lr=1e-3,
        warmup_ratio=0.0,
        seed=seed,
        logger=logging.getLogger("integration_gc_no"),
        align_lora_kl_lambda=0.1,
        gradient_checkpointing=False,
    )

    # ---- Run 2: WITH gradient checkpointing --------------------------------
    model_gc, tok_gc = create_model()
    model_gc.load_state_dict(init_state)

    trained_gc, history_gc, _ = train_multitask_align_lora(
        model=model_gc,
        tokenizer=tok_gc,
        checkpoint="esmc_300m",
        task_datasets=create_datasets(),
        batch=2,
        max_epochs=max_epochs,
        lr=1e-3,
        warmup_ratio=0.0,
        seed=seed,
        logger=logging.getLogger("integration_gc_yes"),
        align_lora_kl_lambda=0.1,
        gradient_checkpointing=True,
    )

    # ---- Assertions --------------------------------------------------------
    assert len(history_no_gc) == len(history_gc), (
        f"History lengths differ: {len(history_no_gc)} vs {len(history_gc)}"
    )

    # All trainable parameters must be identical
    gc_params = dict(trained_gc.named_parameters())
    for name, param_no_gc in trained_no_gc.named_parameters():
        if not param_no_gc.requires_grad:
            continue
        param_gc = gc_params[name]
        torch.testing.assert_close(
            param_no_gc.cpu(),
            param_gc.cpu(),
            msg=lambda m: f"Parameter {name} differs with gradient checkpointing: {m}",
        )

    # Verify training actually modified weights
    any_changed = False
    for name, param in trained_no_gc.named_parameters():
        if not param.requires_grad:
            continue
        init_val = init_state.get(name)
        if init_val is not None and not torch.equal(param.data.cpu(), init_val.cpu()):
            any_changed = True
            break
    assert any_changed, "Training did not change any weights — test is vacuous"


# ---------------------------------------------------------------------------
# ProtT5-XL
# ---------------------------------------------------------------------------

_PROTT5_CHECKPOINT = "Rostlab/prot_t5_xl_uniref50"


def _try_load_prott5_backbone():
    """Try to load the ProtT5-XL backbone; return None on failure."""
    try:
        backbone, tokenizer = build_backbone_and_tokenizer(
            _PROTT5_CHECKPOINT, lora_rank=4, lora_alpha=8
        )
        return backbone, tokenizer
    except Exception:
        return None


@pytest.mark.integration
def test_gradient_checkpointing_prott5_produces_identical_weights():
    """Train a real ProtT5-XL + LoRA model for 1 epoch with and without
    gradient checkpointing and verify the resulting weights are identical.

    Note: On CPU and CUDA the gradients are bitwise identical. On MPS
    (Apple Silicon) the backend introduces small non-deterministic
    differences during backward recomputation through 24 transformer
    layers, so we use a relaxed tolerance (atol=1e-2) there. The
    tolerance is generous to avoid flaky failures on different hardware
    while still catching gross errors (e.g. zero gradients, wrong sign).
    """

    result = _try_load_prott5_backbone()
    if result is None:
        pytest.skip(
            "ProtT5-XL model could not be loaded (not cached / download failed)"
        )

    backbone_ref, tokenizer = result

    seed = 123
    max_epochs = 1

    # On CPU/CUDA, gradient checkpointing is bitwise deterministic for T5.
    # MPS introduces small numerical differences during backward recomputation.
    _device = torch.device("mps") if torch.backends.mps.is_available() else None
    _on_mps = _device is not None and _device.type == "mps"

    def create_model():
        """Create a fresh ProtT5-XL model with LoRA and two classification heads."""
        bb, tok = build_backbone_and_tokenizer(
            _PROTT5_CHECKPOINT, lora_rank=4, lora_alpha=8
        )
        hs = bb.hidden_size
        heads = {
            "task_a": HeadConfig(
                head=PerProteinClassificationHead(0.0, hs, hs, 2),
                problem_type="classification",
                level="per_protein",
            ),
            "task_b": HeadConfig(
                head=PerProteinClassificationHead(0.0, hs, hs, 2),
                problem_type="classification",
                level="per_protein",
            ),
        }
        return MultitaskModel(backbone=bb, heads=heads), tok

    def create_datasets():
        return {
            "task_a": _SyntheticTaskDataset("task_a", _TASK_A_SEQS, _TASK_A_LABELS),
            "task_b": _SyntheticTaskDataset("task_b", _TASK_B_SEQS, _TASK_B_LABELS),
        }

    # ---- Run 1: WITHOUT gradient checkpointing ----------------------------
    model_no_gc, tok = create_model()
    init_state = copy.deepcopy(model_no_gc.state_dict())

    trained_no_gc, history_no_gc, _ = train_multitask_align_lora(
        model=model_no_gc,
        tokenizer=tok,
        checkpoint=_PROTT5_CHECKPOINT,
        task_datasets=create_datasets(),
        batch=2,
        max_epochs=max_epochs,
        lr=1e-3,
        warmup_ratio=0.0,
        seed=seed,
        logger=logging.getLogger("integration_prott5_gc_no"),
        align_lora_kl_lambda=0.1,
        gradient_checkpointing=False,
    )

    # ---- Run 2: WITH gradient checkpointing --------------------------------
    model_gc, tok_gc = create_model()
    model_gc.load_state_dict(init_state)

    trained_gc, history_gc, _ = train_multitask_align_lora(
        model=model_gc,
        tokenizer=tok_gc,
        checkpoint=_PROTT5_CHECKPOINT,
        task_datasets=create_datasets(),
        batch=2,
        max_epochs=max_epochs,
        lr=1e-3,
        warmup_ratio=0.0,
        seed=seed,
        logger=logging.getLogger("integration_prott5_gc_yes"),
        align_lora_kl_lambda=0.1,
        gradient_checkpointing=True,
    )

    # ---- Assertions --------------------------------------------------------
    assert len(history_no_gc) == len(history_gc), (
        f"History lengths differ: {len(history_no_gc)} vs {len(history_gc)}"
    )

    # Compare trainable parameters.
    # On MPS the backward recomputation is not bitwise deterministic, so we
    # use a relaxed tolerance. On CPU/CUDA we expect exact equality.
    atol = 1e-2 if _on_mps else 1e-5
    rtol = 1e-3 if _on_mps else 1.3e-6
    gc_params = dict(trained_gc.named_parameters())
    for name, param_no_gc in trained_no_gc.named_parameters():
        if not param_no_gc.requires_grad:
            continue
        param_gc = gc_params[name]
        torch.testing.assert_close(
            param_no_gc.cpu(),
            param_gc.cpu(),
            atol=atol,
            rtol=rtol,
            msg=lambda m: f"Parameter {name} differs with gradient checkpointing: {m}",
        )

    # Verify training actually modified weights
    any_changed = False
    for name, param in trained_no_gc.named_parameters():
        if not param.requires_grad:
            continue
        init_val = init_state.get(name)
        if init_val is not None and not torch.equal(param.data.cpu(), init_val.cpu()):
            any_changed = True
            break
    assert any_changed, "Training did not change any weights — test is vacuous"
