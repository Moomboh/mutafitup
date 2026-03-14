import pytest
import torch
from torch import nn

from mutafitup.models.multitask_heads import (
    PerProteinClassificationHead,
    PerResidueRegressionHead,
)
from mutafitup.models.multitask_model import (
    HeadConfig,
    MultitaskBackbone,
    MultitaskForwardArgs,
    MultitaskModel,
)


class DummyBackbone(MultitaskBackbone):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(hidden_size, hidden_size)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str,
        lora_config=None,
    ):
        backbone = cls(hidden_size=8)
        backbone.apply_lora_config(lora_config)
        return backbone, None

    def _inject_lora_config(self, lora_config=None) -> None:
        self._injected = lora_config

    def forward(self, args: MultitaskForwardArgs) -> torch.Tensor:
        return self.linear(args.input_ids.float())

    def preprocess_sequences(self, sequences, checkpoint):
        return sequences


def test_apply_lora_config_calls_inject_and_sets_property():
    backbone = DummyBackbone(hidden_size=4)
    marker = object()

    backbone.apply_lora_config(marker)

    assert backbone.lora_config is marker
    assert backbone._injected is marker


def test_multitask_model_forward_and_unknown_task():
    backbone = DummyBackbone(hidden_size=8)
    heads = {
        "task": HeadConfig(
            head=PerProteinClassificationHead(0.0, 8, 8, 2),
            problem_type="classification",
            level="per_protein",
        )
    }
    model = MultitaskModel(backbone=backbone, heads=heads)

    args = MultitaskForwardArgs(
        attention_mask=torch.ones(1, 3, dtype=torch.long),
        input_ids=torch.ones(1, 3, 8, dtype=torch.long),
    )
    labels = torch.zeros(1, dtype=torch.long)

    loss, logits = model.forward("task", args, labels)

    assert loss is not None
    assert loss.shape == ()
    assert logits.shape == (1, 2)

    with pytest.raises(ValueError):
        model.forward("other", args, labels)


def test_multitask_model_forward_from_embeddings():
    backbone = DummyBackbone(hidden_size=8)
    heads = {
        "cls_task": HeadConfig(
            head=PerProteinClassificationHead(0.0, 8, 8, 3),
            problem_type="classification",
            level="per_protein",
        ),
        "reg_task": HeadConfig(
            head=PerResidueRegressionHead(0.0, 8, 8),
            problem_type="regression",
            level="per_residue",
        ),
    }
    model = MultitaskModel(backbone=backbone, heads=heads)

    embeddings = torch.randn(2, 5, 8)
    attention_mask = torch.ones(2, 5, dtype=torch.long)
    attention_mask[1, 3:] = 0

    cls_labels = torch.tensor([0, 2])
    loss, logits = model.forward_from_embeddings(
        task="cls_task",
        embeddings=embeddings,
        attention_mask=attention_mask,
        labels=cls_labels,
    )

    assert loss is not None
    assert loss.shape == ()
    assert logits.shape == (2, 3)

    reg_labels = torch.randn(2, 5, 1)
    loss, logits = model.forward_from_embeddings(
        task="reg_task",
        embeddings=embeddings,
        attention_mask=attention_mask,
        labels=reg_labels,
    )

    assert loss is not None
    assert logits.shape == (2, 5, 1)

    with pytest.raises(ValueError):
        model.forward_from_embeddings(
            task="unknown",
            embeddings=embeddings,
            attention_mask=attention_mask,
        )


def test_classification_loss_per_residue_masks_attention_and_ignore_index():
    backbone = DummyBackbone(hidden_size=2)
    model = MultitaskModel(backbone=backbone, heads={})

    config = HeadConfig(
        head=nn.Linear(1, 1),
        problem_type="classification",
        level="per_residue",
        ignore_index=-1,
    )

    logits = torch.zeros(1, 2, 2)
    labels = torch.tensor([[0, 1]])
    attention_mask = torch.tensor([[1, 0]])

    loss = model._classification_loss(config, logits, labels, attention_mask)

    assert loss.shape == ()
    assert loss.item() >= 0


def test_classification_loss_per_residue_all_masked_returns_zero():
    backbone = DummyBackbone(hidden_size=2)
    model = MultitaskModel(backbone=backbone, heads={})

    config = HeadConfig(
        head=nn.Linear(1, 1),
        problem_type="classification",
        level="per_residue",
        ignore_index=-1,
    )

    logits = torch.zeros(1, 2, 2)
    labels = torch.tensor([[0, 1]])
    attention_mask = torch.zeros(1, 2, dtype=torch.long)

    loss = model._classification_loss(config, logits, labels, attention_mask)

    assert loss.item() == 0.0


def test_regression_loss_per_residue_and_per_protein_masking():
    backbone = DummyBackbone(hidden_size=2)
    model = MultitaskModel(backbone=backbone, heads={})

    per_res_config = HeadConfig(
        head=nn.Linear(1, 1),
        problem_type="regression",
        level="per_residue",
        ignore_index=-1,
    )

    logits_res = torch.tensor([[[1.0], [2.0]]])
    labels_res = torch.tensor([[[0.0], [-1.0]]])
    attention_mask = torch.tensor([[1, 1]])

    loss_res = model._regression_loss(
        per_res_config,
        logits_res,
        labels_res,
        attention_mask,
    )

    assert loss_res.item() > 0

    per_protein_config = HeadConfig(
        head=nn.Linear(1, 1),
        problem_type="regression",
        level="per_protein",
        ignore_index=-1,
    )

    logits_prot = torch.tensor([[1.0], [2.0]])
    labels_prot = torch.tensor([[1.0], [-1.0]])
    mask = torch.tensor([[1], [1]])

    loss_prot = model._regression_loss(
        per_protein_config,
        logits_prot,
        labels_prot,
        mask,
    )

    assert loss_prot.item() >= 0
