import types

import torch
import torch.nn as nn

from mutafitup.models.multitask_backbones import esmc_backbone as module
from mutafitup.models.multitask_backbones.esmc_backbone import EsmcBackbone
from mutafitup.models.multitask_model import MultitaskForwardArgs


class DummyESMCOutput:
    def __init__(self, embeddings: torch.Tensor):
        self.sequence_logits = torch.zeros_like(embeddings[:, :, :64])
        self.embeddings = embeddings
        self.hidden_states = None


class DummyESMC(nn.Module):
    def __init__(self, d_model=960):
        super().__init__()
        self.embed = nn.Embedding(64, d_model)

    @classmethod
    def from_pretrained(cls, checkpoint):
        return cls()

    def forward(self, sequence_tokens=None, sequence_id=None):
        x = self.embed(sequence_tokens)
        return DummyESMCOutput(x)


class DummyTokenizer:
    pass


def test_esmc_backbone_from_pretrained_and_forward(monkeypatch):
    monkeypatch.setattr(module, "ESMC", DummyESMC)

    def fake_get_tokenizers():
        return DummyTokenizer()

    monkeypatch.setattr(module, "get_esmc_model_tokenizers", fake_get_tokenizers)

    injected = {}

    def fake_inject(lora_config, model):
        injected["lora_config"] = lora_config
        injected["model"] = model
        return model

    monkeypatch.setattr(module, "inject_adapter_in_model", fake_inject)

    marker = object()
    backbone, tokenizer = EsmcBackbone.from_pretrained(
        "esmc_300m",
        lora_config=marker,
    )

    assert isinstance(tokenizer, DummyTokenizer)
    assert tokenizer.model_input_names == ["input_ids", "attention_mask"]
    assert isinstance(backbone.model, DummyESMC)
    assert backbone.hidden_size == 960
    assert backbone.lora_config is marker
    assert injected["lora_config"] is marker
    assert injected["model"] is backbone.model

    args = MultitaskForwardArgs(
        attention_mask=torch.ones(1, 3, dtype=torch.long),
        input_ids=torch.ones(1, 3, dtype=torch.long),
    )

    hidden = backbone.forward(args)

    assert hidden.shape == (1, 3, 960)


def test_esmc_backbone_from_pretrained_without_lora(monkeypatch):
    monkeypatch.setattr(module, "ESMC", DummyESMC)

    def fake_get_tokenizers():
        return DummyTokenizer()

    monkeypatch.setattr(module, "get_esmc_model_tokenizers", fake_get_tokenizers)

    backbone, tokenizer = EsmcBackbone.from_pretrained(
        "esmc_300m",
        lora_config=None,
    )

    assert isinstance(tokenizer, DummyTokenizer)
    assert tokenizer.model_input_names == ["input_ids", "attention_mask"]
    assert isinstance(backbone.model, DummyESMC)
    assert backbone.hidden_size == 960
    assert backbone.lora_config is None


def test_esmc_backbone_preprocess_sequences_identity():
    model = DummyESMC()
    backbone = EsmcBackbone(model)

    sequences = ["ACD", "XYZ"]
    processed = backbone.preprocess_sequences(sequences, "esmc_300m")

    assert processed == sequences


def test_esmc_backbone_hidden_size_from_embed():
    model_default = DummyESMC(d_model=960)
    backbone = EsmcBackbone(model_default)
    assert backbone.hidden_size == 960

    model_custom = DummyESMC(d_model=1152)
    backbone_custom = EsmcBackbone(model_custom)
    assert backbone_custom.hidden_size == 1152
