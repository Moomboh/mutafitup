import types

import torch

from mutafitup.models.multitask_backbones import esm_backbone as module
from mutafitup.models.multitask_backbones.esm_backbone import EsmBackbone
from mutafitup.models.multitask_model import MultitaskForwardArgs


class DummyOutputs:
    def __init__(self, hidden_states: torch.Tensor):
        self.last_hidden_state = hidden_states


class DummyEsmModel:
    def __init__(self, torch_dtype=None):
        class Config:
            pass

        self.config = Config()
        self.config.hidden_size = 8
        self.torch_dtype = torch_dtype

    @classmethod
    def from_pretrained(cls, checkpoint, torch_dtype=None):
        return cls(torch_dtype=torch_dtype)

    def __call__(self, **kwargs):
        input_ids = kwargs["input_ids"]
        batch, length = input_ids.shape
        hidden = torch.zeros(batch, length, self.config.hidden_size)
        return DummyOutputs(hidden)


def test_esm_backbone_from_pretrained_and_forward(monkeypatch):
    monkeypatch.setattr(module, "EsmModel", DummyEsmModel)

    class DummyTokenizer:
        pass

    def fake_tokenizer_from_pretrained(checkpoint):
        return DummyTokenizer()

    monkeypatch.setattr(
        module,
        "AutoTokenizer",
        types.SimpleNamespace(from_pretrained=fake_tokenizer_from_pretrained),
    )

    injected = {}

    def fake_inject(lora_config, model):
        injected["lora_config"] = lora_config
        injected["model"] = model
        return model

    monkeypatch.setattr(module, "inject_adapter_in_model", fake_inject)

    marker = object()
    backbone, tokenizer = EsmBackbone.from_pretrained(
        "esm1_t6",
        lora_config=marker,
    )

    assert isinstance(tokenizer, DummyTokenizer)
    assert isinstance(backbone.model, DummyEsmModel)
    assert backbone.hidden_size == 8
    assert backbone.lora_config is marker
    assert injected["lora_config"] is marker
    assert injected["model"] is backbone.model

    args = MultitaskForwardArgs(
        attention_mask=torch.ones(1, 3, dtype=torch.long),
        input_ids=torch.ones(1, 3, dtype=torch.long),
    )

    hidden = backbone.forward(args)

    assert hidden.shape == (1, 3, 8)


def test_esm_backbone_preprocess_sequences_identity():
    model = DummyEsmModel()
    backbone = EsmBackbone(model)

    sequences = ["ACD", "XYZ"]
    processed = backbone.preprocess_sequences(sequences, "any")

    assert processed == sequences
