import types

import torch

from mutafitup.models.multitask_backbones import t5_backbone as module
from mutafitup.models.multitask_backbones.t5_backbone import T5Backbone
from mutafitup.models.multitask_model import MultitaskForwardArgs


class DummyOutputs:
    def __init__(self, hidden_states: torch.Tensor):
        self.last_hidden_state = hidden_states


class DummyT5Model:
    def __init__(self, torch_dtype=None):
        class Config:
            pass

        self.config = Config()
        self.config.d_model = 16
        self.torch_dtype = torch_dtype

    @classmethod
    def from_pretrained(cls, checkpoint, torch_dtype=None):
        return cls(torch_dtype=torch_dtype)

    def __call__(self, **kwargs):
        input_ids = kwargs["input_ids"]
        batch, length = input_ids.shape
        hidden = torch.zeros(batch, length, self.config.d_model)
        return DummyOutputs(hidden)


def _patch_tokenizers(monkeypatch):
    class DummyTokenizer:
        def __init__(self, checkpoint):
            self.checkpoint = checkpoint

    def auto_from_pretrained(checkpoint):
        return DummyTokenizer(checkpoint)

    def t5_from_pretrained(checkpoint, do_lower_case=False):
        return DummyTokenizer(checkpoint)

    monkeypatch.setattr(
        module,
        "AutoTokenizer",
        types.SimpleNamespace(from_pretrained=auto_from_pretrained),
    )
    monkeypatch.setattr(
        module,
        "T5Tokenizer",
        types.SimpleNamespace(from_pretrained=t5_from_pretrained),
    )


def test_t5_backbone_from_pretrained_ankh_and_forward(monkeypatch):
    monkeypatch.setattr(module, "T5EncoderModel", DummyT5Model)
    _patch_tokenizers(monkeypatch)
    monkeypatch.setattr(module, "inject_adapter_in_model", lambda cfg, model: model)

    marker = object()
    backbone, tokenizer = T5Backbone.from_pretrained(
        "ankh-model",
        lora_config=marker,
    )

    assert hasattr(tokenizer, "checkpoint")
    assert tokenizer.checkpoint == "ankh-model"
    assert isinstance(backbone.model, DummyT5Model)
    assert backbone.hidden_size == 16
    assert backbone.lora_config is marker

    args = MultitaskForwardArgs(
        attention_mask=torch.ones(1, 4, dtype=torch.long),
        input_ids=torch.ones(1, 4, dtype=torch.long),
    )

    hidden = backbone.forward(args)

    assert hidden.shape == (1, 4, 16)


def test_t5_backbone_from_pretrained_prot_t5(monkeypatch):
    record = {}

    class RecordingT5Model(DummyT5Model):
        @classmethod
        def from_pretrained(cls, checkpoint, torch_dtype=None):
            record["checkpoint"] = checkpoint
            record["dtype"] = torch_dtype
            return cls(torch_dtype=torch_dtype)

    monkeypatch.setattr(module, "T5EncoderModel", RecordingT5Model)
    _patch_tokenizers(monkeypatch)

    backbone, tokenizer = T5Backbone.from_pretrained(
        "prot_t5_xl_uniref50-enc",
        lora_config=None,
    )

    assert backbone.hidden_size == 16
    assert hasattr(tokenizer, "checkpoint")
    assert record["checkpoint"] == "prot_t5_xl_uniref50-enc"
    assert record["dtype"] is None


def test_t5_backbone_preprocess_sequences_variants():
    model = DummyT5Model()
    backbone = T5Backbone(model)

    seqs = ["OBUZXJ"]
    processed = backbone.preprocess_sequences(seqs, "plain")
    assert processed == ["XXXXXX"]

    rostlab = backbone.preprocess_sequences(["AB"], "Rostlab/prot_t5")
    assert rostlab == ["A X"]

    prost = backbone.preprocess_sequences(["AB"], "ProstT5")
    assert prost == ["<AA2fold> AX"]
