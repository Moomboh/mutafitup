import types

import pytest

from mutafitup import models as models_module


def test_build_backbone_and_tokenizer_requires_both_lora_params():
    with pytest.raises(ValueError):
        models_module.build_backbone_and_tokenizer(
            "esm1_t6", lora_rank=4, lora_alpha=None
        )

    with pytest.raises(ValueError):
        models_module.build_backbone_and_tokenizer(
            "esm1_t6", lora_rank=None, lora_alpha=8
        )


def test_build_backbone_and_tokenizer_esm_without_lora(monkeypatch):
    called = {}

    class DummyBackbone:
        pass

    class DummyTokenizer:
        pass

    def fake_from_pretrained(checkpoint, lora_config):
        called["checkpoint"] = checkpoint
        called["lora_config"] = lora_config
        return DummyBackbone(), DummyTokenizer()

    monkeypatch.setattr(
        models_module,
        "EsmBackbone",
        types.SimpleNamespace(from_pretrained=fake_from_pretrained),
    )

    backbone, tokenizer = models_module.build_backbone_and_tokenizer(
        "esm1_t6", lora_rank=None, lora_alpha=None
    )

    assert isinstance(backbone, DummyBackbone)
    assert isinstance(tokenizer, DummyTokenizer)
    assert called["checkpoint"] == "esm1_t6"
    assert called["lora_config"] is None


def test_build_backbone_and_tokenizer_esm_with_lora_builds_expected_lora_config(
    monkeypatch,
):
    created = {}

    class DummyLoraConfig:
        def __init__(self, r, lora_alpha, target_modules):
            created["r"] = r
            created["alpha"] = lora_alpha
            created["target_modules"] = target_modules

    def fake_from_pretrained(checkpoint, lora_config):
        created["checkpoint"] = checkpoint
        created["lora_config"] = lora_config
        return object(), object()

    monkeypatch.setattr(models_module, "LoraConfig", DummyLoraConfig)
    monkeypatch.setattr(
        models_module,
        "EsmBackbone",
        types.SimpleNamespace(from_pretrained=fake_from_pretrained),
    )

    backbone, tokenizer = models_module.build_backbone_and_tokenizer(
        "esm_model", lora_rank=4, lora_alpha=8
    )

    assert backbone is not None
    assert tokenizer is not None
    assert created["checkpoint"] == "esm_model"
    assert created["r"] == 4
    assert created["alpha"] == 8
    assert created["target_modules"] == ["query", "key", "value", "dense"]
    assert isinstance(created["lora_config"], DummyLoraConfig)


def test_build_backbone_and_tokenizer_t5_with_lora(monkeypatch):
    created = {}

    class DummyLoraConfig:
        def __init__(self, r, lora_alpha, target_modules):
            created["r"] = r
            created["alpha"] = lora_alpha
            created["target_modules"] = target_modules

    def fake_from_pretrained(checkpoint, lora_config):
        created["checkpoint"] = checkpoint
        created["lora_config"] = lora_config
        return object(), object()

    monkeypatch.setattr(models_module, "LoraConfig", DummyLoraConfig)
    monkeypatch.setattr(
        models_module,
        "T5Backbone",
        types.SimpleNamespace(from_pretrained=fake_from_pretrained),
    )

    backbone, tokenizer = models_module.build_backbone_and_tokenizer(
        "some_t5_checkpoint", lora_rank=2, lora_alpha=4
    )

    assert backbone is not None
    assert tokenizer is not None
    assert created["checkpoint"] == "some_t5_checkpoint"
    assert created["r"] == 2
    assert created["alpha"] == 4
    assert created["target_modules"] == ["q", "k", "v", "o"]
    assert isinstance(created["lora_config"], DummyLoraConfig)


def test_build_backbone_and_tokenizer_esmc_without_lora(monkeypatch):
    called = {}

    class DummyBackbone:
        pass

    class DummyTokenizer:
        pass

    def fake_from_pretrained(checkpoint, lora_config):
        called["checkpoint"] = checkpoint
        called["lora_config"] = lora_config
        return DummyBackbone(), DummyTokenizer()

    monkeypatch.setattr(
        models_module,
        "EsmcBackbone",
        types.SimpleNamespace(from_pretrained=fake_from_pretrained),
    )

    backbone, tokenizer = models_module.build_backbone_and_tokenizer(
        "esmc_300m", lora_rank=None, lora_alpha=None
    )

    assert isinstance(backbone, DummyBackbone)
    assert isinstance(tokenizer, DummyTokenizer)
    assert called["checkpoint"] == "esmc_300m"
    assert called["lora_config"] is None


def test_build_backbone_and_tokenizer_esmc_with_lora_builds_expected_lora_config(
    monkeypatch,
):
    created = {}

    class DummyLoraConfig:
        def __init__(self, r, lora_alpha, target_modules):
            created["r"] = r
            created["alpha"] = lora_alpha
            created["target_modules"] = target_modules

    def fake_from_pretrained(checkpoint, lora_config):
        created["checkpoint"] = checkpoint
        created["lora_config"] = lora_config
        return object(), object()

    monkeypatch.setattr(models_module, "LoraConfig", DummyLoraConfig)
    monkeypatch.setattr(
        models_module,
        "EsmcBackbone",
        types.SimpleNamespace(from_pretrained=fake_from_pretrained),
    )

    backbone, tokenizer = models_module.build_backbone_and_tokenizer(
        "esmc_300m", lora_rank=4, lora_alpha=8
    )

    assert backbone is not None
    assert tokenizer is not None
    assert created["checkpoint"] == "esmc_300m"
    assert created["r"] == 4
    assert created["alpha"] == 8
    assert created["target_modules"] == ["layernorm_qkv.1", "out_proj"]
    assert isinstance(created["lora_config"], DummyLoraConfig)


def test_build_backbone_and_tokenizer_esmc_not_routed_to_esm(monkeypatch):
    """Verify that an 'esmc' checkpoint is routed to EsmcBackbone, not EsmBackbone."""
    esmc_called = {}
    esm_called = {}

    def fake_esmc_from_pretrained(checkpoint, lora_config):
        esmc_called["checkpoint"] = checkpoint
        return object(), object()

    def fake_esm_from_pretrained(checkpoint, lora_config):
        esm_called["checkpoint"] = checkpoint
        return object(), object()

    monkeypatch.setattr(
        models_module,
        "EsmcBackbone",
        types.SimpleNamespace(from_pretrained=fake_esmc_from_pretrained),
    )
    monkeypatch.setattr(
        models_module,
        "EsmBackbone",
        types.SimpleNamespace(from_pretrained=fake_esm_from_pretrained),
    )

    models_module.build_backbone_and_tokenizer(
        "esmc_300m", lora_rank=None, lora_alpha=None
    )

    assert "checkpoint" in esmc_called
    assert esmc_called["checkpoint"] == "esmc_300m"
    assert "checkpoint" not in esm_called


def test_build_backbone_and_tokenizer_raises_for_unsupported_lora_checkpoint():
    with pytest.raises(ValueError):
        models_module.build_backbone_and_tokenizer("unknown", lora_rank=1, lora_alpha=1)


def test_build_backbone_and_tokenizer_raises_for_unsupported_checkpoint():
    with pytest.raises(ValueError):
        models_module.build_backbone_and_tokenizer(
            "unknown", lora_rank=None, lora_alpha=None
        )
