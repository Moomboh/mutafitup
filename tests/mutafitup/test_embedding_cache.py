import tempfile
from pathlib import Path

import torch
from torch import nn

from mutafitup.embedding_cache import (
    _compute_sequence_hash,
    _normalize_checkpoint_name,
    compute_and_cache_embeddings,
    ensure_cached_embedding_paths,
    get_cache_path,
    get_default_cache_dir,
    load_cached_embedding,
    save_cached_embedding,
)
from mutafitup.models.multitask_model import MultitaskBackbone, MultitaskForwardArgs


class DummyBackbone(MultitaskBackbone):
    def __init__(self, hidden_size: int = 4):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.call_count = 0

    def forward(self, args: MultitaskForwardArgs) -> torch.Tensor:
        self.call_count += 1
        batch_size, seq_len = args.input_ids.shape
        return torch.randn(batch_size, seq_len, self.hidden_size)

    def preprocess_sequences(self, sequences, checkpoint):
        return sequences

    def _inject_lora_config(self, lora_config=None) -> None:
        pass


class DummyTokenizer:
    def __call__(
        self,
        sequences,
        max_length=None,
        padding=None,
        truncation=None,
        return_tensors=None,
    ):
        batch_size = len(sequences)
        max_len = max(len(s) for s in sequences)
        input_ids = torch.ones(batch_size, max_len, dtype=torch.long)
        attention_mask = torch.ones(batch_size, max_len, dtype=torch.long)
        for i, s in enumerate(sequences):
            attention_mask[i, len(s) :] = 0
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def test_compute_sequence_hash_deterministic():
    seq = "ACDEFGH"
    hash1 = _compute_sequence_hash(seq)
    hash2 = _compute_sequence_hash(seq)
    assert hash1 == hash2
    assert len(hash1) == 16


def test_compute_sequence_hash_different_for_different_sequences():
    hash1 = _compute_sequence_hash("ACDEFGH")
    hash2 = _compute_sequence_hash("HGFEDCA")
    assert hash1 != hash2


def test_normalize_checkpoint_name():
    assert (
        _normalize_checkpoint_name("facebook/esm2_t6_8M_UR50D")
        == "facebook_esm2_t6_8M_UR50D"
    )
    assert _normalize_checkpoint_name("model\\path") == "model_path"


def test_get_cache_path():
    path = get_cache_path("/cache", "facebook/esm2", "ACDEF")
    assert isinstance(path, Path)
    assert str(path).startswith("/cache/facebook_esm2/")
    assert path.suffix == ".pt"


def test_save_and_load_cached_embedding():
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint = "test/checkpoint"
        sequence = "ACDEFGH"
        embedding = torch.randn(7, 320)

        result = load_cached_embedding(tmpdir, checkpoint, sequence)
        assert result is None

        saved_path = save_cached_embedding(tmpdir, checkpoint, sequence, embedding)
        assert saved_path.exists()

        loaded = load_cached_embedding(tmpdir, checkpoint, sequence)
        assert loaded is not None
        assert torch.allclose(loaded, embedding)


def test_compute_and_cache_embeddings_caches_results():
    with tempfile.TemporaryDirectory() as tmpdir:
        backbone = DummyBackbone(hidden_size=4)
        tokenizer = DummyTokenizer()
        sequences = ["ABC", "DEFGH", "IJ"]

        embeddings1 = compute_and_cache_embeddings(
            backbone=backbone,
            tokenizer=tokenizer,
            checkpoint="test/model",
            sequences=sequences,
            cache_dir=tmpdir,
            batch_size=2,
        )

        assert len(embeddings1) == 3
        assert backbone.call_count == 2

        backbone.call_count = 0
        embeddings2 = compute_and_cache_embeddings(
            backbone=backbone,
            tokenizer=tokenizer,
            checkpoint="test/model",
            sequences=sequences,
            cache_dir=tmpdir,
            batch_size=2,
        )

        assert backbone.call_count == 0
        assert len(embeddings2) == 3


def test_compute_and_cache_embeddings_partial_cache():
    with tempfile.TemporaryDirectory() as tmpdir:
        backbone = DummyBackbone(hidden_size=4)
        tokenizer = DummyTokenizer()

        compute_and_cache_embeddings(
            backbone=backbone,
            tokenizer=tokenizer,
            checkpoint="test/model",
            sequences=["ABC"],
            cache_dir=tmpdir,
        )
        assert backbone.call_count == 1

        backbone.call_count = 0
        embeddings = compute_and_cache_embeddings(
            backbone=backbone,
            tokenizer=tokenizer,
            checkpoint="test/model",
            sequences=["ABC", "DEF"],
            cache_dir=tmpdir,
        )

        assert backbone.call_count == 1
        assert len(embeddings) == 2


def test_ensure_cached_embedding_paths_returns_paths():
    with tempfile.TemporaryDirectory() as tmpdir:
        backbone = DummyBackbone(hidden_size=4)
        tokenizer = DummyTokenizer()
        sequences = ["ABC", "DEFGH", "IJ"]

        paths = ensure_cached_embedding_paths(
            backbone=backbone,
            tokenizer=tokenizer,
            checkpoint="test/model",
            sequences=sequences,
            cache_dir=tmpdir,
            batch_size=2,
        )

        assert len(paths) == 3
        assert all(isinstance(p, Path) for p in paths)
        assert all(p.exists() for p in paths)
        assert backbone.call_count == 2

        # Verify the saved files contain valid tensors
        for path in paths:
            emb = torch.load(path, map_location="cpu", weights_only=True)
            assert isinstance(emb, torch.Tensor)
            assert emb.dim() == 2
            assert emb.shape[1] == 4  # hidden_size


def test_ensure_cached_embedding_paths_uses_cache():
    with tempfile.TemporaryDirectory() as tmpdir:
        backbone = DummyBackbone(hidden_size=4)
        tokenizer = DummyTokenizer()
        sequences = ["ABC", "DEFGH"]

        paths1 = ensure_cached_embedding_paths(
            backbone=backbone,
            tokenizer=tokenizer,
            checkpoint="test/model",
            sequences=sequences,
            cache_dir=tmpdir,
        )
        assert backbone.call_count == 1

        backbone.call_count = 0
        paths2 = ensure_cached_embedding_paths(
            backbone=backbone,
            tokenizer=tokenizer,
            checkpoint="test/model",
            sequences=sequences,
            cache_dir=tmpdir,
        )

        # No new computation needed
        assert backbone.call_count == 0
        assert len(paths2) == 2
        assert all(p.exists() for p in paths2)


def test_ensure_cached_embedding_paths_partial_cache():
    with tempfile.TemporaryDirectory() as tmpdir:
        backbone = DummyBackbone(hidden_size=4)
        tokenizer = DummyTokenizer()

        ensure_cached_embedding_paths(
            backbone=backbone,
            tokenizer=tokenizer,
            checkpoint="test/model",
            sequences=["ABC"],
            cache_dir=tmpdir,
        )
        assert backbone.call_count == 1

        backbone.call_count = 0
        paths = ensure_cached_embedding_paths(
            backbone=backbone,
            tokenizer=tokenizer,
            checkpoint="test/model",
            sequences=["ABC", "DEF"],
            cache_dir=tmpdir,
        )

        # Only DEF needed computation
        assert backbone.call_count == 1
        assert len(paths) == 2
        assert all(p.exists() for p in paths)


def test_get_default_cache_dir():
    cache_dir = get_default_cache_dir()
    assert "mutafitup" in cache_dir
    assert "embeddings" in cache_dir
