import tempfile
from pathlib import Path

import pytest
import torch

from mutafitup.datasets.cached_embedding_dataset import (
    CachedEmbeddingDataCollator,
    CachedEmbeddingDataset,
)


def _save_embeddings(tmpdir: str, embeddings: list[torch.Tensor]) -> list[Path]:
    """Save embeddings to .pt files and return their paths."""
    paths = []
    for i, emb in enumerate(embeddings):
        path = Path(tmpdir) / f"emb_{i}.pt"
        torch.save(emb, path)
        paths.append(path)
    return paths


def test_cached_embedding_dataset_length_and_getitem():
    embeddings = [torch.randn(5, 8), torch.randn(7, 8), torch.randn(3, 8)]
    labels = [0, 1, 2]

    with tempfile.TemporaryDirectory() as tmpdir:
        paths = _save_embeddings(tmpdir, embeddings)
        dataset = CachedEmbeddingDataset(embedding_paths=paths, labels=labels)

        assert len(dataset) == 3

        item = dataset[1]
        assert "embeddings" in item
        assert "labels" in item
        assert torch.equal(item["embeddings"], embeddings[1])
        assert item["labels"] == 1


def test_cached_embedding_dataset_mismatched_lengths_raises():
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = _save_embeddings(tmpdir, [torch.randn(5, 8), torch.randn(7, 8)])
        labels = [0, 1, 2]

        with pytest.raises(ValueError, match="must match"):
            CachedEmbeddingDataset(embedding_paths=paths, labels=labels)


def test_cached_embedding_collator_scalar_int_labels():
    collator = CachedEmbeddingDataCollator()

    features = [
        {"embeddings": torch.randn(5, 8), "labels": 0},
        {"embeddings": torch.randn(7, 8), "labels": 1},
        {"embeddings": torch.randn(3, 8), "labels": 2},
    ]

    batch = collator(features)

    assert batch["embeddings"].shape == (3, 7, 8)
    assert batch["attention_mask"].shape == (3, 7)
    assert batch["labels"].shape == (3,)
    assert batch["labels"].dtype == torch.long
    assert batch["labels"].tolist() == [0, 1, 2]

    assert batch["attention_mask"][0, :5].sum() == 5
    assert batch["attention_mask"][0, 5:].sum() == 0
    assert batch["attention_mask"][1, :7].sum() == 7
    assert batch["attention_mask"][2, :3].sum() == 3
    assert batch["attention_mask"][2, 3:].sum() == 0


def test_cached_embedding_collator_scalar_float_labels():
    collator = CachedEmbeddingDataCollator()

    features = [
        {"embeddings": torch.randn(4, 8), "labels": 0.5},
        {"embeddings": torch.randn(6, 8), "labels": 1.5},
    ]

    batch = collator(features)

    assert batch["labels"].shape == (2,)
    assert batch["labels"].dtype == torch.float
    assert batch["labels"].tolist() == [0.5, 1.5]


def test_cached_embedding_collator_sequence_int_labels():
    collator = CachedEmbeddingDataCollator(label_pad_token_id=-100)

    features = [
        {"embeddings": torch.randn(3, 8), "labels": [0, 1, 2]},
        {"embeddings": torch.randn(5, 8), "labels": [3, 4, 5, 6, 7]},
    ]

    batch = collator(features)

    assert batch["embeddings"].shape == (2, 5, 8)
    assert batch["attention_mask"].shape == (2, 5)
    assert batch["labels"].shape == (2, 5)
    assert batch["labels"].dtype == torch.long

    assert batch["labels"][0].tolist() == [0, 1, 2, -100, -100]
    assert batch["labels"][1].tolist() == [3, 4, 5, 6, 7]


def test_cached_embedding_collator_sequence_float_labels():
    collator = CachedEmbeddingDataCollator(label_pad_token_id=-100)

    features = [
        {"embeddings": torch.randn(2, 4), "labels": [0.1, 0.2]},
        {"embeddings": torch.randn(4, 4), "labels": [0.3, 0.4, 0.5, 0.6]},
    ]

    batch = collator(features)

    assert batch["labels"].shape == (2, 4)
    assert batch["labels"].dtype == torch.float

    assert batch["labels"][0, :2].tolist() == pytest.approx([0.1, 0.2])
    assert batch["labels"][0, 2:].tolist() == [-100.0, -100.0]


def test_cached_embedding_collator_tensor_labels():
    collator = CachedEmbeddingDataCollator()

    features = [
        {"embeddings": torch.randn(3, 8), "labels": torch.tensor(5)},
        {"embeddings": torch.randn(3, 8), "labels": torch.tensor(10)},
    ]

    batch = collator(features)

    assert batch["labels"].shape == (2,)
    assert batch["labels"].tolist() == [5, 10]


def test_cached_embedding_collator_preserves_embedding_values():
    collator = CachedEmbeddingDataCollator()

    emb1 = torch.randn(3, 4)
    emb2 = torch.randn(5, 4)

    features = [
        {"embeddings": emb1, "labels": 0},
        {"embeddings": emb2, "labels": 1},
    ]

    batch = collator(features)

    assert torch.allclose(batch["embeddings"][0, :3], emb1)
    assert torch.allclose(batch["embeddings"][1, :5], emb2)
    assert (batch["embeddings"][0, 3:] == 0).all()


def test_cached_embedding_collator_numpy_int_labels():
    import numpy as np

    collator = CachedEmbeddingDataCollator(label_pad_token_id=-100)

    features = [
        {"embeddings": torch.randn(3, 8), "labels": np.array([0, 1, 2])},
        {"embeddings": torch.randn(5, 8), "labels": np.array([3, 4, 5, 6, 7])},
    ]

    batch = collator(features)

    assert batch["embeddings"].shape == (2, 5, 8)
    assert batch["attention_mask"].shape == (2, 5)
    assert batch["labels"].shape == (2, 5)
    assert batch["labels"].dtype == torch.long

    assert batch["labels"][0].tolist() == [0, 1, 2, -100, -100]
    assert batch["labels"][1].tolist() == [3, 4, 5, 6, 7]


def test_cached_embedding_collator_numpy_float_labels():
    import numpy as np

    collator = CachedEmbeddingDataCollator(label_pad_token_id=-100)

    features = [
        {"embeddings": torch.randn(2, 4), "labels": np.array([0.1, 0.2])},
        {"embeddings": torch.randn(4, 4), "labels": np.array([0.3, 0.4, 0.5, 0.6])},
    ]

    batch = collator(features)

    assert batch["labels"].shape == (2, 4)
    assert batch["labels"].dtype == torch.float

    assert batch["labels"][0, :2].tolist() == pytest.approx([0.1, 0.2])
    assert batch["labels"][0, 2:].tolist() == [-100.0, -100.0]


def test_cached_embedding_collator_truncates_labels_to_embedding_length():
    import numpy as np

    collator = CachedEmbeddingDataCollator(label_pad_token_id=-100)

    features = [
        {"embeddings": torch.randn(3, 8), "labels": np.array([0, 1, 2, 3, 4, 5])},
        {
            "embeddings": torch.randn(4, 8),
            "labels": np.array([10, 11, 12, 13, 14, 15, 16]),
        },
    ]

    batch = collator(features)

    assert batch["embeddings"].shape == (2, 4, 8)
    assert batch["labels"].shape == (2, 4)
    assert batch["labels"][0].tolist() == [0, 1, 2, -100]
    assert batch["labels"][1].tolist() == [10, 11, 12, 13]
