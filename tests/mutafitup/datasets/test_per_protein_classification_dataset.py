from typing import List

import pandas as pd
import pytest

from mutafitup.datasets import PerProteinClassificationDataset
from mutafitup.models.multitask_model import MultitaskBackbone, MultitaskForwardArgs


class DummyBackbone(MultitaskBackbone):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, args: MultitaskForwardArgs):
        raise RuntimeError

    def preprocess_sequences(self, sequences: List[str], checkpoint: str) -> List[str]:
        self.calls.append((sequences, checkpoint))
        return [f"proc-{s}" for s in sequences]

    def _inject_lora_config(self, lora_config=None) -> None:
        return None


def test_per_protein_classification_dataset_label_transform(monkeypatch):
    import mutafitup.datasets.per_protein_classification_dataset as module

    train_df = pd.DataFrame({"sequence": ["AAA", "BBB"], "label": [0, 1]})
    valid_df = pd.DataFrame({"sequence": ["CCC"], "label": [2]})

    def fake_load(self, path: str):
        if "train" in path:
            return train_df
        return valid_df

    monkeypatch.setattr(
        module.PerProteinClassificationDataset, "_load_dataframe", fake_load
    )

    ds = PerProteinClassificationDataset(
        name="task",
        train_parquet="train.parquet",
        valid_parquet="valid.parquet",
        label_column="label",
        label_transform=lambda x: x + 1,
    )

    assert ds.name == "task"
    assert ds.train_df["label"].tolist() == [1, 2]
    assert ds.valid_df["label"].tolist() == [3]


def test_per_protein_classification_dataset_get_dataloader(monkeypatch):
    import mutafitup.datasets.per_protein_classification_dataset as module

    train_df = pd.DataFrame({"sequence": ["AAA"], "label": [0]})
    valid_df = pd.DataFrame({"sequence": ["BBB"], "label": [1]})

    def fake_load(self, path: str):
        if "train" in path:
            return train_df
        return valid_df

    monkeypatch.setattr(
        module.PerProteinClassificationDataset, "_load_dataframe", fake_load
    )

    records = {}

    def fake_create_dataset(tokenizer, seqs, labels, checkpoint):
        records["tokenizer"] = tokenizer
        records["seqs"] = seqs
        records["labels"] = labels
        records["checkpoint"] = checkpoint
        return "dummy-dataset"

    monkeypatch.setattr(module, "create_per_protein_dataset", fake_create_dataset)

    class DummyDataLoader:
        def __init__(self, dataset, batch_size, shuffle, collate_fn, pin_memory):
            self.dataset = dataset
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.collate_fn = collate_fn
            self.pin_memory = pin_memory

    monkeypatch.setattr(module, "DataLoader", DummyDataLoader)

    ds = PerProteinClassificationDataset(
        name="task",
        train_parquet="train.parquet",
        valid_parquet="valid.parquet",
        label_column="label",
    )

    backbone = DummyBackbone()
    tokenizer = object()

    loader = ds.get_dataloader(
        split="train",
        backbone=backbone,
        tokenizer=tokenizer,
        checkpoint="ckpt",
        batch_size=4,
    )

    assert isinstance(loader, DummyDataLoader)
    assert loader.dataset == "dummy-dataset"
    assert loader.batch_size == 4
    assert loader.shuffle is True
    assert loader.pin_memory is True

    assert records["tokenizer"] is tokenizer
    assert records["seqs"] == ["proc-AAA"]
    assert records["labels"] == [0]
    assert records["checkpoint"] == "ckpt"

    with pytest.raises(ValueError):
        ds.get_dataloader(
            split="test",
            backbone=backbone,
            tokenizer=tokenizer,
            checkpoint="ckpt",
            batch_size=4,
        )
