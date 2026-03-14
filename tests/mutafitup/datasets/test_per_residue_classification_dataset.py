from typing import List

import pandas as pd
import pytest

from mutafitup.datasets import PerResidueClassificationDataset
from mutafitup.models.multitask_model import MultitaskBackbone, MultitaskForwardArgs


class DummyBackbone(MultitaskBackbone):
    def __init__(self):
        super().__init__()

    def forward(self, args: MultitaskForwardArgs):
        raise RuntimeError

    def preprocess_sequences(self, sequences: List[str], checkpoint: str) -> List[str]:
        return [f"proc-{s}" for s in sequences]

    def _inject_lora_config(self, lora_config=None) -> None:
        return None


def test_per_residue_classification_label_transform(monkeypatch):
    import mutafitup.datasets.per_residue_classification_dataset as module

    train_df = pd.DataFrame({"sequence": ["AAA"], "labels": [[0, 1, 2]]})
    valid_df = pd.DataFrame({"sequence": ["BBB"], "labels": [[3, 4, 5]]})

    def fake_load(self, path: str):
        if "train" in path:
            return train_df
        return valid_df

    monkeypatch.setattr(module.PerResidueClassificationDataset, "_load_dataframe", fake_load)

    def transform(values):
        return [v + 1 for v in values]

    ds = PerResidueClassificationDataset(
        name="task",
        train_parquet="train.parquet",
        valid_parquet="valid.parquet",
        label_column="labels",
        label_transform=transform,
    )

    assert ds.train_df["labels"].tolist() == [[1, 2, 3]]
    assert ds.valid_df["labels"].tolist() == [[4, 5, 6]]


def test_per_residue_classification_get_dataloader(monkeypatch):
    import mutafitup.datasets.per_residue_classification_dataset as module

    train_df = pd.DataFrame({"sequence": ["AAA"], "labels": [[0, 1]]})
    valid_df = pd.DataFrame({"sequence": ["BBB"], "labels": [[2, 3]]})

    def fake_load(self, path: str):
        if "train" in path:
            return train_df
        return valid_df

    monkeypatch.setattr(module.PerResidueClassificationDataset, "_load_dataframe", fake_load)

    records = {}

    def fake_create_dataset(tokenizer, seqs, labels, checkpoint):
        records["seqs"] = seqs
        records["labels"] = labels
        records["checkpoint"] = checkpoint
        return "dummy-dataset"

    monkeypatch.setattr(module, "create_per_residue_dataset", fake_create_dataset)

    class DummyDataLoader:
        def __init__(self, dataset, batch_size, shuffle, collate_fn, pin_memory):
            self.dataset = dataset
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.collate_fn = collate_fn
            self.pin_memory = pin_memory

    monkeypatch.setattr(module, "DataLoader", DummyDataLoader)

    ds = PerResidueClassificationDataset(
        name="task",
        train_parquet="train.parquet",
        valid_parquet="valid.parquet",
        label_column="labels",
    )

    backbone = DummyBackbone()
    tokenizer = object()

    loader = ds.get_dataloader(
        split="train",
        backbone=backbone,
        tokenizer=tokenizer,
        checkpoint="ckpt",
        batch_size=2,
    )

    assert isinstance(loader, DummyDataLoader)
    assert loader.dataset == "dummy-dataset"
    assert loader.batch_size == 2
    assert loader.shuffle is True
    assert records["seqs"] == ["proc-AAA"]
    assert records["labels"] == [[0, 1]]
    assert records["checkpoint"] == "ckpt"
