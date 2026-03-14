from typing import List, cast

import pandas as pd
import pytest

from transformers import PreTrainedTokenizerBase

from mutafitup.datasets import PerProteinRegressionDataset
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


def test_per_protein_regression_normalize_and_mask(monkeypatch):
    import mutafitup.datasets.per_protein_regression_dataset as module

    train_df = pd.DataFrame({"sequence": ["AAA", "BBB"], "score": [1.0, 999.0]})
    valid_df = pd.DataFrame({"sequence": ["CCC", "DDD"], "score": [2.0, 999.0]})

    def fake_load(self, path: str):
        if "train" in path:
            return train_df
        return valid_df

    monkeypatch.setattr(
        module.PerProteinRegressionDataset, "_load_dataframe", fake_load
    )

    ds = PerProteinRegressionDataset(
        name="task",
        train_parquet="train.parquet",
        valid_parquet="valid.parquet",
        label_column="score",
        unresolved_marker=999.0,
    )

    assert ds.label_min == 1.0
    assert ds.label_max == 1.0
    assert ds.train_df["score"].tolist() == [1.0, -100.0]
    assert ds.valid_df["score"].tolist() == [2.0, -100.0]


def test_per_protein_regression_raises_if_all_unresolved(monkeypatch):
    import mutafitup.datasets.per_protein_regression_dataset as module

    train_df = pd.DataFrame({"sequence": ["AAA"], "score": [999.0]})
    valid_df = pd.DataFrame({"sequence": ["BBB"], "score": [999.0]})

    def fake_load(self, path: str):
        if "train" in path:
            return train_df
        return valid_df

    monkeypatch.setattr(
        module.PerProteinRegressionDataset, "_load_dataframe", fake_load
    )

    with pytest.raises(ValueError):
        PerProteinRegressionDataset(
            name="task",
            train_parquet="train.parquet",
            valid_parquet="valid.parquet",
            label_column="score",
            unresolved_marker=999.0,
        )


def test_per_protein_regression_get_dataloader(monkeypatch):
    import mutafitup.datasets.per_protein_regression_dataset as module

    train_df = pd.DataFrame({"sequence": ["AAA"], "score": [1.0]})
    valid_df = pd.DataFrame({"sequence": ["BBB"], "score": [2.0]})

    def fake_load(self, path: str):
        if "train" in path:
            return train_df
        return valid_df

    monkeypatch.setattr(
        module.PerProteinRegressionDataset, "_load_dataframe", fake_load
    )

    records = {}

    def fake_create_dataset(tokenizer, seqs, labels, checkpoint):
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

    ds = PerProteinRegressionDataset(
        name="task",
        train_parquet="train.parquet",
        valid_parquet="valid.parquet",
        label_column="score",
        unresolved_marker=999.0,
    )

    backbone = DummyBackbone()
    tokenizer = cast(PreTrainedTokenizerBase, object())

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
    assert records["labels"] == [1.0]
    assert records["checkpoint"] == "ckpt"


def test_per_protein_regression_test_split_uses_train_scale(monkeypatch):
    import mutafitup.datasets.per_protein_regression_dataset as module

    train_df = pd.DataFrame({"sequence": ["AAA", "BBB"], "score": [-2.0, 1.0]})
    valid_df = pd.DataFrame({"sequence": ["CCC"], "score": [4.0]})
    test_df = pd.DataFrame({"sequence": ["DDD", "EEE"], "score": [6.0, 999.0]})

    def fake_load(self, path: str):
        if "train" in path:
            return train_df
        if "valid" in path:
            return valid_df
        return test_df

    monkeypatch.setattr(
        module.PerProteinRegressionDataset, "_load_dataframe", fake_load
    )

    ds = PerProteinRegressionDataset(
        name="task",
        train_parquet="train.parquet",
        valid_parquet="valid.parquet",
        test_parquet="test.parquet",
        label_column="score",
        unresolved_marker=999.0,
    )

    assert ds.label_min == -2.0
    assert ds.label_max == 1.0
    assert ds.test_df is not None
    assert ds.train_df["score"].tolist() == [-1.0, 0.5]
    assert ds.valid_df["score"].tolist() == [2.0]
    assert ds.test_df["score"].tolist() == [3.0, -100.0]


def test_per_protein_regression_all_splits_share_same_normalized_scale(monkeypatch):
    import mutafitup.datasets.per_protein_regression_dataset as module

    train_df = pd.DataFrame({"sequence": ["AAA", "BBB"], "score": [-4.0, 2.0]})
    valid_df = pd.DataFrame({"sequence": ["CCC"], "score": [8.0]})
    test_df = pd.DataFrame({"sequence": ["DDD"], "score": [-12.0]})

    def fake_load(self, path: str):
        if "train" in path:
            return train_df
        if "valid" in path:
            return valid_df
        return test_df

    monkeypatch.setattr(
        module.PerProteinRegressionDataset, "_load_dataframe", fake_load
    )

    ds = PerProteinRegressionDataset(
        name="task",
        train_parquet="train.parquet",
        valid_parquet="valid.parquet",
        test_parquet="test.parquet",
        label_column="score",
        unresolved_marker=999.0,
    )

    scale_factor = max(abs(ds.label_min), abs(ds.label_max))
    assert scale_factor == 4.0
    assert ds.test_df is not None
    assert ds.train_df["score"].tolist() == [-1.0, 0.5]
    assert ds.valid_df["score"].tolist() == [2.0]
    assert ds.test_df["score"].tolist() == [-3.0]
