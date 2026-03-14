from typing import List, cast

import pandas as pd
import pytest

from transformers import PreTrainedTokenizerBase

from mutafitup.datasets import PerResidueRegressionDataset
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


def test_per_residue_regression_normalize_and_mask(monkeypatch):
    import mutafitup.datasets.per_residue_regression_dataset as module

    train_df = pd.DataFrame({"sequence": ["AAA"], "scores": [[1.0, -2.0]]})
    valid_df = pd.DataFrame({"sequence": ["BBB"], "scores": [[3.0, -4.0]]})

    def fake_load(self, path: str):
        if "train" in path:
            return train_df
        return valid_df

    monkeypatch.setattr(
        module.PerResidueRegressionDataset, "_load_dataframe", fake_load
    )

    ds = PerResidueRegressionDataset(
        name="task",
        train_parquet="train.parquet",
        valid_parquet="valid.parquet",
        label_column="scores",
        unresolved_marker=999.0,
    )

    assert ds.label_min == -2.0
    assert ds.label_max == 1.0
    train_scores = ds.train_df["scores"].iloc[0]
    assert max(abs(v) for v in train_scores) == 1.0


def test_per_residue_regression_raises_if_all_unresolved(monkeypatch):
    import mutafitup.datasets.per_residue_regression_dataset as module

    train_df = pd.DataFrame({"sequence": ["AAA"], "scores": [[999.0]]})
    valid_df = pd.DataFrame({"sequence": ["BBB"], "scores": [[999.0]]})

    def fake_load(self, path: str):
        if "train" in path:
            return train_df
        return valid_df

    monkeypatch.setattr(
        module.PerResidueRegressionDataset, "_load_dataframe", fake_load
    )

    with pytest.raises(ValueError):
        PerResidueRegressionDataset(
            name="task",
            train_parquet="train.parquet",
            valid_parquet="valid.parquet",
            label_column="scores",
            unresolved_marker=999.0,
        )


def test_per_residue_regression_get_dataloader(monkeypatch):
    import mutafitup.datasets.per_residue_regression_dataset as module

    train_df = pd.DataFrame({"sequence": ["AAA"], "scores": [[1.0, 2.0]]})
    valid_df = pd.DataFrame({"sequence": ["BBB"], "scores": [[3.0, 4.0]]})

    def fake_load(self, path: str):
        if "train" in path:
            return train_df
        return valid_df

    monkeypatch.setattr(
        module.PerResidueRegressionDataset, "_load_dataframe", fake_load
    )

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

    ds = PerResidueRegressionDataset(
        name="task",
        train_parquet="train.parquet",
        valid_parquet="valid.parquet",
        label_column="scores",
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
    assert records["labels"] == [[0.5, 1.0]]
    assert records["checkpoint"] == "ckpt"


def test_per_residue_regression_test_split_uses_train_scale(monkeypatch):
    import mutafitup.datasets.per_residue_regression_dataset as module

    train_df = pd.DataFrame({"sequence": ["AAA"], "scores": [[1.0, -2.0]]})
    valid_df = pd.DataFrame({"sequence": ["BBB"], "scores": [[4.0, 999.0]]})
    test_df = pd.DataFrame({"sequence": ["CCC"], "scores": [[6.0, 999.0]]})

    def fake_load(self, path: str):
        if "train" in path:
            return train_df
        if "valid" in path:
            return valid_df
        return test_df

    monkeypatch.setattr(
        module.PerResidueRegressionDataset, "_load_dataframe", fake_load
    )

    ds = PerResidueRegressionDataset(
        name="task",
        train_parquet="train.parquet",
        valid_parquet="valid.parquet",
        test_parquet="test.parquet",
        label_column="scores",
        unresolved_marker=999.0,
    )

    assert ds.label_min == -2.0
    assert ds.label_max == 1.0
    assert ds.test_df is not None
    assert ds.train_df["scores"].iloc[0] == [0.5, -1.0]
    assert ds.valid_df["scores"].iloc[0] == [2.0, -100.0]
    assert ds.test_df["scores"].iloc[0] == [3.0, -100.0]


def test_per_residue_regression_all_splits_share_same_normalized_scale(monkeypatch):
    import mutafitup.datasets.per_residue_regression_dataset as module

    train_df = pd.DataFrame({"sequence": ["AAA"], "scores": [[2.0, -4.0]]})
    valid_df = pd.DataFrame({"sequence": ["BBB"], "scores": [[8.0, -12.0]]})
    test_df = pd.DataFrame({"sequence": ["CCC"], "scores": [[16.0, -20.0]]})

    def fake_load(self, path: str):
        if "train" in path:
            return train_df
        if "valid" in path:
            return valid_df
        return test_df

    monkeypatch.setattr(
        module.PerResidueRegressionDataset, "_load_dataframe", fake_load
    )

    ds = PerResidueRegressionDataset(
        name="task",
        train_parquet="train.parquet",
        valid_parquet="valid.parquet",
        test_parquet="test.parquet",
        label_column="scores",
        unresolved_marker=999.0,
    )

    scale_factor = max(abs(ds.label_min), abs(ds.label_max))
    assert scale_factor == 4.0
    assert ds.test_df is not None
    assert ds.train_df["scores"].iloc[0] == [0.5, -1.0]
    assert ds.valid_df["scores"].iloc[0] == [2.0, -3.0]
    assert ds.test_df["scores"].iloc[0] == [4.0, -5.0]
