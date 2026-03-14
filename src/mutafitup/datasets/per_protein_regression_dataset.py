from typing import Callable, List, Optional

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from mutafitup.dataset import create_per_protein_dataset
from mutafitup.datasets.base_multitask_dataset import BaseMultitaskDataset
from mutafitup.datasets.data_collator import DataCollator
from mutafitup.models.multitask_model import MultitaskBackbone


class PerProteinRegressionDataset(BaseMultitaskDataset):
    def __init__(
        self,
        name: str,
        train_parquet: str,
        valid_parquet: str,
        label_column: str,
        unresolved_marker: float = 999.0,
        label_transform: Optional[Callable[[List], List]] = None,
        test_parquet: Optional[str] = None,
    ):
        super().__init__(name)
        self.label_column = label_column
        self.label_transform = label_transform
        self.unresolved_marker = unresolved_marker
        self.train_df = self._prepare_dataframe(train_parquet)
        self.valid_df = self._prepare_dataframe(valid_parquet)
        self._raw_test_df = (
            self._prepare_dataframe(test_parquet) if test_parquet else None
        )
        (
            self.train_df,
            self.valid_df,
            self.label_min,
            self.label_max,
        ) = self._normalize_and_mask(self.train_df, self.valid_df)
        # Normalize test with same stats derived from train
        if self._raw_test_df is not None:
            self.test_df = self._normalize_df(self._raw_test_df)
        else:
            self.test_df = None

    def _load_dataframe(self, path: str) -> pd.DataFrame:
        return pd.read_parquet(path)

    def _apply_label_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.label_transform is None:
            return df
        df = df.copy()
        df[self.label_column] = df[self.label_column].apply(self.label_transform)
        return df

    def _prepare_dataframe(self, path: str) -> pd.DataFrame:
        df = self._load_dataframe(path)
        return self._apply_label_transform(df)

    def _normalize_and_mask(self, train_df, valid_df):
        train_values = [
            v for v in train_df[self.label_column] if v != self.unresolved_marker
        ]

        if not train_values:
            raise ValueError("No resolved scores found for normalization")

        label_min = float(np.min(train_values))
        label_max = float(np.max(train_values))
        scale_factor = max(abs(label_min), abs(label_max)) or 1.0

        def normalize_value(v):
            if v == self.unresolved_marker:
                return v
            return v / scale_factor

        def mask_unresolved(v):
            if v == self.unresolved_marker:
                return -100.0
            return v

        train_df = train_df.copy()
        valid_df = valid_df.copy()

        train_df[self.label_column] = train_df[self.label_column].apply(normalize_value)
        valid_df[self.label_column] = valid_df[self.label_column].apply(normalize_value)

        train_df[self.label_column] = train_df[self.label_column].apply(mask_unresolved)
        valid_df[self.label_column] = valid_df[self.label_column].apply(mask_unresolved)

        return train_df, valid_df, label_min, label_max

    def _normalize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize a dataframe using the scale factor derived from training."""
        scale_factor = max(abs(self.label_min), abs(self.label_max)) or 1.0

        def normalize_value(v):
            if v == self.unresolved_marker:
                return v
            return v / scale_factor

        def mask_unresolved(v):
            if v == self.unresolved_marker:
                return -100.0
            return v

        df = df.copy()
        df[self.label_column] = df[self.label_column].apply(normalize_value)
        df[self.label_column] = df[self.label_column].apply(mask_unresolved)
        return df

    def get_dataloader(
        self,
        split: str,
        backbone: MultitaskBackbone,
        tokenizer: PreTrainedTokenizerBase,
        checkpoint: str,
        batch_size: int,
    ) -> DataLoader:
        if split == "train":
            df = self.train_df
            shuffle = True
        elif split == "valid":
            df = self.valid_df
            shuffle = False
        elif split == "test":
            if self.test_df is None:
                raise ValueError("No test_parquet was provided")
            df = self.test_df
            shuffle = False
        else:
            raise ValueError(f"Unsupported split {split}")

        seqs = backbone.preprocess_sequences(list(df["sequence"]), checkpoint)
        labels = list(df[self.label_column])

        dataset = create_per_protein_dataset(tokenizer, seqs, labels, checkpoint)
        data_collator = DataCollator(tokenizer)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=data_collator,
            pin_memory=True,
        )
