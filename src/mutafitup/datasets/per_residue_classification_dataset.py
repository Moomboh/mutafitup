from typing import Callable, List, Optional

import pandas as pd
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from mutafitup.dataset import create_per_residue_dataset
from mutafitup.datasets.base_multitask_dataset import BaseMultitaskDataset
from mutafitup.datasets.data_collator import DataCollator
from mutafitup.models.multitask_model import MultitaskBackbone


class PerResidueClassificationDataset(BaseMultitaskDataset):
    def __init__(
        self,
        name: str,
        train_parquet: str,
        valid_parquet: str,
        label_column: str,
        label_transform: Optional[Callable[[List], List]] = None,
        test_parquet: Optional[str] = None,
    ):
        super().__init__(name)
        self.label_column = label_column
        self.label_transform = label_transform
        self.train_df = self._prepare_dataframe(train_parquet)
        self.valid_df = self._prepare_dataframe(valid_parquet)
        self.test_df = self._prepare_dataframe(test_parquet) if test_parquet else None

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

        dataset = create_per_residue_dataset(tokenizer, seqs, labels, checkpoint)
        data_collator = DataCollator(tokenizer)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=data_collator,
            pin_memory=True,
        )
