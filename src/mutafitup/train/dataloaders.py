import logging
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from mutafitup.datasets import BaseMultitaskDataset
from mutafitup.datasets.cached_embedding_dataset import (
    CachedEmbeddingDataCollator,
    CachedEmbeddingDataset,
)
from mutafitup.embedding_cache import ensure_cached_embedding_paths
from mutafitup.models.multitask_model import MultitaskModel


def build_cached_dataloaders(
    model: MultitaskModel,
    tokenizer: PreTrainedTokenizerBase,
    checkpoint: str,
    task_datasets: Dict[str, BaseMultitaskDataset],
    task_names: List[str],
    batch_size: int,
    cache_dir: str,
    device: torch.device,
    logger: Optional[logging.Logger] = None,
    num_workers: int = 1,
) -> Tuple[Dict[str, DataLoader], Dict[str, DataLoader]]:
    """Build train and validation dataloaders using cached embeddings.

    Embeddings are lazy-loaded from disk in each DataLoader worker to avoid
    holding all embeddings in RAM simultaneously. The ``num_workers``
    parameter controls how many parallel workers prefetch batches so that
    disk I/O is overlapped with GPU compute.
    """
    # MPS does not reliably support multi-worker data loading
    if device.type == "mps":
        num_workers = 0

    train_loaders: Dict[str, DataLoader] = {}
    valid_loaders: Dict[str, DataLoader] = {}

    for name in task_names:
        dataset = task_datasets[name]

        for split in ["train", "valid"]:
            if split == "train":
                df = dataset.train_df
                shuffle = True
            else:
                df = dataset.valid_df
                shuffle = False

            sequences = list(df["sequence"])
            labels = list(df[dataset.label_column])

            if logger is not None:
                logger.info(
                    f"Computing/loading cached embeddings for {name} {split} "
                    f"({len(sequences)} sequences)"
                )

            embedding_paths = ensure_cached_embedding_paths(
                backbone=model.backbone,
                tokenizer=tokenizer,
                checkpoint=checkpoint,
                sequences=sequences,
                cache_dir=cache_dir,
                device=device,
            )

            cached_dataset = CachedEmbeddingDataset(
                embedding_paths=embedding_paths, labels=labels
            )
            collator = CachedEmbeddingDataCollator()

            loader = DataLoader(
                cached_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                collate_fn=collator,
                num_workers=num_workers,
                persistent_workers=num_workers > 0,
                pin_memory=True,
            )

            if split == "train":
                train_loaders[name] = loader
            else:
                valid_loaders[name] = loader

    return train_loaders, valid_loaders


def build_task_dataloaders(
    model: MultitaskModel,
    tokenizer: PreTrainedTokenizerBase,
    checkpoint: str,
    task_datasets: Dict[str, BaseMultitaskDataset],
    task_names: List[str],
    batch: int,
    use_embedding_cache: bool,
    embedding_cache_dir: Optional[str],
    device: torch.device,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Dict[str, DataLoader], Dict[str, DataLoader]]:
    """Build train and validation dataloaders for all tasks.

    If use_embedding_cache is True and embedding_cache_dir is provided,
    uses cached embeddings for faster training.
    """
    if use_embedding_cache:
        if embedding_cache_dir is None:
            raise ValueError("embedding_cache_dir must be provided when using cache")
        return build_cached_dataloaders(
            model=model,
            tokenizer=tokenizer,
            checkpoint=checkpoint,
            task_datasets=task_datasets,
            task_names=task_names,
            batch_size=batch,
            cache_dir=embedding_cache_dir,
            device=device,
            logger=logger,
        )

    train_loaders: Dict[str, DataLoader] = {
        name: task_datasets[name].get_dataloader(
            split="train",
            backbone=model.backbone,
            tokenizer=tokenizer,
            checkpoint=checkpoint,
            batch_size=batch,
        )
        for name in task_names
    }
    valid_loaders: Dict[str, DataLoader] = {
        name: task_datasets[name].get_dataloader(
            split="valid",
            backbone=model.backbone,
            tokenizer=tokenizer,
            checkpoint=checkpoint,
            batch_size=batch,
        )
        for name in task_names
    }
    return train_loaders, valid_loaders


# Backward compatibility aliases (prefixed with underscore for internal use)
_build_cached_dataloaders = build_cached_dataloaders
_build_task_dataloaders = build_task_dataloaders
