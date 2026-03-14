from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from torch.utils.data import Dataset


class CachedEmbeddingDataset(Dataset):
    """Dataset that lazy-loads pre-computed embeddings from disk.

    Embeddings are stored as individual .pt files and loaded on-demand in
    __getitem__ to avoid holding all embeddings in RAM simultaneously.
    This is critical when training across many tasks with overlapping
    sequences, where the total embedding memory would otherwise exceed
    available RAM.
    """

    def __init__(
        self,
        embedding_paths: List[Path],
        labels: List[Union[int, float, List[int], List[float]]],
    ):
        if len(embedding_paths) != len(labels):
            raise ValueError(
                f"Number of embedding paths ({len(embedding_paths)}) must match "
                f"number of labels ({len(labels)})"
            )
        self.embedding_paths = embedding_paths
        self.labels = labels

    def __len__(self) -> int:
        return len(self.embedding_paths)

    def __getitem__(self, idx: int) -> dict:
        embedding = torch.load(
            self.embedding_paths[idx], map_location="cpu", weights_only=True
        )
        return {
            "embeddings": embedding,
            "labels": self.labels[idx],
        }


class CachedEmbeddingDataCollator:
    def __init__(self, label_pad_token_id: int = -100):
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features: List[dict]) -> dict:
        embeddings = [f["embeddings"] for f in features]
        labels = [f["labels"] for f in features]

        max_len = max(emb.size(0) for emb in embeddings)
        hidden_size = embeddings[0].size(-1)

        first_embedding = embeddings[0]
        padded_embeddings = torch.zeros(
            len(embeddings),
            max_len,
            hidden_size,
            dtype=first_embedding.dtype,
            device=first_embedding.device,
        )
        attention_mask = torch.zeros(
            len(embeddings),
            max_len,
            dtype=torch.long,
            device=first_embedding.device,
        )

        for i, emb in enumerate(embeddings):
            seq_len = emb.size(0)
            padded_embeddings[i, :seq_len] = emb
            attention_mask[i, :seq_len] = 1

        first_label = labels[0]
        is_sequence_label = isinstance(first_label, (list, tuple, np.ndarray)) or (
            isinstance(first_label, torch.Tensor) and first_label.dim() > 0
        )

        if is_sequence_label:
            padded_labels = []
            for i, label in enumerate(labels):
                if isinstance(label, (torch.Tensor, np.ndarray)):
                    label = label.tolist()
                emb_len = embeddings[i].size(0)
                label = list(label)[:emb_len]
                label_len = len(label)
                padding_len = max_len - label_len
                padded_label = label + [self.label_pad_token_id] * padding_len
                padded_labels.append(padded_label)

            if isinstance(first_label, torch.Tensor):
                dtype = first_label.dtype
            elif isinstance(first_label, np.ndarray):
                dtype = (
                    torch.float
                    if np.issubdtype(first_label.dtype, np.floating)
                    else torch.long
                )
            elif isinstance(first_label[0], float):
                dtype = torch.float
            else:
                dtype = torch.long

            labels_tensor = torch.tensor(padded_labels, dtype=dtype)
        else:
            if isinstance(first_label, torch.Tensor):
                if first_label.dtype in (torch.int32, torch.int64):
                    labels_tensor = torch.tensor(
                        [int(l) for l in labels], dtype=torch.long
                    )
                else:
                    labels_tensor = torch.tensor(
                        [float(l) for l in labels], dtype=torch.float
                    )
            elif isinstance(first_label, float):
                labels_tensor = torch.tensor(labels, dtype=torch.float)
            else:
                labels_tensor = torch.tensor(labels, dtype=torch.long)

        return {
            "embeddings": padded_embeddings,
            "attention_mask": attention_mask,
            "labels": labels_tensor,
        }
