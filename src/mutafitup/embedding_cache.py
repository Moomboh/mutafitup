import hashlib
import os
from pathlib import Path
from typing import Optional, Tuple

import torch


def _compute_sequence_hash(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()[:16]


def _normalize_checkpoint_name(checkpoint: str) -> str:
    return checkpoint.replace("/", "_").replace("\\", "_")


def get_cache_path(
    cache_dir: str,
    checkpoint: str,
    sequence: str,
) -> Path:
    checkpoint_normalized = _normalize_checkpoint_name(checkpoint)
    sequence_hash = _compute_sequence_hash(sequence)
    return Path(cache_dir) / checkpoint_normalized / f"{sequence_hash}.pt"


def load_cached_embedding(
    cache_dir: str,
    checkpoint: str,
    sequence: str,
) -> Optional[torch.Tensor]:
    cache_path = get_cache_path(cache_dir, checkpoint, sequence)
    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu", weights_only=True)
    return None


def save_cached_embedding(
    cache_dir: str,
    checkpoint: str,
    sequence: str,
    embedding: torch.Tensor,
) -> Path:
    cache_path = get_cache_path(cache_dir, checkpoint, sequence)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embedding.cpu(), cache_path)
    return cache_path


def compute_and_cache_embeddings(
    backbone: "MultitaskBackbone",
    tokenizer: "PreTrainedTokenizerBase",
    checkpoint: str,
    sequences: list[str],
    cache_dir: str,
    device: Optional[torch.device] = None,
    batch_size: int = 8,
) -> list[torch.Tensor]:
    from mutafitup.models.multitask_model import MultitaskBackbone, MultitaskForwardArgs

    embeddings = []
    to_compute_indices = []
    to_compute_sequences = []

    for i, seq in enumerate(sequences):
        cached = load_cached_embedding(cache_dir, checkpoint, seq)
        if cached is not None:
            embeddings.append((i, cached))
        else:
            to_compute_indices.append(i)
            to_compute_sequences.append(seq)

    if to_compute_sequences:
        backbone_device = device or next(backbone.parameters()).device
        backbone.eval()

        with torch.no_grad():
            for batch_start in range(0, len(to_compute_sequences), batch_size):
                batch_end = min(batch_start + batch_size, len(to_compute_sequences))
                batch_seqs = to_compute_sequences[batch_start:batch_end]
                batch_indices = to_compute_indices[batch_start:batch_end]

                preprocessed = backbone.preprocess_sequences(batch_seqs, checkpoint)
                tokenized = tokenizer(
                    preprocessed,
                    max_length=1024,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )

                input_ids = tokenized["input_ids"].to(backbone_device)
                attention_mask = tokenized["attention_mask"].to(backbone_device)

                args = MultitaskForwardArgs(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                hidden_states = backbone.forward(args)

                for j, (idx, seq) in enumerate(zip(batch_indices, batch_seqs)):
                    seq_len = attention_mask[j].sum().item()
                    emb = hidden_states[j, :seq_len].cpu()
                    save_cached_embedding(cache_dir, checkpoint, seq, emb)
                    embeddings.append((idx, emb))

    embeddings.sort(key=lambda x: x[0])
    return [emb for _, emb in embeddings]


def ensure_cached_embedding_paths(
    backbone: "MultitaskBackbone",
    tokenizer: "PreTrainedTokenizerBase",
    checkpoint: str,
    sequences: list[str],
    cache_dir: str,
    device: Optional[torch.device] = None,
    batch_size: int = 8,
) -> list[Path]:
    """Ensure all embeddings are computed and cached, returning file paths.

    Like compute_and_cache_embeddings, but returns a list of Path objects
    pointing to the cached .pt files instead of loading all tensors into
    memory. This enables lazy-loading in the dataset to avoid OOM when
    many tasks share large embedding sets.
    """
    from mutafitup.models.multitask_model import MultitaskBackbone, MultitaskForwardArgs

    to_compute_indices = []
    to_compute_sequences = []

    for i, seq in enumerate(sequences):
        cache_path = get_cache_path(cache_dir, checkpoint, seq)
        if not cache_path.exists():
            to_compute_indices.append(i)
            to_compute_sequences.append(seq)

    if to_compute_sequences:
        backbone_device = device or next(backbone.parameters()).device
        backbone.eval()

        with torch.no_grad():
            for batch_start in range(0, len(to_compute_sequences), batch_size):
                batch_end = min(batch_start + batch_size, len(to_compute_sequences))
                batch_seqs = to_compute_sequences[batch_start:batch_end]
                batch_indices = to_compute_indices[batch_start:batch_end]

                preprocessed = backbone.preprocess_sequences(batch_seqs, checkpoint)
                tokenized = tokenizer(
                    preprocessed,
                    max_length=1024,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )

                input_ids = tokenized["input_ids"].to(backbone_device)
                attention_mask = tokenized["attention_mask"].to(backbone_device)

                args = MultitaskForwardArgs(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                hidden_states = backbone.forward(args)

                for j, (idx, seq) in enumerate(zip(batch_indices, batch_seqs)):
                    seq_len = attention_mask[j].sum().item()
                    emb = hidden_states[j, :seq_len].cpu()
                    save_cached_embedding(cache_dir, checkpoint, seq, emb)

    return [get_cache_path(cache_dir, checkpoint, seq) for seq in sequences]


def get_default_cache_dir() -> str:
    return os.path.join(os.path.expanduser("~"), ".cache", "mutafitup", "embeddings")
