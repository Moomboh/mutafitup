"""Export pooled per-protein embeddings for matched Swiss-Prot sequences."""

import json
from pathlib import Path

import pandas as pd
import torch
from snakemake.script import snakemake

from mutafitup.device import get_device
from mutafitup.models import build_backbone_and_tokenizer
from mutafitup.models.multitask_model import MultitaskForwardArgs, MultitaskModel
from wfutils.logging import get_logger, log_snakemake_info

logger = get_logger(__name__)
log_snakemake_info(logger)


def _iter_fasta_records(path: Path):
    header = None
    seq_lines: list[str] = []
    with path.open() as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_lines)
                header = line[1:]
                seq_lines = []
            else:
                seq_lines.append(line.strip())
    if header is not None:
        yield header, "".join(seq_lines)


def _load_tokenizer_for_model(model_path: Path):
    saved_data = torch.load(model_path, map_location="cpu", weights_only=False)
    base_checkpoint = saved_data["config"]["base_checkpoint"]
    backbone_class_path = saved_data["config"]["backbone_class"]
    module_name, attr_name = backbone_class_path.rsplit(".", 1)
    backbone_module = __import__(module_name, fromlist=[attr_name])
    backbone_cls = getattr(backbone_module, attr_name)
    _, tokenizer = backbone_cls.from_pretrained(
        checkpoint=base_checkpoint,
        lora_config=None,
    )
    return tokenizer, base_checkpoint


def _build_frozen_backbone(checkpoint: str, device: torch.device):
    backbone, tokenizer = build_backbone_and_tokenizer(
        checkpoint=checkpoint,
        lora_rank=None,
        lora_alpha=None,
    )
    backbone.to(device)
    backbone.eval()
    return backbone, tokenizer, checkpoint


def _build_trained_backbone(model_dir: Path, device: torch.device):
    model_path = model_dir / "model.pt"
    model = MultitaskModel.load_from_file(model_path, device=device)
    model.eval()
    tokenizer, base_checkpoint = _load_tokenizer_for_model(model_path)
    return model.backbone, tokenizer, base_checkpoint


def _build_valid_token_mask(
    attention_mask: torch.Tensor, tokenized: dict
) -> torch.Tensor:
    valid_mask = attention_mask.bool()
    special_tokens_mask = tokenized.get("special_tokens_mask")
    if special_tokens_mask is not None:
        valid_mask = valid_mask & ~special_tokens_mask.to(attention_mask.device).bool()
        fallback = valid_mask.sum(dim=1) == 0
        if torch.any(fallback):
            valid_mask[fallback] = attention_mask[fallback].bool()
    return valid_mask


def _pool_hidden_states(
    hidden_states: torch.Tensor, valid_mask: torch.Tensor
) -> torch.Tensor:
    weights = valid_mask.unsqueeze(-1).to(hidden_states.dtype)
    summed = (hidden_states * weights).sum(dim=1)
    counts = weights.sum(dim=1).clamp(min=1)
    return summed / counts


def main() -> None:
    matched_sequences_path = Path(str(snakemake.input["matched_sequences"]))
    matched_entries_path = Path(str(snakemake.input["matched_entries"]))
    manifest_path = Path(str(snakemake.output["manifest"]))
    vectors_dir = Path(str(snakemake.output["vectors"]))
    summary_path = Path(str(snakemake.output["summary"]))
    spec = dict(snakemake.params["spec"])
    batch_size = int(snakemake.params["batch_size"])

    device = get_device()
    logger.info("Using device: %s", device)

    if spec["model_kind"] == "frozen":
        backbone, tokenizer, checkpoint = _build_frozen_backbone(
            spec["checkpoint"], device
        )
    else:
        model_dir = Path(str(snakemake.input["model_dir"]))
        backbone, tokenizer, checkpoint = _build_trained_backbone(model_dir, device)

    matched_entries = pd.read_csv(matched_entries_path, sep="\t")
    unique_entries = matched_entries.groupby("accession", as_index=False).agg(
        entry_name=("entry_name", "first"),
        protein_names=("protein_names", "first"),
        gene_names=("gene_names", "first"),
        organism=("organism", "first"),
        go_ids=("go_ids", "first"),
        ec_numbers=("ec_numbers", "first"),
        interpro_ids=("interpro_ids", "first"),
        pfam_ids=("pfam_ids", "first"),
        query_count=("query_id", "nunique"),
        datasets=(
            "datasets",
            lambda s: ",".join(
                sorted({item for cell in s for item in str(cell).split(",") if item})
            ),
        ),
    )

    sequences = {}
    for header, sequence in _iter_fasta_records(matched_sequences_path):
        accession = header.split("|", 1)[0]
        sequences[accession] = sequence

    unique_entries["sequence"] = unique_entries["accession"].map(sequences)
    missing = unique_entries[unique_entries["sequence"].isna()]
    if not missing.empty:
        raise ValueError(
            f"Missing matched sequence for {len(missing)} accessions; first few: "
            f"{missing['accession'].tolist()[:10]}"
        )

    vectors_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    embedding_dim = None

    with torch.no_grad():
        for start in range(0, len(unique_entries), batch_size):
            batch = unique_entries.iloc[start : start + batch_size].copy()
            batch_sequences = batch["sequence"].tolist()
            processed_sequences = backbone.preprocess_sequences(
                batch_sequences, checkpoint
            )
            tokenized = tokenizer(
                processed_sequences,
                padding=True,
                truncation=True,
                max_length=1024,
                return_tensors="pt",
                return_special_tokens_mask=True,
            )
            input_ids = tokenized["input_ids"].to(device)
            attention_mask = tokenized["attention_mask"].to(device)
            args = MultitaskForwardArgs(
                input_ids=input_ids, attention_mask=attention_mask
            )
            hidden_states = backbone.forward(args)
            valid_mask = _build_valid_token_mask(attention_mask, tokenized)
            pooled = _pool_hidden_states(hidden_states, valid_mask).cpu()

            if embedding_dim is None:
                embedding_dim = int(pooled.shape[1])

            for row, embedding in zip(
                batch.to_dict(orient="records"), pooled, strict=True
            ):
                vector_path = vectors_dir / f"{row['accession']}.pt"
                torch.save(embedding, vector_path)
                manifest_rows.append(
                    {
                        **row,
                        "model_key": spec["model_key"],
                        "model_kind": spec["model_kind"],
                        "category": spec["category"],
                        "checkpoint": checkpoint,
                        "source_section": spec["section"],
                        "source_run_id": spec["run_id"],
                        "embedding_dim": int(embedding.shape[0]),
                        "embedding_path": str(
                            vector_path.relative_to(manifest_path.parent)
                        ),
                    }
                )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(manifest_rows).to_csv(manifest_path, sep="\t", index=False)
    summary_path.write_text(
        json.dumps(
            {
                "model_key": spec["model_key"],
                "model_kind": spec["model_kind"],
                "category": spec["category"],
                "checkpoint": checkpoint,
                "source_section": spec["section"],
                "source_run_id": spec["run_id"],
                "num_sequences": len(manifest_rows),
                "embedding_dim": embedding_dim,
            },
            indent=2,
        )
    )
    logger.info("Wrote %d pooled embeddings to %s", len(manifest_rows), manifest_path)


if __name__ == "__main__":
    main()
