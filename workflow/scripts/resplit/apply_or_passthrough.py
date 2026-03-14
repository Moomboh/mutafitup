"""Apply resplit assignments or symlink from subsampled (passthrough).

For resplit-configured datasets:
- Reads split_assignments.tsv and sequence_metadata.tsv.
- Reads ALL source parquets for this dataset from datasets_subsampled/
  (train + valid + test).
- Filters rows: keeps only sequences whose assignment for this dataset
  matches the requested output split.
- Writes output parquet to results/datasets_resplit/{type}/{name}/{split}.parquet.

For non-resplit datasets:
- Creates a relative symlink from datasets_subsampled/ to datasets/.
"""

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from snakemake.script import snakemake

from wfutils import get_logger
from wfutils.logging import log_snakemake_info

logger = get_logger()
log_snakemake_info(logger)


def passthrough(subsampled_path: Path, out_path: Path):
    """Create a relative symlink from subsampled to final output."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    target = os.path.relpath(subsampled_path.resolve(), out_path.parent.resolve())
    out_path.symlink_to(target)
    logger.info("Passthrough (symlink) -> %s", target)


def _aggregate_duplicates(
    df: pd.DataFrame,
    strategy: str,
    dataset: str,
) -> pd.DataFrame:
    """Aggregate duplicate sequences using the specified strategy.

    For 'mean': groups by sequence and averages the score column.  Works for
    scalar scores (per-protein regression) and array scores (per-residue
    regression, element-wise mean).
    """
    if strategy != "mean":
        raise ValueError(
            f"Unknown aggregation strategy {strategy!r} for dataset {dataset}"
        )

    score_col = "score" if "score" in df.columns else "label"
    n_before = len(df)

    # Check whether scores are array-valued (per-residue) or scalar
    sample = df[score_col].iloc[0]
    if isinstance(sample, (np.ndarray, list)):
        # Per-residue: element-wise mean across duplicate rows
        grouped = df.groupby("sequence", sort=False)
        rows = []
        for seq, group in grouped:
            if len(group) == 1:
                rows.append(group.iloc[0].to_dict())
            else:
                arrays = [np.asarray(v, dtype=float) for v in group[score_col]]
                mean_arr = np.mean(arrays, axis=0)
                row = {"sequence": seq, score_col: mean_arr}
                rows.append(row)
        result = pd.DataFrame(rows)
    else:
        # Scalar: simple groupby mean
        result = df.groupby("sequence", sort=False)[score_col].mean().reset_index()

    n_after = len(result)
    if n_before != n_after:
        logger.info(
            "Aggregated duplicates (%s): %d -> %d rows (%.1f%% reduction)",
            strategy,
            n_before,
            n_after,
            100.0 * (n_before - n_after) / n_before,
        )
    return result


def apply_resplit(
    dataset: str,
    split: str,
    subset_type: str,
    assignments_path: Path,
    metadata_path: Path,
    out_path: Path,
    aggregate_duplicates: dict[str, Any] | None = None,
):
    """Filter source parquets based on split assignments and write output."""
    # Load assignments for this dataset
    assignments_df = pd.read_csv(assignments_path, sep="\t")
    assignments_df["seq_id"] = assignments_df["seq_id"].astype(str)
    ds_assignments = assignments_df[assignments_df["dataset"] == dataset].copy()

    # Get seq_ids assigned to the requested split
    target_seqs = set(
        ds_assignments[ds_assignments["new_split"] == split]["seq_id"].tolist()
    )
    n_assigned = len(target_seqs)

    if not target_seqs:
        # Write an empty parquet with the right schema
        # Read one source parquet to get the schema
        subsampled_dir = Path(f"results/datasets_subsampled/{subset_type}/{dataset}")
        for src_split in ("train", "valid", "test"):
            src_path = subsampled_dir / f"{src_split}.parquet"
            if src_path.exists():
                df = pd.read_parquet(src_path)
                empty_df = df.head(0)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                empty_df.to_parquet(out_path, index=False)
                logger.info(
                    "Resplit: %d assigned, 0 written (empty parquet)",
                    n_assigned,
                )
                return
        raise FileNotFoundError(
            f"No source parquets found for {dataset} in {subsampled_dir}"
        )

    # Load metadata to map seq_id -> sequence
    metadata_df = pd.read_csv(metadata_path, sep="\t")
    metadata_df["seq_id"] = metadata_df["seq_id"].astype(str)
    target_sequences = set(
        metadata_df[metadata_df["seq_id"].isin(target_seqs)]["sequence"].tolist()
    )

    # Read ALL source parquets and filter by sequence
    subsampled_dir = Path(f"results/datasets_subsampled/{subset_type}/{dataset}")
    dfs = []
    for src_split in ("train", "valid", "test"):
        src_path = subsampled_dir / f"{src_split}.parquet"
        if src_path.exists():
            df = pd.read_parquet(src_path)
            # Filter to sequences that are assigned to the target split
            mask = df["sequence"].isin(target_sequences)
            filtered = df[mask]
            if len(filtered) > 0:
                dfs.append(filtered)

    if dfs:
        result_df = pd.concat(dfs, ignore_index=True)

        # Check if this dataset should aggregate duplicates (e.g. mean)
        agg_cfg = (aggregate_duplicates or {}).get(dataset)
        if agg_cfg:
            result_df = _aggregate_duplicates(
                result_df, strategy=agg_cfg["strategy"], dataset=dataset
            )
        else:
            # Drop duplicates where both the sequence and its annotation are
            # identical (same sequence could appear in multiple source splits).
            # Rows with the same sequence but different labels/scores are kept.
            # The label/score column contains lists/arrays (per-residue
            # annotations), which are unhashable — convert to tuples for dedup.
            label_col = "label" if "label" in result_df.columns else "score"
            result_df["_dedup_key"] = result_df[label_col].apply(
                lambda x: tuple(x) if hasattr(x, "__iter__") else x
            )
            result_df = result_df.drop_duplicates(subset=["sequence", "_dedup_key"])
            result_df = result_df.drop(columns=["_dedup_key"])
    else:
        # Fallback: empty dataframe
        src_path = subsampled_dir / "train.parquet"
        result_df = pd.read_parquet(src_path).head(0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(out_path, index=False)
    logger.info(
        "Resplit: %d assigned, %d written",
        n_assigned,
        len(result_df),
    )


def main():
    out_path = Path(snakemake.output.dataset)
    dataset = str(snakemake.wildcards.dataset)
    split = str(snakemake.wildcards.split)
    subset_type = str(snakemake.wildcards.subset_type)
    resplit_datasets = list(snakemake.params.resplit_datasets)
    aggregate_duplicates = dict(snakemake.params.aggregate_duplicates)

    logger.info(
        "[%s/%s] mode=%s",
        dataset,
        split,
        "resplit" if dataset in resplit_datasets else "passthrough",
    )

    if dataset not in resplit_datasets:
        # Passthrough mode
        subsampled_path = Path(snakemake.input.subsampled)
        passthrough(subsampled_path, out_path)
    else:
        # Resplit mode
        assignments_path = Path(snakemake.input.split_assignments)
        metadata_path = Path(snakemake.input.sequence_metadata)
        apply_resplit(
            dataset=dataset,
            split=split,
            subset_type=subset_type,
            assignments_path=assignments_path,
            metadata_path=metadata_path,
            out_path=out_path,
            aggregate_duplicates=aggregate_duplicates,
        )


main()
