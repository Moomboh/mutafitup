"""Generate duplicate-sequence analysis CSVs for each standardized dataset.

Outputs (written to the directory specified by snakemake.output[0]):
  duplicate_summary.csv  – one row per (dataset, split)
  duplicate_groups.csv   – one row per duplicate group
"""

import numpy as np
import pandas as pd
from pathlib import Path
from snakemake.script import snakemake

from wfutils import get_logger
from wfutils.logging import log_snakemake_info

logger = get_logger()
log_snakemake_info(logger)

datasets_root = Path(str(snakemake.params["datasets_root"]))
standardized_meta = dict(snakemake.params["standardized_datasets"])

# Map dataset name -> subset_type
dataset_type_by_name: dict[str, str] = {}
for subset_type, datasets in standardized_meta.items():
    for dataset_name in datasets.keys():
        dataset_type_by_name[dataset_name] = subset_type

DEFAULT_SPLITS = ("train", "valid", "test")

# Sentinel value used in per-residue regression datasets (e.g. RSA from NetSurfP-2)
# to mark unresolved/disordered residues.  Must be filtered before computing stats.
# See also: workflow/scripts/prepare_data/netsurfp2/convert.py::UNRESOLVED_SCORE
_UNRESOLVED_SCORE = 999.0

output_dir = Path(snakemake.output[0])
output_dir.mkdir(parents=True, exist_ok=True)


def _is_regression(subset_type: str) -> bool:
    return "regression" in subset_type


def _make_dedup_key(series: pd.Series) -> pd.Series:
    """Convert a label/score column to a hashable series for grouping."""
    return series.apply(lambda x: tuple(x) if hasattr(x, "__iter__") else x)


def _score_stats_for_group(scores) -> dict:
    """Compute score statistics for a duplicate group (regression only).

    *scores* is a list/Series of per-row score values. Each value may be a
    scalar (per-protein regression) or an array (per-residue regression).
    For per-residue we compute the **per-protein mean** first (excluding
    unresolved sentinel values), then derive group-level statistics over
    those means.
    """
    values = []
    for s in scores:
        if isinstance(s, (np.ndarray, list)):
            arr = np.asarray(s, dtype=float)
            arr = arr[arr != _UNRESOLVED_SCORE]
            if arr.size > 0:
                values.append(float(np.nanmean(arr)))
        else:
            if float(s) != _UNRESOLVED_SCORE:
                values.append(float(s))

    if not values:
        return {
            "score_mean": None,
            "score_std": None,
            "score_min": None,
            "score_max": None,
            "score_range": None,
        }

    arr = np.array(values, dtype=float)
    return {
        "score_mean": round(float(np.nanmean(arr)), 6),
        "score_std": round(float(np.nanstd(arr, ddof=0)), 6),
        "score_min": round(float(np.nanmin(arr)), 6),
        "score_max": round(float(np.nanmax(arr)), 6),
        "score_range": round(float(np.nanmax(arr) - np.nanmin(arr)), 6),
    }


def _labels_identical(labels) -> bool:
    """Return True if all label values in the group are identical."""
    items = list(labels)
    if len(items) <= 1:
        return True
    first = items[0]
    for item in items[1:]:
        if isinstance(first, np.ndarray) and isinstance(item, np.ndarray):
            if not np.array_equal(first, item):
                return False
        elif isinstance(first, np.ndarray) or isinstance(item, np.ndarray):
            return False
        else:
            if first != item:
                return False
    return True


def analyse_dataset_split(
    dataset_name: str,
    split: str,
    subset_type: str,
) -> tuple[dict | None, list[dict]]:
    """Analyse duplicates in a single dataset split.

    Returns (summary_row, list_of_group_rows).
    """
    parquet_path = datasets_root / subset_type / dataset_name / f"{split}.parquet"
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        logger.warning("Could not load %s: %s", parquet_path, e)
        return None, []

    total_rows = len(df)
    if total_rows == 0:
        return None, []

    unique_sequences = df["sequence"].nunique()

    # Find duplicate groups (same sequence appearing more than once)
    seq_counts = df["sequence"].value_counts()
    dup_seqs = seq_counts[seq_counts > 1]
    n_dup_groups = len(dup_seqs)
    n_dup_rows = int(dup_seqs.sum())
    n_extra_rows = n_dup_rows - n_dup_groups  # rows beyond the first occurrence
    dup_pct = round(100.0 * n_dup_rows / total_rows, 2) if total_rows > 0 else 0.0

    regression = _is_regression(subset_type)
    label_col = "score" if regression else "label"

    # --- summary row ---
    summary: dict = {
        "dataset": dataset_name,
        "split": split,
        "subset_type": subset_type,
        "total_rows": total_rows,
        "unique_sequences": unique_sequences,
        "duplicate_groups": n_dup_groups,
        "duplicate_rows": n_dup_rows,
        "extra_rows": n_extra_rows,
        "duplicate_pct": dup_pct,
    }

    if n_dup_groups > 0:
        summary["mean_group_size"] = round(float(dup_seqs.mean()), 2)
        summary["max_group_size"] = int(dup_seqs.max())
    else:
        summary["mean_group_size"] = 0.0
        summary["max_group_size"] = 0

    # Score-range statistics across duplicate groups (regression only)
    if regression and n_dup_groups > 0 and label_col in df.columns:
        score_ranges = []
        score_stds = []
        for seq, group_df in df[df["sequence"].isin(dup_seqs.index)].groupby(
            "sequence"
        ):
            stats = _score_stats_for_group(group_df[label_col])
            score_ranges.append(stats["score_range"])
            score_stds.append(stats["score_std"])
        summary["mean_score_range"] = round(float(np.mean(score_ranges)), 6)
        summary["max_score_range"] = round(float(np.max(score_ranges)), 6)
        summary["mean_score_std"] = round(float(np.mean(score_stds)), 6)
    else:
        summary["mean_score_range"] = None
        summary["max_score_range"] = None
        summary["mean_score_std"] = None

    # --- per-group rows ---
    group_rows: list[dict] = []
    if n_dup_groups > 0:
        for seq, group_df in df[df["sequence"].isin(dup_seqs.index)].groupby(
            "sequence"
        ):
            row: dict = {
                "dataset": dataset_name,
                "split": split,
                "seq_length": len(str(seq)),
                "count": len(group_df),
            }
            if regression and label_col in group_df.columns:
                stats = _score_stats_for_group(group_df[label_col])
                row.update(stats)
                row["labels_identical"] = None
            else:
                row["score_mean"] = None
                row["score_std"] = None
                row["score_min"] = None
                row["score_max"] = None
                row["score_range"] = None
                if label_col in group_df.columns:
                    row["labels_identical"] = _labels_identical(
                        group_df[label_col].tolist()
                    )
                else:
                    row["labels_identical"] = None
            group_rows.append(row)

    logger.info(
        "%s/%s: %d rows, %d unique seqs, %d dup groups (%d rows, %.1f%%)",
        dataset_name,
        split,
        total_rows,
        unique_sequences,
        n_dup_groups,
        n_dup_rows,
        dup_pct,
    )

    return summary, group_rows


def main():
    logger.info("Starting duplicate analysis...")

    all_summaries: list[dict] = []
    all_groups: list[dict] = []

    for dataset_name, subset_type in sorted(dataset_type_by_name.items()):
        for split in DEFAULT_SPLITS:
            summary, groups = analyse_dataset_split(dataset_name, split, subset_type)
            if summary is not None:
                all_summaries.append(summary)
            all_groups.extend(groups)

    # Write CSVs
    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(output_dir / "duplicate_summary.csv", index=False)
    logger.info("Saved duplicate_summary.csv (%d rows)", len(summary_df))

    groups_df = pd.DataFrame(all_groups)
    groups_df.to_csv(output_dir / "duplicate_groups.csv", index=False)
    logger.info("Saved duplicate_groups.csv (%d rows)", len(groups_df))

    # Log overall statistics
    total_dup_groups = summary_df["duplicate_groups"].sum()
    total_extra = summary_df["extra_rows"].sum()
    total_rows = summary_df["total_rows"].sum()
    logger.info(
        "Overall: %d duplicate groups, %d extra rows out of %d total (%.1f%%)",
        total_dup_groups,
        total_extra,
        total_rows,
        100.0 * total_extra / total_rows if total_rows > 0 else 0.0,
    )

    logger.info("Duplicate analysis complete.")


main()
