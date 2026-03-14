import pandas as pd
import numpy as np
from pathlib import Path
from snakemake.script import snakemake

from wfutils import get_logger
from wfutils.logging import log_snakemake_info

logger = get_logger()
log_snakemake_info(logger)

datasets_root = Path(str(snakemake.params["datasets_root"]))
histogram_bins = dict(snakemake.params["histogram_bins"])
standardized_meta = dict(snakemake.params["standardized_datasets"])

# Map standardized dataset name -> subset_type (per_protein_regression, ...)
dataset_type_by_name: dict[str, str] = {}
for subset_type, datasets in standardized_meta.items():
    for dataset_name in datasets.keys():
        dataset_type_by_name[dataset_name] = subset_type

DEFAULT_SPLITS = ("train", "valid", "test")

# Sentinel value used in per-residue regression datasets (e.g. RSA from NetSurfP-2)
# to mark unresolved/disordered residues.  Must be filtered before computing stats.
# See also: workflow/scripts/prepare_data/netsurfp2/convert.py::UNRESOLVED_SCORE
_UNRESOLVED_SCORE = 999.0

# Create output directory
output_dir = Path(snakemake.output[0])
output_dir.mkdir(parents=True, exist_ok=True)


def _get_dataset_type(dataset_name: str) -> str:
    """Return standardized subset_type for a dataset (e.g. per_protein_regression)."""

    try:
        return dataset_type_by_name[dataset_name]
    except KeyError as e:
        raise KeyError(f"Unknown standardized dataset {dataset_name!r}") from e


def _get_dataset_color(dataset_name: str) -> str:
    dataset_type = _get_dataset_type(dataset_name)
    return (
        standardized_meta.get(dataset_type, {})
        .get(dataset_name, {})
        .get("color", "#636EFA")
    )


def load_all_data():
    """Load all parquet data from results/datasets_resplit and add computed columns.

    Iterates over *all* standardized datasets defined in config.standardized_datasets,
    matching the layout under results/datasets_resplit/<subset_type>/<dataset>/<split>.parquet.
    """

    all_data = {}

    for dataset_name, dataset_type in dataset_type_by_name.items():
        all_data[dataset_name] = {}
        for split in DEFAULT_SPLITS:
            parquet_path = (
                datasets_root / dataset_type / dataset_name / f"{split}.parquet"
            )
            try:
                df = pd.read_parquet(parquet_path)
                df["seq_length"] = df["sequence"].str.len()
                df["split"] = split
                df["dataset"] = dataset_name
                df["scoring_method"] = dataset_type
                all_data[dataset_name][split] = df
                logger.info("Loaded %s/%s: %d rows", dataset_name, split, len(df))
            except Exception as e:
                logger.warning("Could not load %s: %s", parquet_path, e)
                all_data[dataset_name][split] = None

    return all_data


def calculate_statistics(all_data):
    """Calculate general statistics for each dataset"""
    stats_data = []

    for dataset, splits_data in all_data.items():
        for split in DEFAULT_SPLITS:
            df = splits_data.get(split)
            if df is None:
                continue

            stats = {
                "Dataset": dataset,
                "Split": split,
                "Num Sequences": len(df),
                "Min Seq Length": df["seq_length"].min(),
                "Max Seq Length": df["seq_length"].max(),
                "Avg Seq Length": round(df["seq_length"].mean(), 2),
            }

            # Handle per-protein vs per-residue scores
            if "score" in df.columns and not df["score"].isna().all():
                non_na_scores = df["score"].dropna()
                if non_na_scores.empty:
                    num_samples = 0
                    min_score = max_score = avg_score = None
                else:
                    sample_score = non_na_scores.iloc[0]

                    # Per-residue regression: score is an array per sequence
                    if isinstance(sample_score, (np.ndarray, list)):
                        flattened_scores = []
                        for s in non_na_scores:
                            if isinstance(s, (np.ndarray, list)):
                                flattened_scores.extend(list(s))
                            else:
                                flattened_scores.append(s)

                        if flattened_scores:
                            arr = np.asarray(flattened_scores, dtype=float)
                            arr = arr[arr != _UNRESOLVED_SCORE]
                            num_samples = int(arr.size)
                            if num_samples > 0:
                                min_score = round(float(np.nanmin(arr)), 4)
                                max_score = round(float(np.nanmax(arr)), 4)
                                avg_score = round(float(np.nanmean(arr)), 4)
                            else:
                                min_score = max_score = avg_score = None
                        else:
                            num_samples = 0
                            min_score = max_score = avg_score = None

                    # Per-protein regression: scalar score per sequence
                    else:
                        numeric_scores = pd.to_numeric(
                            non_na_scores, errors="coerce"
                        ).dropna()
                        numeric_scores = numeric_scores[
                            numeric_scores != _UNRESOLVED_SCORE
                        ]
                        if numeric_scores.empty:
                            num_samples = 0
                            min_score = max_score = avg_score = None
                        else:
                            num_samples = int(len(numeric_scores))
                            min_score = round(float(numeric_scores.min()), 4)
                            max_score = round(float(numeric_scores.max()), 4)
                            avg_score = round(float(numeric_scores.mean()), 4)

                stats.update(
                    {
                        "Num Samples": num_samples,
                        "Min Score": min_score,
                        "Max Score": max_score,
                        "Avg Score": avg_score,
                    }
                )
            else:
                # No score column: fall back to label-based or sequence-based counting
                if "label" in df.columns and not df["label"].isna().all():
                    sample_label = df["label"].dropna().iloc[0]
                    if isinstance(sample_label, np.ndarray):
                        # Per-residue labels: count total residues
                        total_labels = 0
                        for _, row in df.iterrows():
                            labels_array = row["label"]
                            if isinstance(labels_array, np.ndarray):
                                total_labels += len(labels_array)
                        num_samples = total_labels
                    else:
                        # Per-protein labels: count sequences with labels
                        num_samples = len(df.dropna(subset=["label"]))
                else:
                    # No score/label: fall back to number of sequences
                    num_samples = len(df)

                stats.update(
                    {
                        "Num Samples": num_samples,
                        "Min Score": None,
                        "Max Score": None,
                        "Avg Score": None,
                    }
                )

            stats_data.append(stats)

    return pd.DataFrame(stats_data)


def generate_split_proportions(all_data):
    """Generate split proportions data for pie charts"""
    split_data = []

    for dataset, splits_data in all_data.items():
        color = _get_dataset_color(dataset)

        for split in DEFAULT_SPLITS:
            df = splits_data.get(split)
            if df is not None:
                split_data.append(
                    {
                        "dataset": dataset,
                        "split": split,
                        "count": len(df),
                        "color": color,
                    }
                )

    return pd.DataFrame(split_data)


def generate_sequence_length_data(all_data):
    """Generate sequence length data for histograms and boxplots"""
    seq_data = []

    for dataset, splits_data in all_data.items():
        color = _get_dataset_color(dataset)

        for split in DEFAULT_SPLITS:
            df = splits_data.get(split)
            if df is not None:
                for _, row in df.iterrows():
                    seq_data.append(
                        {
                            "dataset": dataset,
                            "split": split,
                            "seq_length": row["seq_length"],
                            "color": color,
                        }
                    )

    return pd.DataFrame(seq_data)


def generate_score_data(all_data):
    """Generate score data for histograms and boxplots"""
    score_data = []

    for dataset, splits_data in all_data.items():
        color = _get_dataset_color(dataset)

        for split in DEFAULT_SPLITS:
            df = splits_data.get(split)
            if (
                df is not None
                and "score" in df.columns
                and not df["score"].isna().all()
            ):
                non_na_scores = df["score"].dropna()
                if non_na_scores.empty:
                    continue

                sample_score = non_na_scores.iloc[0]

                # Per-residue regression: score is an array per sequence
                if isinstance(sample_score, (np.ndarray, list)):
                    for s in non_na_scores:
                        if isinstance(s, (np.ndarray, list)):
                            for val in s:
                                if float(val) == _UNRESOLVED_SCORE:
                                    continue
                                score_data.append(
                                    {
                                        "dataset": dataset,
                                        "split": split,
                                        "score": float(val),
                                        "color": color,
                                    }
                                )
                        else:
                            if float(s) == _UNRESOLVED_SCORE:
                                continue
                            score_data.append(
                                {
                                    "dataset": dataset,
                                    "split": split,
                                    "score": float(s),
                                    "color": color,
                                }
                            )

                # Per-protein regression: scalar score per sequence
                else:
                    for s in non_na_scores:
                        if pd.isna(s) or float(s) == _UNRESOLVED_SCORE:
                            continue
                        score_data.append(
                            {
                                "dataset": dataset,
                                "split": split,
                                "score": float(s),
                                "color": color,
                            }
                        )

    return pd.DataFrame(score_data)


def generate_label_counts(all_data):
    """Generate label count data for bar charts"""
    label_data = []

    for dataset, splits_data in all_data.items():
        color = _get_dataset_color(dataset)

        for split in DEFAULT_SPLITS:
            df = splits_data.get(split)
            if (
                df is not None
                and "label" in df.columns
                and not df["label"].isna().all()
            ):
                # Handle different label formats
                sample_label = df["label"].iloc[0]
                if isinstance(sample_label, np.ndarray):
                    # Per-residue labels
                    all_labels = []
                    for _, row in df.iterrows():
                        labels_array = row["label"]
                        if isinstance(labels_array, np.ndarray):
                            all_labels.extend(labels_array)

                    unique_labels, counts = np.unique(all_labels, return_counts=True)
                    for label, count in zip(unique_labels, counts):
                        label_data.append(
                            {
                                "dataset": dataset,
                                "split": split,
                                "label": str(label),
                                "count": int(count),
                                "color": color,
                            }
                        )
                else:
                    # Per-protein labels
                    label_counts = df["label"].value_counts()
                    for label, count in label_counts.items():
                        label_data.append(
                            {
                                "dataset": dataset,
                                "split": split,
                                "label": str(label),
                                "count": int(count),
                                "color": color,
                            }
                        )

    return pd.DataFrame(label_data)


def main():
    """Main function to generate CSV files for dashboard"""
    logger.info("Starting CSV generation for dashboard...")

    logger.info("Loading data...")
    all_data = load_all_data()

    logger.info("Calculating statistics...")
    stats_df = calculate_statistics(all_data)
    stats_df.to_csv(output_dir / "statistics.csv", index=False)
    logger.info(f"Saved statistics to {output_dir / 'statistics.csv'}")

    logger.info("Generating split proportions...")
    split_df = generate_split_proportions(all_data)
    split_df.to_csv(output_dir / "split_proportions.csv", index=False)
    logger.info(f"Saved split proportions to {output_dir / 'split_proportions.csv'}")

    logger.info("Generating sequence length data...")
    seq_df = generate_sequence_length_data(all_data)
    seq_df.to_csv(output_dir / "sequence_lengths.csv", index=False)
    logger.info(f"Saved sequence lengths to {output_dir / 'sequence_lengths.csv'}")

    logger.info("Generating score data...")
    score_df = generate_score_data(all_data)
    if not score_df.empty:
        score_df.to_csv(output_dir / "scores.csv", index=False)
        logger.info(f"Saved scores to {output_dir / 'scores.csv'}")
    else:
        logger.info("No score data found, skipping scores.csv")

    logger.info("Generating label counts...")
    label_df = generate_label_counts(all_data)
    if not label_df.empty:
        label_df.to_csv(output_dir / "label_counts.csv", index=False)
        logger.info(f"Saved label counts to {output_dir / 'label_counts.csv'}")
    else:
        logger.info("No label data found, skipping label_counts.csv")

    logger.info("CSV generation completed successfully!")


if __name__ == "__main__":
    main()
