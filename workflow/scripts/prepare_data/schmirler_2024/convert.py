import pickle
from pathlib import Path

import pandas as pd
from snakemake.script import snakemake

from wfutils import get_logger
from wfutils.logging import log_snakemake_info

logger = get_logger()
log_snakemake_info(logger)

pickle_path = Path(snakemake.input["pickle"])
parquet_path = Path(snakemake.output["parquet"])

subsets = list(snakemake.params["schmirler_subsets"])

# Create mapping from dataset name to scoring method
dataset_to_scoring_method = {}
for subset in subsets:
    dataset_to_scoring_method[subset["name"]] = subset["type"]


def standardize_columns(df: pd.DataFrame, scoring_method: str) -> pd.DataFrame:
    """Standardize column names and remove redundant columns"""
    sequence_columns = ["primary", "Sequence", "sequence"]
    for seq_col in sequence_columns:
        if seq_col in df.columns:
            if seq_col != "sequence":
                df = df.rename(columns={seq_col: "sequence"})
            break

    if scoring_method == "per_protein_regression":
        score_mapping = {
            "thermo_score": "score",
        }
        for old_name, new_name in score_mapping.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})
                break

    elif scoring_method == "per_protein_classification":
        label_mapping = {"Location": "label", "loc_num": "label_numeric"}
        for old_name, new_name in label_mapping.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})

    elif scoring_method == "per_residue_regression":
        score_mapping = {
            "disorder": "score",
        }
        for old_name, new_name in score_mapping.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})
                break

    elif scoring_method == "per_residue_classification":
        label_mapping = {"3state": "label"}
        for old_name, new_name in label_mapping.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})

        df["resolved"] = df["resolved"].apply(
            lambda resolved_str: [int(c) for c in resolved_str]
        )

        assert df["resolved"].apply(len).equals(df["label"].apply(len)), (
            "resolved and label must have the same length"
        )

    redundant_columns = [
        "thermo_score_extra",
        "loc_extra",
        "thermo_score_str",
        "loc_num_str",
    ]

    columns_to_drop = [col for col in redundant_columns if col in df.columns]

    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)

    return df


for dataset in pickle_path.iterdir():
    if dataset.is_dir():
        raw_name = dataset.name
        # Config uses lowercase names; raw pickle dirs use original casing
        dataset_name = raw_name.lower()
        scoring_method = dataset_to_scoring_method.get(dataset_name)

        if scoring_method is None:
            logger.warning(
                f"Dataset {raw_name} (-> {dataset_name}) not found in config, skipping"
            )
            continue

        logger.info(
            f"Processing dataset {raw_name} as {dataset_name} with scoring method {scoring_method}"
        )

        for split in dataset.iterdir():
            with open(split, "rb") as f:
                data: pd.DataFrame = pickle.load(f)

                data = standardize_columns(data, scoring_method)

                output_path = parquet_path / scoring_method / dataset_name

                output_path.mkdir(parents=True, exist_ok=True)

                data.to_parquet(output_path / split.with_suffix(".parquet").name)
