from pathlib import Path
import shutil

from snakemake.script import snakemake

from wfutils import get_logger
from wfutils.logging import log_snakemake_info


logger = get_logger()
log_snakemake_info(logger)


schmirler_parquet = Path(str(snakemake.input["schmirler_parquet"]))
gpsite_parquets = [Path(str(p)) for p in snakemake.input["gpsite_parquets"]]
netsurfp2_parquets = [Path(str(p)) for p in snakemake.input["netsurfp2_parquets"]]
datasets_root = Path(str(snakemake.output["datasets_root"]))


DEFAULT_SPLITS = ("train", "valid", "test")


# Copy existing schmirler_2024 parquets into the datasets root
schmirler_subsets = list(snakemake.params["schmirler_subsets"])
for subset in schmirler_subsets:
    subset_name = subset["name"]
    subset_type = subset["type"]

    for split in DEFAULT_SPLITS:
        src = schmirler_parquet / subset_type / subset_name / f"{split}.parquet"
        if not src.exists():
            logger.warning("schmirler_2024 parquet missing: %s", src)
            continue

        dst_dir = datasets_root / subset_type / subset_name
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name
        logger.info("Copying %s -> %s", src, dst)
        shutil.copy2(src, dst)


# Copy GPSite parquets into the datasets root
# GPSite parquets are already at results/datasets_preprocess/gpsite/parquet/{type}/{name}/{split}.parquet
# and are listed explicitly via the input list.
for src in gpsite_parquets:
    # Extract type/name/split from the path:
    # .../results/datasets_preprocess/gpsite/parquet/per_residue_classification/GPSite_DNA/train.parquet
    split_name = src.name  # e.g. "train.parquet"
    dataset_name = src.parent.name  # e.g. "GPSite_DNA"
    subset_type = src.parent.parent.name  # e.g. "per_residue_classification"

    dst_dir = datasets_root / subset_type / dataset_name
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / split_name
    logger.info("Copying %s -> %s", src, dst)
    shutil.copy2(src, dst)


# Copy NetSurfP-2.0 parquets into the datasets root
# NetSurfP-2.0 parquets are at results/datasets_preprocess/netsurfp2/parquet/{type}/{name}/{split}.parquet
for src in netsurfp2_parquets:
    # Extract type/name/split from the path:
    # .../results/datasets_preprocess/netsurfp2/parquet/per_residue_classification/SecStr8/train.parquet
    split_name = src.name  # e.g. "train.parquet"
    dataset_name = src.parent.name  # e.g. "SecStr8"
    subset_type = src.parent.parent.name  # e.g. "per_residue_classification"

    dst_dir = datasets_root / subset_type / dataset_name
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / split_name
    logger.info("Copying %s -> %s", src, dst)
    shutil.copy2(src, dst)
