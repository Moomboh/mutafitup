"""Rules for fetching and processing NetSurfP-2.0 structural annotation data.

Data source: NetSurfP-2.0 NPZ files from DTU (Klausen et al., 2019)
             + NEW364 CSV from ProtTrans (Elnaggar et al., 2021)

Pipeline:  fetch -> convert -> cluster -> split
                                            |
                          results/datasets_preprocess/netsurfp2/parquet/
                              per_residue_classification/secstr8/{train,valid,test}.parquet
                              per_residue_classification/secstr/{train,valid,test}.parquet
                              per_residue_regression/rsa/{train,valid,test}.parquet
"""


def _netsurfp2_parquet_targets():
    """Return list of all NetSurfP-2.0 parquet output paths."""
    targets = []
    for subset in config.get("datasets", {}).get("netsurfp2", {}).get("subsets", []):
        name = subset["name"]
        subset_type = subset["type"]
        for split in ("train", "valid", "test"):
            targets.append(
                f"results/datasets_preprocess/netsurfp2/parquet/{subset_type}/{name}/{split}.parquet"
            )
    return targets


rule netsurfp2_fetch:
    """Download NetSurfP-2.0 NPZ files from DTU and NEW364 CSV from Dropbox."""
    output:
        raw_dir=directory("results/datasets_preprocess/netsurfp2/raw"),
    params:
        urls=config["datasets"]["netsurfp2"]["urls"],
        hashes=config["datasets"]["netsurfp2"].get("hashes", {}),
    conda:
        "../../envs/prepare_data/fetch.yml"
    log:
        f"{LOG_PREFIX}/prepare_data/netsurfp2/fetch.log"
    script:
        "../../scripts/prepare_data/netsurfp2/fetch.py"


rule netsurfp2_convert:
    """Parse NetSurfP-2.0 NPZ/CSV into standardized parquet format (secstr8 + secstr + rsa)."""
    input:
        raw_dir="results/datasets_preprocess/netsurfp2/raw",
    output:
        secstr8_train_full="results/datasets_preprocess/netsurfp2/parsed/secstr8/train_full.parquet",
        secstr8_test="results/datasets_preprocess/netsurfp2/parsed/secstr8/test.parquet",
        secstr_train_full="results/datasets_preprocess/netsurfp2/parsed/secstr/train_full.parquet",
        secstr_test="results/datasets_preprocess/netsurfp2/parsed/secstr/test.parquet",
        rsa_train_full="results/datasets_preprocess/netsurfp2/parsed/rsa/train_full.parquet",
        rsa_test="results/datasets_preprocess/netsurfp2/parsed/rsa/test.parquet",
    conda:
        "../../envs/prepare_data/process.yml"
    log:
        f"{LOG_PREFIX}/prepare_data/netsurfp2/convert.log"
    script:
        "../../scripts/prepare_data/netsurfp2/convert.py"


rule netsurfp2_cluster:
    """Cluster NetSurfP-2.0 training sequences using MMseqs2."""
    input:
        train_full="results/datasets_preprocess/netsurfp2/parsed/secstr8/train_full.parquet",
    output:
        cluster_tsv="results/datasets_preprocess/netsurfp2/clusters/cluster.tsv",
    params:
        cluster_min_seq_id=config["datasets"]["netsurfp2"]["cluster_min_seq_id"],
    conda:
        "../../envs/prepare_data/process.yml"
    log:
        f"{LOG_PREFIX}/prepare_data/netsurfp2/cluster.log"
    script:
        "../../scripts/prepare_data/netsurfp2/cluster.py"


rule netsurfp2_split:
    """Split NetSurfP-2.0 training data into train/valid using cluster assignments."""
    input:
        secstr8_train_full="results/datasets_preprocess/netsurfp2/parsed/secstr8/train_full.parquet",
        secstr8_test="results/datasets_preprocess/netsurfp2/parsed/secstr8/test.parquet",
        secstr_train_full="results/datasets_preprocess/netsurfp2/parsed/secstr/train_full.parquet",
        secstr_test="results/datasets_preprocess/netsurfp2/parsed/secstr/test.parquet",
        rsa_train_full="results/datasets_preprocess/netsurfp2/parsed/rsa/train_full.parquet",
        rsa_test="results/datasets_preprocess/netsurfp2/parsed/rsa/test.parquet",
        cluster_tsv="results/datasets_preprocess/netsurfp2/clusters/cluster.tsv",
    params:
        valid_cluster_fraction=config["datasets"]["netsurfp2"]["valid_cluster_fraction"],
        random_seed=config["datasets"]["netsurfp2"]["random_seed"],
    output:
        secstr8_train="results/datasets_preprocess/netsurfp2/parquet/per_residue_classification/secstr8/train.parquet",
        secstr8_valid="results/datasets_preprocess/netsurfp2/parquet/per_residue_classification/secstr8/valid.parquet",
        secstr8_test="results/datasets_preprocess/netsurfp2/parquet/per_residue_classification/secstr8/test.parquet",
        secstr_train="results/datasets_preprocess/netsurfp2/parquet/per_residue_classification/secstr/train.parquet",
        secstr_valid="results/datasets_preprocess/netsurfp2/parquet/per_residue_classification/secstr/valid.parquet",
        secstr_test="results/datasets_preprocess/netsurfp2/parquet/per_residue_classification/secstr/test.parquet",
        rsa_train="results/datasets_preprocess/netsurfp2/parquet/per_residue_regression/rsa/train.parquet",
        rsa_valid="results/datasets_preprocess/netsurfp2/parquet/per_residue_regression/rsa/valid.parquet",
        rsa_test="results/datasets_preprocess/netsurfp2/parquet/per_residue_regression/rsa/test.parquet",
    conda:
        "../../envs/prepare_data/process.yml"
    log:
        f"{LOG_PREFIX}/prepare_data/netsurfp2/split.log"
    script:
        "../../scripts/prepare_data/netsurfp2/split.py"
