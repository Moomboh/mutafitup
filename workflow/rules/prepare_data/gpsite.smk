"""Rules for fetching and processing GPSite binding-site datasets.

Pipeline:  fetch -> parse -> cluster -> split
                                          |
                        results/datasets_preprocess/gpsite/parquet/per_residue_classification/
                            {dataset_name}/{train,valid,test}.parquet
"""


def _gpsite_ligands():
    """Return sorted list of ligand abbreviations from config."""
    return sorted(
        s["ligand"]
        for s in config.get("datasets", {}).get("gpsite", {}).get("subsets", [])
    )


def _gpsite_ligand_to_name():
    """Return mapping from ligand abbreviation to dataset name."""
    return {
        s["ligand"]: s["name"]
        for s in config.get("datasets", {}).get("gpsite", {}).get("subsets", [])
    }


def _gpsite_name_to_ligand():
    """Return mapping from dataset name to ligand abbreviation."""
    return {
        s["name"]: s["ligand"]
        for s in config.get("datasets", {}).get("gpsite", {}).get("subsets", [])
    }


def _gpsite_names():
    """Return sorted list of dataset names from config."""
    return sorted(
        s["name"]
        for s in config.get("datasets", {}).get("gpsite", {}).get("subsets", [])
    )


def _gpsite_parquet_targets():
    """Return list of all GPSite parquet output paths."""
    targets = []
    for subset in config.get("datasets", {}).get("gpsite", {}).get("subsets", []):
        name = subset["name"]
        for split in ("train", "valid", "test"):
            targets.append(
                f"results/datasets_preprocess/gpsite/parquet/per_residue_classification/{name}/{split}.parquet"
            )
    return targets


rule gpsite_fetch:
    """Download all GPSite raw .txt files from GitHub."""
    output:
        raw_dir=directory("results/datasets_preprocess/gpsite/raw"),
    params:
        commit=config["datasets"]["gpsite"]["commit"],
        base_url=config["datasets"]["gpsite"]["base_url"],
        subsets=config["datasets"]["gpsite"]["subsets"],
        hashes=config["datasets"]["gpsite"].get("hashes", {}),
    conda:
        "../../envs/prepare_data/fetch.yml"
    log:
        f"{LOG_PREFIX}/prepare_data/gpsite/fetch.log"
    script:
        "../../scripts/prepare_data/gpsite/fetch.py"


rule gpsite_parse:
    """Parse raw GPSite text files into parquet for a single ligand type."""
    input:
        raw_dir="results/datasets_preprocess/gpsite/raw",
    output:
        train_full="results/datasets_preprocess/gpsite/parsed/{ligand}/train_full.parquet",
        test="results/datasets_preprocess/gpsite/parsed/{ligand}/test.parquet",
    conda:
        "../../envs/prepare_data/process.yml"
    log:
        f"{LOG_PREFIX}/prepare_data/gpsite/parse_{{ligand}}.log"
    script:
        "../../scripts/prepare_data/gpsite/parse.py"


rule gpsite_cluster:
    """Cluster GPSite training sequences using MMseqs2 at configured identity."""
    input:
        train_full="results/datasets_preprocess/gpsite/parsed/{ligand}/train_full.parquet",
    output:
        cluster_tsv="results/datasets_preprocess/gpsite/clusters/{ligand}/cluster.tsv",
    params:
        cluster_min_seq_id=config["datasets"]["gpsite"]["cluster_min_seq_id"],
    conda:
        "../../envs/prepare_data/process.yml"
    log:
        f"{LOG_PREFIX}/prepare_data/gpsite/cluster_{{ligand}}.log"
    script:
        "../../scripts/prepare_data/gpsite/cluster.py"


rule gpsite_split:
    """Split GPSite training data into train/valid using cluster assignments."""
    input:
        train_full=lambda wc: f"results/datasets_preprocess/gpsite/parsed/{_gpsite_name_to_ligand()[wc.gpsite_name]}/train_full.parquet",
        test=lambda wc: f"results/datasets_preprocess/gpsite/parsed/{_gpsite_name_to_ligand()[wc.gpsite_name]}/test.parquet",
        cluster_tsv=lambda wc: f"results/datasets_preprocess/gpsite/clusters/{_gpsite_name_to_ligand()[wc.gpsite_name]}/cluster.tsv",
    output:
        train="results/datasets_preprocess/gpsite/parquet/per_residue_classification/{gpsite_name}/train.parquet",
        valid="results/datasets_preprocess/gpsite/parquet/per_residue_classification/{gpsite_name}/valid.parquet",
        test="results/datasets_preprocess/gpsite/parquet/per_residue_classification/{gpsite_name}/test.parquet",
    wildcard_constraints:
        gpsite_name="|".join(_gpsite_names()) if _gpsite_names() else "NOMATCH",
    params:
        dataset_name=lambda wc: wc.gpsite_name,
        ligand=lambda wc: _gpsite_name_to_ligand()[wc.gpsite_name],
        valid_cluster_fraction=config["datasets"]["gpsite"]["valid_cluster_fraction"],
        random_seed=config["datasets"]["gpsite"]["random_seed"],
    conda:
        "../../envs/prepare_data/process.yml"
    log:
        f"{LOG_PREFIX}/prepare_data/gpsite/split_{{gpsite_name}}.log"
    script:
        "../../scripts/prepare_data/gpsite/split.py"
