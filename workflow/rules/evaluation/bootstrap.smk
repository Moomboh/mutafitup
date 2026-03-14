rule bootstrap:
    """Generate protein-level bootstrap indices for a dataset split."""
    input:
        dataset="results/datasets_resplit/{subset_type}/{dataset}/{split}.parquet",
    output:
        bootstraps="results/bootstraps/{subset_type}/{dataset}/{split}.parquet",
    params:
        num_bootstraps=config.get("metrics", {}).get("num_bootstraps", 1000),
        bootstrap_seed=config.get("metrics", {}).get("bootstrap_seed", 42),
    log:
        f"{LOG_PREFIX}/evaluation/bootstraps/{{subset_type}}_{{dataset}}_{{split}}.log",
    script:
        "../../scripts/evaluation/bootstrap.py"
