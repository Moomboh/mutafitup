"""Rules for subsampling (dev) or symlinking (prod) full datasets.

When ``config.subsample`` is defined, each parquet file in
``results/datasets/`` is subsampled and written to
``results/datasets_subsampled/``.  When the key is absent the output file
is a symlink to the full parquet, adding zero overhead.
"""


rule subsample_dataset:
    """Subsample or symlink a single dataset parquet file."""
    input:
        full="results/datasets/{subset_type}/{dataset}/{split}.parquet",
    output:
        dataset="results/datasets_subsampled/{subset_type}/{dataset}/{split}.parquet",
    params:
        train_frac=config.get("subsample", {}).get("train_frac", None),
        valid_frac=config.get("subsample", {}).get("valid_frac", None),
        test_frac=config.get("subsample", {}).get("test_frac", None),
        min_rows=config.get("subsample", {}).get("min_rows", 8),
        random_seed=config.get("subsample", {}).get("random_seed", None),
    conda:
        "../envs/resplit/resplit.yml"
    log:
        f"{LOG_PREFIX}/subsample_dataset/{{subset_type}}/{{dataset}}/{{split}}.log"
    script:
        "../scripts/subsample_dataset.py"
