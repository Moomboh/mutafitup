include: "prepare_data/schmirler_2024.smk"
include: "prepare_data/gpsite.smk"
include: "prepare_data/netsurfp2.smk"


def get_standardized_parquet_targets():
    """Concrete dataset parquet targets produced by this workflow.

    We derive targets from enabled dataset sources (config.datasets) and
    training tasks so dev configs that narrow datasets don't expect files
    from the base config.
    """

    splits = ("train", "valid", "test")
    targets = []

    datasets_by_type = {}

    for source in ("schmirler_2024", "gpsite", "netsurfp2"):
        source_subsets = (
            config.get("datasets", {})
            .get(source, {})
            .get("subsets", [])
        )
        for subset in source_subsets:
            subset_type = subset.get("type")
            subset_name = subset.get("name")
            if not subset_type or not subset_name:
                continue
            datasets_by_type.setdefault(subset_type, set()).add(subset_name)

    for section in ("heads_only", "lora", "accgrad_lora", "align_lora"):
        for run in config.get("train", {}).get(section, []):
            for task_name in run.get("tasks", []):
                task = expand_task(task_name)
                subset_type = task["subset_type"]
                dataset_name = task["dataset"]
                datasets_by_type.setdefault(subset_type, set()).add(dataset_name)

    for subset_type in sorted(datasets_by_type.keys()):
        for dataset_name in sorted(datasets_by_type[subset_type]):
            for split in splits:
                targets.append(
                    f"results/datasets/{subset_type}/{dataset_name}/{split}.parquet"
                )

    return targets


rule collect_datasets:
    input:
        schmirler_parquet="results/datasets_preprocess/schmirler_2024/parquet",
        gpsite_parquets=_gpsite_parquet_targets(),
        netsurfp2_parquets=_netsurfp2_parquet_targets(),
    output:
        # Root directory that will hold all full (unsubsampled) parquet datasets
        datasets_root=directory("results/datasets"),
        # Explicit list of standardized parquet files produced inside datasets_root.
        standardized_parquets=get_standardized_parquet_targets(),
    params:
        schmirler_subsets=config["datasets"]["schmirler_2024"]["subsets"],
    conda:
        "../envs/resplit/resplit.yml"
    log:
        f"{LOG_PREFIX}/prepare_data/collect_datasets.log"
    script:
        "../scripts/prepare_data/collect_datasets.py"
