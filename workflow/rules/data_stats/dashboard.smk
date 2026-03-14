def _data_stats_inputs():
    """Collect all dataset parquets for the dashboard stats rule."""
    inputs = {}
    for subset_type, datasets in config.get("standardized_datasets", {}).items():
        for dataset_name in datasets:
            for split in ("train", "valid", "test"):
                key = f"{dataset_name}_{split}"
                inputs[key] = (
                    f"results/datasets_resplit/{subset_type}/{dataset_name}/{split}.parquet"
                )
    return inputs


rule data_stats:
    input:
        **_data_stats_inputs(),
    output:
        directory("results/data_stats/resplit/data")
    params:
        datasets_root="results/datasets_resplit",
        standardized_datasets=config.get("standardized_datasets", {}),
        histogram_bins=config["data_stats"]["histogram_bins"],
    conda:
        "../../envs/data_stats/dashboard/stats.yml"
    log:
        f"{LOG_PREFIX}/data_stats/resplit/dashboard_csvs.log"
    script:
        "../../scripts/data_stats/dashboard/data_stats.py"


rule dash:
    message:
        "Launching Dash app. Press Ctrl+C to stop."
    input:
        "results/data_stats/resplit/data"
    params:
        host = config.get("dash_host", "127.0.0.1"),
        port = config.get("dash_port", 8050)
    conda:
        "../../envs/data_stats/dashboard/stats.yml"
    shell:
        r"""
        exec python workflow/scripts/data_stats/dashboard/dashboard_app.py \
            --data {input} \
            --host {params.host} \
            --port {params.port}
        """


rule data_stats_dashboard:
    input:
        data="results/data_stats/resplit/data",
    output:
        "results/data_stats/resplit/dashboard.html"
    conda:
        "../../envs/data_stats/dashboard/stats.yml"
    log:
        f"{LOG_PREFIX}/data_stats/resplit/dashboard.log"
    script:
        "../../scripts/data_stats/dashboard/data_stats_dashboard.py"


# ---------------------------------------------------------------------------
# Original (pre-resplit) dashboard – same scripts, different data root
# ---------------------------------------------------------------------------

def _data_stats_original_inputs():
    """Collect all dataset parquets from datasets_subsampled/ for the original-splits dashboard."""
    inputs = {}
    for subset_type, datasets in config.get("standardized_datasets", {}).items():
        for dataset_name in datasets:
            for split in ("train", "valid", "test"):
                key = f"{dataset_name}_{split}"
                inputs[key] = (
                    f"results/datasets_subsampled/{subset_type}/{dataset_name}/{split}.parquet"
                )
    return inputs


rule data_stats_original:
    input:
        **_data_stats_original_inputs(),
    output:
        directory("results/data_stats/original/data")
    params:
        datasets_root="results/datasets_subsampled",
        standardized_datasets=config.get("standardized_datasets", {}),
        histogram_bins=config["data_stats"]["histogram_bins"],
    conda:
        "../../envs/data_stats/dashboard/stats.yml"
    log:
        f"{LOG_PREFIX}/data_stats/original/dashboard_csvs.log"
    script:
        "../../scripts/data_stats/dashboard/data_stats.py"


rule data_stats_dashboard_original:
    input:
        data="results/data_stats/original/data",
    output:
        "results/data_stats/original/dashboard.html"
    conda:
        "../../envs/data_stats/dashboard/stats.yml"
    log:
        f"{LOG_PREFIX}/data_stats/original/dashboard.log"
    script:
        "../../scripts/data_stats/dashboard/data_stats_dashboard.py"
