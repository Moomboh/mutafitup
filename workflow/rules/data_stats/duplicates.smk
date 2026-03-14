# ---------------------------------------------------------------------------
# Duplicate sequence analysis – post-resplit (results/datasets_resplit/)
# ---------------------------------------------------------------------------

rule duplicate_stats:
    input:
        **_data_stats_inputs(),
    output:
        directory("results/data_stats/duplicates/resplit/data")
    params:
        datasets_root="results/datasets_resplit",
        standardized_datasets=config.get("standardized_datasets", {}),
    conda:
        "../../envs/data_stats/dashboard/stats.yml"
    log:
        f"{LOG_PREFIX}/data_stats/duplicates/resplit/duplicate_stats.log"
    script:
        "../../scripts/data_stats/duplicates/duplicate_stats.py"


rule duplicate_dashboard:
    input:
        data="results/data_stats/duplicates/resplit/data",
    output:
        "results/data_stats/duplicates/resplit/dashboard.html"
    conda:
        "../../envs/data_stats/dashboard/stats.yml"
    log:
        f"{LOG_PREFIX}/data_stats/duplicates/resplit/duplicate_dashboard.log"
    script:
        "../../scripts/data_stats/duplicates/duplicate_dashboard.py"


# ---------------------------------------------------------------------------
# Duplicate sequence analysis – pre-resplit (results/datasets_subsampled/)
# ---------------------------------------------------------------------------

rule duplicate_stats_original:
    input:
        **_data_stats_original_inputs(),
    output:
        directory("results/data_stats/duplicates/original/data")
    params:
        datasets_root="results/datasets_subsampled",
        standardized_datasets=config.get("standardized_datasets", {}),
    conda:
        "../../envs/data_stats/dashboard/stats.yml"
    log:
        f"{LOG_PREFIX}/data_stats/duplicates/original/duplicate_stats.log"
    script:
        "../../scripts/data_stats/duplicates/duplicate_stats.py"


rule duplicate_dashboard_original:
    input:
        data="results/data_stats/duplicates/original/data",
    output:
        "results/data_stats/duplicates/original/dashboard.html"
    conda:
        "../../envs/data_stats/dashboard/stats.yml"
    log:
        f"{LOG_PREFIX}/data_stats/duplicates/original/duplicate_dashboard.log"
    script:
        "../../scripts/data_stats/duplicates/duplicate_dashboard.py"
