# ---------------------------------------------------------------------------
# Pre/post-resplit comparison dashboard
# ---------------------------------------------------------------------------

rule data_stats_comparison:
    """Side-by-side comparison of original (pre-resplit) vs resplit dataset statistics."""
    input:
        original="results/data_stats/original/data",
        resplit="results/data_stats/resplit/data",
    output:
        "results/data_stats/comparison/dashboard.html"
    conda:
        "../../envs/data_stats/dashboard/stats.yml"
    log:
        f"{LOG_PREFIX}/data_stats/comparison/dashboard.log"
    script:
        "../../scripts/data_stats/comparison/comparison_dashboard.py"
