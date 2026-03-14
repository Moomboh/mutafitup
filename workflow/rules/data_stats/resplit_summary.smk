# ---------------------------------------------------------------------------
# Resplit summary dashboard – visualises resplit_summary.json
# ---------------------------------------------------------------------------

rule resplit_summary_dashboard:
    input:
        json="results/resplit/resplit_summary.json",
    output:
        "results/data_stats/resplit_summary/dashboard.html"
    conda:
        "../../envs/data_stats/dashboard/stats.yml"
    log:
        f"{LOG_PREFIX}/data_stats/resplit_summary/dashboard.log"
    script:
        "../../scripts/data_stats/resplit_summary/resplit_summary_dashboard.py"
