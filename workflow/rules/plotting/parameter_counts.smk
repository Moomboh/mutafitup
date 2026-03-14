def _get_parameter_run_entries():
    entries = []
    for section, run in iter_train_runs():
        entries.append(
            {
                "section": section,
                "id": run["id"],
                "checkpoint": run["checkpoint"],
                "dropout": run["dropout"],
                "lora_rank": run.get("lora_rank"),
                "lora_alpha": run.get("lora_alpha"),
                "tasks": expand_tasks(run["tasks"]),
            }
        )
    return entries


def _get_parameter_include_run_keys():
    return config.get("parameter_counts", {}).get("include_runs", [])


rule plot_parameter_counts:
    output:
        summary_json="results/plots/parameters/summary.json",
        summary_csv="results/plots/parameters/summary.csv",
        summary_typst="results/thesis_tables/parameters/summary.typ",
        plot="results/plots/parameters/trainable_vs_frozen.png",
    params:
        run_entries=_get_parameter_run_entries(),
        include_run_keys=_get_parameter_include_run_keys(),
    conda:
        "../../envs/finetune/train.yml"
    log:
        f"{LOG_PREFIX}/plotting/parameter_counts.log"
    script:
        "../../scripts/plotting/plot_parameter_counts.py"
