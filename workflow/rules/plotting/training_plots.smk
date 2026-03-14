rule plot_training_history:
    input:
        history="results/train/{section}/{run}/history.json",
    output:
        losses="results/plots/train/{section}/{run}/losses.png",
        metrics="results/plots/train/{section}/{run}/metrics.png",
        learning_rate="results/plots/train/{section}/{run}/learning_rate.png",
    params:
        run_id=lambda wc: f"{wc.section}/{wc.run}",
        task_names=lambda wc: get_train_run(wc.section, wc.run)["tasks"],
    conda:
        "../../envs/finetune/train.yml"
    script:
        "../../scripts/plotting/plot_training_history.py"
