from snakemake.script import snakemake

from mutafitup.plotting import plot_learning_rate, plot_losses, plot_metrics


history_path = snakemake.input["history"]
output_losses = snakemake.output["losses"]
output_metrics = snakemake.output["metrics"]
output_lr = snakemake.output["learning_rate"]
run_id = snakemake.params.get("run_id", None)
task_names = snakemake.params.get("task_names", None)

plot_losses(history_path, output_losses, run_id, task_names)
plot_metrics(history_path, output_metrics, run_id, task_names)
plot_learning_rate(history_path, output_lr, run_id)
