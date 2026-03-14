# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
import datetime as _dt

TRAIN_SECTIONS = ("heads_only", "lora", "accgrad_lora", "align_lora")

# Per-run timestamped log directory so logs from different pipeline
# invocations never overwrite each other.
LOG_PREFIX = f"logs/{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
def resolve_dataset_type(dataset_name):
    """Look up subset_type for a dataset from standardized_datasets config."""
    for subset_type in ("per_protein_regression", "per_protein_classification",
                        "per_residue_regression", "per_residue_classification"):
        if dataset_name in config.get("standardized_datasets", {}).get(subset_type, {}):
            return subset_type
    raise ValueError(f"Dataset {dataset_name!r} not found in standardized_datasets")


def resolve_dataset_meta(dataset_name):
    """Return the full metadata dict for a dataset from standardized_datasets.

    The returned dict includes at least: color, primary_metric.
    Classification datasets also have: num_labels.
    Optionally: early_stopping_metric.
    """
    subset_type = resolve_dataset_type(dataset_name)
    return config["standardized_datasets"][subset_type][dataset_name]


def expand_task(task_name):
    """Expand a task name string to a full task dict.

    Returns a dict with keys: name, dataset, subset_type, primary_metric,
    and num_labels (for classification tasks).
    """
    subset_type = resolve_dataset_type(task_name)
    ds_meta = resolve_dataset_meta(task_name)
    result = {
        "name": task_name,
        "dataset": task_name,
        "subset_type": subset_type,
        "primary_metric": ds_meta["primary_metric"],
    }
    if "classification" in subset_type:
        num_labels = ds_meta.get("num_labels")
        if num_labels is None:
            raise ValueError(
                f"num_labels not found for classification dataset {task_name!r}"
            )
        result["num_labels"] = num_labels
    return result


def expand_tasks(tasks):
    """Expand a list of task name strings to full task dicts."""
    return [expand_task(t) for t in tasks]


# ---------------------------------------------------------------------------
# Training-run helpers (moved from training_plots.smk / Snakefile)
# ---------------------------------------------------------------------------
def iter_train_runs():
    """Yield (section, run_dict) for every training run across all sections."""
    for section in TRAIN_SECTIONS:
        for run in config.get("train", {}).get(section, []):
            yield section, run


def get_train_run(section, run_id):
    """Look up a specific training run config dict by section and id."""
    for run in config.get("train", {}).get(section, []):
        if run["id"] == run_id:
            return run
    raise ValueError(f"Unknown train.{section} run id: {run_id}")


def make_run_key(section, run_id):
    """Create a flat string key for a training run: '{section}__{run_id}'."""
    return f"{section}__{run_id}"


def get_train_runs():
    """Return list of (section, run_id) tuples for all training runs."""
    return [(section, run["id"]) for section, run in iter_train_runs()]


def get_train_run_ids():
    """Return list of all training run IDs."""
    return [run_id for _, run_id in get_train_runs()]


def get_train_sections():
    """Return list of sections corresponding to get_train_run_ids()."""
    return [section for section, _ in get_train_runs()]


def get_all_run_keys():
    """Return list of all run keys ('{section}__{run_id}')."""
    return [make_run_key(section, run["id"]) for section, run in iter_train_runs()]


# ---------------------------------------------------------------------------
# Shared tool rules
# ---------------------------------------------------------------------------
rule download_prostt5:
    """Download the ProstT5 model for Foldseek structural similarity.

    Used by both the leakage analysis and the resplit pipeline.
    """
    output:
        model=".cache/foldseek_prostt5/prostt5/prostt5-f16.gguf",
    params:
        db_prefix=".cache/foldseek_prostt5/prostt5",
        tmp_dir=".cache/foldseek_prostt5/tmp",
    conda:
        "../envs/data_stats/leakage/foldseek.yml"
    log:
        f"{LOG_PREFIX}/download_prostt5.log"
    shell:
        r"""
        mkdir -p "$(dirname {params.db_prefix})" "{params.tmp_dir}" 2>&1 | tee {log}
        foldseek databases ProstT5 {params.db_prefix} {params.tmp_dir} 2>&1 | tee -a {log}
        """
