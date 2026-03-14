"""Rules for cross-dataset reclustering and resplitting.

Reads from ``results/datasets_subsampled/`` and writes final outputs to
``results/datasets_resplit/``.  When a dataset is listed in ``config.resplit.datasets``
the full clustering + resplit pipeline runs; otherwise the output is a symlink
to ``datasets_subsampled/`` (zero-overhead passthrough).

Rule DAG::

    datasets_subsampled/
            |
      resplit_extract_all_fasta
            |
      +-----+----------+
      |                 |
    resplit_     resplit_foldseek_
    mmseqs_         createdb
    cluster            |
      |          resplit_foldseek_
      |              cluster
      +-----+----------+
            |
      resplit_merge_clusters
            |
      resplit_compute_split
            |
         resplit  ->  datasets_resplit/

Note: mmseqs and foldseek branches are conditional — only enabled
tools run (see ``resplit.tools`` in config).
"""


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _resplit_datasets():
    """Return the list of dataset names configured for resplitting."""
    return config.get("resplit", {}).get("datasets", [])


def _resplit_tool_config(tool_name):
    """Return config dict for a resplit clustering tool (mmseqs or foldseek)."""
    return config.get("resplit", {}).get("tools", {}).get(tool_name, {})


def _resplit_enabled_tools():
    """Return list of enabled tool names for resplit clustering."""
    tools = config.get("resplit", {}).get("tools", {})
    return [name for name, cfg in tools.items() if cfg.get("enabled", False)]


def _resplit_merge_inputs():
    """Return dict of enabled tools' cluster TSV paths for the merge rule."""
    inputs = {}
    enabled = _resplit_enabled_tools()
    if "mmseqs" in enabled:
        inputs["mmseqs"] = "results/resplit/mmseqs_clusters.tsv"
    if "foldseek" in enabled:
        inputs["foldseek"] = "results/resplit/foldseek_clusters.tsv"
    return inputs


# Config-time validation: datasets listed but no tools enabled is an error.
if _resplit_datasets() and not _resplit_enabled_tools():
    raise ValueError(
        "resplit.datasets is non-empty but no clustering tools are enabled in "
        "resplit.tools. Enable at least one of mmseqs or foldseek."
    )


def _resplit_extract_inputs():
    """Collect all subsampled parquets needed for the global FASTA extraction."""
    datasets = _resplit_datasets()
    inputs = {}
    for ds in datasets:
        ds_type = resolve_dataset_type(ds)
        for split in ("train", "valid", "test"):
            inputs[f"{ds}_{split}"] = (
                f"results/datasets_subsampled/{ds_type}/{ds}/{split}.parquet"
            )
    return inputs


def _resplit_input(wildcards):
    """Conditionally depend on split_assignments only for resplit datasets."""
    inputs = {
        "subsampled": (
            f"results/datasets_subsampled/{wildcards.subset_type}"
            f"/{wildcards.dataset}/{wildcards.split}.parquet"
        ),
    }
    resplit_datasets = _resplit_datasets()
    if wildcards.dataset in resplit_datasets:
        inputs["split_assignments"] = "results/resplit/split_assignments.tsv"
        inputs["sequence_metadata"] = "results/resplit/sequence_metadata.tsv"
    return inputs


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

rule resplit_extract_all_fasta:
    """Extract all sequences from resplit-configured datasets into one FASTA + metadata TSV."""
    input:
        **_resplit_extract_inputs(),
    output:
        fasta="results/resplit/all_sequences.fasta",
        metadata="results/resplit/sequence_metadata.tsv",
    params:
        datasets=_resplit_datasets(),
    conda:
        "../envs/resplit/resplit.yml"
    log:
        f"{LOG_PREFIX}/resplit/extract_all_fasta.log"
    script:
        "../scripts/resplit/extract_all_fasta.py"


rule resplit_mmseqs_cluster:
    """Run MMseqs2 easy-cluster on all sequences."""
    input:
        fasta="results/resplit/all_sequences.fasta",
    output:
        clusters="results/resplit/mmseqs_clusters.tsv",
    params:
        min_seq_id=_resplit_tool_config("mmseqs").get("min_seq_id", 0.2),
        gpu=_resplit_tool_config("mmseqs").get("gpu", 0),
    conda:
        "../envs/prepare_data/process.yml"
    log:
        f"{LOG_PREFIX}/resplit/mmseqs_cluster.log"
    script:
        "../scripts/resplit/mmseqs_cluster.py"


rule resplit_foldseek_createdb:
    """Create Foldseek DB with ProstT5 structure prediction (expensive, cached)."""
    input:
        fasta="results/resplit/all_sequences.fasta",
        prostt5_model=".cache/foldseek_prostt5/prostt5/prostt5-f16.gguf",
    output:
        db=directory("results/resplit/foldseek_db"),
    params:
        gpu=_resplit_tool_config("foldseek").get("gpu", 0),
    conda:
        "../envs/data_stats/leakage/foldseek.yml"
    log:
        f"{LOG_PREFIX}/resplit/foldseek_createdb.log"
    script:
        "../scripts/resplit/foldseek_createdb.py"


rule resplit_foldseek_cluster:
    """Run Foldseek clustering on pre-built DB."""
    input:
        db="results/resplit/foldseek_db",
    output:
        clusters="results/resplit/foldseek_clusters.tsv",
    params:
        min_seq_id=_resplit_tool_config("foldseek").get("min_seq_id", 0.2),
        gpu=_resplit_tool_config("foldseek").get("gpu", 0),
    conda:
        "../envs/data_stats/leakage/foldseek.yml"
    log:
        f"{LOG_PREFIX}/resplit/foldseek_cluster.log"
    script:
        "../scripts/resplit/foldseek_cluster.py"


rule resplit_merge_clusters:
    """Merge enabled clustering tools' results via union of similarity edges."""
    input:
        **_resplit_merge_inputs(),
    output:
        merged="results/resplit/merged_clusters.tsv",
    params:
        tools=_resplit_enabled_tools(),
    conda:
        "../envs/resplit/resplit.yml"
    log:
        f"{LOG_PREFIX}/resplit/merge_clusters.log"
    script:
        "../scripts/resplit/merge_clusters.py"


rule resplit_compute_split:
    """Compute new split assignments (Phase A meltome + Phase B main resplit)."""
    input:
        clusters="results/resplit/merged_clusters.tsv",
        metadata="results/resplit/sequence_metadata.tsv",
    output:
        assignments="results/resplit/split_assignments.tsv",
        summary="results/resplit/resplit_summary.json",
    params:
        min_valid_fraction=config.get("resplit", {}).get("min_valid_fraction", 0.1),
        random_seed=config.get("resplit", {}).get("random_seed", 42),
        shared_protein_groups=config.get("resplit", {}).get("shared_protein_groups", []),
        reconstruct_test_datasets=config.get("resplit", {}).get("reconstruct_test", {}).get("datasets", []),
        reconstruct_test_min_test_fraction=config.get("resplit", {}).get("reconstruct_test", {}).get("min_test_fraction", 0.1),
    conda:
        "../envs/resplit/resplit.yml"
    log:
        f"{LOG_PREFIX}/resplit/compute_split.log"
    script:
        "../scripts/resplit/compute_resplit.py"


rule resplit:
    """Apply resplit assignments or symlink from subsampled (passthrough).

    For resplit-configured datasets: reads split_assignments.tsv and all 3
    source parquets, filters rows for the requested split, and writes the
    output parquet.

    For non-resplit datasets: creates a relative symlink from
    datasets_subsampled/ to datasets_resplit/.
    """
    input:
        unpack(_resplit_input),
    output:
        dataset="results/datasets_resplit/{subset_type}/{dataset}/{split}.parquet",
    params:
        resplit_datasets=_resplit_datasets(),
        aggregate_duplicates=config.get("resplit", {}).get("aggregate_duplicates", {}),
    conda:
        "../envs/resplit/resplit.yml"
    log:
        f"{LOG_PREFIX}/resplit/{{subset_type}}/{{dataset}}/{{split}}.log"
    script:
        "../scripts/resplit/apply_or_passthrough.py"
