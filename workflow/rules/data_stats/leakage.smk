import re


def get_leakage_profiles():
    """Return dict of profile_name -> profile_config from leakage_analysis config."""
    return config.get("leakage_analysis", {}).get("profiles", {})


def get_leakage_profile_names():
    """Return ordered list of profile names."""
    return list(get_leakage_profiles().keys())


def get_leakage_datasets(profile):
    """Return list of dataset name strings for a given profile."""
    return get_leakage_profiles().get(profile, {}).get("datasets", [])


def get_leakage_dataset_names(profile):
    """Return ordered list of dataset names for a given profile."""
    return get_leakage_datasets(profile)


def get_leakage_dataset_type(profile, dataset_name):
    """Return the subset type for a given dataset name within a profile."""
    return resolve_dataset_type(dataset_name)


def get_leakage_tool_config(profile, tool_name):
    """Return the config dict for a specific tool within a profile."""
    return (
        get_leakage_profiles()
        .get(profile, {})
        .get("tools", {})
        .get(tool_name, {})
    )


def get_leakage_enabled_tools(profile):
    """Return list of tool names that are enabled for a given profile."""
    tools = get_leakage_profiles().get(profile, {}).get("tools", {})
    return [name for name, cfg in tools.items() if cfg.get("enabled", False)]


def _all_leakage_dataset_names():
    """Return the union of all dataset names across all profiles (for wildcard constraints)."""
    names = set()
    for profile_cfg in get_leakage_profiles().values():
        for dataset_name in profile_cfg.get("datasets", []):
            names.add(dataset_name)
    return sorted(names)


def _get_leakage_dataset_type_global(dataset_name):
    """Look up dataset type from any profile (dataset type is profile-independent)."""
    return resolve_dataset_type(dataset_name)


def _get_leakage_search_gpu(tool_name, min_seq_id):
    """Find gpu setting for a tool+threshold from any matching profile."""
    for profile_cfg in get_leakage_profiles().values():
        tool_cfg = profile_cfg.get("tools", {}).get(tool_name, {})
        if tool_cfg.get("enabled") and tool_cfg.get("min_seq_id") == min_seq_id:
            return tool_cfg.get("gpu", 0)
    return 0


def _all_leakage_min_seq_ids():
    """Return unique min_seq_id values (as strings) for wildcard constraints."""
    ids = set()
    for profile_cfg in get_leakage_profiles().values():
        for tool_cfg in profile_cfg.get("tools", {}).values():
            if tool_cfg.get("enabled") and "min_seq_id" in tool_cfg:
                ids.add(str(tool_cfg["min_seq_id"]))
    return sorted(ids)


LEAKAGE_SPLITS = ["train", "valid", "test"]
LEAKAGE_VARIANTS = ["all", "cross_task"]

_LEAKAGE_DS_CONSTRAINT = (
    "|".join(_all_leakage_dataset_names())
    if _all_leakage_dataset_names()
    else "NOMATCH"
)

_LEAKAGE_MIN_SEQ_ID_CONSTRAINT = (
    "|".join(re.escape(s) for s in _all_leakage_min_seq_ids())
    if _all_leakage_min_seq_ids()
    else "NOMATCH"
)


# ---------------------------------------------------------------------------
# Step 1: Extract per-split FASTAs from pre-resplit (original) data
# ---------------------------------------------------------------------------

rule leakage_extract_fasta_original:
    input:
        parquet=lambda wc: (
            f"results/datasets_subsampled/{_get_leakage_dataset_type_global(wc.dataset)}"
            f"/{wc.dataset}/{wc.split}.parquet"
        ),
    output:
        fasta="results/leakage_analysis/fasta_original/{dataset}_{split}.fasta",
    wildcard_constraints:
        dataset=_LEAKAGE_DS_CONSTRAINT,
        split="train|valid|test",
    conda:
        "../../envs/resplit/resplit.yml"
    log:
        f"{LOG_PREFIX}/leakage_analysis/extract_fasta_original/{{dataset}}_{{split}}.log"
    script:
        "../../scripts/data_stats/leakage/extract_fasta.py"


# ---------------------------------------------------------------------------
# Step 2: Merge per-split FASTAs into one per-dataset FASTA
# ---------------------------------------------------------------------------

rule leakage_merge_fasta:
    input:
        train="results/leakage_analysis/fasta_original/{dataset}_train.fasta",
        valid="results/leakage_analysis/fasta_original/{dataset}_valid.fasta",
        test="results/leakage_analysis/fasta_original/{dataset}_test.fasta",
    output:
        fasta="results/leakage_analysis/fasta_merged/{dataset}.fasta",
    wildcard_constraints:
        dataset=_LEAKAGE_DS_CONSTRAINT,
    log:
        f"{LOG_PREFIX}/leakage_analysis/merge_fasta/{{dataset}}.log"
    shell:
        "cat {input.train} {input.valid} {input.test} > {output.fasta} 2> {log}"


# ---------------------------------------------------------------------------
# Step 3: Pairwise search (one per dataset pair, all splits combined)
# ---------------------------------------------------------------------------

rule leakage_mmseqs_search:
    input:
        query="results/leakage_analysis/fasta_merged/{query_dataset}.fasta",
        target="results/leakage_analysis/fasta_merged/{target_dataset}.fasta",
    output:
        result="results/leakage_analysis/mmseqs/{min_seq_id}/{query_dataset}_vs_{target_dataset}.tsv",
    params:
        min_seq_id=lambda wc: float(wc.min_seq_id),
        gpu=lambda wc: _get_leakage_search_gpu("mmseqs", float(wc.min_seq_id)),
    wildcard_constraints:
        min_seq_id=_LEAKAGE_MIN_SEQ_ID_CONSTRAINT,
        query_dataset=_LEAKAGE_DS_CONSTRAINT,
        target_dataset=_LEAKAGE_DS_CONSTRAINT,
    conda:
        "../../envs/prepare_data/process.yml"
    log:
        f"{LOG_PREFIX}/leakage_analysis/mmseqs/{{min_seq_id}}/{{query_dataset}}_vs_{{target_dataset}}.log"
    script:
        "../../scripts/data_stats/leakage/mmseqs_search.py"


rule leakage_foldseek_search:
    input:
        query="results/leakage_analysis/fasta_merged/{query_dataset}.fasta",
        target="results/leakage_analysis/fasta_merged/{target_dataset}.fasta",
        prostt5_model=".cache/foldseek_prostt5/prostt5/prostt5-f16.gguf",
    output:
        result="results/leakage_analysis/foldseek/{min_seq_id}/{query_dataset}_vs_{target_dataset}.tsv",
    params:
        min_seq_id=lambda wc: float(wc.min_seq_id),
        gpu=lambda wc: _get_leakage_search_gpu("foldseek", float(wc.min_seq_id)),
    wildcard_constraints:
        min_seq_id=_LEAKAGE_MIN_SEQ_ID_CONSTRAINT,
        query_dataset=_LEAKAGE_DS_CONSTRAINT,
        target_dataset=_LEAKAGE_DS_CONSTRAINT,
    conda:
        "../../envs/data_stats/leakage/foldseek.yml"
    log:
        f"{LOG_PREFIX}/leakage_analysis/foldseek/{{min_seq_id}}/{{query_dataset}}_vs_{{target_dataset}}.log"
    script:
        "../../scripts/data_stats/leakage/foldseek_search.py"


# ---------------------------------------------------------------------------
# Step 4: Comparison plotting (pre-resplit vs post-resplit)
# ---------------------------------------------------------------------------

def get_leakage_comparison_inputs(wildcards):
    """Collect search result TSVs, original per-split FASTAs, and resplit
    metadata for a comparison figure."""
    profile = wildcards.profile
    datasets = get_leakage_dataset_names(profile)
    tool = wildcards.tool
    query_split = wildcards.query_split
    target_split = wildcards.target_split
    min_seq_id = str(get_leakage_tool_config(profile, tool).get("min_seq_id", 0.3))

    inputs = {}

    # One result file per dataset pair (all splits combined, reused for both)
    result_files = []
    for query_ds in datasets:
        for target_ds in datasets:
            result_files.append(
                f"results/leakage_analysis/{tool}/{min_seq_id}/{query_ds}_vs_{target_ds}.tsv"
            )
    inputs["results"] = result_files

    # Pre-resplit (original) per-split FASTAs — all 3 splits for all datasets.
    # Needed for: (a) original heatmap split sizes & hit filtering,
    #             (b) building the resplit ID mapping (sequence lookup per ID).
    orig_fastas = []
    for ds in datasets:
        for split in LEAKAGE_SPLITS:
            orig_fastas.append(
                f"results/leakage_analysis/fasta_original/{ds}_{split}.fasta"
            )
    inputs["orig_fastas"] = orig_fastas

    # Resplit metadata — used to map original FASTA IDs to resplit splits
    inputs["sequence_metadata"] = "results/resplit/sequence_metadata.tsv"
    inputs["split_assignments"] = "results/resplit/split_assignments.tsv"

    return inputs


def _all_leakage_enabled_tools():
    """Return the union of all enabled tool names across all profiles (for wildcard constraints)."""
    tools = set()
    for profile_name in get_leakage_profile_names():
        tools.update(get_leakage_enabled_tools(profile_name))
    return sorted(tools)


rule leakage_comparison:
    input:
        unpack(get_leakage_comparison_inputs),
    output:
        comparison="results/leakage_analysis/{profile}/comparison_{variant}_{tool}_{query_split}_vs_{target_split}.png",
    params:
        datasets=lambda wc: get_leakage_dataset_names(wc.profile),
        fasta_original_dir="results/leakage_analysis/fasta_original",
        results_dir=lambda wc: f"results/leakage_analysis/{wc.tool}/{get_leakage_tool_config(wc.profile, wc.tool).get('min_seq_id', 0.3)}",
        sequence_metadata="results/resplit/sequence_metadata.tsv",
        split_assignments="results/resplit/split_assignments.tsv",
        min_seq_id=lambda wc: get_leakage_tool_config(wc.profile, wc.tool).get(
            "min_seq_id", 0.3
        ),
        profile=lambda wc: wc.profile,
        variant=lambda wc: wc.variant,
    wildcard_constraints:
        profile="|".join(get_leakage_profile_names()) if get_leakage_profile_names() else "NOMATCH",
        tool="|".join(_all_leakage_enabled_tools()) if _all_leakage_enabled_tools() else "NOMATCH",
        variant="all|cross_task",
        query_split="train|valid|test",
        target_split="train|valid|test",
    conda:
        "../../envs/data_stats/leakage/plotting.yml"
    log:
        f"{LOG_PREFIX}/leakage_analysis/{{profile}}/comparison_{{variant}}_{{tool}}_{{query_split}}_vs_{{target_split}}.log"
    script:
        "../../scripts/data_stats/leakage/plot_comparison.py"
