"""Rules for preparing Swiss-Prot homolog-based representation evaluation."""


def _representation_eval_enabled():
    return config.get("representation_eval", {}).get("enabled", False)


def _representation_eval_dataset_names():
    names = []
    for subset_type in (
        "per_protein_regression",
        "per_protein_classification",
        "per_residue_regression",
        "per_residue_classification",
    ):
        names.extend(
            sorted(config.get("standardized_datasets", {}).get(subset_type, {}).keys())
        )
    return names


def _representation_eval_extract_inputs():
    inputs = {}
    for dataset in _representation_eval_dataset_names():
        subset_type = resolve_dataset_type(dataset)
        inputs[f"{dataset}_test"] = (
            f"results/datasets_resplit/{subset_type}/{dataset}/test.parquet"
        )
    return inputs


def _representation_eval_embedding_specs():
    cfg = config["representation_eval"]["embeddings"]
    include_sections = set(cfg.get("include_sections", list(TRAIN_SECTIONS)))
    specs = []

    for checkpoint in cfg.get("frozen_checkpoints", []):
        specs.append(
            {
                "model_key": f"frozen__{checkpoint.replace('/', '__')}",
                "model_kind": "frozen",
                "category": "frozen",
                "checkpoint": checkpoint,
                "section": None,
                "run_id": None,
            }
        )

    for section in TRAIN_SECTIONS:
        if section not in include_sections:
            continue
        for run in config.get("train", {}).get(section, []):
            task_count = len(run.get("tasks", []))
            category = "single_task" if task_count == 1 else "multitask"
            specs.append(
                {
                    "model_key": f"trained__{section}__{run['id']}",
                    "model_kind": "trained",
                    "category": category,
                    "checkpoint": run["checkpoint"],
                    "section": section,
                    "run_id": run["id"],
                }
            )

    return specs


def _get_representation_eval_embedding_spec(model_key):
    for spec in _representation_eval_embedding_specs():
        if spec["model_key"] == model_key:
            return spec
    raise ValueError(f"Unknown representation-eval model key: {model_key!r}")


def get_representation_eval_embedding_targets():
    return [
        f"results/representation_eval/embeddings/{spec['model_key']}/manifest.tsv"
        for spec in _representation_eval_embedding_specs()
    ]


def _representation_eval_embedding_inputs(wildcards):
    spec = _get_representation_eval_embedding_spec(wildcards.model_key)
    inputs = {
        "matched_sequences": "results/representation_eval/swissprot/matched_sequences.fasta",
        "matched_entries": "results/representation_eval/swissprot/matched_entries.tsv",
    }
    if spec["model_kind"] == "trained":
        inputs["model_dir"] = (
            f"results/train/{spec['section']}/{spec['run_id']}/best_overall_model"
        )
    return inputs


if _representation_eval_enabled():

    rule swissprot_fetch:
        output:
            fasta_gz="results/reference_data/uniprot_sprot/raw/uniprot_sprot.fasta.gz",
            fasta="results/reference_data/uniprot_sprot/uniprot_sprot.fasta",
        params:
            url=config["representation_eval"]["swissprot"]["url"],
            hash=config["representation_eval"]["swissprot"]["hash"],
        conda:
            "../envs/prepare_data/fetch.yml"
        log:
            f"{LOG_PREFIX}/representation_eval/swissprot_fetch.log"
        script:
            "../scripts/representation_eval/fetch_swissprot.py"


    rule representation_eval_extract_test_queries:
        input:
            **_representation_eval_extract_inputs(),
        output:
            fasta="results/representation_eval/test_queries/all_test_sequences.fasta",
            metadata="results/representation_eval/test_queries/sequence_metadata.tsv",
        params:
            datasets=_representation_eval_dataset_names(),
        conda:
            "../envs/resplit/resplit.yml"
        log:
            f"{LOG_PREFIX}/representation_eval/extract_test_queries.log"
        script:
            "../scripts/representation_eval/extract_test_queries.py"


    rule representation_eval_mmseqs_search:
        input:
            query="results/representation_eval/test_queries/all_test_sequences.fasta",
            target="results/reference_data/uniprot_sprot/uniprot_sprot.fasta",
        output:
            hits="results/representation_eval/swissprot/mmseqs_hits.tsv",
        params:
            min_seq_id=config["representation_eval"]["search"]["min_seq_id"],
            gpu=config["representation_eval"]["search"].get("gpu", 0),
        conda:
            "../envs/prepare_data/process.yml"
        log:
            f"{LOG_PREFIX}/representation_eval/mmseqs_search.log"
        script:
            "../scripts/representation_eval/mmseqs_search_swissprot.py"


    rule representation_eval_select_best_hits:
        input:
            hits="results/representation_eval/swissprot/mmseqs_hits.tsv",
            metadata="results/representation_eval/test_queries/sequence_metadata.tsv",
        output:
            best_hits="results/representation_eval/swissprot/best_hits.tsv",
            accessions="results/representation_eval/swissprot/accessions.txt",
        params:
            min_query_coverage=config["representation_eval"]["search"]["min_query_coverage"],
            min_target_coverage=config["representation_eval"]["search"]["min_target_coverage"],
        conda:
            "../envs/resplit/resplit.yml"
        log:
            f"{LOG_PREFIX}/representation_eval/select_best_hits.log"
        script:
            "../scripts/representation_eval/select_best_swissprot_hits.py"


    rule representation_eval_fetch_annotations:
        input:
            accessions="results/representation_eval/swissprot/accessions.txt",
        output:
            entries="results/representation_eval/swissprot/entries.tsv",
        params:
            batch_size=config["representation_eval"]["annotations"].get(
                "batch_size", 200
            ),
        conda:
            "../envs/prepare_data/fetch.yml"
        log:
            f"{LOG_PREFIX}/representation_eval/fetch_annotations.log"
        script:
            "../scripts/representation_eval/fetch_uniprot_entries.py"


    rule representation_eval_join_matched_entries:
        input:
            best_hits="results/representation_eval/swissprot/best_hits.tsv",
            entries="results/representation_eval/swissprot/entries.tsv",
        output:
            matched_entries="results/representation_eval/swissprot/matched_entries.tsv",
        conda:
            "../envs/resplit/resplit.yml"
        log:
            f"{LOG_PREFIX}/representation_eval/join_matched_entries.log"
        script:
            "../scripts/representation_eval/join_matched_entries.py"


    rule representation_eval_annotation_summary:
        input:
            matched_entries="results/representation_eval/swissprot/matched_entries.tsv",
        output:
            coverage_summary="results/representation_eval/swissprot/annotation_coverage_summary.tsv",
            term_supports="results/representation_eval/swissprot/annotation_term_supports.tsv",
        conda:
            "../envs/resplit/resplit.yml"
        log:
            f"{LOG_PREFIX}/representation_eval/annotation_summary.log"
        script:
            "../scripts/representation_eval/summarize_annotations.py"


    rule representation_eval_annotation_dashboard:
        input:
            coverage_summary="results/representation_eval/swissprot/annotation_coverage_summary.tsv",
            term_supports="results/representation_eval/swissprot/annotation_term_supports.tsv",
        output:
            dashboard="results/representation_eval/swissprot/annotation_coverage_dashboard.html",
        conda:
            "../envs/data_stats/dashboard/stats.yml"
        log:
            f"{LOG_PREFIX}/representation_eval/annotation_dashboard.log"
        script:
            "../scripts/representation_eval/annotation_coverage_dashboard.py"


    rule representation_eval_extract_matched_sequences:
        input:
            swissprot_fasta="results/reference_data/uniprot_sprot/uniprot_sprot.fasta",
            best_hits="results/representation_eval/swissprot/best_hits.tsv",
            entries="results/representation_eval/swissprot/entries.tsv",
            matched_entries="results/representation_eval/swissprot/matched_entries.tsv",
        output:
            fasta="results/representation_eval/swissprot/matched_sequences.fasta",
        conda:
            "../envs/resplit/resplit.yml"
        log:
            f"{LOG_PREFIX}/representation_eval/extract_matched_sequences.log"
        script:
            "../scripts/representation_eval/extract_swissprot_matches.py"


    rule representation_eval_export_embeddings:
        input:
            unpack(_representation_eval_embedding_inputs),
        output:
            manifest="results/representation_eval/embeddings/{model_key}/manifest.tsv",
            vectors=directory("results/representation_eval/embeddings/{model_key}/vectors"),
            summary="results/representation_eval/embeddings/{model_key}/summary.json",
        params:
            spec=lambda wc: _get_representation_eval_embedding_spec(wc.model_key),
            batch_size=config["representation_eval"]["embeddings"].get("batch_size", 8),
        wildcard_constraints:
            model_key="|".join(
                spec["model_key"] for spec in _representation_eval_embedding_specs()
            ) if _representation_eval_embedding_specs() else "NOMATCH",
        conda:
            "../envs/representation_eval/embeddings.yml"
        log:
            f"{LOG_PREFIX}/representation_eval/embeddings/{{model_key}}.log"
        script:
            "../scripts/representation_eval/export_model_embeddings.py"
