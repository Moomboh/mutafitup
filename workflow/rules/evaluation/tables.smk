from mutafitup.evaluation.resolve import resolve_evaluation_inputs


def _get_evaluation_by_id(eval_id):
    for ev in config.get("evaluations", []):
        if ev["id"] == eval_id:
            return ev
    raise ValueError(f"Unknown evaluation ID: {eval_id!r}")


def _get_evaluation_metric_inputs_for_tables(wildcards):
    ev = _get_evaluation_by_id(wildcards.eval_id)
    return resolve_evaluation_inputs(ev, train_config=config.get("train", {}))


rule export_evaluation_typst_tables:
    input:
        metric_jsons=_get_evaluation_metric_inputs_for_tables,
        summary="results/evaluation/{eval_id}/summary_{split}.json",
    output:
        full_results_table="results/thesis_tables/{eval_id}/full_results_{split}.typ",
        delta_summary_table="results/thesis_tables/{eval_id}/delta_summary_{split}.typ",
    params:
        evaluation=lambda wc: _get_evaluation_by_id(wc.eval_id),
    log:
        f"{LOG_PREFIX}/evaluation/tables/{{eval_id}}_{{split}}.log",
    script:
        "../../scripts/evaluation/export_typst_tables.py"
