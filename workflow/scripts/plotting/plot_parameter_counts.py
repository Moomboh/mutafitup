import json
from pathlib import Path

from snakemake.script import snakemake

from mutafitup.plotting import export_parameter_summary_typst, plot_trainable_vs_frozen
from mutafitup.plotting.parameter_counts import (
    filter_run_entries,
    summarize_parameter_counts,
)
from mutafitup.plotting.typst_tables import write_text
from wfutils.logging import get_logger, log_snakemake_info


logger = get_logger(__name__)
log_snakemake_info(logger)


def main() -> None:
    run_entries = snakemake.params["run_entries"]
    include_run_keys = snakemake.params.get("include_run_keys")
    summary_json = Path(str(snakemake.output["summary_json"]))
    summary_csv = Path(str(snakemake.output["summary_csv"]))
    summary_typst = Path(str(snakemake.output["summary_typst"]))
    plot_path = str(snakemake.output["plot"])

    filtered_entries = filter_run_entries(run_entries, include_run_keys)
    df = summarize_parameter_counts(filtered_entries)

    summary_json.parent.mkdir(parents=True, exist_ok=True)
    with summary_json.open("w") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2)
        f.write("\n")

    df.to_csv(summary_csv, index=False)
    write_text(summary_typst, export_parameter_summary_typst(df))
    plot_trainable_vs_frozen(df, plot_path)

    logger.info(
        "Wrote parameter summary to %s, %s, and %s",
        summary_json,
        summary_csv,
        summary_typst,
    )
    logger.info("Saved parameter plot to %s", plot_path)


main()
