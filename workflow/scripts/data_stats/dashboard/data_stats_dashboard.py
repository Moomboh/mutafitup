from pathlib import Path

import pandas as pd
from snakemake.script import snakemake

from wfutils import get_logger
from wfutils.logging import log_snakemake_info

import dashboard_app


logger = get_logger()
log_snakemake_info(logger)


data_dir = Path(str(snakemake.input["data"]))
output_html = Path(str(snakemake.output[0]))


def build_html() -> str:
    data = dashboard_app.load_csv_data(data_dir)
    (
        stats_df,
        pie_graphs,
        seq_histograms,
        seq_boxplots,
        score_histograms,
        score_boxplots,
        label_barcharts,
    ) = dashboard_app.get_dashboard_components(data)

    sections: list[str] = []
    sections.append("<h1>Dataset Statistics Dashboard</h1>")

    if not stats_df.empty:
        sections.append("<h2>General Statistics</h2>")
        sections.append(stats_df.to_html(index=False, classes="stats-table"))

    first_fig = {"value": True}

    def fig_to_div(fig) -> str:
        include_js = "inline" if first_fig["value"] else False
        first_fig["value"] = False
        return fig.to_html(full_html=False, include_plotlyjs=include_js)

    def add_section(title: str, graphs) -> None:
        if not graphs:
            return
        sections.append(f"<h2>{title}</h2>")
        for g in graphs:
            fig = getattr(g, "figure", None)
            if fig is None:
                continue
            sections.append(fig_to_div(fig))

    add_section("Split Proportions", pie_graphs)
    add_section("Sequence Length Distributions", seq_histograms)
    add_section("Sequence Length Box Plots", seq_boxplots)
    add_section("Score Distributions", score_histograms)
    add_section("Score Box Plots", score_boxplots)
    add_section("Label Counts", label_barcharts)

    head = """<!DOCTYPE html>
<html>
<head>
<meta charset=\"utf-8\">
<title>Dataset Statistics Dashboard</title>
<style>
body { font-family: Arial, sans-serif; margin: 20px; }
h1, h2 { text-align: center; }
.stats-table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
.stats-table th, .stats-table td { border: 1px solid #ddd; padding: 8px; }
.stats-table th { background-color: #f2f2f2; }
.section { margin-bottom: 40px; }
</style>
</head>
<body>
"""

    tail = "</body></html>"

    html_parts: list[str] = [head]
    for section in sections:
        html_parts.append('<div class="section">')
        html_parts.append(section)
        html_parts.append("</div>")
    html_parts.append(tail)

    return "".join(html_parts)


def main() -> None:
    output_html.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Loading dashboard data from {data_dir}")
    html_str = build_html()
    logger.info(f"Writing dashboard HTML to {output_html}")
    output_html.write_text(html_str, encoding="utf-8")


if __name__ == "__main__":
    main()
