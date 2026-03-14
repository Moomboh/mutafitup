"""Generate a self-contained static HTML dashboard for duplicate sequence analysis.

Reads the CSVs produced by ``duplicate_stats.py`` and renders sections with
Plotly figures embedded inline.

Sections
--------
1. Summary table (from ``duplicate_summary.csv``)
2. Group-size distribution bar chart per dataset
3. Score-range histogram per dataset (regression only)
4. Score-std boxplot per dataset (regression only)
5. Score-range vs group-size scatter per dataset (regression only)
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from snakemake.script import snakemake

from wfutils import get_logger
from wfutils.logging import log_snakemake_info

logger = get_logger()
log_snakemake_info(logger)

data_dir = Path(str(snakemake.input["data"]))
output_html = Path(str(snakemake.output[0]))

# ---------------------------------------------------------------------------
# Plotly JS inclusion helper
# ---------------------------------------------------------------------------

_first_fig = True


def _fig_to_div(fig: go.Figure) -> str:
    """Convert a Plotly figure to an HTML div, including plotly.js only once."""
    global _first_fig
    include_js = "inline" if _first_fig else False
    _first_fig = False
    return fig.to_html(full_html=False, include_plotlyjs=include_js)


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------


def _build_summary_table(summary_df: pd.DataFrame) -> str:
    """Render the summary dataframe as an HTML table."""
    display_cols = [
        "dataset",
        "split",
        "subset_type",
        "total_rows",
        "unique_sequences",
        "duplicate_groups",
        "duplicate_rows",
        "extra_rows",
        "duplicate_pct",
        "mean_group_size",
        "max_group_size",
        "mean_score_range",
        "max_score_range",
        "mean_score_std",
    ]
    cols = [c for c in display_cols if c in summary_df.columns]
    return summary_df[cols].to_html(index=False, classes="stats-table", na_rep="-")


def _build_group_size_charts(groups_df: pd.DataFrame) -> list[str]:
    """Bar chart of duplicate-group sizes per dataset (all splits combined)."""
    divs: list[str] = []
    if groups_df.empty:
        return divs

    for dataset in sorted(groups_df["dataset"].unique()):
        ds_df = groups_df[groups_df["dataset"] == dataset]
        size_counts = ds_df["count"].value_counts().sort_index()
        fig = go.Figure(
            go.Bar(
                x=size_counts.index.astype(str),
                y=size_counts.values,
                marker_color="#636EFA",
            )
        )
        fig.update_layout(
            title=f"{dataset} — Duplicate group sizes",
            xaxis_title="Group size (sequences with identical AA sequence)",
            yaxis_title="Number of groups",
            template="plotly_white",
            height=350,
            margin=dict(t=50, b=50),
        )
        divs.append(_fig_to_div(fig))
    return divs


def _build_score_range_histograms(groups_df: pd.DataFrame) -> list[str]:
    """Histogram of score ranges within duplicate groups (regression only)."""
    divs: list[str] = []
    if groups_df.empty:
        return divs
    reg_df = groups_df.dropna(subset=["score_range"])
    if reg_df.empty:
        return divs

    for dataset in sorted(reg_df["dataset"].unique()):
        ds_df = reg_df[reg_df["dataset"] == dataset]
        if ds_df.empty or ds_df["score_range"].isna().all():
            continue
        fig = px.histogram(
            ds_df,
            x="score_range",
            nbins=30,
            title=f"{dataset} — Score range within duplicate groups",
            labels={"score_range": "Score range (max - min)"},
            template="plotly_white",
            color_discrete_sequence=["#EF553B"],
        )
        fig.update_layout(
            yaxis_title="Number of groups",
            height=350,
            margin=dict(t=50, b=50),
        )
        divs.append(_fig_to_div(fig))
    return divs


def _build_score_std_boxplots(groups_df: pd.DataFrame) -> list[str]:
    """Boxplot of score std-dev across duplicate groups (regression only)."""
    divs: list[str] = []
    if groups_df.empty:
        return divs
    reg_df = groups_df.dropna(subset=["score_std"])
    if reg_df.empty:
        return divs

    # One boxplot with all regression datasets side by side
    datasets_with_data = sorted(reg_df[reg_df["score_std"].notna()]["dataset"].unique())
    if not datasets_with_data:
        return divs

    fig = go.Figure()
    for ds in datasets_with_data:
        ds_df = reg_df[reg_df["dataset"] == ds]
        fig.add_trace(
            go.Box(
                y=ds_df["score_std"],
                name=ds,
                boxmean="sd",
            )
        )
    fig.update_layout(
        title="Score std-dev within duplicate groups (regression datasets)",
        yaxis_title="Standard deviation",
        template="plotly_white",
        height=400,
        margin=dict(t=50, b=50),
        showlegend=False,
    )
    divs.append(_fig_to_div(fig))
    return divs


def _build_score_range_vs_size(groups_df: pd.DataFrame) -> list[str]:
    """Scatter of score-range vs group-size (regression only)."""
    divs: list[str] = []
    if groups_df.empty:
        return divs
    reg_df = groups_df.dropna(subset=["score_range"])
    if reg_df.empty:
        return divs

    for dataset in sorted(reg_df["dataset"].unique()):
        ds_df = reg_df[reg_df["dataset"] == dataset]
        if ds_df.empty:
            continue
        fig = px.scatter(
            ds_df,
            x="count",
            y="score_range",
            title=f"{dataset} — Score range vs group size",
            labels={
                "count": "Group size",
                "score_range": "Score range (max - min)",
            },
            template="plotly_white",
            color_discrete_sequence=["#00CC96"],
            opacity=0.6,
        )
        fig.update_layout(
            height=400,
            margin=dict(t=50, b=50),
        )
        divs.append(_fig_to_div(fig))
    return divs


# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------

_HEAD = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Duplicate Sequence Analysis</title>
<style>
body { font-family: Arial, sans-serif; margin: 20px; max-width: 1400px; margin-left: auto; margin-right: auto; }
h1, h2 { text-align: center; }
.stats-table { border-collapse: collapse; width: 100%; margin-bottom: 20px; font-size: 13px; }
.stats-table th, .stats-table td { border: 1px solid #ddd; padding: 6px 8px; text-align: right; }
.stats-table th { background-color: #f2f2f2; text-align: center; }
.stats-table td:first-child, .stats-table td:nth-child(2), .stats-table td:nth-child(3) {
    text-align: left;
}
.section { margin-bottom: 40px; }
</style>
</head>
<body>
"""

_TAIL = "</body></html>"


def build_html() -> str:
    summary_path = data_dir / "duplicate_summary.csv"
    groups_path = data_dir / "duplicate_groups.csv"

    try:
        summary_df = (
            pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
        )
    except pd.errors.EmptyDataError:
        summary_df = pd.DataFrame()

    try:
        groups_df = pd.read_csv(groups_path) if groups_path.exists() else pd.DataFrame()
    except pd.errors.EmptyDataError:
        groups_df = pd.DataFrame()

    sections: list[str] = []
    sections.append("<h1>Duplicate Sequence Analysis</h1>")

    # 1. Summary table
    if not summary_df.empty:
        sections.append("<h2>Summary</h2>")
        sections.append(_build_summary_table(summary_df))

    # 2. Group-size distribution
    group_size_divs = _build_group_size_charts(groups_df)
    if group_size_divs:
        sections.append("<h2>Group Size Distribution</h2>")
        sections.extend(group_size_divs)

    # 3. Score-range histograms (regression)
    score_range_divs = _build_score_range_histograms(groups_df)
    if score_range_divs:
        sections.append("<h2>Score Range Within Duplicate Groups (Regression)</h2>")
        sections.extend(score_range_divs)

    # 4. Score-std boxplots (regression)
    score_std_divs = _build_score_std_boxplots(groups_df)
    if score_std_divs:
        sections.append(
            "<h2>Score Standard Deviation Within Duplicate Groups (Regression)</h2>"
        )
        sections.extend(score_std_divs)

    # 5. Score-range vs group-size scatter (regression)
    scatter_divs = _build_score_range_vs_size(groups_df)
    if scatter_divs:
        sections.append("<h2>Score Range vs Group Size (Regression)</h2>")
        sections.extend(scatter_divs)

    parts: list[str] = [_HEAD]
    for section in sections:
        parts.append('<div class="section">')
        parts.append(section)
        parts.append("</div>")
    parts.append(_TAIL)
    return "".join(parts)


def main() -> None:
    output_html.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Loading duplicate analysis data from %s", data_dir)
    html_str = build_html()
    logger.info("Writing duplicate dashboard HTML to %s", output_html)
    output_html.write_text(html_str, encoding="utf-8")
    logger.info("Done.")


if __name__ == "__main__":
    main()
