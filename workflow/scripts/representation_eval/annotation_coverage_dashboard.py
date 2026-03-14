"""Self-contained HTML dashboard for annotation coverage of matched Swiss-Prot entries.

Reads the two summary TSVs produced by ``summarize_annotations.py`` and renders
an interactive Plotly-based report with:

1. Coverage overview table (counts & percentages per annotation family)
2. Annotation family coverage bar chart (grouped by scope)
3. Unique terms per family bar chart
4. Term support distribution histograms (per family)
5. Top-N most frequent terms (per family, horizontal bar charts)
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from snakemake.script import snakemake

from wfutils import get_logger
from wfutils.logging import log_snakemake_info

logger = get_logger()
log_snakemake_info(logger)

coverage_summary_path = Path(str(snakemake.input["coverage_summary"]))
term_supports_path = Path(str(snakemake.input["term_supports"]))
output_html = Path(str(snakemake.output[0]))

FAMILIES = ("go", "ec", "interpro", "pfam")
FAMILY_LABELS = {"go": "GO", "ec": "EC", "interpro": "InterPro", "pfam": "Pfam"}
FAMILY_COLORS = {
    "go": "#636EFA",
    "ec": "#EF553B",
    "interpro": "#00CC96",
    "pfam": "#AB63FA",
}

TOP_N = 20

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
# Section builders
# ---------------------------------------------------------------------------


def _build_coverage_table(coverage_df: pd.DataFrame) -> str:
    """Render coverage summary as an HTML table with counts and percentages."""
    rows_html: list[str] = []

    header = "<tr><th>Scope</th><th>Records</th><th>Any Annotation</th>"
    for fam in FAMILIES:
        label = FAMILY_LABELS[fam]
        header += (
            f"<th>{label} (count)</th><th>{label} (%)</th><th>{label} unique terms</th>"
        )
    header += "</tr>"

    for _, row in coverage_df.iterrows():
        n = int(row["n_records"])
        any_ann = int(row["with_any_annotation"])
        any_pct = f"{100 * any_ann / n:.1f}" if n else "0.0"

        cells = (
            f"<td>{row['scope']}</td><td>{n:,}</td><td>{any_ann:,} ({any_pct}%)</td>"
        )
        for fam in FAMILIES:
            count = int(row[f"with_{fam}"])
            pct = f"{100 * count / n:.1f}" if n else "0.0"
            unique = int(row[f"unique_{fam}_terms"])
            cells += f"<td>{count:,}</td><td>{pct}%</td><td>{unique:,}</td>"
        rows_html.append(f"<tr>{cells}</tr>")

    return (
        '<table class="stats-table">'
        f"<thead>{header}</thead>"
        f"<tbody>{''.join(rows_html)}</tbody>"
        "</table>"
    )


def _build_coverage_bar(coverage_df: pd.DataFrame) -> str:
    """Grouped bar chart: coverage count per annotation family, grouped by scope."""
    fig = go.Figure()

    for _, row in coverage_df.iterrows():
        scope = str(row["scope"])
        counts = [int(row[f"with_{fam}"]) for fam in FAMILIES]
        labels = [FAMILY_LABELS[fam] for fam in FAMILIES]
        fig.add_trace(go.Bar(name=scope, x=labels, y=counts))

    fig.update_layout(
        title="Annotation Coverage by Family",
        xaxis_title="Annotation Family",
        yaxis_title="Number of Proteins",
        barmode="group",
        template="plotly_white",
        height=400,
        margin=dict(t=50, b=50),
    )
    return _fig_to_div(fig)


def _build_unique_terms_bar(coverage_df: pd.DataFrame) -> str:
    """Bar chart of unique term counts per family (unique_accessions scope only)."""
    acc_row = coverage_df[coverage_df["scope"] == "unique_accessions"]
    if acc_row.empty:
        acc_row = coverage_df.iloc[:1]
    acc_row = acc_row.iloc[0]

    labels = [FAMILY_LABELS[fam] for fam in FAMILIES]
    counts = [int(acc_row[f"unique_{fam}_terms"]) for fam in FAMILIES]
    colors = [FAMILY_COLORS[fam] for fam in FAMILIES]

    fig = go.Figure(go.Bar(x=labels, y=counts, marker_color=colors))
    fig.update_layout(
        title="Unique Annotation Terms per Family (unique accessions)",
        xaxis_title="Annotation Family",
        yaxis_title="Number of Unique Terms",
        template="plotly_white",
        height=400,
        margin=dict(t=50, b=50),
    )
    return _fig_to_div(fig)


def _build_support_distributions(term_supports_df: pd.DataFrame) -> list[str]:
    """Per-family histogram of term support counts (how many proteins per term)."""
    divs: list[str] = []
    if term_supports_df.empty:
        return divs

    families_present = [
        fam for fam in FAMILIES if fam in term_supports_df["family"].values
    ]
    if not families_present:
        return divs

    n_fam = len(families_present)
    fig = make_subplots(
        rows=1,
        cols=n_fam,
        subplot_titles=[FAMILY_LABELS[fam] for fam in families_present],
        horizontal_spacing=0.08,
    )

    for i, fam in enumerate(families_present, 1):
        fam_df = term_supports_df[term_supports_df["family"] == fam]
        support_counts = fam_df["n_accessions"].value_counts().sort_index()
        fig.add_trace(
            go.Bar(
                x=support_counts.index.astype(str),
                y=support_counts.values,
                marker_color=FAMILY_COLORS[fam],
                showlegend=False,
            ),
            row=1,
            col=i,
        )
        fig.update_xaxes(title_text="Support (proteins)", row=1, col=i)
        fig.update_yaxes(title_text="Number of terms" if i == 1 else "", row=1, col=i)

    fig.update_layout(
        title_text="Term Support Distribution (how many proteins annotated per term)",
        template="plotly_white",
        height=400,
        margin=dict(t=80, b=50),
    )
    divs.append(_fig_to_div(fig))
    return divs


def _build_top_terms(term_supports_df: pd.DataFrame) -> list[str]:
    """Per-family horizontal bar chart of top-N most frequent terms."""
    divs: list[str] = []
    if term_supports_df.empty:
        return divs

    for fam in FAMILIES:
        fam_df = term_supports_df[term_supports_df["family"] == fam]
        if fam_df.empty:
            continue

        top = fam_df.head(TOP_N).copy()
        # Reverse for horizontal bar (highest at top)
        top = top.iloc[::-1]

        fig = go.Figure(
            go.Bar(
                x=top["n_accessions"],
                y=top["term"],
                orientation="h",
                marker_color=FAMILY_COLORS[fam],
            )
        )
        n_total = len(fam_df)
        fig.update_layout(
            title=f"{FAMILY_LABELS[fam]} — Top {min(TOP_N, len(top))} Terms (of {n_total:,} total)",
            xaxis_title="Number of Proteins",
            yaxis_title="",
            template="plotly_white",
            height=max(350, 20 * len(top) + 100),
            margin=dict(t=50, b=50, l=200),
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
<title>Annotation Coverage — Representation Eval</title>
<style>
body { font-family: Arial, sans-serif; margin: 20px; max-width: 1600px; margin-left: auto; margin-right: auto; }
h1, h2 { text-align: center; }
.stats-table { border-collapse: collapse; width: 100%; margin-bottom: 20px; font-size: 13px; }
.stats-table th, .stats-table td { border: 1px solid #ddd; padding: 6px 8px; text-align: right; }
.stats-table th { background-color: #f2f2f2; text-align: center; }
.stats-table td:first-child { text-align: left; }
.section { margin-bottom: 40px; }
</style>
</head>
<body>
"""

_TAIL = "</body></html>"


def build_html() -> str:
    coverage_df = pd.read_csv(coverage_summary_path, sep="\t")
    term_supports_df = pd.read_csv(term_supports_path, sep="\t")

    sections: list[str] = []
    sections.append("<h1>Annotation Coverage — Matched Swiss-Prot Entries</h1>")

    # 1. Coverage overview table
    sections.append("<h2>Coverage Overview</h2>")
    sections.append(_build_coverage_table(coverage_df))

    # 2. Coverage bar chart
    sections.append("<h2>Annotation Family Coverage</h2>")
    sections.append(_build_coverage_bar(coverage_df))

    # 3. Unique terms per family
    sections.append("<h2>Unique Terms per Family</h2>")
    sections.append(_build_unique_terms_bar(coverage_df))

    # 4. Term support distributions
    support_divs = _build_support_distributions(term_supports_df)
    if support_divs:
        sections.append("<h2>Term Support Distributions</h2>")
        sections.extend(support_divs)

    # 5. Top-N most frequent terms
    top_divs = _build_top_terms(term_supports_df)
    if top_divs:
        sections.append("<h2>Most Frequent Terms</h2>")
        sections.extend(top_divs)

    parts: list[str] = [_HEAD]
    for section in sections:
        parts.append('<div class="section">')
        parts.append(section)
        parts.append("</div>")
    parts.append(_TAIL)
    return "".join(parts)


def main() -> None:
    output_html.parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Loading annotation summaries from %s, %s",
        coverage_summary_path,
        term_supports_path,
    )
    html_str = build_html()
    logger.info("Writing annotation coverage dashboard to %s", output_html)
    output_html.write_text(html_str, encoding="utf-8")
    logger.info("Done.")


if __name__ == "__main__":
    main()
