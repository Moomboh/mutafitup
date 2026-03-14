"""Side-by-side comparison dashboard: Original (pre-resplit) vs Resplit.

Reads the CSV directories produced by ``data_stats`` (post-resplit) and
``data_stats_original`` (pre-resplit) and renders a single self-contained
HTML page with paired Plotly subplots for every section.

Sections
--------
1. Statistics comparison table (merged with Source column)
2. Split proportions (paired pie charts)
3. Sequence length distributions (paired histograms)
4. Sequence length box plots (paired box plots)
5. Score distributions (paired histograms, regression only)
6. Score box plots (paired box plots, regression only)
7. Label counts (paired bar charts, classification only)
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from snakemake.script import snakemake

from wfutils import get_logger
from wfutils.logging import log_snakemake_info

# ---------------------------------------------------------------------------
# Import helpers from the sibling dashboard_app module
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dashboard"))
from dashboard_app import generate_color_variations, load_csv_data  # noqa: E402

logger = get_logger()
log_snakemake_info(logger)

original_dir = Path(str(snakemake.input.original))
resplit_dir = Path(str(snakemake.input.resplit))
output_html = Path(str(snakemake.output[0]))

SOURCES = ("Original", "Resplit")

# ---------------------------------------------------------------------------
# Plotly JS inclusion helper
# ---------------------------------------------------------------------------

_first_fig = True


def _fig_to_div(fig: go.Figure) -> str:
    global _first_fig
    include_js = "inline" if _first_fig else False
    _first_fig = False
    return fig.to_html(full_html=False, include_plotlyjs=include_js)


# ---------------------------------------------------------------------------
# 1. Statistics comparison table
# ---------------------------------------------------------------------------


def _get_color_map(orig_data: dict, resplit_data: dict) -> dict[str, str]:
    """Build a dataset -> hex color mapping from split_proportions data."""
    color_map: dict[str, str] = {}
    for sp_df in (orig_data["split_proportions"], resplit_data["split_proportions"]):
        if sp_df.empty:
            continue
        for _, row in sp_df.drop_duplicates("dataset").iterrows():
            color_map.setdefault(row["dataset"], row["color"])
    return color_map


def _lighten_hex(hex_color: str, factor: float = 0.85) -> str:
    """Blend a hex color towards white.  factor=0 gives original, 1 gives white."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


def _fmt_pct_diff(orig_val, resplit_val) -> str:
    """Format a percent-difference string for a numeric cell."""
    try:
        o = float(orig_val)
        r = float(resplit_val)
    except (TypeError, ValueError):
        return "-"
    if o == 0:
        return "new" if r != 0 else "-"
    diff = 100.0 * (r - o) / abs(o)
    sign = "+" if diff >= 0 else ""
    return f"{sign}{diff:.1f}%"


def _build_stats_table(orig_data: dict, resplit_data: dict) -> str:
    orig_df = orig_data["statistics"].copy()
    resplit_df = resplit_data["statistics"].copy()

    if orig_df.empty and resplit_df.empty:
        return ""

    color_map = _get_color_map(orig_data, resplit_data)

    orig_df.insert(0, "Source", "Original")
    resplit_df.insert(0, "Source", "Resplit")

    merged = pd.concat([orig_df, resplit_df], ignore_index=True)
    # Order: Dataset alphabetical, Split as test > valid > train, Source as Original > Resplit
    split_order = pd.CategoricalDtype(["test", "valid", "train"], ordered=True)
    source_order = pd.CategoricalDtype(["Original", "Resplit"], ordered=True)
    merged["Split"] = merged["Split"].astype(split_order)
    merged["Source"] = merged["Source"].astype(source_order)
    merged = merged.sort_values(["Dataset", "Split", "Source"]).reset_index(drop=True)

    # Identify numeric columns for percent-difference calculation
    skip_cols = {"Source", "Dataset", "Split"}
    numeric_cols = [
        c
        for c in merged.columns
        if c not in skip_cols and pd.api.types.is_numeric_dtype(merged[c])
    ]

    # Add a "% Diff" column for Resplit rows
    diff_vals: list[str] = []
    for idx, row in merged.iterrows():
        if row["Source"] != "Resplit":
            diff_vals.append("")
            continue
        # Find the matching Original row
        orig_row = orig_df[
            (orig_df["Dataset"] == row["Dataset"]) & (orig_df["Split"] == row["Split"])
        ]
        if orig_row.empty:
            diff_vals.append("-")
            continue
        orig_row = orig_row.iloc[0]
        # Compute diff based on "Num Sequences" as the primary metric
        diff_vals.append(
            _fmt_pct_diff(orig_row.get("Num Sequences"), row.get("Num Sequences"))
        )
    merged.insert(merged.columns.get_loc("Source") + 1, "% Diff", diff_vals)

    # Build HTML table manually for per-row styling
    header_cols = list(merged.columns)
    lines: list[str] = []
    lines.append('<table class="stats-table">')
    lines.append("<thead><tr>")
    for col in header_cols:
        lines.append(f"<th>{col}</th>")
    lines.append("</tr></thead>")
    lines.append("<tbody>")

    # Lighten factor per split: train lightest, valid medium, test darkest
    split_lighten = {"train": 0.90, "valid": 0.82, "test": 0.74}

    for _, row in merged.iterrows():
        dataset = row.get("Dataset", "")
        split = str(row.get("Split", ""))
        is_resplit = row.get("Source") == "Resplit"
        factor = split_lighten.get(split, 0.85)
        bg_color = _lighten_hex(color_map.get(dataset, "#636EFA"), factor)
        style = f'style="background-color: {bg_color};'
        if is_resplit:
            style += " font-weight: bold;"
        style += '"'
        lines.append(f"<tr {style}>")
        for col in header_cols:
            val = row[col]
            if pd.isna(val):
                cell = "-"
            elif isinstance(val, float):
                cell = f"{val:g}"
            else:
                cell = str(val)
            lines.append(f"<td>{cell}</td>")
        lines.append("</tr>")

    lines.append("</tbody></table>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. Split proportions (paired pie charts)
# ---------------------------------------------------------------------------


def _build_pie_comparisons(orig_data: dict, resplit_data: dict) -> list[str]:
    divs: list[str] = []
    orig_sp = orig_data["split_proportions"]
    resplit_sp = resplit_data["split_proportions"]

    if orig_sp.empty and resplit_sp.empty:
        return divs

    all_datasets = sorted(
        set(orig_sp["dataset"].unique()) | set(resplit_sp["dataset"].unique())
    )

    for dataset in all_datasets:
        orig_ds = orig_sp[orig_sp["dataset"] == dataset]
        resplit_ds = resplit_sp[resplit_sp["dataset"] == dataset]

        orig_total = int(orig_ds["count"].sum()) if not orig_ds.empty else 0
        resplit_total = int(resplit_ds["count"].sum()) if not resplit_ds.empty else 0

        # Build per-split count lookup from original for computing differences
        orig_counts: dict[str, int] = {}
        if not orig_ds.empty:
            for _, row in orig_ds.iterrows():
                orig_counts[row["split"]] = int(row["count"])

        # Subplot titles: just the number, resplit also gets % diff
        orig_title = f"Original ({orig_total:,})"
        if orig_total > 0:
            diff_pct = 100.0 * (resplit_total - orig_total) / orig_total
            sign = "+" if diff_pct >= 0 else ""
            resplit_title = f"Resplit ({resplit_total:,}, {sign}{diff_pct:.1f}%)"
        else:
            resplit_title = f"Resplit ({resplit_total:,})"

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=[orig_title, resplit_title],
            specs=[[{"type": "pie"}, {"type": "pie"}]],
        )

        # Original pie: label + percent + value
        if not orig_ds.empty:
            base_color = orig_ds.iloc[0]["color"]
            colors = generate_color_variations(base_color, len(orig_ds))
            fig.add_trace(
                go.Pie(
                    labels=orig_ds["split"],
                    values=orig_ds["count"],
                    marker_colors=colors,
                    textinfo="label+percent+value",
                    textposition="inside",
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

        # Resplit pie: label + percent + value + per-split % diff from original
        if not resplit_ds.empty:
            base_color = resplit_ds.iloc[0]["color"]
            colors = generate_color_variations(base_color, len(resplit_ds))

            custom_text: list[str] = []
            resplit_vals = []
            for _, row in resplit_ds.iterrows():
                split_name = row["split"]
                count = int(row["count"])
                resplit_vals.append(count)
                total_here = resplit_total if resplit_total > 0 else 1
                pct_of_total = 100.0 * count / total_here
                orig_count = orig_counts.get(split_name, 0)
                if orig_count > 0:
                    split_diff = 100.0 * (count - orig_count) / orig_count
                    sign = "+" if split_diff >= 0 else ""
                    custom_text.append(
                        f"{split_name}<br>{pct_of_total:.1f}%<br>"
                        f"{count:,} ({sign}{split_diff:.1f}%)"
                    )
                else:
                    custom_text.append(
                        f"{split_name}<br>{pct_of_total:.1f}%<br>{count:,}"
                    )

            fig.add_trace(
                go.Pie(
                    labels=resplit_ds["split"],
                    values=resplit_ds["count"],
                    marker_colors=colors,
                    text=custom_text,
                    textinfo="text",
                    textposition="inside",
                    showlegend=False,
                ),
                row=1,
                col=2,
            )

        fig.update_layout(
            title=f"{dataset} — Split Distribution",
            width=800,
            height=400,
            margin=dict(t=80, b=40, l=40, r=40),
        )
        divs.append(_fig_to_div(fig))

    return divs


# ---------------------------------------------------------------------------
# 3 & 5. Histogram comparisons (seq_length or score)
# ---------------------------------------------------------------------------


def _build_histogram_comparisons(
    orig_data: dict,
    resplit_data: dict,
    key: str,
    column: str,
    title_suffix: str,
    bins: int = 20,
) -> list[str]:
    divs: list[str] = []
    orig_df = orig_data[key]
    resplit_df = resplit_data[key]

    if orig_df.empty and resplit_df.empty:
        return divs
    if column not in orig_df.columns and column not in resplit_df.columns:
        return divs

    all_datasets = sorted(
        set(orig_df["dataset"].unique() if not orig_df.empty else [])
        | set(resplit_df["dataset"].unique() if not resplit_df.empty else [])
    )

    for dataset in all_datasets:
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=list(SOURCES),
            shared_yaxes=True,
            horizontal_spacing=0.08,
        )

        # Determine shared x-range across both sources
        orig_ds = (
            orig_df[orig_df["dataset"] == dataset]
            if not orig_df.empty
            else pd.DataFrame()
        )
        resplit_ds = (
            resplit_df[resplit_df["dataset"] == dataset]
            if not resplit_df.empty
            else pd.DataFrame()
        )

        all_vals = pd.concat(
            [
                orig_ds[column]
                if not orig_ds.empty and column in orig_ds.columns
                else pd.Series(dtype=float),
                resplit_ds[column]
                if not resplit_ds.empty and column in resplit_ds.columns
                else pd.Series(dtype=float),
            ]
        )
        if all_vals.empty:
            continue

        x_min, x_max = float(all_vals.min()), float(all_vals.max())

        for col_idx, (source, ds_df) in enumerate(
            zip(SOURCES, [orig_ds, resplit_ds]), start=1
        ):
            if ds_df.empty:
                continue
            splits = ds_df["split"].unique()
            base_color = (
                ds_df.iloc[0]["color"]
                if "color" in ds_df.columns and not ds_df.empty
                else "#636EFA"
            )
            colors = generate_color_variations(base_color, len(splits))

            for i, split in enumerate(splits):
                split_data = ds_df[ds_df["split"] == split]
                if split_data.empty:
                    continue
                fig.add_trace(
                    go.Histogram(
                        x=split_data[column],
                        nbinsx=bins,
                        name=f"{split}",
                        marker_color=colors[i],
                        showlegend=(col_idx == 1),
                        legendgroup=split,
                    ),
                    row=1,
                    col=col_idx,
                )

            # Apply shared x-range
            axis_key = "xaxis" if col_idx == 1 else f"xaxis{col_idx}"
            fig.update_layout(**{f"{axis_key}_range": [x_min, x_max]})

        fig.update_layout(
            title=f"{dataset} — {title_suffix}",
            width=900,
            height=400,
            margin=dict(t=60, b=50, l=50, r=50),
        )
        divs.append(_fig_to_div(fig))

    return divs


# ---------------------------------------------------------------------------
# 4 & 6. Box plot comparisons (seq_length or score)
# ---------------------------------------------------------------------------


def _build_boxplot_comparisons(
    orig_data: dict,
    resplit_data: dict,
    key: str,
    column: str,
    title_suffix: str,
) -> list[str]:
    divs: list[str] = []
    orig_df = orig_data[key]
    resplit_df = resplit_data[key]

    if orig_df.empty and resplit_df.empty:
        return divs
    if column not in orig_df.columns and column not in resplit_df.columns:
        return divs

    all_datasets = sorted(
        set(orig_df["dataset"].unique() if not orig_df.empty else [])
        | set(resplit_df["dataset"].unique() if not resplit_df.empty else [])
    )

    for dataset in all_datasets:
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=list(SOURCES),
            shared_yaxes=True,
            horizontal_spacing=0.12,
        )

        orig_ds = (
            orig_df[orig_df["dataset"] == dataset]
            if not orig_df.empty
            else pd.DataFrame()
        )
        resplit_ds = (
            resplit_df[resplit_df["dataset"] == dataset]
            if not resplit_df.empty
            else pd.DataFrame()
        )

        for col_idx, (source, ds_df) in enumerate(
            zip(SOURCES, [orig_ds, resplit_ds]), start=1
        ):
            if ds_df.empty:
                continue
            splits = ds_df["split"].unique()
            base_color = (
                ds_df.iloc[0]["color"]
                if "color" in ds_df.columns and not ds_df.empty
                else "#636EFA"
            )
            colors = generate_color_variations(base_color, len(splits))

            for i, split in enumerate(splits):
                split_data = ds_df[ds_df["split"] == split]
                if split_data.empty:
                    continue
                fig.add_trace(
                    go.Box(
                        y=split_data[column],
                        name=split,
                        marker_color=colors[i],
                        showlegend=(col_idx == 1),
                        legendgroup=split,
                    ),
                    row=1,
                    col=col_idx,
                )

        fig.update_layout(
            title=f"{dataset} — {title_suffix}",
            width=800,
            height=400,
            margin=dict(t=60, b=50, l=50, r=50),
        )
        divs.append(_fig_to_div(fig))

    return divs


# ---------------------------------------------------------------------------
# 7. Label count comparisons (paired bar charts)
# ---------------------------------------------------------------------------


def _build_label_comparisons(orig_data: dict, resplit_data: dict) -> list[str]:
    divs: list[str] = []
    orig_lc = orig_data["label_counts"]
    resplit_lc = resplit_data["label_counts"]

    if orig_lc.empty and resplit_lc.empty:
        return divs

    all_datasets = sorted(
        set(orig_lc["dataset"].unique() if not orig_lc.empty else [])
        | set(resplit_lc["dataset"].unique() if not resplit_lc.empty else [])
    )

    for dataset in all_datasets:
        orig_ds = (
            orig_lc[orig_lc["dataset"] == dataset]
            if not orig_lc.empty
            else pd.DataFrame()
        )
        resplit_ds = (
            resplit_lc[resplit_lc["dataset"] == dataset]
            if not resplit_lc.empty
            else pd.DataFrame()
        )

        # Determine consistent label ordering from the union of both sources
        all_labels = sorted(
            set(orig_ds["label"].unique() if not orig_ds.empty else [])
            | set(resplit_ds["label"].unique() if not resplit_ds.empty else [])
        )

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=list(SOURCES),
            shared_yaxes=True,
            horizontal_spacing=0.08,
        )

        for col_idx, (source, ds_df) in enumerate(
            zip(SOURCES, [orig_ds, resplit_ds]), start=1
        ):
            if ds_df.empty:
                continue
            splits = ds_df["split"].unique()
            base_color = (
                ds_df.iloc[0]["color"]
                if "color" in ds_df.columns and not ds_df.empty
                else "#636EFA"
            )
            colors = generate_color_variations(base_color, len(splits))

            for i, split in enumerate(splits):
                split_data = ds_df[ds_df["split"] == split]
                if split_data.empty:
                    continue
                fig.add_trace(
                    go.Bar(
                        x=split_data["label"],
                        y=split_data["count"],
                        name=split,
                        marker_color=colors[i],
                        showlegend=(col_idx == 1),
                        legendgroup=split,
                    ),
                    row=1,
                    col=col_idx,
                )

            # Ensure consistent x-axis ordering
            axis_key = "xaxis" if col_idx == 1 else f"xaxis{col_idx}"
            fig.update_layout(
                **{
                    f"{axis_key}_categoryorder": "array",
                    f"{axis_key}_categoryarray": all_labels,
                }
            )

        fig.update_layout(
            title=f"{dataset} — Label Counts",
            width=max(800, 120 * len(all_labels)),
            height=400,
            margin=dict(t=60, b=50, l=50, r=50),
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
<title>Dataset Comparison: Original vs Resplit</title>
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
.charts { display: flex; flex-wrap: wrap; justify-content: center; gap: 10px; }
</style>
</head>
<body>
"""

_TAIL = "</body></html>"


def build_html() -> str:
    logger.info("Loading original data from %s", original_dir)
    orig_data = load_csv_data(original_dir)
    logger.info("Loading resplit data from %s", resplit_dir)
    resplit_data = load_csv_data(resplit_dir)

    sections: list[str] = []
    sections.append("<h1>Dataset Comparison: Original vs Resplit</h1>")

    # 1. Statistics table
    table_html = _build_stats_table(orig_data, resplit_data)
    if table_html:
        sections.append("<h2>General Statistics</h2>")
        sections.append(table_html)

    # Helper to add a chart section
    def _add_chart_section(title: str, chart_divs: list[str]) -> None:
        if not chart_divs:
            return
        sections.append(f"<h2>{title}</h2>")
        sections.append('<div class="charts">')
        sections.extend(chart_divs)
        sections.append("</div>")

    # 2. Split proportions
    _add_chart_section(
        "Split Proportions",
        _build_pie_comparisons(orig_data, resplit_data),
    )

    # 3. Sequence length distributions
    _add_chart_section(
        "Sequence Length Distributions",
        _build_histogram_comparisons(
            orig_data,
            resplit_data,
            "sequence_lengths",
            "seq_length",
            "Sequence Length Distribution",
        ),
    )

    # 4. Sequence length box plots
    _add_chart_section(
        "Sequence Length Box Plots",
        _build_boxplot_comparisons(
            orig_data,
            resplit_data,
            "sequence_lengths",
            "seq_length",
            "Sequence Length Distribution",
        ),
    )

    # 5. Score distributions
    _add_chart_section(
        "Score Distributions",
        _build_histogram_comparisons(
            orig_data,
            resplit_data,
            "scores",
            "score",
            "Score Distribution",
        ),
    )

    # 6. Score box plots
    _add_chart_section(
        "Score Box Plots",
        _build_boxplot_comparisons(
            orig_data,
            resplit_data,
            "scores",
            "score",
            "Score Distribution",
        ),
    )

    # 7. Label counts
    _add_chart_section(
        "Label Counts",
        _build_label_comparisons(orig_data, resplit_data),
    )

    parts: list[str] = [_HEAD]
    for section in sections:
        parts.append('<div class="section">')
        parts.append(section)
        parts.append("</div>")
    parts.append(_TAIL)
    return "".join(parts)


def main() -> None:
    output_html.parent.mkdir(parents=True, exist_ok=True)
    html_str = build_html()
    logger.info("Writing comparison dashboard to %s", output_html)
    output_html.write_text(html_str, encoding="utf-8")
    logger.info("Done.")


if __name__ == "__main__":
    main()
