"""Generate a self-contained static HTML dashboard for resplit summary statistics.

Reads ``resplit_summary.json`` produced by the resplit pipeline and renders
sections with Plotly figures embedded inline.

Sections
--------
1. Overview table – per-dataset original vs new split counts
2. Split comparison – grouped bar chart of original vs new sizes
3. Step-by-step flow – per-dataset waterfall showing sequence movement
4. Phase A reconstruction – details for datasets with test reconstruction
"""

import json
from pathlib import Path


import math

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from snakemake.script import snakemake

from wfutils import get_logger
from wfutils.logging import log_snakemake_info

logger = get_logger()
log_snakemake_info(logger)

input_json = Path(str(snakemake.input["json"]))
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
# Colour palette
# ---------------------------------------------------------------------------

COLORS = {
    "train": "#636EFA",
    "valid": "#EF553B",
    "test": "#00CC96",
    "dropped": "#AB63FA",
}

STEP_COLORS = {
    "phase_a": "#FFA15A",
    "step1_merge": "#636EFA",
    "step2_drop": "#EF553B",
    "step3_cross": "#00CC96",
    "step4_topup": "#AB63FA",
}


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------


def _pct_change(old: int, new: int) -> str:
    """Format a relative percentage change like ``+5.2%`` or ``-12.3%``."""
    if old == 0:
        return "n/a"
    pct = (new - old) / old * 100
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}%"


def _build_overview_table(summary: dict) -> str:
    """Render an HTML table with one row per dataset."""
    rows: list[str] = []
    for ds_name in sorted(summary):
        ds = summary[ds_name]
        orig = ds["original_counts"]
        new = ds["new_counts"]
        total = ds["total"]
        dropped = new.get("dropped", 0)
        total_kept = new["train"] + new["valid"] + new["test"]
        rows.append(
            "<tr>"
            f"<td style='text-align:left'>{ds_name}</td>"
            # Original
            f"<td>{orig['train']:,}</td>"
            f"<td>{orig['valid']:,}</td>"
            f"<td>{orig['test']:,}</td>"
            f"<td>{total:,}</td>"
            # After resplit
            f"<td>{new['train']:,}</td>"
            f"<td>{new['valid']:,}</td>"
            f"<td>{new['test']:,}</td>"
            f"<td>{dropped:,}</td>"
            f"<td>{total_kept:,}</td>"
            # Relative change
            f"<td>{_pct_change(orig['train'], new['train'])}</td>"
            f"<td>{_pct_change(orig['valid'], new['valid'])}</td>"
            f"<td>{_pct_change(orig['test'], new['test'])}</td>"
            f"<td>{_pct_change(total, total_kept)}</td>"
            "</tr>"
        )

    return (
        '<table class="stats-table">'
        "<thead><tr>"
        '<th rowspan="2">Dataset</th>'
        '<th colspan="3">Original</th>'
        '<th rowspan="2">Total</th>'
        '<th colspan="4">After Resplit</th>'
        '<th rowspan="2">Kept</th>'
        '<th colspan="4">Relative Change</th>'
        "</tr><tr>"
        "<th>Train</th><th>Valid</th><th>Test</th>"
        "<th>Train</th><th>Valid</th><th>Test</th><th>Dropped</th>"
        "<th>Train</th><th>Valid</th><th>Test</th><th>Total</th>"
        "</tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody>"
        "</table>"
    )


def _build_split_comparison(summary: dict) -> str:
    """Per-dataset subplots: original vs new split sizes, each with own y-axis."""
    datasets = sorted(summary)
    n = len(datasets)
    cols = min(4, n)
    rows = math.ceil(n / cols)
    splits = ["train", "valid", "test", "dropped"]

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=datasets,
        vertical_spacing=0.08,
        horizontal_spacing=0.06,
    )

    for idx, ds_name in enumerate(datasets):
        r = idx // cols + 1
        c = idx % cols + 1
        show_legend = idx == 0

        orig = summary[ds_name]["original_counts"]
        new = summary[ds_name]["new_counts"]

        for split in splits:
            color = COLORS[split]
            fig.add_trace(
                go.Bar(
                    name=f"{split} (original)",
                    x=[split],
                    y=[orig.get(split, 0)],
                    marker_color=color,
                    opacity=0.45,
                    legendgroup=f"{split}_orig",
                    showlegend=show_legend,
                ),
                row=r,
                col=c,
            )
            fig.add_trace(
                go.Bar(
                    name=f"{split} (resplit)",
                    x=[split],
                    y=[new.get(split, 0)],
                    marker_color=color,
                    opacity=1.0,
                    legendgroup=f"{split}_new",
                    showlegend=show_legend,
                ),
                row=r,
                col=c,
            )

    fig.update_layout(
        barmode="group",
        title_text="Original vs Resplit Split Sizes",
        template="plotly_white",
        height=280 * rows,
        margin=dict(t=80, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return _fig_to_div(fig)


def _build_step_composition(summary: dict) -> list[str]:
    """Horizontal stacked bars showing split composition at each pipeline stage.

    Each dataset gets a subplot.  Rows (stages) from bottom to top:
    Original → (Phase A) → Step 1 → Step 2 → Step 3 → Step 4 (final).
    Segments: train, valid, test, dropped.
    """
    ds_with_steps = {
        name: data for name, data in sorted(summary.items()) if data.get("steps")
    }
    if not ds_with_steps:
        return []

    datasets = list(ds_with_steps.keys())
    n = len(datasets)
    cols = min(4, n)
    rows = math.ceil(n / cols)

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=datasets,
        horizontal_spacing=0.06,
        vertical_spacing=0.06,
    )

    for idx, ds_name in enumerate(datasets):
        r = idx // cols + 1
        c = idx % cols + 1
        show_legend = idx == 0
        ds = ds_with_steps[ds_name]
        steps = ds["steps"]
        orig = ds["original_counts"]

        # ------------------------------------------------------------------
        # Build stage data: list of (label, train, valid, test, dropped)
        # Stages are appended top-to-bottom so they display bottom-to-top
        # in the horizontal bar (Plotly categorical y order).
        # ------------------------------------------------------------------
        stages: list[tuple] = []

        # Final (Step 4)
        s4 = steps.get("step4_topup", {})
        new = ds["new_counts"]
        stages.append(
            (
                "Final",
                s4.get("final_train", new["train"]),
                s4.get("final_valid", new["valid"]),
                new["test"],
                new.get("dropped", 0),
            )
        )

        # Step 3: cross-contamination — train/valid separated, test unchanged
        s3 = steps.get("step3_cross_contamination", {})
        s1 = steps.get("step1_merge", {})
        s2 = steps.get("step2_drop_within_test_similar", {})
        test_after_s2 = s1.get("test", orig["test"]) - s2.get("dropped", 0)
        if s3:
            stages.append(
                (
                    "3: cross-contam",
                    s3.get("train", 0),
                    s3.get("valid", 0),
                    test_after_s2,
                    s2.get("dropped", 0),
                )
            )

        # Step 2: drop within-test similar — pool still merged
        if s2:
            stages.append(
                (
                    "2: drop similar",
                    s2.get("train_valid_after", 0),
                    0,
                    test_after_s2,
                    s2.get("dropped", 0),
                )
            )

        # Step 1: merge — train+valid pooled
        if s1:
            stages.append(
                (
                    "1: merge",
                    s1.get("train_valid", 0),
                    0,
                    s1.get("test", 0),
                    0,
                )
            )

        # Phase A (if present)
        pa = steps.get("phase_a")
        if pa:
            stages.append(
                (
                    "Phase A",
                    pa["train_valid_count"],
                    0,
                    pa["final_test_count"],
                    0,
                )
            )

        # Original
        stages.append(
            (
                "Original",
                orig["train"],
                orig["valid"],
                orig["test"],
                0,
            )
        )

        # Reverse so Original is at the bottom
        stages.reverse()
        stage_labels = [s[0] for s in stages]

        # Add traces — one per segment type
        segment_defs = [
            ("Train", [s[1] for s in stages], COLORS["train"]),
            ("Valid", [s[2] for s in stages], COLORS["valid"]),
            ("Test", [s[3] for s in stages], COLORS["test"]),
            ("Dropped", [s[4] for s in stages], COLORS["dropped"]),
        ]

        for seg_name, seg_vals, seg_color in segment_defs:
            # Skip the dropped trace entirely when all zeros for this dataset
            if seg_name == "Dropped" and all(v == 0 for v in seg_vals):
                continue
            fig.add_trace(
                go.Bar(
                    name=seg_name,
                    y=stage_labels,
                    x=seg_vals,
                    orientation="h",
                    marker_color=seg_color,
                    legendgroup=seg_name,
                    showlegend=show_legend,
                    text=[f"{v:,}" if v > 0 else "" for v in seg_vals],
                    textposition="inside",
                    insidetextanchor="middle",
                    textfont=dict(color="white", size=10),
                ),
                row=r,
                col=c,
            )

    fig.update_layout(
        barmode="stack",
        title_text="Per-Dataset Split Composition at Each Stage",
        template="plotly_white",
        height=max(500, 220 * rows),
        margin=dict(t=80, b=40, l=120),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return [_fig_to_div(fig)]


def _build_step_detail_bars(summary: dict) -> list[str]:
    """Horizontal stacked bar showing per-step numbers for all datasets."""
    divs: list[str] = []

    # Collect datasets that have step data
    ds_with_steps = {
        name: data for name, data in sorted(summary.items()) if data.get("steps")
    }
    if not ds_with_steps:
        return divs

    datasets = list(ds_with_steps.keys())

    # Build a bar for each step metric
    s2_dropped = []
    s3_moved = []
    s4_topped = []

    for ds_name in datasets:
        steps = ds_with_steps[ds_name]["steps"]
        s2_dropped.append(
            steps.get("step2_drop_within_test_similar", {}).get("dropped", 0)
        )
        s3_moved.append(
            steps.get("step3_cross_contamination", {}).get("moved_to_valid", 0)
        )
        s4_topped.append(steps.get("step4_topup", {}).get("topped_up", 0))

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Step 2: dropped (within-test similar)",
            y=datasets,
            x=s2_dropped,
            orientation="h",
            marker_color="#EF553B",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Step 3: moved to valid (cross-contamination)",
            y=datasets,
            x=s3_moved,
            orientation="h",
            marker_color="#00CC96",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Step 4: topped up to valid",
            y=datasets,
            x=s4_topped,
            orientation="h",
            marker_color="#AB63FA",
        )
    )

    fig.update_layout(
        barmode="stack",
        title="Sequences affected per step (all datasets)",
        xaxis_title="Number of sequences",
        template="plotly_white",
        height=max(350, 40 * len(datasets) + 120),
        margin=dict(t=60, b=50, l=180),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    divs.append(_fig_to_div(fig))
    return divs


def _build_phase_a_section(summary: dict) -> list[str]:
    """Bar charts for datasets that went through Phase A test reconstruction."""
    divs: list[str] = []

    phase_a_ds = {
        name: data
        for name, data in sorted(summary.items())
        if data.get("steps", {}).get("phase_a")
    }
    if not phase_a_ds:
        return divs

    datasets = list(phase_a_ds.keys())
    total_pool = []
    overlap_test = []
    topup_test = []
    final_test = []
    train_valid = []

    for ds_name in datasets:
        pa = phase_a_ds[ds_name]["steps"]["phase_a"]
        total_pool.append(pa["total_pool"])
        overlap_test.append(pa["overlap_test_count"])
        topup_test.append(pa["topup_test_count"])
        final_test.append(pa["final_test_count"])
        train_valid.append(pa["train_valid_count"])

    fig = go.Figure()
    fig.add_trace(
        go.Bar(name="Total pool", x=datasets, y=total_pool, marker_color="#636EFA")
    )
    fig.add_trace(
        go.Bar(
            name="Overlap test (kept)",
            x=datasets,
            y=overlap_test,
            marker_color="#00CC96",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Top-up test (added)",
            x=datasets,
            y=topup_test,
            marker_color="#FFA15A",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Final test set",
            x=datasets,
            y=final_test,
            marker_color="#AB63FA",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Train+valid pool",
            x=datasets,
            y=train_valid,
            marker_color="#EF553B",
        )
    )

    fig.update_layout(
        barmode="group",
        title="Phase A — Test Set Reconstruction",
        xaxis_title="Dataset",
        yaxis_title="Number of sequences",
        template="plotly_white",
        height=450,
        margin=dict(t=60, b=80),
        xaxis_tickangle=-45,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
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
<title>Resplit Summary Dashboard</title>
<style>
body { font-family: Arial, sans-serif; margin: 20px; max-width: 1400px; margin-left: auto; margin-right: auto; }
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


def build_html(summary: dict) -> str:
    """Assemble the full HTML dashboard from the summary JSON data."""
    sections: list[str] = []
    sections.append("<h1>Resplit Summary Dashboard</h1>")

    if not summary:
        sections.append("<p>No resplit summary data available.</p>")
        parts = [_HEAD]
        for s in sections:
            parts.append('<div class="section">')
            parts.append(s)
            parts.append("</div>")
        parts.append(_TAIL)
        return "".join(parts)

    # 1. Overview table
    sections.append("<h2>Overview</h2>")
    sections.append(_build_overview_table(summary))

    # 2. Split comparison
    sections.append("<h2>Original vs Resplit Split Sizes</h2>")
    sections.append(_build_split_comparison(summary))

    # 3. Step detail bars (all datasets)
    step_divs = _build_step_detail_bars(summary)
    if step_divs:
        sections.append("<h2>Sequences Affected per Step</h2>")
        sections.extend(step_divs)

    # 4. Per-dataset split composition at each stage
    composition_divs = _build_step_composition(summary)
    if composition_divs:
        sections.append("<h2>Per-Dataset Split Composition at Each Stage</h2>")
        sections.extend(composition_divs)

    # 5. Phase A reconstruction
    phase_a_divs = _build_phase_a_section(summary)
    if phase_a_divs:
        sections.append("<h2>Phase A — Test Set Reconstruction</h2>")
        sections.extend(phase_a_divs)

    parts: list[str] = [_HEAD]
    for section in sections:
        parts.append('<div class="section">')
        parts.append(section)
        parts.append("</div>")
    parts.append(_TAIL)
    return "".join(parts)


def main() -> None:
    output_html.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Loading resplit summary from %s", input_json)
    with open(input_json, encoding="utf-8") as fh:
        summary = json.load(fh)
    logger.info("Building resplit summary dashboard for %d datasets", len(summary))
    html_str = build_html(summary)
    logger.info("Writing resplit summary dashboard to %s", output_html)
    output_html.write_text(html_str, encoding="utf-8")
    logger.info("Done.")


if __name__ == "__main__":
    main()
