"""Helpers for exporting evaluation summaries as reusable Typst table fragments."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Union

from mutafitup.evaluation.plotting import _collect_absolute_values
from mutafitup.task_display_names import task_display_name


def _escape_typst(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _render_cell(value: str, *, bold: bool = False) -> str:
    escaped = _escape_typst(value)
    if bold:
        return f"[#strong[{escaped}]]"
    return f"[{escaped}]"


def _format_metric(value: float | None, ci95: float | None) -> str:
    if value is None or ci95 is None or math.isnan(value) or math.isnan(ci95):
        return "N/A"
    return f"{value:.3f} +- {ci95:.3f}"


def _format_number(value: float | None, digits: int = 2, suffix: str = "") -> str:
    if value is None or math.isnan(value):
        return "N/A"
    return f"{value:.{digits}f}{suffix}"


def _render_typst_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    aligns: Sequence[str],
) -> str:
    header_cells = ",\n    ".join(_render_cell(h, bold=True) for h in headers)
    row_cells: List[str] = []
    for row in rows:
        row_cells.extend(row)

    body = ",\n    ".join([header_cells] + row_cells)
    columns = ", ".join(["auto"] * len(headers))
    align = ", ".join(aligns)
    return (
        "#table(\n"
        f"  columns: ({columns}),\n"
        f"  align: ({align}),\n"
        "  stroke: (x, y) => if y == 1 { (bottom: 0.8pt + black) } else { (bottom: 0.3pt + luma(180)) },\n"
        "  inset: (x: 0.45em, y: 0.30em),\n"
        f"  {body},\n"
        ")\n"
    )


def export_full_results_typst(
    evaluation: Dict[str, Any],
    metric_jsons: Dict[str, Dict[str, Any]],
) -> str:
    labels, task_names, values, ci95s = _collect_absolute_values(
        evaluation, metric_jsons
    )
    task_metric_labels = {
        ds["task"]: ds.get("metric_label", ds["metric"])
        for ds in evaluation["datasets"]
    }

    headers = ["Task", "Metric", *labels]
    rows: List[List[str]] = []
    for j, task in enumerate(task_names):
        column = values[:, j]
        valid_values = [v for v in column if not math.isnan(v)]
        best_value = max(valid_values) if valid_values else None

        rendered_row = [
            _render_cell(task_display_name(task)),
            _render_cell(task_metric_labels[task]),
        ]
        for i in range(len(labels)):
            value = None if math.isnan(values[i, j]) else float(values[i, j])
            ci = None if math.isnan(ci95s[i, j]) else float(ci95s[i, j])
            cell_text = _format_metric(value, ci)
            is_best = (
                value is not None
                and best_value is not None
                and math.isclose(value, best_value)
            )
            rendered_row.append(_render_cell(cell_text, bold=is_best))
        rows.append(rendered_row)

    return _render_typst_table(
        headers,
        rows,
        aligns=["left", "left", *(["right"] * len(labels))],
    )


def export_delta_summary_typst(summary: Dict[str, Any]) -> str:
    headers = [
        "Method",
        "Mean delta (%)",
        "Median delta (%)",
        "CI95 of mean",
        "Wins",
        "Losses",
        "Neutral",
        "Tasks",
    ]
    rows: List[List[str]] = []
    delta_rows = summary.get("delta_summary", {}).get("rows", [])
    best_mean = None
    available = [
        row["mean_delta_percent"]
        for row in delta_rows
        if row["mean_delta_percent"] is not None
    ]
    if available:
        best_mean = max(available)

    for row in delta_rows:
        mean_delta = row["mean_delta_percent"]
        rows.append(
            [
                _render_cell(row["label"]),
                _render_cell(
                    _format_number(mean_delta, digits=2, suffix="%"),
                    bold=mean_delta is not None
                    and best_mean is not None
                    and math.isclose(mean_delta, best_mean),
                ),
                _render_cell(
                    _format_number(row["median_delta_percent"], digits=2, suffix="%")
                ),
                _render_cell(
                    _format_number(row["ci95_mean_delta_percent"], digits=2, suffix="%")
                ),
                _render_cell(str(row["significant_wins"])),
                _render_cell(str(row["significant_losses"])),
                _render_cell(str(row["nonsignificant"])),
                _render_cell(str(row["tasks_compared"])),
            ]
        )

    return _render_typst_table(
        headers,
        rows,
        aligns=["left", "right", "right", "right", "right", "right", "right", "right"],
    )


def write_text(path: Union[str, Path], text: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
