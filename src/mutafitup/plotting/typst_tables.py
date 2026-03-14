"""Helpers for exporting parameter summaries as reusable Typst tables."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd


SECTION_LABELS = {
    "heads_only": "Heads only",
    "lora": "LoRA",
    "accgrad_lora": "AccGrad LoRA",
    "align_lora": "Align LoRA",
}


def _escape_typst(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _render_cell(value: str, *, bold: bool = False) -> str:
    escaped = _escape_typst(value)
    if bold:
        return f"[#strong[{escaped}]]"
    return f"[{escaped}]"


def _render_typst_table(headers: list[str], rows: list[list[str]]) -> str:
    header_cells = ",\n    ".join(_render_cell(h, bold=True) for h in headers)
    row_cells: list[str] = []
    for row in rows:
        row_cells.extend(row)

    body = ",\n    ".join([header_cells] + row_cells)
    columns = ", ".join(["auto"] * len(headers))
    align = "left, left, right, right, right, right, right, right"
    return (
        "#table(\n"
        f"  columns: ({columns}),\n"
        f"  align: ({align}),\n"
        "  stroke: (x, y) => if y == 1 { (bottom: 0.8pt + black) } else { (bottom: 0.3pt + luma(180)) },\n"
        "  inset: (x: 0.45em, y: 0.30em),\n"
        f"  {body},\n"
        ")\n"
    )


def _format_int(value: int) -> str:
    return f"{value:,}"


def _format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def export_parameter_summary_typst(summary_df: pd.DataFrame) -> str:
    headers = [
        "Method",
        "Backbone",
        "Tasks",
        "LoRA rank",
        "Total params",
        "Trainable",
        "Frozen",
        "Trainable %",
    ]
    rows: list[list[str]] = []

    for row in summary_df.itertuples(index=False):
        method = f"{SECTION_LABELS.get(row.section, row.section)} / {row.run_id}"
        lora_rank = "-" if pd.isna(row.lora_rank) else str(int(row.lora_rank))
        rows.append(
            [
                _render_cell(method),
                _render_cell(str(row.checkpoint)),
                _render_cell(str(int(row.num_tasks))),
                _render_cell(lora_rank),
                _render_cell(_format_int(int(row.total_params))),
                _render_cell(_format_int(int(row.trainable_params))),
                _render_cell(_format_int(int(row.frozen_params))),
                _render_cell(_format_percent(float(row.trainable_fraction))),
            ]
        )

    return _render_typst_table(headers, rows)


def write_text(path: Union[str, Path], text: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
