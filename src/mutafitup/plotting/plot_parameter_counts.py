from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


SECTION_COLORS = {
    "heads_only": "#666666",
    "lora": "#4363d8",
    "accgrad_lora": "#ffe119",
    "align_lora": "#800000",
}


def plot_trainable_vs_frozen(
    summary_df: pd.DataFrame,
    output_path: str,
    figsize: Optional[tuple[float, float]] = None,
) -> None:
    if summary_df.empty:
        raise ValueError("summary_df must not be empty")

    df = summary_df.copy()
    n_runs = len(df)
    if figsize is None:
        figsize = (14, max(8.0, 0.32 * n_runs + 2.5))

    fig, ax = plt.subplots(figsize=figsize)

    y = list(range(n_runs))
    frozen_m = df["frozen_params"] / 1_000_000
    trainable_m = df["trainable_params"] / 1_000_000
    colors = [SECTION_COLORS.get(section, "#1f77b4") for section in df["section"]]

    ax.barh(y, frozen_m, color="#d9d9d9", label="Frozen")
    ax.barh(y, trainable_m, left=frozen_m, color=colors, label="Trainable")

    labels = [f"{row.section} / {row.run_id}" for row in df.itertuples(index=False)]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Parameters (millions)")
    ax.set_title("Trainable vs frozen parameters by configured run")
    ax.grid(axis="x", alpha=0.25)

    for idx, row in enumerate(df.itertuples(index=False)):
        total_m = row.total_params / 1_000_000
        trainable_m_here = row.trainable_params / 1_000_000
        pct = row.trainable_fraction * 100
        ax.text(
            total_m + max(total_m * 0.01, 0.2),
            idx,
            f"{trainable_m_here:.2f}M ({pct:.2f}%)",
            va="center",
            fontsize=8,
        )

    # Section separators
    prev_section = None
    for idx, section in enumerate(df["section"]):
        if prev_section is not None and section != prev_section:
            ax.axhline(idx - 0.5, color="#999999", linewidth=0.8)
        prev_section = section

    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
