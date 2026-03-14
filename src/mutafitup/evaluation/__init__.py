"""Evaluation infrastructure for mutafitup.

Provides bootstrap index generation, per-run metric computation with
protein-level bootstraps, config resolution, and evaluation plotting
(delta heatmaps + absolute bar charts).
"""

from .bootstrap import generate_bootstrap_indices, load_bootstrap_indices
from .metrics import compute_metrics
from .plotting import plot_delta_heatmap, plot_absolute_bars
from .resolve import resolve_evaluation_inputs

__all__ = [
    "generate_bootstrap_indices",
    "load_bootstrap_indices",
    "compute_metrics",
    "plot_delta_heatmap",
    "plot_absolute_bars",
    "resolve_evaluation_inputs",
]
