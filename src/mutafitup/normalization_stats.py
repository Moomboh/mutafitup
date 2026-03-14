"""Compute per-task normalization statistics from training parquets.

During training, regression labels are normalized using max-absolute scaling
derived from the training split. The model outputs normalized values, so we
need the same scale factors to recover original-scale predictions at inference
time.

Some upstream datasets also ship labels in a display-space transform that is
different from the scientific unit shown to users. In those cases we export an
additional display scaling factor so inference clients can recover both the
training label scale and the final user-facing scale.

The statistics are written to a JSON file that is loaded alongside the ONNX
model by the inference frontend.
"""

import argparse
import json
import logging
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)

UNRESOLVED_MARKER = 999.0
REGRESSION_SUBSET_TYPES = ("per_protein_regression", "per_residue_regression")
LABEL_COLUMN = "score"
DISPLAY_SCALE_FACTORS = {
    # Schmirler/FLIP meltome labels are stored as temperature / 10 and rounded
    # to two decimals in the distributed training package, while the app should
    # display predictions in degrees Celsius.
    "meltome": 10.0,
}


def compute_normalization_stats(
    tasks: List[Dict[str, Any]],
    datasets_resplit_dir: str,
) -> Dict[str, Dict[str, float]]:
    """Compute normalization statistics for regression tasks.

    Parameters
    ----------
    tasks:
        List of expanded task dicts (as returned by ``expand_tasks()``).
        Each dict must have ``name`` and ``subset_type`` keys.
    datasets_resplit_dir:
        Path to the ``results/datasets_resplit`` directory containing
        ``{subset_type}/{task_name}/train.parquet`` files.

    Returns
    -------
    dict
        Mapping from task name to
        ``{"label_min", "label_max", "scale_factor", "display_scale_factor"}``.
        Only regression tasks are included.
    """
    stats: Dict[str, Dict[str, float]] = {}

    for task in tasks:
        name = task["name"]
        subset_type = task["subset_type"]

        if subset_type not in REGRESSION_SUBSET_TYPES:
            logger.debug("Skipping non-regression task %r (%s)", name, subset_type)
            continue

        parquet_path = os.path.join(
            datasets_resplit_dir, subset_type, name, "train.parquet"
        )
        logger.info("Reading training data for %r from %s", name, parquet_path)

        df = pd.read_parquet(parquet_path)
        values = _extract_resolved_values(df[LABEL_COLUMN], subset_type)

        if len(values) == 0:
            raise ValueError(
                f"No resolved scores found for task {name!r} in {parquet_path}"
            )

        label_min = float(np.min(values))
        label_max = float(np.max(values))
        scale_factor = max(abs(label_min), abs(label_max)) or 1.0
        display_scale_factor = float(DISPLAY_SCALE_FACTORS.get(name, 1.0))

        stats[name] = {
            "label_min": label_min,
            "label_max": label_max,
            "scale_factor": scale_factor,
            "display_scale_factor": display_scale_factor,
        }
        logger.info(
            "  %s: min=%.6f max=%.6f scale_factor=%.6f display_scale_factor=%.6f",
            name,
            label_min,
            label_max,
            scale_factor,
            display_scale_factor,
        )

    return stats


def _extract_resolved_values(
    series: pd.Series,
    subset_type: str,
) -> np.ndarray:
    """Flatten a label column and filter out the unresolved marker."""
    if subset_type == "per_residue_regression":
        all_values = []
        for row in series:
            for v in row:
                if v != UNRESOLVED_MARKER:
                    all_values.append(v)
        return np.array(all_values, dtype=np.float64)

    return np.array(
        [v for v in series if v != UNRESOLVED_MARKER],
        dtype=np.float64,
    )


def main() -> None:
    """CLI entry point for standalone testing."""
    parser = argparse.ArgumentParser(
        description="Compute normalization statistics from training parquets."
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        required=True,
        help="Task names and their subset types as name:subset_type pairs "
        "(e.g. meltome:per_protein_regression disorder:per_residue_regression)",
    )
    parser.add_argument(
        "--datasets-resplit-dir",
        default="results/datasets_resplit",
        help="Path to the datasets_resplit directory (default: results/datasets_resplit)",
    )
    parser.add_argument(
        "--output",
        default="-",
        help="Output JSON file path, or - for stdout (default: -)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    tasks = []
    for spec in args.tasks:
        name, subset_type = spec.split(":", 1)
        tasks.append({"name": name, "subset_type": subset_type})

    stats = compute_normalization_stats(tasks, args.datasets_resplit_dir)

    json_str = json.dumps(stats, indent=2)
    if args.output == "-":
        print(json_str)
    else:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            f.write(json_str)
            f.write("\n")
        logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
