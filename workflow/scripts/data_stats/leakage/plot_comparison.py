"""Plot a side-by-side leakage comparison: pre-resplit vs post-resplit.

Produces a single figure with two panels:

1. **Left** — Pre-resplit (original) leakage heatmap (N+1) x (N+1)
2. **Right** — Post-resplit leakage heatmap (N+1) x (N+1)

Searches run on **original (pre-resplit) merged FASTAs**, so all FASTA IDs
encode original split assignments (``{dataset}_{split}_{index}``).  The
original heatmap uses ``parse_split_from_id()`` directly.  The resplit
heatmap uses ``split_assignments.tsv`` + ``sequence_metadata.tsv`` to map
original FASTA IDs to their new (resplit) split assignments.

Supports two variants:
- ``all``: all dataset pairs (diagonal included)
- ``cross_task``: diagonal (within-dataset) zeroed out and excluded from totals
"""

from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from snakemake.script import snakemake

from wfutils import get_logger
from wfutils.logging import log_snakemake_info

logger = get_logger()
log_snakemake_info(logger)

# Increase base font size for all plot elements
plt.rcParams.update({"font.size": 16})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def count_fasta_sequences(fasta_path: Path) -> int:
    """Count the number of sequences in a FASTA file."""
    count = 0
    with fasta_path.open() as fh:
        for line in fh:
            if line.startswith(">"):
                count += 1
    return count


def read_fasta_id_sequences(fasta_path: Path) -> dict[str, str]:
    """Read a FASTA file and return ``{header_id: sequence}`` mapping."""
    result: dict[str, str] = {}
    current_id: str | None = None
    seq_parts: list[str] = []

    with fasta_path.open() as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if current_id is not None:
                    result[current_id] = "".join(seq_parts)
                current_id = line[1:].split()[0]  # first word after >
                seq_parts = []
            elif current_id is not None:
                seq_parts.append(line)
        if current_id is not None:
            result[current_id] = "".join(seq_parts)

    return result


def parse_split_from_id(seq_id: str, dataset: str) -> str | None:
    """Extract the split from a FASTA ID like ``{dataset}_{split}_{index}``."""
    prefix = dataset + "_"
    if not seq_id.startswith(prefix):
        return None
    suffix = seq_id[len(prefix) :]
    for s in ("train", "valid", "test"):
        if suffix.startswith(s + "_") or suffix == s:
            return s
    return None


def build_resplit_id_mapping(
    datasets: list[str],
    fasta_original_dir: Path,
    metadata_path: Path,
    assignments_path: Path,
) -> dict[str, str | None]:
    """Map original FASTA IDs to their resplit split assignment.

    Uses ``sequence_metadata.tsv`` and ``split_assignments.tsv`` from the
    resplit pipeline to avoid loading actual resplit parquet/FASTA files.

    Returns:
        ``{fasta_id: "train"|"valid"|"test"|None}`` where ``None`` means
        the sequence was dropped during resplit.
    """
    # 1. Build sequence -> seq_id from metadata
    logger.info("Loading sequence_metadata.tsv for resplit mapping...")
    meta_df = pd.read_csv(metadata_path, sep="\t", dtype={"seq_id": str})
    seq_to_seqid: dict[str, str] = dict(zip(meta_df["sequence"], meta_df["seq_id"]))
    logger.info("  %d unique sequences in metadata", len(seq_to_seqid))

    # 2. Build (seq_id, dataset) -> new_split from assignments
    logger.info("Loading split_assignments.tsv for resplit mapping...")
    assign_df = pd.read_csv(assignments_path, sep="\t", dtype={"seq_id": str})
    seqid_ds_to_split: dict[tuple[str, str], str] = {}
    for _, row in assign_df.iterrows():
        seqid_ds_to_split[(str(row["seq_id"]), str(row["dataset"]))] = str(
            row["new_split"]
        )
    logger.info("  %d assignment entries", len(seqid_ds_to_split))

    # 3. For each dataset, read original FASTAs and compose the mapping
    id_to_split: dict[str, str | None] = {}
    n_mapped = 0
    n_dropped = 0
    n_missing_seq = 0
    n_missing_assign = 0

    for ds in datasets:
        for split in ("train", "valid", "test"):
            fasta_path = fasta_original_dir / f"{ds}_{split}.fasta"
            if not fasta_path.exists():
                logger.warning("Missing original FASTA: %s", fasta_path)
                continue

            id_seqs = read_fasta_id_sequences(fasta_path)
            for fasta_id, seq in id_seqs.items():
                seq_id = seq_to_seqid.get(seq)
                if seq_id is None:
                    # Sequence not in resplit metadata (shouldn't happen if
                    # all leakage datasets are in the resplit config)
                    n_missing_seq += 1
                    id_to_split[fasta_id] = None
                    continue

                new_split = seqid_ds_to_split.get((seq_id, ds))
                if new_split is None:
                    # No assignment for this (seq_id, dataset) pair
                    n_missing_assign += 1
                    id_to_split[fasta_id] = None
                    continue

                if new_split == "dropped":
                    id_to_split[fasta_id] = None
                    n_dropped += 1
                else:
                    id_to_split[fasta_id] = new_split
                    n_mapped += 1

    logger.info(
        "Resplit ID mapping: %d mapped, %d dropped, "
        "%d missing-in-metadata, %d missing-assignment",
        n_mapped,
        n_dropped,
        n_missing_seq,
        n_missing_assign,
    )

    return id_to_split


def load_filtered_hits(
    result_tsv: Path,
    query_dataset: str,
    target_dataset: str,
    query_split: str,
    target_split: str,
    query_id_to_split: dict[str, str | None] | None = None,
    target_id_to_split: dict[str, str | None] | None = None,
) -> tuple[set[str], set[str]]:
    """Read a combined result TSV and return (query_ids, target_ids) for the
    requested split combination.

    When ``query_id_to_split`` or ``target_id_to_split`` are provided, they
    are used to look up the split of each hit ID instead of parsing it from
    the FASTA ID string.  This supports the resplit heatmap where the FASTA
    IDs encode original splits but we need resplit split assignments.
    """
    query_ids: set[str] = set()
    target_ids: set[str] = set()
    if not result_tsv.exists():
        return query_ids, target_ids

    with result_tsv.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            qid, tid = parts[0], parts[1]

            if query_id_to_split is not None:
                q_split = query_id_to_split.get(qid)
            else:
                q_split = parse_split_from_id(qid, query_dataset)

            if target_id_to_split is not None:
                t_split = target_id_to_split.get(tid)
            else:
                t_split = parse_split_from_id(tid, target_dataset)

            if q_split == query_split and t_split == target_split:
                query_ids.add(qid)
                target_ids.add(tid)

    return query_ids, target_ids


# ---------------------------------------------------------------------------
# Leakage data computation
# ---------------------------------------------------------------------------


def _count_ids_per_dataset_split(
    id_to_split: dict[str, str | None],
    datasets: list[str],
    split: str,
) -> list[int]:
    """Count FASTA IDs per dataset that map to *split* via the mapping.

    IDs are associated with a dataset by parsing the ``{dataset}_`` prefix
    from the original FASTA ID.
    """
    counts = {ds: 0 for ds in datasets}
    for fasta_id, s in id_to_split.items():
        if s != split:
            continue
        for ds in datasets:
            prefix = ds + "_"
            if fasta_id.startswith(prefix):
                counts[ds] += 1
                break
    return [counts[ds] for ds in datasets]


def compute_leakage_data(
    datasets: list[str],
    query_split: str,
    target_split: str,
    results_dir: Path,
    cross_task_only: bool,
    fasta_dir: Path | None = None,
    id_to_split: dict[str, str | None] | None = None,
) -> tuple[
    np.ndarray,
    list[float],
    list[int],
    list[float],
    list[int],
    list[int],
    list[int],
]:
    """Compute the leakage percentage matrix and Total row/column data.

    For the **original** heatmap, pass ``fasta_dir`` — split sizes are
    counted from FASTA files and hit filtering uses ``parse_split_from_id()``.

    For the **resplit** heatmap, pass ``id_to_split`` — split sizes are
    derived from the mapping and hit filtering uses the mapping dict.

    Exactly one of ``fasta_dir`` or ``id_to_split`` must be provided.

    Returns:
        body_matrix, total_col_pcts, total_col_counts,
        total_row_pcts, total_row_counts, query_sizes, target_sizes
    """
    if (fasta_dir is None) == (id_to_split is None):
        raise ValueError("Exactly one of fasta_dir or id_to_split must be provided")

    n = len(datasets)
    body_matrix = np.zeros((n, n), dtype=float)

    # Compute split sizes
    if fasta_dir is not None:
        query_sizes = []
        target_sizes = []
        for ds in datasets:
            query_sizes.append(
                count_fasta_sequences(fasta_dir / f"{ds}_{query_split}.fasta")
            )
            target_sizes.append(
                count_fasta_sequences(fasta_dir / f"{ds}_{target_split}.fasta")
            )
    else:
        assert id_to_split is not None
        query_sizes = _count_ids_per_dataset_split(id_to_split, datasets, query_split)
        target_sizes = _count_ids_per_dataset_split(id_to_split, datasets, target_split)

    query_id_sets: list[list[set[str]]] = [[set() for _ in range(n)] for _ in range(n)]
    target_id_sets: list[list[set[str]]] = [[set() for _ in range(n)] for _ in range(n)]

    for i, query_ds in enumerate(datasets):
        for j, target_ds in enumerate(datasets):
            if cross_task_only and i == j:
                continue

            result_file = results_dir / f"{query_ds}_vs_{target_ds}.tsv"
            if not result_file.exists():
                logger.warning("Missing result file: %s", result_file)
                continue

            q_ids, t_ids = load_filtered_hits(
                result_file,
                query_ds,
                target_ds,
                query_split,
                target_split,
                query_id_to_split=id_to_split,
                target_id_to_split=id_to_split,
            )
            query_id_sets[i][j] = q_ids
            target_id_sets[i][j] = t_ids

            n_query = query_sizes[i]
            if n_query > 0:
                body_matrix[i, j] = (len(q_ids) / n_query) * 100.0

    # Total column
    total_col_pcts: list[float] = []
    total_col_counts: list[int] = []
    for i in range(n):
        union_ids: set[str] = set()
        for j in range(n):
            union_ids |= query_id_sets[i][j]
        count = len(union_ids)
        total_col_counts.append(count)
        n_query = query_sizes[i]
        total_col_pcts.append((count / n_query * 100.0) if n_query > 0 else 0.0)

    # Total row
    total_row_pcts: list[float] = []
    total_row_counts: list[int] = []
    for j in range(n):
        union_ids = set()
        for i in range(n):
            union_ids |= target_id_sets[i][j]
        count = len(union_ids)
        total_row_counts.append(count)
        n_target = target_sizes[j]
        total_row_pcts.append((count / n_target * 100.0) if n_target > 0 else 0.0)

    return (
        body_matrix,
        total_col_pcts,
        total_col_counts,
        total_row_pcts,
        total_row_counts,
        query_sizes,
        target_sizes,
    )


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _build_heatmap_arrays(
    body_matrix: np.ndarray,
    datasets: list[str],
    total_col_pcts: list[float],
    total_col_counts: list[int],
    total_row_pcts: list[float],
    total_row_counts: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the (N+1)x(N+1) numeric matrix, annotation array, and mask."""
    n = len(datasets)
    rows = n + 1
    cols = n + 1

    full_matrix = np.full((rows, cols), np.nan, dtype=float)
    full_matrix[:n, :n] = body_matrix
    for i in range(n):
        full_matrix[i, n] = total_col_pcts[i]
    for j in range(n):
        full_matrix[n, j] = total_row_pcts[j]

    annot = np.empty((rows, cols), dtype=object)
    for i in range(n):
        for j in range(n):
            annot[i, j] = f"{body_matrix[i, j]:.1f}"
    for i in range(n):
        annot[i, n] = f"{total_col_pcts[i]:.1f}%\n({total_col_counts[i]})"
    for j in range(n):
        annot[n, j] = f"{total_row_pcts[j]:.1f}%\n({total_row_counts[j]})"
    annot[n, n] = ""

    mask = np.zeros((rows, cols), dtype=bool)
    mask[n, n] = True

    return full_matrix, annot, mask


def _draw_heatmap(
    ax: plt.Axes,
    full_matrix: np.ndarray,
    annot: np.ndarray,
    mask: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    n: int,
    title: str,
    xlabel: str,
    ylabel: str,
    show_cbar: bool = False,
):
    """Draw a single leakage heatmap on the given axes."""
    sns.heatmap(
        full_matrix,
        annot=annot,
        fmt="",
        cmap="YlOrRd",
        xticklabels=col_labels,
        yticklabels=row_labels,
        vmin=0,
        vmax=100,
        cbar=show_cbar,
        cbar_kws={"label": "% sequences with match"} if show_cbar else {},
        ax=ax,
        mask=mask,
        linewidths=0.5,
        linecolor="white",
    )

    ax.axhline(y=n, color="black", linewidth=2)
    ax.axvline(x=n, color="black", linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=2, fontsize=18)
    ax.tick_params(axis="y", rotation=0)


# ---------------------------------------------------------------------------
# Main comparison figure
# ---------------------------------------------------------------------------


def plot_comparison(
    datasets: list[str],
    orig_data: tuple,
    resplit_data: tuple,
    title_prefix: str,
    output_path: Path,
    query_split: str,
    target_split: str,
):
    """Render the 2-panel comparison figure."""
    n = len(datasets)
    row_labels = datasets + ["Total"]
    col_labels = datasets + ["Total"]

    (
        orig_matrix,
        orig_col_pcts,
        orig_col_counts,
        orig_row_pcts,
        orig_row_counts,
    ) = orig_data

    (
        resplit_matrix,
        resplit_col_pcts,
        resplit_col_counts,
        resplit_row_pcts,
        resplit_row_counts,
    ) = resplit_data

    orig_full, orig_annot, orig_mask = _build_heatmap_arrays(
        orig_matrix,
        datasets,
        orig_col_pcts,
        orig_col_counts,
        orig_row_pcts,
        orig_row_counts,
    )
    resplit_full, resplit_annot, resplit_mask = _build_heatmap_arrays(
        resplit_matrix,
        datasets,
        resplit_col_pcts,
        resplit_col_counts,
        resplit_row_pcts,
        resplit_row_counts,
    )

    # Figure layout: [heatmap | heatmap] [colorbar]
    # Use nested GridSpec so heatmap-to-heatmap and heatmap-to-colorbar
    # spacing can be controlled independently.
    heatmap_w = n + 1
    cbar_w = 0.6

    fig = plt.figure(
        figsize=(
            max(16, (heatmap_w * 2 + cbar_w) * 1.25),
            max(6, (n + 1) * 1.1),
        )
    )

    gs_outer = gridspec.GridSpec(
        1,
        2,
        width_ratios=[heatmap_w * 2, cbar_w],
        wspace=0.08,
        figure=fig,
    )

    gs_heatmaps = gridspec.GridSpecFromSubplotSpec(
        1,
        2,
        subplot_spec=gs_outer[0, 0],
        wspace=0.15,
    )

    ax_orig = fig.add_subplot(gs_heatmaps[0, 0])
    ax_resplit = fig.add_subplot(gs_heatmaps[0, 1], sharey=ax_orig)
    ax_cbar = fig.add_subplot(gs_outer[0, 1])

    xlabel = f"{target_split.capitalize()} set (target)"
    ylabel = f"{query_split.capitalize()} set (query)"

    # Left: original (pre-resplit)
    _draw_heatmap(
        ax_orig,
        orig_full,
        orig_annot,
        orig_mask,
        row_labels,
        col_labels,
        n,
        title="Before Re-splitting",
        xlabel=xlabel,
        ylabel=ylabel,
        show_cbar=False,
    )

    # Right: resplit (post-resplit)
    _draw_heatmap(
        ax_resplit,
        resplit_full,
        resplit_annot,
        resplit_mask,
        row_labels,
        col_labels,
        n,
        title="After Re-splitting",
        xlabel=xlabel,
        ylabel="",
        show_cbar=False,
    )

    # Shared colorbar
    norm = plt.Normalize(vmin=0, vmax=100)
    sm = plt.cm.ScalarMappable(cmap="YlOrRd", norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=ax_cbar, label="% sequences with match")

    fig.suptitle(title_prefix, fontsize=25, fontweight="bold", y=0.98)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Comparison figure saved to %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    datasets = list(snakemake.params["datasets"])
    query_split = str(snakemake.wildcards.query_split)
    target_split = str(snakemake.wildcards.target_split)
    tool = str(snakemake.wildcards.tool)
    variant = str(snakemake.params["variant"])
    min_seq_id = float(snakemake.params["min_seq_id"])
    profile = str(snakemake.params["profile"])
    output_path = Path(snakemake.output["comparison"])

    fasta_original_dir = Path(snakemake.params["fasta_original_dir"])
    results_dir = Path(snakemake.params["results_dir"])
    metadata_path = Path(snakemake.params["sequence_metadata"])
    assignments_path = Path(snakemake.params["split_assignments"])

    cross_task_only = variant == "cross_task"

    tool_label = "MMseqs2 easy-search" if tool == "mmseqs" else "Foldseek easy-search"
    variant_label = "cross-task pairs" if cross_task_only else "all dataset pairs"
    title_prefix = (
        f"Sequence Similarity Leakage \u2014 "
        f"{query_split.capitalize()} vs {target_split.capitalize()}\n"
        f"{tool_label}, {min_seq_id:.0%} min. sequence identity, {variant_label}"
    )

    # Build the resplit ID mapping: original FASTA ID -> resplit split
    logger.info("Building resplit ID mapping from split_assignments.tsv...")
    resplit_mapping = build_resplit_id_mapping(
        datasets, fasta_original_dir, metadata_path, assignments_path
    )

    logger.info("Computing original (pre-resplit) leakage...")
    (
        orig_matrix,
        orig_col_pcts,
        orig_col_counts,
        orig_row_pcts,
        orig_row_counts,
        orig_query_sizes,
        orig_target_sizes,
    ) = compute_leakage_data(
        datasets,
        query_split,
        target_split,
        results_dir,
        cross_task_only,
        fasta_dir=fasta_original_dir,
    )

    logger.info("Computing resplit (post-resplit) leakage...")
    (
        resplit_matrix,
        resplit_col_pcts,
        resplit_col_counts,
        resplit_row_pcts,
        resplit_row_counts,
        resplit_query_sizes,
        resplit_target_sizes,
    ) = compute_leakage_data(
        datasets,
        query_split,
        target_split,
        results_dir,
        cross_task_only,
        id_to_split=resplit_mapping,
    )

    # Log summaries
    for label, q_sizes, t_sizes, col_counts, col_pcts, row_counts, row_pcts in [
        (
            "Original",
            orig_query_sizes,
            orig_target_sizes,
            orig_col_counts,
            orig_col_pcts,
            orig_row_counts,
            orig_row_pcts,
        ),
        (
            "Resplit",
            resplit_query_sizes,
            resplit_target_sizes,
            resplit_col_counts,
            resplit_col_pcts,
            resplit_row_counts,
            resplit_row_pcts,
        ),
    ]:
        logger.info(
            "%s query sizes: %s",
            label,
            ", ".join(f"{ds}={sz}" for ds, sz in zip(datasets, q_sizes)),
        )
        logger.info(
            "%s target sizes: %s",
            label,
            ", ".join(f"{ds}={sz}" for ds, sz in zip(datasets, t_sizes)),
        )
        logger.info(
            "%s total col (query removal): %s",
            label,
            ", ".join(
                f"{ds}={c} ({p:.1f}%)"
                for ds, c, p in zip(datasets, col_counts, col_pcts)
            ),
        )
        logger.info(
            "%s total row (target removal): %s",
            label,
            ", ".join(
                f"{ds}={c} ({p:.1f}%)"
                for ds, c, p in zip(datasets, row_counts, row_pcts)
            ),
        )

    plot_comparison(
        datasets,
        orig_data=(
            orig_matrix,
            orig_col_pcts,
            orig_col_counts,
            orig_row_pcts,
            orig_row_counts,
        ),
        resplit_data=(
            resplit_matrix,
            resplit_col_pcts,
            resplit_col_counts,
            resplit_row_pcts,
            resplit_row_counts,
        ),
        title_prefix=title_prefix,
        output_path=output_path,
        query_split=query_split,
        target_split=target_split,
    )


if __name__ == "__main__":
    main()
