"""Compute new split assignments: Phase A (meltome test reconstruction) + Phase B (main resplit).

Inputs:
- merged_clusters.tsv (cluster_id, member)
- sequence_metadata.tsv (seq_id, sequence, datasets)

Outputs:
- split_assignments.tsv (seq_id, dataset, new_split)
  One row per seq_id x dataset combination. new_split is one of:
  train, valid, test, dropped.
- resplit_summary.json (per-dataset statistics including per-step detail)

Algorithm — see docs/plans/resplit.md for the full specification.
"""

import json
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd
from snakemake.script import snakemake

from wfutils import get_logger
from wfutils.logging import log_snakemake_info

logger = get_logger()
log_snakemake_info(logger)


def _pct(part: int, total: int) -> str:
    """Format a percentage string, safe against zero division."""
    return f"{part / total * 100:.1f}%" if total else "0.0%"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(clusters_path: Path, metadata_path: Path):
    """Load and parse cluster and metadata files.

    Returns:
        clusters_df: DataFrame with columns (cluster_id, member)
        seq_to_clusters: dict mapping seq_id (str) -> set of cluster_ids (int)
        cluster_to_seqs: dict mapping cluster_id (int) -> set of seq_ids (str)
        seq_ds_split: dict mapping (seq_id_str, dataset) -> original_split
        ds_seqs: dict mapping dataset -> set of seq_ids (str)
    """
    clusters_df = pd.read_csv(clusters_path, sep="\t")
    clusters_df["member"] = clusters_df["member"].astype(str)
    clusters_df["cluster_id"] = clusters_df["cluster_id"].astype(int)

    metadata_df = pd.read_csv(metadata_path, sep="\t")
    metadata_df["seq_id"] = metadata_df["seq_id"].astype(str)

    # Build cluster lookups
    seq_to_clusters: dict[str, set[int]] = defaultdict(set)
    cluster_to_seqs: dict[int, set[str]] = defaultdict(set)
    for _, row in clusters_df.iterrows():
        sid = row["member"]
        cid = row["cluster_id"]
        seq_to_clusters[sid].add(cid)
        cluster_to_seqs[cid].add(sid)

    # Build sequence-dataset-split mapping
    seq_ds_split: dict[tuple[str, str], str] = {}
    ds_seqs: dict[str, set[str]] = defaultdict(set)
    for _, row in metadata_df.iterrows():
        sid = str(row["seq_id"])
        for tag in str(row["datasets"]).split(","):
            tag = tag.strip()
            if ":" not in tag:
                continue
            ds, split = tag.split(":", 1)
            seq_ds_split[(sid, ds)] = split
            ds_seqs[ds].add(sid)

    n_clusters = clusters_df["cluster_id"].nunique()
    n_seqs = len(set(clusters_df["member"]))
    n_datasets = len(ds_seqs)
    logger.info(
        "Loaded %d clusters covering %d sequences across %d datasets",
        n_clusters,
        n_seqs,
        n_datasets,
    )

    return clusters_df, seq_to_clusters, cluster_to_seqs, seq_ds_split, ds_seqs


# ---------------------------------------------------------------------------
# Phase A: Test reconstruction
# ---------------------------------------------------------------------------


def phase_a(
    reconstruct_datasets: list[str],
    min_test_fraction: float,
    all_resplit_datasets: list[str],
    seq_to_clusters: dict[str, set[int]],
    cluster_to_seqs: dict[int, set[str]],
    seq_ds_split: dict[tuple[str, str], str],
    ds_seqs: dict[str, set[str]],
    rng: random.Random,
) -> tuple[dict[str, dict[str, set[str]]], dict[str, dict]]:
    """Reconstruct test sets for specified datasets.

    Returns:
        results: dataset -> {"test": set[seq_id], "train_valid": set[seq_id]}
        step_stats: dataset -> {
            "total_pool", "overlap_test_count", "topup_test_count",
            "final_test_count", "train_valid_count"
        }
    """
    if not reconstruct_datasets:
        logger.info("Phase A: no datasets configured for test reconstruction")
        return {}, {}

    logger.info("=== Phase A: Test reconstruction ===")
    results = {}
    step_stats: dict[str, dict] = {}

    for ds in reconstruct_datasets:
        # 1. Pool ALL sequences from this dataset
        pool = ds_seqs.get(ds, set()).copy()
        total_count = len(pool)

        if total_count == 0:
            results[ds] = {"test": set(), "train_valid": set()}
            step_stats[ds] = {
                "total_pool": 0,
                "overlap_test_count": 0,
                "topup_test_count": 0,
                "final_test_count": 0,
                "train_valid_count": 0,
            }
            logger.info("Phase A [%s]: 0 sequences, skipping", ds)
            continue

        # 2. Collect test sequences from all OTHER datasets
        other_test_seqs: set[str] = set()
        for other_ds in all_resplit_datasets:
            if other_ds == ds:
                continue
            for sid in ds_seqs.get(other_ds, set()):
                if seq_ds_split.get((sid, other_ds)) == "test":
                    other_test_seqs.add(sid)

        # 3. Find clusters containing other-dataset test sequences
        test_contaminated_clusters: set[int] = set()
        for sid in other_test_seqs:
            test_contaminated_clusters.update(seq_to_clusters.get(sid, set()))

        # 4. Assign this-dataset sequences in contaminated clusters to new test
        new_test: set[str] = set()
        for sid in pool:
            for cid in seq_to_clusters.get(sid, set()):
                if cid in test_contaminated_clusters:
                    new_test.add(sid)
                    break

        overlap_test_count = len(new_test)

        # 5. Top up if needed
        min_test_count = int(min_test_fraction * total_count)
        topup_test_count = 0
        if len(new_test) < min_test_count:
            # Collect clusters that contain this-dataset sequences but are
            # NOT test-contaminated
            candidate_clusters: set[int] = set()
            for sid in pool:
                if sid in new_test:
                    continue
                candidate_clusters.update(seq_to_clusters.get(sid, set()))
            candidate_clusters -= test_contaminated_clusters

            # Sort for determinism, then randomly select
            sorted_candidates = sorted(candidate_clusters)
            rng.shuffle(sorted_candidates)

            before_topup = len(new_test)
            for cid in sorted_candidates:
                if len(new_test) >= min_test_count:
                    break
                # Move all of this dataset's sequences from this cluster to test
                for sid in cluster_to_seqs.get(cid, set()):
                    if sid in pool and sid not in new_test:
                        new_test.add(sid)
            topup_test_count = len(new_test) - before_topup

        # 6. Remaining -> train_valid
        train_valid = pool - new_test

        results[ds] = {"test": new_test, "train_valid": train_valid}
        step_stats[ds] = {
            "total_pool": total_count,
            "overlap_test_count": overlap_test_count,
            "topup_test_count": topup_test_count,
            "final_test_count": len(new_test),
            "train_valid_count": len(train_valid),
        }
        logger.info(
            "Phase A [%s]: %d total -> %d test (%s), %d train_valid (%s)",
            ds,
            total_count,
            len(new_test),
            _pct(len(new_test), total_count),
            len(train_valid),
            _pct(len(train_valid), total_count),
        )

    return results, step_stats


# ---------------------------------------------------------------------------
# Phase B: Main resplit
# ---------------------------------------------------------------------------


def phase_b(
    all_datasets: list[str],
    phase_a_results: dict[str, dict[str, set[str]]],
    min_valid_fraction: float,
    shared_protein_groups: list[list[str]],
    seq_to_clusters: dict[str, set[int]],
    cluster_to_seqs: dict[int, set[str]],
    seq_ds_split: dict[tuple[str, str], str],
    ds_seqs: dict[str, set[str]],
    rng: random.Random,
) -> tuple[dict[tuple[str, str], str], dict[str, dict]]:
    """Run the main resplit algorithm.

    Returns:
        assignments: (seq_id, dataset) -> new_split
        step_stats: dataset -> {
            "step1_merge": {"test": int, "train_valid": int},
            "step2_drop_within_test_similar": {"dropped": int, "train_valid_after": int},
            "step3_cross_contamination": {"moved_to_valid": int, "train": int, "valid": int},
            "step4_topup": {"topped_up": int, "final_train": int, "final_valid": int},
        }
    """
    logger.info("=== Phase B: Main resplit ===")

    step_stats: dict[str, dict] = {ds: {} for ds in all_datasets}

    # Build per-dataset pools: test (fixed) and train_valid (to be split)
    ds_test: dict[str, set[str]] = {}
    ds_train_valid: dict[str, set[str]] = {}

    # Step 1: Merge train+valid, keep test as-is
    logger.info("Step 1 — Merge train+valid, keep test:")
    for ds in all_datasets:
        if ds in phase_a_results:
            ds_test[ds] = phase_a_results[ds]["test"].copy()
            ds_train_valid[ds] = phase_a_results[ds]["train_valid"].copy()
            logger.info(
                "  [%s] test: %d (from Phase A), train_valid: %d (from Phase A)",
                ds,
                len(ds_test[ds]),
                len(ds_train_valid[ds]),
            )
        else:
            ds_test[ds] = set()
            ds_train_valid[ds] = set()
            orig_train = 0
            orig_valid = 0
            for sid in ds_seqs.get(ds, set()):
                orig_split = seq_ds_split.get((sid, ds))
                if orig_split == "test":
                    ds_test[ds].add(sid)
                else:
                    ds_train_valid[ds].add(sid)
                    if orig_split == "train":
                        orig_train += 1
                    elif orig_split == "valid":
                        orig_valid += 1
            logger.info(
                "  [%s] test: %d, train_valid: %d (from train: %d + valid: %d)",
                ds,
                len(ds_test[ds]),
                len(ds_train_valid[ds]),
                orig_train,
                orig_valid,
            )

        step_stats[ds]["step1_merge"] = {
            "test": len(ds_test[ds]),
            "train_valid": len(ds_train_valid[ds]),
        }

    # Record original train_valid counts (before drops) for valid top-up
    original_tv_counts = {ds: len(ds_train_valid[ds]) for ds in all_datasets}

    # Step 2: Drop within-dataset test-similar sequences
    logger.info("Step 2 — Drop within-dataset test-similar:")
    for ds in all_datasets:
        test_clusters: set[int] = set()
        for sid in ds_test[ds]:
            test_clusters.update(seq_to_clusters.get(sid, set()))

        to_drop: set[str] = set()
        for sid in ds_train_valid[ds]:
            for cid in seq_to_clusters.get(sid, set()):
                if cid in test_clusters:
                    to_drop.add(sid)
                    break

        n_before = len(ds_train_valid[ds])
        ds_train_valid[ds] -= to_drop
        logger.info(
            "  [%s] dropped %d/%d (%s)",
            ds,
            len(to_drop),
            n_before,
            _pct(len(to_drop), n_before),
        )

        step_stats[ds]["step2_drop_within_test_similar"] = {
            "dropped": len(to_drop),
            "train_valid_after": len(ds_train_valid[ds]),
        }

    # Step 3: Move cross-dataset test-contaminated cluster members to valid
    logger.info("Step 3 — Move cross-dataset test-contaminated to valid:")
    # First, collect ALL test clusters across ALL datasets
    all_test_clusters: set[int] = set()
    for ds in all_datasets:
        for sid in ds_test[ds]:
            all_test_clusters.update(seq_to_clusters.get(sid, set()))

    ds_train: dict[str, set[str]] = {}
    ds_valid: dict[str, set[str]] = {}

    for ds in all_datasets:
        ds_valid[ds] = set()
        ds_train[ds] = set()

        for sid in ds_train_valid[ds]:
            in_test_cluster = False
            for cid in seq_to_clusters.get(sid, set()):
                if cid in all_test_clusters:
                    in_test_cluster = True
                    break
            if in_test_cluster:
                ds_valid[ds].add(sid)
            else:
                ds_train[ds].add(sid)

        n_tv = len(ds_train_valid[ds])
        logger.info(
            "  [%s] moved %d/%d to valid (%s)",
            ds,
            len(ds_valid[ds]),
            n_tv,
            _pct(len(ds_valid[ds]), n_tv),
        )

        step_stats[ds]["step3_cross_contamination"] = {
            "moved_to_valid": len(ds_valid[ds]),
            "train": len(ds_train[ds]),
            "valid": len(ds_valid[ds]),
        }

    # Step 4: Top up valid to min_valid_fraction per dataset
    logger.info(
        "Step 4 — Top up valid to %s:",
        _pct(int(min_valid_fraction * 100), 100).replace(".0%", "%"),
    )
    # Build a mapping of which shared_protein_group each dataset belongs to
    ds_to_group: dict[str, int] = {}
    for gidx, group in enumerate(shared_protein_groups):
        for ds in group:
            ds_to_group[ds] = gidx

    processed_groups: set[int] = set()

    for ds in all_datasets:
        gidx = ds_to_group.get(ds)

        if gidx is not None and gidx in processed_groups:
            continue  # Already handled as part of a shared group

        if gidx is not None:
            # Shared protein group: compute combined counts
            group = shared_protein_groups[gidx]
            combined_orig_tv = sum(original_tv_counts.get(g, 0) for g in group)
            combined_valid_before = sum(len(ds_valid.get(g, set())) for g in group)
            min_valid_count = int(min_valid_fraction * combined_orig_tv)

            if combined_valid_before < min_valid_count:
                # Select clusters from train that have NO test members,
                # using sequences from the first dataset in the group as
                # the reference (they share the same proteins)
                ref_ds = group[0]
                candidate_clusters: set[int] = set()
                for sid in ds_train.get(ref_ds, set()):
                    candidate_clusters.update(seq_to_clusters.get(sid, set()))
                candidate_clusters -= all_test_clusters

                sorted_candidates = sorted(candidate_clusters)
                rng.shuffle(sorted_candidates)

                for cid in sorted_candidates:
                    combined_valid = sum(len(ds_valid.get(g, set())) for g in group)
                    if combined_valid >= min_valid_count:
                        break
                    # Move cluster members from train to valid for ALL group datasets
                    for g_ds in group:
                        for sid in cluster_to_seqs.get(cid, set()):
                            if sid in ds_train.get(g_ds, set()):
                                ds_train[g_ds].discard(sid)
                                ds_valid[g_ds].add(sid)

            combined_valid_after = sum(len(ds_valid.get(g, set())) for g in group)
            logger.info(
                "  [group %s] combined valid: %d -> %d (%s of %d train+valid)",
                group,
                combined_valid_before,
                combined_valid_after,
                _pct(combined_valid_after, combined_orig_tv),
                combined_orig_tv,
            )

            # Record step 4 stats for each dataset in the group
            for g_ds in group:
                valid_before_ds = step_stats[g_ds]["step3_cross_contamination"]["valid"]
                topped_up_ds = len(ds_valid[g_ds]) - valid_before_ds
                step_stats[g_ds]["step4_topup"] = {
                    "topped_up": topped_up_ds,
                    "final_train": len(ds_train[g_ds]),
                    "final_valid": len(ds_valid[g_ds]),
                }

            processed_groups.add(gidx)
        else:
            # Individual dataset
            orig_tv = original_tv_counts.get(ds, 0)
            min_valid_count = int(min_valid_fraction * orig_tv)
            valid_before = len(ds_valid[ds])

            if valid_before < min_valid_count:
                candidate_clusters: set[int] = set()
                for sid in ds_train[ds]:
                    candidate_clusters.update(seq_to_clusters.get(sid, set()))
                candidate_clusters -= all_test_clusters

                sorted_candidates = sorted(candidate_clusters)
                rng.shuffle(sorted_candidates)

                for cid in sorted_candidates:
                    if len(ds_valid[ds]) >= min_valid_count:
                        break
                    for sid in cluster_to_seqs.get(cid, set()):
                        if sid in ds_train[ds]:
                            ds_train[ds].discard(sid)
                            ds_valid[ds].add(sid)

            valid_after = len(ds_valid[ds])
            logger.info(
                "  [%s] valid: %d -> %d (%s of %d train+valid)",
                ds,
                valid_before,
                valid_after,
                _pct(valid_after, orig_tv),
                orig_tv,
            )

            step_stats[ds]["step4_topup"] = {
                "topped_up": valid_after - valid_before,
                "final_train": len(ds_train[ds]),
                "final_valid": len(ds_valid[ds]),
            }

    # For any datasets in shared groups that were skipped (already processed),
    # ensure they still get step4 stats
    for ds in all_datasets:
        if "step4_topup" not in step_stats[ds]:
            # Dataset was in a group that was processed before its turn
            valid_before = step_stats[ds]["step3_cross_contamination"]["valid"]
            step_stats[ds]["step4_topup"] = {
                "topped_up": len(ds_valid[ds]) - valid_before,
                "final_train": len(ds_train[ds]),
                "final_valid": len(ds_valid[ds]),
            }

    # Build final assignments
    assignments: dict[tuple[str, str], str] = {}

    for ds in all_datasets:
        for sid in ds_test[ds]:
            assignments[(sid, ds)] = "test"
        for sid in ds_valid[ds]:
            assignments[(sid, ds)] = "valid"
        for sid in ds_train[ds]:
            assignments[(sid, ds)] = "train"
        # Mark dropped sequences
        all_assigned = ds_test[ds] | ds_valid[ds] | ds_train[ds]
        all_original = ds_seqs.get(ds, set())
        for sid in all_original - all_assigned:
            assignments[(sid, ds)] = "dropped"

    # Log final summary table
    logger.info("=== Final split assignments ===")
    max_ds_len = max(len(ds) for ds in all_datasets)
    totals = {"train": 0, "valid": 0, "test": 0, "dropped": 0, "total": 0}

    for ds in all_datasets:
        total = len(ds_seqs.get(ds, set()))
        counts = {"train": 0, "valid": 0, "test": 0, "dropped": 0}
        for sid in ds_seqs.get(ds, set()):
            s = assignments.get((sid, ds), "unknown")
            if s in counts:
                counts[s] += 1
        for k in counts:
            totals[k] += counts[k]
        totals["total"] += total

        logger.info(
            "  %-*s: train=%d (%s) valid=%d (%s) test=%d (%s) dropped=%d (%s) | total=%d",
            max_ds_len,
            ds,
            counts["train"],
            _pct(counts["train"], total),
            counts["valid"],
            _pct(counts["valid"], total),
            counts["test"],
            _pct(counts["test"], total),
            counts["dropped"],
            _pct(counts["dropped"], total),
            total,
        )

    logger.info(
        "  %-*s: train=%d (%s) valid=%d (%s) test=%d (%s) dropped=%d (%s) | total=%d",
        max_ds_len,
        "TOTAL",
        totals["train"],
        _pct(totals["train"], totals["total"]),
        totals["valid"],
        _pct(totals["valid"], totals["total"]),
        totals["test"],
        _pct(totals["test"], totals["total"]),
        totals["dropped"],
        _pct(totals["dropped"], totals["total"]),
        totals["total"],
    )

    return assignments, step_stats


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------


def build_summary(
    all_datasets: list[str],
    assignments: dict[tuple[str, str], str],
    seq_ds_split: dict[tuple[str, str], str],
    ds_seqs: dict[str, set[str]],
    phase_a_results: dict[str, dict[str, set[str]]],
    phase_a_step_stats: dict[str, dict],
    phase_b_step_stats: dict[str, dict],
) -> dict:
    """Build per-dataset summary statistics with per-step detail."""
    summary = {}

    for ds in all_datasets:
        original = {"train": 0, "valid": 0, "test": 0}
        for sid in ds_seqs.get(ds, set()):
            orig_split = seq_ds_split.get((sid, ds), "unknown")
            if orig_split in original:
                original[orig_split] += 1

        new = {"train": 0, "valid": 0, "test": 0, "dropped": 0}
        for sid in ds_seqs.get(ds, set()):
            new_split = assignments.get((sid, ds), "unknown")
            if new_split in new:
                new[new_split] += 1

        ds_summary: dict = {
            "original_counts": original,
            "new_counts": new,
            "total": sum(original.values()),
        }

        if ds in phase_a_results:
            ds_summary["reconstruct_test"] = {
                "new_test_count": len(phase_a_results[ds]["test"]),
                "train_valid_count": len(phase_a_results[ds]["train_valid"]),
            }

        # Per-step statistics
        steps: dict = {}
        if ds in phase_a_step_stats:
            steps["phase_a"] = phase_a_step_stats[ds]
        if ds in phase_b_step_stats:
            steps.update(phase_b_step_stats[ds])
        if steps:
            ds_summary["steps"] = steps

        summary[ds] = ds_summary

    return summary


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_assignments(
    all_datasets: list[str],
    assignments: dict[tuple[str, str], str],
    ds_seqs: dict[str, set[str]],
    seq_ds_split: dict[tuple[str, str], str],
    phase_a_results: dict[str, dict[str, set[str]]],
    min_valid_fraction: float,
    shared_protein_groups: list[list[str]],
):
    """Run sanity checks on the final assignments."""
    errors = []

    for ds in all_datasets:
        seqs = ds_seqs.get(ds, set())

        # Every sequence must be accounted for
        assigned = {sid for sid in seqs if (sid, ds) in assignments}
        if assigned != seqs:
            missing = seqs - assigned
            errors.append(f"[{ds}] {len(missing)} sequences not assigned")

        # No overlap between train and valid
        train_seqs = {sid for sid in seqs if assignments.get((sid, ds)) == "train"}
        valid_seqs = {sid for sid in seqs if assignments.get((sid, ds)) == "valid"}
        test_seqs = {sid for sid in seqs if assignments.get((sid, ds)) == "test"}

        if train_seqs & valid_seqs:
            errors.append(
                f"[{ds}] {len(train_seqs & valid_seqs)} in both train & valid"
            )
        if train_seqs & test_seqs:
            errors.append(f"[{ds}] {len(train_seqs & test_seqs)} in both train & test")

        # For normal datasets: test set must match original exactly
        if ds not in phase_a_results:
            orig_test = {sid for sid in seqs if seq_ds_split.get((sid, ds)) == "test"}
            if test_seqs != orig_test:
                errors.append(
                    f"[{ds}] test set changed: orig={len(orig_test)}, new={len(test_seqs)}"
                )

    if errors:
        for e in errors:
            logger.error("VALIDATION ERROR: %s", e)
        raise ValueError(
            f"Resplit validation failed with {len(errors)} errors. See log for details."
        )

    logger.info("Validation passed: all checks OK")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    clusters_path = Path(snakemake.input.clusters)
    metadata_path = Path(snakemake.input.metadata)
    assignments_path = Path(snakemake.output.assignments)
    summary_path = Path(snakemake.output.summary)

    min_valid_fraction = float(snakemake.params.min_valid_fraction)
    random_seed = int(snakemake.params.random_seed)
    shared_protein_groups = list(snakemake.params.shared_protein_groups)
    reconstruct_datasets = list(snakemake.params.reconstruct_test_datasets)
    reconstruct_min_test_frac = float(
        snakemake.params.reconstruct_test_min_test_fraction
    )

    rng = random.Random(random_seed)

    logger.info(
        "Parameters: min_valid_fraction=%s, random_seed=%d, "
        "shared_protein_groups=%s, reconstruct_test_datasets=%s",
        min_valid_fraction,
        random_seed,
        shared_protein_groups,
        reconstruct_datasets,
    )

    # Load data
    clusters_df, seq_to_clusters, cluster_to_seqs, seq_ds_split, ds_seqs = load_data(
        clusters_path, metadata_path
    )

    # Determine all datasets from metadata
    all_datasets = sorted({ds for (_, ds) in seq_ds_split.keys()})
    logger.info("Processing %d datasets: %s", len(all_datasets), all_datasets)

    # Phase A: Test reconstruction
    phase_a_results, phase_a_step_stats = phase_a(
        reconstruct_datasets=reconstruct_datasets,
        min_test_fraction=reconstruct_min_test_frac,
        all_resplit_datasets=all_datasets,
        seq_to_clusters=seq_to_clusters,
        cluster_to_seqs=cluster_to_seqs,
        seq_ds_split=seq_ds_split,
        ds_seqs=ds_seqs,
        rng=rng,
    )

    # Phase B: Main resplit
    assignments, phase_b_step_stats = phase_b(
        all_datasets=all_datasets,
        phase_a_results=phase_a_results,
        min_valid_fraction=min_valid_fraction,
        shared_protein_groups=shared_protein_groups,
        seq_to_clusters=seq_to_clusters,
        cluster_to_seqs=cluster_to_seqs,
        seq_ds_split=seq_ds_split,
        ds_seqs=ds_seqs,
        rng=rng,
    )

    # Validate
    validate_assignments(
        all_datasets=all_datasets,
        assignments=assignments,
        ds_seqs=ds_seqs,
        seq_ds_split=seq_ds_split,
        phase_a_results=phase_a_results,
        min_valid_fraction=min_valid_fraction,
        shared_protein_groups=shared_protein_groups,
    )

    # Write split_assignments.tsv
    assignments_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for (sid, ds), new_split in sorted(assignments.items()):
        rows.append({"seq_id": sid, "dataset": ds, "new_split": new_split})
    pd.DataFrame(rows).to_csv(assignments_path, sep="\t", index=False)

    # Write summary JSON
    summary = build_summary(
        all_datasets=all_datasets,
        assignments=assignments,
        seq_ds_split=seq_ds_split,
        ds_seqs=ds_seqs,
        phase_a_results=phase_a_results,
        phase_a_step_stats=phase_a_step_stats,
        phase_b_step_stats=phase_b_step_stats,
    )
    with summary_path.open("w") as fh:
        json.dump(summary, fh, indent=2)

    logger.info("Split assignments written to %s", assignments_path)
    logger.info("Summary written to %s", summary_path)


main()
