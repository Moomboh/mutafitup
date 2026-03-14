#!/usr/bin/env python3
"""Upload resplit datasets to HuggingFace Hub as a dataset repository.

Reads the pipeline configuration to determine which datasets to upload,
stages them into a HuggingFace-compatible directory layout, generates a
dataset card (README.md), and uploads via ``huggingface_hub``.

Usage::

    python scripts/upload_datasets_to_huggingface.py --config config/config.yml
    python scripts/upload_datasets_to_huggingface.py --config config/config.yml --dry-run
    python scripts/upload_datasets_to_huggingface.py --config config/config.yml --repo-id user/repo

The script expects that the resplit pipeline has already been run (i.e.
``results/datasets_resplit/`` and ``results/resplit/`` are populated).

Authentication: set the ``HF_TOKEN`` environment variable or run
``huggingface-cli login`` beforehand.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("upload_datasets_to_huggingface")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_STAGING_DIR = _PROJECT_ROOT / ".hf_staging"

# HuggingFace uses "validation" rather than "valid"
_SPLIT_RENAME = {"valid": "validation"}

# Metadata files to include from the resplit directory
_METADATA_FILES = [
    "split_assignments.tsv",
    "resplit_summary.json",
    "merged_clusters.tsv",
]


# ── Dataset card ─────────────────────────────────────────────────────

# fmt: off
_DATASET_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "secstr": {
        "description": "3-state secondary structure (helix / strand / coil)",
        "task_type": "per-residue classification",
        "num_labels": "3",
        "primary_metric": "accuracy",
        "provenance": (
            "DSSP assignments from [NetSurfP-2.0](https://doi.org/10.1002/prot.25674) "
            "(Klausen et al., 2019), test set from "
            "[ProtTrans](https://doi.org/10.1109/TPAMI.2021.3095381) NEW364 "
            "(Elnaggar et al., 2022)"
        ),
    },
    "secstr8": {
        "description": "8-state secondary structure (DSSP Q8)",
        "task_type": "per-residue classification",
        "num_labels": "8",
        "primary_metric": "accuracy",
        "provenance": (
            "DSSP assignments from [NetSurfP-2.0](https://doi.org/10.1002/prot.25674) "
            "(Klausen et al., 2019), test set from "
            "[ProtTrans](https://doi.org/10.1109/TPAMI.2021.3095381) NEW364 "
            "(Elnaggar et al., 2022)"
        ),
    },
    "rsa": {
        "description": "Relative solvent accessibility (isolated chain)",
        "task_type": "per-residue regression",
        "num_labels": "continuous",
        "primary_metric": "spearman",
        "provenance": (
            "Isolated-chain RSA from [NetSurfP-2.0](https://doi.org/10.1002/prot.25674) "
            "(Klausen et al., 2019), test set CB513"
        ),
    },
    "disorder": {
        "description": "Intrinsic disorder (CheZOD Z-scores)",
        "task_type": "per-residue regression",
        "num_labels": "continuous",
        "primary_metric": "spearman",
        "provenance": (
            "[CheZOD database](https://doi.org/10.1038/s41598-020-71716-1) "
            "(Dass et al., 2020) "
            "-> [SETH](https://doi.org/10.3389/fbinf.2022.1019597) "
            "(Ilzhofer et al., 2022) "
            "-> [Schmirler et al., 2024](https://doi.org/10.1038/s41467-024-51844-2)"
        ),
    },
    "meltome": {
        "description": "Melting temperature (thermal stability)",
        "task_type": "per-protein regression",
        "num_labels": "continuous",
        "primary_metric": "spearman",
        "provenance": (
            "[Meltome Atlas](https://doi.org/10.1038/s41592-020-0801-4) "
            "(Jarzab et al., 2020) "
            "-> [FLIP](https://doi.org/10.1101/2021.11.09.467890) "
            "(Dallago et al., 2021) "
            "-> [Schmirler et al., 2024](https://doi.org/10.1038/s41467-024-51844-2)"
        ),
    },
    "subloc": {
        "description": "Subcellular localization (10 compartments)",
        "task_type": "per-protein classification",
        "num_labels": "10",
        "primary_metric": "accuracy",
        "provenance": (
            "[Light Attention](https://doi.org/10.1093/bioadv/vbab035) "
            u"(St\u00e4rk et al., 2021) "
            "-> [Schmirler et al., 2024](https://doi.org/10.1038/s41467-024-51844-2)"
        ),
    },
    "gpsite_dna": {
        "description": "DNA-binding residues",
        "task_type": "per-residue classification",
        "num_labels": "2",
        "primary_metric": "accuracy",
        "provenance": "[GPSite](https://doi.org/10.7554/eLife.93695) (Yuan et al., 2024)",
    },
    "gpsite_rna": {
        "description": "RNA-binding residues",
        "task_type": "per-residue classification",
        "num_labels": "2",
        "primary_metric": "accuracy",
        "provenance": "[GPSite](https://doi.org/10.7554/eLife.93695) (Yuan et al., 2024)",
    },
    "gpsite_pep": {
        "description": "Peptide-binding residues",
        "task_type": "per-residue classification",
        "num_labels": "2",
        "primary_metric": "accuracy",
        "provenance": "[GPSite](https://doi.org/10.7554/eLife.93695) (Yuan et al., 2024)",
    },
    "gpsite_pro": {
        "description": "Protein-binding residues",
        "task_type": "per-residue classification",
        "num_labels": "2",
        "primary_metric": "accuracy",
        "provenance": "[GPSite](https://doi.org/10.7554/eLife.93695) (Yuan et al., 2024)",
    },
    "gpsite_atp": {
        "description": "ATP-binding residues",
        "task_type": "per-residue classification",
        "num_labels": "2",
        "primary_metric": "accuracy",
        "provenance": "[GPSite](https://doi.org/10.7554/eLife.93695) (Yuan et al., 2024)",
    },
    "gpsite_hem": {
        "description": "Heme-binding residues",
        "task_type": "per-residue classification",
        "num_labels": "2",
        "primary_metric": "accuracy",
        "provenance": "[GPSite](https://doi.org/10.7554/eLife.93695) (Yuan et al., 2024)",
    },
    "gpsite_zn": {
        "description": "Zinc-binding residues",
        "task_type": "per-residue classification",
        "num_labels": "2",
        "primary_metric": "accuracy",
        "provenance": "[GPSite](https://doi.org/10.7554/eLife.93695) (Yuan et al., 2024)",
    },
    "gpsite_ca": {
        "description": "Calcium-binding residues",
        "task_type": "per-residue classification",
        "num_labels": "2",
        "primary_metric": "accuracy",
        "provenance": "[GPSite](https://doi.org/10.7554/eLife.93695) (Yuan et al., 2024)",
    },
    "gpsite_mg": {
        "description": "Magnesium-binding residues",
        "task_type": "per-residue classification",
        "num_labels": "2",
        "primary_metric": "accuracy",
        "provenance": "[GPSite](https://doi.org/10.7554/eLife.93695) (Yuan et al., 2024)",
    },
    "gpsite_mn": {
        "description": "Manganese-binding residues",
        "task_type": "per-residue classification",
        "num_labels": "2",
        "primary_metric": "accuracy",
        "provenance": "[GPSite](https://doi.org/10.7554/eLife.93695) (Yuan et al., 2024)",
    },
}
# fmt: on


def _build_dataset_card(
    repo_id: str,
    datasets: list[str],
    dataset_type_map: dict[str, str],
    resplit_config: dict,
) -> str:
    """Generate the HuggingFace dataset card (README.md)."""

    # Build configs YAML block for frontmatter
    configs_yaml_lines: list[str] = []
    for ds in datasets:
        configs_yaml_lines.append(f"  - config_name: {ds}")
        configs_yaml_lines.append("    data_files:")
        for split in ("train", "validation", "test"):
            configs_yaml_lines.append(f"      - split: {split}")
            configs_yaml_lines.append(f"        path: data/{ds}/{split}.parquet")
    configs_yaml = "\n".join(configs_yaml_lines)

    # Build dataset table
    table_lines = [
        "| Dataset | Task type | Labels | Metric | Provenance |",
        "|---------|-----------|--------|--------|------------|",
    ]
    for ds in datasets:
        info = _DATASET_DESCRIPTIONS.get(ds, {})
        table_lines.append(
            f"| `{ds}` "
            f"| {info.get('task_type', 'unknown')} "
            f"| {info.get('num_labels', '?')} "
            f"| {info.get('primary_metric', '?')} "
            f"| {info.get('provenance', '?')} |"
        )
    dataset_table = "\n".join(table_lines)

    # Resplit parameters
    tools = resplit_config.get("tools", {})
    enabled_tools = [t for t, cfg in tools.items() if cfg.get("enabled", False)]
    min_seq_id = tools.get("mmseqs", {}).get("min_seq_id", "?")
    min_valid_frac = resplit_config.get("min_valid_fraction", "?")
    seed = resplit_config.get("random_seed", "?")
    shared_groups = resplit_config.get("shared_protein_groups", [])
    reconstruct_test_ds = resplit_config.get("reconstruct_test", {}).get("datasets", [])
    aggregate_ds = list(resplit_config.get("aggregate_duplicates", {}).keys())

    card = f"""\
---
configs:
{configs_yaml}
license: mit
tags:
  - protein
  - biology
  - benchmark
  - data-leakage
  - protein-language-model
  - multi-task
---

# {repo_id}

Leakage-free protein function prediction benchmarks for multi-task
protein language model (pLM) fine-tuning.

This dataset contains **{len(datasets)} prediction tasks** drawn from
three independent data families, resplit to eliminate cross-dataset
sequence similarity-based data leakage between training, validation,
and test sets.

## Datasets

{dataset_table}

## Loading

```python
from datasets import load_dataset

# Load a single dataset (e.g. meltome)
ds = load_dataset("{repo_id}", "meltome")

# Access splits
train = ds["train"]
valid = ds["validation"]
test = ds["test"]
```

## Resplit methodology

The original datasets were collected from three independent sources
(see provenance chain below). Because these sources share overlapping
protein sequences, naively combining their original train/test splits
introduces cross-dataset data leakage: a protein in one dataset's
training set may appear in another dataset's test set.

To eliminate this leakage, all sequences across all {len(datasets)}
datasets were pooled, deduplicated, and clustered using
**{", ".join(enabled_tools)}** at **{min_seq_id * 100 if isinstance(min_seq_id, (int, float)) else min_seq_id}% minimum sequence identity**.
New training, validation, and test splits were then assigned at the
cluster level so that no two splits share sequence-similar proteins:

1. **Cluster** all unique sequences across datasets using {", ".join(enabled_tools)}
   at {min_seq_id * 100 if isinstance(min_seq_id, (int, float)) else min_seq_id}% minimum sequence identity.
2. **Merge** original train + valid pools per dataset.
3. **Remove** within-dataset test-similar sequences from the pool.
4. **Reassign** cross-dataset test-contaminated cluster members to
   validation.
5. **Top up** validation to at least {min_valid_frac * 100 if isinstance(min_valid_frac, (int, float)) else min_valid_frac}% of the non-test data
   (random seed: {seed}).

**Shared protein groups:** {", ".join(f"({', '.join(g)})" for g in shared_groups) if shared_groups else "none"} --
datasets sharing the same underlying proteins are coordinated during
split assignment.

**Test set reconstruction:** {", ".join(f"`{d}`" for d in reconstruct_test_ds) if reconstruct_test_ds else "none"} --
test sets are reconstructed from original test sequences with
cluster-aware decontamination.

**Duplicate aggregation:** {", ".join(f"`{d}`" for d in aggregate_ds) if aggregate_ds else "none"} --
duplicate sequences within a split are aggregated (mean strategy for
regression scores).

The `metadata/` directory contains the full clustering and split
assignment artifacts for reproducibility.

## Parquet schemas

### Per-residue classification

Used by: `secstr`, `secstr8`, `gpsite_*`

| Column | Type | Description |
|--------|------|-------------|
| `sequence` | `str` | Amino acid sequence |
| `label` | `list[int]` | Per-residue integer class labels |
| `resolved` | `list[int]` | Per-residue binary mask (1 = ordered, 0 = disordered) |

### Per-residue regression

Used by: `disorder`, `rsa`

| Column | Type | Description |
|--------|------|-------------|
| `sequence` | `str` | Amino acid sequence |
| `score` | `list[float]` | Per-residue float scores (rsa uses 999.0 sentinel for unresolved residues) |

### Per-protein regression

Used by: `meltome`

| Column | Type | Description |
|--------|------|-------------|
| `sequence` | `str` | Amino acid sequence |
| `score` | `float` | Scalar score (melting temperature in degrees Celsius) |

### Per-protein classification

Used by: `subloc`

| Column | Type | Description |
|--------|------|-------------|
| `sequence` | `str` | Amino acid sequence |
| `label` | `str` | Class name (subcellular localization compartment) |
| `label_numeric` | `int` | Integer-encoded label |

## Data provenance

### Schmirler 2024 (meltome, subloc, disorder)

Training data from Schmirler, Heinzinger & Rost (2024), downloaded from
the companion GitHub repository
[RSchmirler/data-repo_plm-finetune-eval](https://github.com/RSchmirler/data-repo_plm-finetune-eval),
also archived at [Zenodo](https://doi.org/10.5281/zenodo.12770310).
Original train/valid/test splits were constructed by the authors using
MMseqs2 clustering at >20% pairwise sequence identity.

**Full provenance chains:**

- **meltome**: Experimental thermal proteome profiling from the
  [Meltome Atlas](https://doi.org/10.1038/s41592-020-0801-4)
  (Jarzab et al., 2020) -- standardized into benchmark splits by
  [FLIP](https://doi.org/10.1101/2021.11.09.467890)
  (Dallago et al., 2021) -- adopted with added validation set by
  [Schmirler et al., 2024](https://doi.org/10.1038/s41467-024-51844-2).
- **subloc**: Curated subcellular localization annotations from
  [Light Attention](https://doi.org/10.1093/bioadv/vbab035)
  (Starck et al., 2021) -- included directly by
  [Schmirler et al., 2024](https://doi.org/10.1038/s41467-024-51844-2).
- **disorder**: NMR-derived per-residue disorder Z-scores from the
  [CheZOD database / ODiNPred](https://doi.org/10.1038/s41598-020-71716-1)
  (Dass et al., 2020) -- used for pLM embedding prediction by
  [SETH](https://doi.org/10.3389/fbinf.2022.1019597)
  (Ilzhofer et al., 2022) -- re-split by
  [Schmirler et al., 2024](https://doi.org/10.1038/s41467-024-51844-2).

### NetSurfP-2.0 (secstr, secstr8, rsa)

Structure-derived annotations from
[NetSurfP-2.0](https://doi.org/10.1002/prot.25674)
(Klausen et al., 2019), hosted at DTU Health Tech. Secondary structure
labels are DSSP-based categorical assignments; RSA targets are
solvent-accessibility values for resolved residues.

The secondary-structure test set (NEW364) comes from the
[ProtTrans](https://doi.org/10.1109/TPAMI.2021.3095381)
project (Elnaggar et al., 2022). The RSA test set is CB513.

### GPSite (gpsite_*)

Residue-level binding-site annotations from
[GPSite](https://doi.org/10.7554/eLife.93695)
(Yuan et al., 2024), pinned to commit
[`58cfa4e`](https://github.com/biomed-AI/GPSite/tree/58cfa4e59f077e531fb38cf4a04bec6aea706454).
Labels correspond to experimentally resolved protein-ligand interactions
from the PDB across 10 ligand types (DNA, RNA, peptide, protein, ATP,
heme, Zn, Ca, Mg, Mn).

## References

- Dallago, C. et al. (2021). FLIP: Benchmark tasks in fitness landscape
  inference for proteins. *bioRxiv* 2021.11.09.467890.
  [doi:10.1101/2021.11.09.467890](https://doi.org/10.1101/2021.11.09.467890)
- Dass, R., Mulder, F.A.A. & Nielsen, J.T. (2020). ODiNPred:
  comprehensive prediction of protein order and disorder. *Sci. Rep.*
  10, 14780.
  [doi:10.1038/s41598-020-71716-1](https://doi.org/10.1038/s41598-020-71716-1)
- Elnaggar, A. et al. (2022). ProtTrans: Toward Understanding the
  Language of Life Through Self-Supervised Learning. *IEEE TPAMI*
  44(10), 7112--7127.
  [doi:10.1109/TPAMI.2021.3095381](https://doi.org/10.1109/TPAMI.2021.3095381)
- Ilzhofer, D., Heinzinger, M. & Rost, B. (2022). SETH predicts
  nuances of residue disorder from protein embeddings. *Front.
  Bioinform.* 2, 1019597.
  [doi:10.3389/fbinf.2022.1019597](https://doi.org/10.3389/fbinf.2022.1019597)
- Jarzab, A. et al. (2020). Meltome atlas -- thermal proteome stability
  across the tree of life. *Nat. Methods* 17, 495--503.
  [doi:10.1038/s41592-020-0801-4](https://doi.org/10.1038/s41592-020-0801-4)
- Klausen, M.S. et al. (2019). NetSurfP-2.0: Improved prediction of
  protein structural features by integrated deep learning. *Proteins*
  87(6), 520--527.
  [doi:10.1002/prot.25674](https://doi.org/10.1002/prot.25674)
- Schmirler, R., Heinzinger, M. & Rost, B. (2024). Fine-tuning protein
  language models boosts predictions across diverse tasks. *Nat.
  Commun.* 15, 7407.
  [doi:10.1038/s41467-024-51844-2](https://doi.org/10.1038/s41467-024-51844-2)
- Starck, H. et al. (2021). Light attention predicts protein location
  from the language of life. *Bioinformatics Advances* 1(1), vbab035.
  [doi:10.1093/bioadv/vbab035](https://doi.org/10.1093/bioadv/vbab035)
- Steinegger, M. & Soding, J. (2017). MMseqs2 enables sensitive protein
  sequence searching for the analysis of massive data sets. *Nat.
  Biotechnol.* 35, 1026--1028.
  [doi:10.1038/nbt.3988](https://doi.org/10.1038/nbt.3988)
- Yuan, Q., Tian, C. & Yang, Y. (2024). Genome-scale annotation of
  protein binding sites via language model and geometric deep learning.
  *eLife* 13, e93695.
  [doi:10.7554/eLife.93695](https://doi.org/10.7554/eLife.93695)
"""
    return card


# ── Helpers ───────────────────────────────────────────────────────────


def _load_config(config_path: str) -> dict:
    """Load and return the YAML config."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def _resolve_dataset_type_map(config: dict) -> dict[str, str]:
    """Build a mapping from dataset name -> subset_type directory name.

    Reads the ``standardized_datasets`` section of the config to
    determine which subset_type directory each dataset lives under.
    """
    ds_map: dict[str, str] = {}
    for subset_type, datasets in config.get("standardized_datasets", {}).items():
        if isinstance(datasets, dict):
            for ds_name in datasets:
                ds_map[ds_name] = subset_type
    return ds_map


def _symlink_file(src: Path, dest: Path) -> None:
    """Create a symlink at ``dest`` pointing to ``src``."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    os.symlink(src.resolve(), dest)


def _stage_datasets(
    datasets: list[str],
    dataset_type_map: dict[str, str],
    resplit_dir: Path,
    staging_dir: Path,
) -> list[str]:
    """Symlink resplit parquets into HF-compatible layout.

    Returns the list of successfully staged dataset names.
    """
    staged: list[str] = []
    splits = ("train", "valid", "test")

    for ds in datasets:
        subset_type = dataset_type_map.get(ds)
        if subset_type is None:
            log.warning("Dataset %r not found in standardized_datasets, skipping", ds)
            continue

        ds_out = staging_dir / "data" / ds
        ds_out.mkdir(parents=True, exist_ok=True)

        all_found = True
        for split in splits:
            src = resplit_dir / subset_type / ds / f"{split}.parquet"
            hf_split = _SPLIT_RENAME.get(split, split)
            dest = ds_out / f"{hf_split}.parquet"

            if not src.exists():
                log.warning("Missing parquet: %s", src)
                all_found = False
                continue

            _symlink_file(src, dest)

        if all_found:
            staged.append(ds)
            log.info("Staged dataset: %s (%s)", ds, subset_type)
        else:
            log.warning("Dataset %s staged with missing splits", ds)
            staged.append(ds)

    return staged


def _stage_metadata(resplit_meta_dir: Path, staging_dir: Path) -> list[str]:
    """Symlink resplit metadata artifacts into staging_dir/metadata/.

    Returns the list of successfully staged file names.
    """
    meta_out = staging_dir / "metadata"
    meta_out.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    for fname in _METADATA_FILES:
        src = resplit_meta_dir / fname
        if src.exists():
            _symlink_file(src, meta_out / fname)
            copied.append(fname)
            log.info("Staged metadata: %s", fname)
        else:
            log.warning("Metadata file not found: %s", src)

    return copied


# ── Main ──────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Upload resplit datasets to HuggingFace Hub."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the pipeline config YAML (e.g. config/config.yml)",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Override the HuggingFace datasets repo ID from config",
    )
    parser.add_argument(
        "--datasets-resplit-dir",
        default="results/datasets_resplit",
        help=("Root directory of resplit datasets (default: results/datasets_resplit)"),
    )
    parser.add_argument(
        "--resplit-dir",
        default="results/resplit",
        help=(
            "Directory containing resplit metadata artifacts (default: results/resplit)"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Stage files but do not upload to HuggingFace",
    )
    parser.add_argument(
        "--commit-message",
        default=None,
        help="Custom commit message for the HF upload",
    )
    args = parser.parse_args(argv)

    config = _load_config(args.config)

    # Resolve repo ID
    repo_id = args.repo_id
    if repo_id is None:
        hf_config = config.get("huggingface", {})
        repo_id = hf_config.get("datasets_repo_id")
    if not repo_id:
        log.error(
            "No datasets repo_id specified. "
            "Set huggingface.datasets_repo_id in config or use --repo-id."
        )
        sys.exit(1)

    # Resolve dataset list from resplit config
    resplit_config = config.get("resplit", {})
    datasets = resplit_config.get("datasets", [])
    if not datasets:
        log.error("No datasets listed under resplit.datasets in config.")
        sys.exit(1)

    # Build dataset -> subset_type mapping
    dataset_type_map = _resolve_dataset_type_map(config)

    resplit_dir = Path(args.datasets_resplit_dir)
    if not resplit_dir.exists():
        log.error("Resplit datasets directory does not exist: %s", resplit_dir)
        sys.exit(1)

    resplit_meta_dir = Path(args.resplit_dir)

    # Prepare staging directory (gitignored, in project root, symlinks only)
    staging_dir = _STAGING_DIR
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)
    log.info("Staging directory: %s", staging_dir)

    try:
        # Stage dataset parquets (symlinks)
        staged = _stage_datasets(datasets, dataset_type_map, resplit_dir, staging_dir)
        if not staged:
            log.error("No datasets were staged. Check resplit directory.")
            sys.exit(1)

        # Stage metadata (symlinks)
        _stage_metadata(resplit_meta_dir, staging_dir)

        # Generate dataset card (written, not symlinked)
        card = _build_dataset_card(repo_id, staged, dataset_type_map, resplit_config)
        readme = staging_dir / "README.md"
        readme.write_text(card)
        log.info("Wrote dataset card: %s", readme)

        log.info("Staged %d dataset(s) for upload to %s", len(staged), repo_id)

        if args.dry_run:
            log.info("Dry run -- listing staged files:")
            for p in sorted(staging_dir.rglob("*")):
                if p.is_symlink() or p.is_file():
                    rel = p.relative_to(staging_dir)
                    target = p.resolve()
                    size_mb = target.stat().st_size / (1024 * 1024)
                    log.info("  %s (%.2f MB)", rel, size_mb)
            log.info("Dry run complete. No files were uploaded.")
            return

        # Upload to HuggingFace
        try:
            from huggingface_hub import HfApi
        except ImportError:
            log.error(
                "huggingface_hub is not installed. "
                "Install it with: pip install huggingface_hub"
            )
            sys.exit(1)

        api = HfApi()

        # Create repo if it doesn't exist
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            exist_ok=True,
        )

        commit_message = args.commit_message or (
            f"Upload {len(staged)} resplit dataset(s): " + ", ".join(staged)
        )

        log.info("Uploading to %s ...", repo_id)
        api.upload_folder(
            repo_id=repo_id,
            folder_path=str(staging_dir),
            commit_message=commit_message,
            repo_type="dataset",
        )

        log.info("Upload complete: https://huggingface.co/datasets/%s", repo_id)

    finally:
        # Clean up staging directory
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
            log.info("Cleaned up staging directory: %s", staging_dir)


if __name__ == "__main__":
    main()
