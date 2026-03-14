"""Tests for Snakefile helper functions related to leakage analysis.

These tests verify that `get_leakage_heatmap_targets` produces the
correct list of output files given various config states with profiles.
"""

import importlib
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest


SNAKEFILE_PATH = Path(__file__).resolve().parents[2] / "workflow" / "Snakefile"

LEAKAGE_COMPARISONS = [
    ("valid", "train"),
    ("test", "train"),
    ("test", "valid"),
    ("valid", "valid"),
    ("valid", "test"),
    ("test", "test"),
]

LEAKAGE_REMOVAL_VARIANTS = [
    "target_all",
    "target_cross_task",
    "query_all",
    "query_cross_task",
]


def _get_heatmap_targets(config: dict) -> list[str]:
    """Simulate get_leakage_heatmap_targets with a given config dict.

    We cannot source the Snakefile directly (it requires Snakemake), so we
    replicate the logic from the function defined there.
    """
    # This mirrors the implementation in workflow/Snakefile:
    profiles = config.get("leakage_analysis", {}).get("profiles", {})
    targets = []
    for profile_name, profile_cfg in profiles.items():
        if not profile_cfg.get("datasets"):
            continue
        tools = profile_cfg.get("tools", {})
        enabled = [name for name, cfg in tools.items() if cfg.get("enabled", False)]
        for tool in enabled:
            for query_split, target_split in LEAKAGE_COMPARISONS:
                targets.append(
                    f"results/leakage_analysis/{profile_name}/heatmap_{tool}_{query_split}_vs_{target_split}.png"
                )
    return targets


def _get_removal_targets(config: dict) -> list[str]:
    """Simulate get_leakage_removal_targets with a given config dict.

    Mirrors the implementation in workflow/Snakefile.
    """
    profiles = config.get("leakage_analysis", {}).get("profiles", {})
    targets = []
    for profile_name, profile_cfg in profiles.items():
        if not profile_cfg.get("datasets"):
            continue
        tools = profile_cfg.get("tools", {})
        enabled = [name for name, cfg in tools.items() if cfg.get("enabled", False)]
        for tool in enabled:
            for query_split, target_split in LEAKAGE_COMPARISONS:
                for variant in LEAKAGE_REMOVAL_VARIANTS:
                    targets.append(
                        f"results/leakage_analysis/{profile_name}/removal_{variant}_{tool}_{query_split}_vs_{target_split}.png"
                    )
                    targets.append(
                        f"results/leakage_analysis/{profile_name}/removal_bar_{variant}_{tool}_{query_split}_vs_{target_split}.png"
                    )
    return targets


def _two_dataset_list():
    return [
        {"name": "A", "type": "per_protein_regression"},
        {"name": "B", "type": "per_protein_regression"},
    ]


class TestGetLeakageHeatmapTargets:
    def test_both_tools_enabled_single_profile(self):
        """Returns 12 heatmap targets when both tools are enabled in one profile."""
        config = {
            "leakage_analysis": {
                "profiles": {
                    "default": {
                        "datasets": _two_dataset_list(),
                        "tools": {
                            "mmseqs": {"enabled": True, "min_seq_id": 0.3},
                            "foldseek": {"enabled": True, "min_seq_id": 0.3},
                        },
                    }
                }
            }
        }
        targets = _get_heatmap_targets(config)
        # 2 tools * 6 comparisons = 12
        assert len(targets) == 12
        assert (
            "results/leakage_analysis/default/heatmap_mmseqs_valid_vs_train.png"
            in targets
        )
        assert (
            "results/leakage_analysis/default/heatmap_mmseqs_test_vs_train.png"
            in targets
        )
        assert (
            "results/leakage_analysis/default/heatmap_mmseqs_test_vs_valid.png"
            in targets
        )
        assert (
            "results/leakage_analysis/default/heatmap_foldseek_valid_vs_train.png"
            in targets
        )
        assert (
            "results/leakage_analysis/default/heatmap_foldseek_test_vs_valid.png"
            in targets
        )
        assert (
            "results/leakage_analysis/default/heatmap_foldseek_test_vs_test.png"
            in targets
        )

    def test_only_mmseqs_enabled(self):
        """Returns 6 heatmap targets when only mmseqs is enabled."""
        config = {
            "leakage_analysis": {
                "profiles": {
                    "test": {
                        "datasets": _two_dataset_list(),
                        "tools": {
                            "mmseqs": {"enabled": True, "min_seq_id": 0.3},
                            "foldseek": {"enabled": False, "min_seq_id": 0.3},
                        },
                    }
                }
            }
        }
        targets = _get_heatmap_targets(config)
        # 1 tool * 6 comparisons = 6
        assert len(targets) == 6
        assert (
            "results/leakage_analysis/test/heatmap_mmseqs_valid_vs_train.png" in targets
        )
        assert (
            "results/leakage_analysis/test/heatmap_mmseqs_test_vs_train.png" in targets
        )
        assert (
            "results/leakage_analysis/test/heatmap_mmseqs_test_vs_valid.png" in targets
        )
        assert all("foldseek" not in t for t in targets)

    def test_only_foldseek_enabled(self):
        """Returns 6 heatmap targets when only foldseek is enabled."""
        config = {
            "leakage_analysis": {
                "profiles": {
                    "fs": {
                        "datasets": _two_dataset_list(),
                        "tools": {
                            "mmseqs": {"enabled": False, "min_seq_id": 0.3},
                            "foldseek": {"enabled": True, "min_seq_id": 0.3},
                        },
                    }
                }
            }
        }
        targets = _get_heatmap_targets(config)
        # 1 tool * 6 comparisons = 6
        assert len(targets) == 6
        assert (
            "results/leakage_analysis/fs/heatmap_foldseek_valid_vs_train.png" in targets
        )
        assert (
            "results/leakage_analysis/fs/heatmap_foldseek_test_vs_train.png" in targets
        )
        assert all("mmseqs" not in t for t in targets)

    def test_both_disabled(self):
        """Returns empty list when both tools are disabled."""
        config = {
            "leakage_analysis": {
                "profiles": {
                    "none": {
                        "datasets": _two_dataset_list(),
                        "tools": {
                            "mmseqs": {"enabled": False, "min_seq_id": 0.3},
                            "foldseek": {"enabled": False, "min_seq_id": 0.3},
                        },
                    }
                }
            }
        }
        targets = _get_heatmap_targets(config)
        assert targets == []

    def test_no_leakage_analysis(self):
        """Returns empty list when leakage_analysis is not in config."""
        config = {}
        targets = _get_heatmap_targets(config)
        assert targets == []

    def test_empty_datasets(self):
        """Returns empty list when datasets list is empty."""
        config = {
            "leakage_analysis": {
                "profiles": {
                    "empty": {
                        "datasets": [],
                        "tools": {
                            "mmseqs": {"enabled": True, "min_seq_id": 0.3},
                            "foldseek": {"enabled": True, "min_seq_id": 0.3},
                        },
                    }
                }
            }
        }
        targets = _get_heatmap_targets(config)
        assert targets == []

    def test_multiple_profiles(self):
        """Multiple profiles each produce their own heatmap targets."""
        config = {
            "leakage_analysis": {
                "profiles": {
                    "relaxed": {
                        "datasets": _two_dataset_list(),
                        "tools": {
                            "mmseqs": {"enabled": True, "min_seq_id": 0.3},
                            "foldseek": {"enabled": False, "min_seq_id": 0.3},
                        },
                    },
                    "strict": {
                        "datasets": _two_dataset_list(),
                        "tools": {
                            "mmseqs": {"enabled": True, "min_seq_id": 0.5},
                            "foldseek": {"enabled": True, "min_seq_id": 0.5},
                        },
                    },
                }
            }
        }
        targets = _get_heatmap_targets(config)
        # relaxed: 1 tool * 6 comparisons = 6
        # strict: 2 tools * 6 comparisons = 12
        assert len(targets) == 18
        assert (
            "results/leakage_analysis/relaxed/heatmap_mmseqs_valid_vs_train.png"
            in targets
        )
        assert (
            "results/leakage_analysis/relaxed/heatmap_mmseqs_test_vs_train.png"
            in targets
        )
        assert (
            "results/leakage_analysis/strict/heatmap_mmseqs_valid_vs_train.png"
            in targets
        )
        assert (
            "results/leakage_analysis/strict/heatmap_mmseqs_test_vs_valid.png"
            in targets
        )
        assert (
            "results/leakage_analysis/strict/heatmap_foldseek_valid_vs_train.png"
            in targets
        )
        assert (
            "results/leakage_analysis/strict/heatmap_foldseek_test_vs_valid.png"
            in targets
        )
        # relaxed should not have foldseek
        assert all("relaxed" not in t or "foldseek" not in t for t in targets)

    def test_target_format(self):
        """Verify exact file path format including profile."""
        config = {
            "leakage_analysis": {
                "profiles": {
                    "myprofile": {
                        "datasets": [
                            {"name": "SecStr", "type": "per_residue_classification"},
                            {"name": "Disorder", "type": "per_residue_regression"},
                        ],
                        "tools": {
                            "mmseqs": {"enabled": True, "min_seq_id": 0.3},
                            "foldseek": {"enabled": True, "min_seq_id": 0.3},
                        },
                    }
                }
            }
        }
        targets = _get_heatmap_targets(config)
        # 2 tools * 6 comparisons = 12
        expected = set()
        for tool in ("mmseqs", "foldseek"):
            for qs, ts in LEAKAGE_COMPARISONS:
                expected.add(
                    f"results/leakage_analysis/myprofile/heatmap_{tool}_{qs}_vs_{ts}.png"
                )
        assert set(targets) == expected


class TestGetLeakageRemovalTargets:
    def test_both_tools_enabled_single_profile(self):
        """Returns 96 removal targets when both tools enabled."""
        config = {
            "leakage_analysis": {
                "profiles": {
                    "default": {
                        "datasets": _two_dataset_list(),
                        "tools": {
                            "mmseqs": {"enabled": True, "min_seq_id": 0.3},
                            "foldseek": {"enabled": True, "min_seq_id": 0.3},
                        },
                    }
                }
            }
        }
        targets = _get_removal_targets(config)
        # 2 tools * 6 comparisons * 4 variants * 2 files (heatmap + bar) = 96
        assert len(targets) == 96
        # Spot-check representative targets
        assert (
            "results/leakage_analysis/default/removal_target_all_mmseqs_valid_vs_train.png"
            in targets
        )
        assert (
            "results/leakage_analysis/default/removal_bar_target_all_mmseqs_valid_vs_train.png"
            in targets
        )
        assert (
            "results/leakage_analysis/default/removal_target_cross_task_mmseqs_test_vs_train.png"
            in targets
        )
        assert (
            "results/leakage_analysis/default/removal_query_all_foldseek_test_vs_valid.png"
            in targets
        )
        assert (
            "results/leakage_analysis/default/removal_bar_query_cross_task_foldseek_test_vs_test.png"
            in targets
        )

    def test_only_mmseqs_enabled(self):
        """Returns 48 removal targets when only mmseqs is enabled."""
        config = {
            "leakage_analysis": {
                "profiles": {
                    "test": {
                        "datasets": _two_dataset_list(),
                        "tools": {
                            "mmseqs": {"enabled": True, "min_seq_id": 0.3},
                            "foldseek": {"enabled": False, "min_seq_id": 0.3},
                        },
                    }
                }
            }
        }
        targets = _get_removal_targets(config)
        # 1 tool * 6 comparisons * 4 variants * 2 files = 48
        assert len(targets) == 48
        assert all("foldseek" not in t for t in targets)

    def test_both_disabled(self):
        """Returns empty list when both tools are disabled."""
        config = {
            "leakage_analysis": {
                "profiles": {
                    "none": {
                        "datasets": _two_dataset_list(),
                        "tools": {
                            "mmseqs": {"enabled": False, "min_seq_id": 0.3},
                            "foldseek": {"enabled": False, "min_seq_id": 0.3},
                        },
                    }
                }
            }
        }
        targets = _get_removal_targets(config)
        assert targets == []

    def test_no_leakage_analysis(self):
        """Returns empty list when leakage_analysis is not in config."""
        assert _get_removal_targets({}) == []

    def test_multiple_profiles(self):
        """Multiple profiles each produce their own removal targets."""
        config = {
            "leakage_analysis": {
                "profiles": {
                    "relaxed": {
                        "datasets": _two_dataset_list(),
                        "tools": {
                            "mmseqs": {"enabled": True, "min_seq_id": 0.2},
                            "foldseek": {"enabled": False, "min_seq_id": 0.2},
                        },
                    },
                    "strict": {
                        "datasets": _two_dataset_list(),
                        "tools": {
                            "mmseqs": {"enabled": True, "min_seq_id": 0.9},
                            "foldseek": {"enabled": True, "min_seq_id": 0.9},
                        },
                    },
                }
            }
        }
        targets = _get_removal_targets(config)
        # relaxed: 1 tool * 6 comparisons * 4 variants * 2 files = 48
        # strict: 2 tools * 6 comparisons * 4 variants * 2 files = 96
        assert len(targets) == 144
        assert (
            "results/leakage_analysis/relaxed/removal_target_all_mmseqs_valid_vs_train.png"
            in targets
        )
        assert (
            "results/leakage_analysis/relaxed/removal_bar_target_all_mmseqs_valid_vs_train.png"
            in targets
        )
        assert (
            "results/leakage_analysis/strict/removal_query_all_foldseek_test_vs_valid.png"
            in targets
        )
        assert (
            "results/leakage_analysis/strict/removal_bar_query_cross_task_foldseek_test_vs_test.png"
            in targets
        )
