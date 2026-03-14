import pandas as pd
import pytest

from mutafitup.normalization_stats import compute_normalization_stats


def test_compute_normalization_stats_matches_dataset_scaling(tmp_path):
    base = tmp_path / "datasets_resplit"

    meltome_dir = base / "per_protein_regression" / "meltome"
    meltome_dir.mkdir(parents=True)
    pd.DataFrame(
        {"sequence": ["AAA", "BBB", "CCC"], "score": [-4.0, 2.0, 999.0]}
    ).to_parquet(meltome_dir / "train.parquet")

    rsa_dir = base / "per_residue_regression" / "rsa"
    rsa_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "sequence": ["DDD", "EEE"],
            "score": [[1.0, -2.0, 999.0], [3.0, -5.0, 4.0]],
        }
    ).to_parquet(rsa_dir / "train.parquet")

    tasks = [
        {"name": "meltome", "subset_type": "per_protein_regression"},
        {"name": "rsa", "subset_type": "per_residue_regression"},
        {"name": "secstr", "subset_type": "per_residue_classification"},
    ]

    stats = compute_normalization_stats(tasks, str(base))

    assert set(stats) == {"meltome", "rsa"}
    assert stats["meltome"] == {
        "label_min": -4.0,
        "label_max": 2.0,
        "scale_factor": 4.0,
        "display_scale_factor": 10.0,
    }
    assert stats["rsa"] == {
        "label_min": -5.0,
        "label_max": 4.0,
        "scale_factor": 5.0,
        "display_scale_factor": 1.0,
    }


def test_compute_normalization_stats_raises_when_all_scores_unresolved(tmp_path):
    base = tmp_path / "datasets_resplit"
    disorder_dir = base / "per_residue_regression" / "disorder"
    disorder_dir.mkdir(parents=True)
    pd.DataFrame({"sequence": ["AAA"], "score": [[999.0, 999.0]]}).to_parquet(
        disorder_dir / "train.parquet"
    )

    tasks = [{"name": "disorder", "subset_type": "per_residue_regression"}]

    with pytest.raises(ValueError, match="No resolved scores found"):
        compute_normalization_stats(tasks, str(base))
