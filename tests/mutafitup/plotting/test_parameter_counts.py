from pathlib import Path

import pandas as pd
import pytest
import torch
import torch.nn as nn

from mutafitup.plotting import plot_trainable_vs_frozen
from mutafitup.plotting import parameter_counts as pc
from mutafitup.plotting.typst_tables import export_parameter_summary_typst


class DummyBackbone(nn.Module):
    def __init__(self, hidden_size: int, trainable_backbone_params: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.frozen = nn.Parameter(torch.zeros(10), requires_grad=False)
        self.trainable = nn.Parameter(torch.zeros(trainable_backbone_params))


def test_count_module_parameters_counts_total_and_trainable():
    module = nn.Sequential(
        nn.Linear(3, 2),
        nn.Linear(2, 1),
    )
    for p in module[1].parameters():
        p.requires_grad = False

    total, trainable = pc.count_module_parameters(module)

    assert total == sum(p.numel() for p in module.parameters())
    assert trainable == sum(p.numel() for p in module.parameters() if p.requires_grad)


def test_summarize_parameter_counts_orders_sections_and_combines_counts(monkeypatch):
    def fake_build_backbone_and_tokenizer(checkpoint, lora_rank, lora_alpha):
        trainable = 0 if lora_rank is None else 4
        return DummyBackbone(
            hidden_size=8, trainable_backbone_params=trainable
        ), object()

    monkeypatch.setattr(
        pc, "build_backbone_and_tokenizer", fake_build_backbone_and_tokenizer
    )

    run_entries = [
        {
            "section": "align_lora",
            "id": "z_run",
            "checkpoint": "esmc_300m",
            "dropout": 0.1,
            "lora_rank": 4,
            "lora_alpha": 4,
            "tasks": [
                {
                    "name": "secstr",
                    "subset_type": "per_residue_classification",
                    "num_labels": 3,
                }
            ],
        },
        {
            "section": "heads_only",
            "id": "a_run",
            "checkpoint": "esmc_300m",
            "dropout": 0.1,
            "lora_rank": None,
            "lora_alpha": None,
            "tasks": [{"name": "meltome", "subset_type": "per_protein_regression"}],
        },
    ]

    df = pc.summarize_parameter_counts(run_entries)

    assert list(df["section"]) == ["heads_only", "align_lora"]
    assert list(df["run_id"]) == ["a_run", "z_run"]

    heads_only = df.iloc[0]
    align = df.iloc[1]
    assert heads_only["backbone_trainable_params"] == 0
    assert heads_only["trainable_params"] == heads_only["head_trainable_params"]
    assert align["backbone_trainable_params"] == 4
    assert (
        align["trainable_params"]
        == align["backbone_trainable_params"] + align["head_trainable_params"]
    )
    assert (
        heads_only["frozen_params"]
        == heads_only["total_params"] - heads_only["trainable_params"]
    )
    assert align["trainable_fraction"] > heads_only["trainable_fraction"]


def test_summarize_parameter_counts_reuses_backbone_cache(monkeypatch):
    calls = []

    def fake_build_backbone_and_tokenizer(checkpoint, lora_rank, lora_alpha):
        calls.append((checkpoint, lora_rank, lora_alpha))
        return DummyBackbone(hidden_size=8, trainable_backbone_params=2), object()

    monkeypatch.setattr(
        pc, "build_backbone_and_tokenizer", fake_build_backbone_and_tokenizer
    )

    shared = {
        "section": "lora",
        "checkpoint": "esmc_300m",
        "dropout": 0.1,
        "lora_rank": 4,
        "lora_alpha": 4,
    }
    run_entries = [
        {
            **shared,
            "id": "run1",
            "tasks": [{"name": "rsa", "subset_type": "per_residue_regression"}],
        },
        {
            **shared,
            "id": "run2",
            "tasks": [{"name": "meltome", "subset_type": "per_protein_regression"}],
        },
    ]

    pc.summarize_parameter_counts(run_entries)

    assert calls == [("esmc_300m", 4, 4)]


def test_filter_run_entries_returns_all_when_filter_empty():
    run_entries = [
        {"section": "heads_only", "id": "run_a"},
        {"section": "lora", "id": "run_b"},
    ]

    filtered = pc.filter_run_entries(run_entries, [])

    assert filtered == run_entries


def test_filter_run_entries_keeps_requested_order_subset():
    run_entries = [
        {"section": "heads_only", "id": "run_a"},
        {"section": "lora", "id": "run_b"},
        {"section": "align_lora", "id": "run_c"},
    ]

    filtered = pc.filter_run_entries(
        run_entries,
        ["heads_only/run_a", "align_lora/run_c"],
    )

    assert filtered == [
        {"section": "heads_only", "id": "run_a"},
        {"section": "align_lora", "id": "run_c"},
    ]


def test_filter_run_entries_raises_on_unknown_run_key():
    run_entries = [{"section": "heads_only", "id": "run_a"}]

    with pytest.raises(ValueError, match="Unknown parameter_counts.include_runs"):
        pc.filter_run_entries(run_entries, ["lora/missing"])


def test_plot_trainable_vs_frozen_writes_png(tmp_path: Path):
    df = pd.DataFrame(
        [
            {
                "section": "heads_only",
                "run_id": "run_a",
                "total_params": 1_000_000,
                "trainable_params": 20_000,
                "frozen_params": 980_000,
                "trainable_fraction": 0.02,
            },
            {
                "section": "lora",
                "run_id": "run_b",
                "total_params": 1_000_000,
                "trainable_params": 40_000,
                "frozen_params": 960_000,
                "trainable_fraction": 0.04,
            },
        ]
    )
    out = tmp_path / "params.png"

    plot_trainable_vs_frozen(df, str(out))

    assert out.is_file()
    assert out.stat().st_size > 0


def test_export_parameter_summary_typst_renders_expected_cells():
    df = pd.DataFrame(
        [
            {
                "section": "heads_only",
                "run_id": "run_a",
                "checkpoint": "esmc_300m",
                "num_tasks": 1,
                "lora_rank": None,
                "total_params": 1_000_000,
                "trainable_params": 20_000,
                "frozen_params": 980_000,
                "trainable_fraction": 0.02,
            },
            {
                "section": "align_lora",
                "run_id": "run_b",
                "checkpoint": "esmc_300m",
                "num_tasks": 15,
                "lora_rank": 16,
                "total_params": 1_500_000,
                "trainable_params": 60_000,
                "frozen_params": 1_440_000,
                "trainable_fraction": 0.04,
            },
        ]
    )

    table = export_parameter_summary_typst(df)

    assert "#table(" in table
    assert "[Heads only / run_a]" in table
    assert "[Align LoRA / run_b]" in table
    assert "[1,000,000]" in table
    assert "[4.00%]" in table
    assert "[-]" in table
