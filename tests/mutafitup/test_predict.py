import json
import logging

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from mutafitup.datasets import BaseMultitaskDataset
from mutafitup.models.multitask_heads import (
    PerProteinClassificationHead,
    PerProteinRegressionHead,
    PerResidueClassificationHead,
    PerResidueRegressionHead,
)
from mutafitup.models.multitask_model import (
    HeadConfig,
    MultitaskBackbone,
    MultitaskForwardArgs,
    MultitaskModel,
)
from mutafitup.predict import predict, save_predictions_jsonl


class DummyBackbone(MultitaskBackbone):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, args: MultitaskForwardArgs) -> torch.Tensor:
        return self.linear(args.input_ids.float())

    def preprocess_sequences(self, sequences, checkpoint):
        return sequences

    def _inject_lora_config(self, lora_config=None) -> None:
        self._injected = lora_config


class DummyBatchDataset(Dataset):
    def __init__(
        self,
        num_samples: int,
        seq_len: int,
        hidden_size: int,
        level: str,
        problem_type: str,
        num_classes: int = 2,
    ):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.level = level
        self.problem_type = problem_type
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        attention_mask = torch.ones(self.seq_len, dtype=torch.long)
        input_ids = torch.randn(self.seq_len, self.hidden_size)

        if self.level == "per_protein":
            if self.problem_type == "classification":
                label = torch.tensor(idx % self.num_classes, dtype=torch.long)
            else:
                label = torch.tensor(float(idx) * 0.1, dtype=torch.float)
        else:
            if self.problem_type == "classification":
                label = torch.randint(0, self.num_classes, (self.seq_len,))
            else:
                label = torch.randn(self.seq_len)

        return {
            "attention_mask": attention_mask,
            "input_ids": input_ids,
            "labels": label,
        }


class DummyTaskDataset(BaseMultitaskDataset):
    def __init__(
        self,
        name: str,
        num_samples: int,
        seq_len: int,
        hidden_size: int,
        level: str,
        problem_type: str,
        num_classes: int = 2,
    ):
        super().__init__(name)
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.level = level
        self.problem_type = problem_type
        self.num_classes = num_classes

        sequences = [f"ACDEF{'G' * i}" for i in range(num_samples)]
        df = pd.DataFrame({"sequence": sequences})
        self.train_df = df
        self.valid_df = df
        self.test_df = df

    def get_dataloader(
        self,
        split: str,
        backbone: MultitaskBackbone,
        tokenizer,
        checkpoint: str,
        batch_size: int,
    ) -> DataLoader:
        dataset = DummyBatchDataset(
            self.num_samples,
            self.seq_len,
            self.hidden_size,
            self.level,
            self.problem_type,
            self.num_classes,
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def test_predict_per_protein_classification():
    hidden_size = 8
    num_classes = 3
    num_samples = 6

    backbone = DummyBackbone(hidden_size)
    heads = {
        "task_a": HeadConfig(
            head=PerProteinClassificationHead(
                0.0, hidden_size, hidden_size, num_classes
            ),
            problem_type="classification",
            level="per_protein",
        ),
    }
    model = MultitaskModel(backbone=backbone, heads=heads)

    task_datasets = {
        "task_a": DummyTaskDataset(
            "task_a",
            num_samples=num_samples,
            seq_len=3,
            hidden_size=hidden_size,
            level="per_protein",
            problem_type="classification",
            num_classes=num_classes,
        ),
    }

    results = predict(
        model=model,
        tokenizer=None,
        checkpoint="dummy",
        task_datasets=task_datasets,
        split="valid",
        batch_size=2,
    )

    assert "task_a" in results
    preds = results["task_a"]
    assert len(preds) == num_samples

    for row in preds:
        assert "id" in row
        assert "prediction" in row
        assert "target" in row
        assert "sequence" in row
        assert isinstance(row["prediction"], int)
        assert isinstance(row["target"], int)
        assert 0 <= row["prediction"] < num_classes


def test_predict_per_protein_regression():
    hidden_size = 8
    num_samples = 4

    backbone = DummyBackbone(hidden_size)
    heads = {
        "task_a": HeadConfig(
            head=PerProteinRegressionHead(0.0, hidden_size, hidden_size),
            problem_type="regression",
            level="per_protein",
        ),
    }
    model = MultitaskModel(backbone=backbone, heads=heads)

    task_datasets = {
        "task_a": DummyTaskDataset(
            "task_a",
            num_samples=num_samples,
            seq_len=3,
            hidden_size=hidden_size,
            level="per_protein",
            problem_type="regression",
        ),
    }

    results = predict(
        model=model,
        tokenizer=None,
        checkpoint="dummy",
        task_datasets=task_datasets,
        split="valid",
        batch_size=2,
    )

    assert "task_a" in results
    preds = results["task_a"]
    assert len(preds) == num_samples

    for row in preds:
        assert isinstance(row["prediction"], float)
        assert isinstance(row["target"], float)


def test_predict_per_residue_classification():
    hidden_size = 8
    num_classes = 3
    num_samples = 4
    seq_len = 5

    backbone = DummyBackbone(hidden_size)
    heads = {
        "task_a": HeadConfig(
            head=PerResidueClassificationHead(
                0.0, hidden_size, hidden_size, num_classes
            ),
            problem_type="classification",
            level="per_residue",
        ),
    }
    model = MultitaskModel(backbone=backbone, heads=heads)

    task_datasets = {
        "task_a": DummyTaskDataset(
            "task_a",
            num_samples=num_samples,
            seq_len=seq_len,
            hidden_size=hidden_size,
            level="per_residue",
            problem_type="classification",
            num_classes=num_classes,
        ),
    }

    results = predict(
        model=model,
        tokenizer=None,
        checkpoint="dummy",
        task_datasets=task_datasets,
        split="valid",
        batch_size=2,
    )

    assert "task_a" in results
    preds = results["task_a"]
    assert len(preds) == num_samples

    for row in preds:
        assert isinstance(row["prediction"], list)
        assert isinstance(row["target"], list)
        assert len(row["prediction"]) == len(row["target"])
        for v in row["prediction"]:
            assert 0 <= v < num_classes


def test_predict_per_residue_regression():
    hidden_size = 8
    num_samples = 4
    seq_len = 5

    backbone = DummyBackbone(hidden_size)
    heads = {
        "task_a": HeadConfig(
            head=PerResidueRegressionHead(0.0, hidden_size, hidden_size),
            problem_type="regression",
            level="per_residue",
        ),
    }
    model = MultitaskModel(backbone=backbone, heads=heads)

    task_datasets = {
        "task_a": DummyTaskDataset(
            "task_a",
            num_samples=num_samples,
            seq_len=seq_len,
            hidden_size=hidden_size,
            level="per_residue",
            problem_type="regression",
        ),
    }

    results = predict(
        model=model,
        tokenizer=None,
        checkpoint="dummy",
        task_datasets=task_datasets,
        split="valid",
        batch_size=2,
    )

    assert "task_a" in results
    preds = results["task_a"]
    assert len(preds) == num_samples

    for row in preds:
        assert isinstance(row["prediction"], list)
        assert isinstance(row["target"], list)
        assert len(row["prediction"]) == len(row["target"])


def test_predict_multitask():
    hidden_size = 8
    num_classes = 3
    num_samples = 4

    backbone = DummyBackbone(hidden_size)
    heads = {
        "cls_task": HeadConfig(
            head=PerProteinClassificationHead(
                0.0, hidden_size, hidden_size, num_classes
            ),
            problem_type="classification",
            level="per_protein",
        ),
        "reg_task": HeadConfig(
            head=PerProteinRegressionHead(0.0, hidden_size, hidden_size),
            problem_type="regression",
            level="per_protein",
        ),
    }
    model = MultitaskModel(backbone=backbone, heads=heads)

    task_datasets = {
        "cls_task": DummyTaskDataset(
            "cls_task",
            num_samples=num_samples,
            seq_len=3,
            hidden_size=hidden_size,
            level="per_protein",
            problem_type="classification",
            num_classes=num_classes,
        ),
        "reg_task": DummyTaskDataset(
            "reg_task",
            num_samples=num_samples,
            seq_len=3,
            hidden_size=hidden_size,
            level="per_protein",
            problem_type="regression",
        ),
    }

    results = predict(
        model=model,
        tokenizer=None,
        checkpoint="dummy",
        task_datasets=task_datasets,
        split="valid",
        batch_size=2,
    )

    assert "cls_task" in results
    assert "reg_task" in results
    assert len(results["cls_task"]) == num_samples
    assert len(results["reg_task"]) == num_samples


def test_save_predictions_jsonl(tmp_path):
    predictions = {
        "task_a": [
            {"id": 0, "prediction": 1, "target": 0, "sequence": "ACDEF"},
            {"id": 1, "prediction": 2, "target": 1, "sequence": "ACDEFG"},
        ],
        "task_b": [
            {"id": 0, "prediction": 0.5, "target": 0.3, "sequence": "ACDEF"},
        ],
    }

    written = save_predictions_jsonl(
        predictions=predictions,
        output_dir=str(tmp_path / "predictions" / "run1" / "valid"),
    )

    assert "task_a" in written
    assert "task_b" in written

    for task_name, path in written.items():
        assert path.endswith(f"{task_name}.jsonl")
        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == len(predictions[task_name])
        for line in lines:
            row = json.loads(line)
            assert "id" in row
            assert "prediction" in row
            assert "target" in row
            assert "sequence" in row


def test_predict_with_logger():
    hidden_size = 8
    num_samples = 4

    backbone = DummyBackbone(hidden_size)
    heads = {
        "task_a": HeadConfig(
            head=PerProteinClassificationHead(0.0, hidden_size, hidden_size, 2),
            problem_type="classification",
            level="per_protein",
        ),
    }
    model = MultitaskModel(backbone=backbone, heads=heads)

    task_datasets = {
        "task_a": DummyTaskDataset(
            "task_a",
            num_samples=num_samples,
            seq_len=3,
            hidden_size=hidden_size,
            level="per_protein",
            problem_type="classification",
        ),
    }

    logger = logging.getLogger("test_predict")

    results = predict(
        model=model,
        tokenizer=None,
        checkpoint="dummy",
        task_datasets=task_datasets,
        split="valid",
        batch_size=2,
        logger=logger,
    )

    assert len(results["task_a"]) == num_samples


def test_predict_test_split():
    hidden_size = 8
    num_samples = 4

    backbone = DummyBackbone(hidden_size)
    heads = {
        "task_a": HeadConfig(
            head=PerProteinClassificationHead(0.0, hidden_size, hidden_size, 2),
            problem_type="classification",
            level="per_protein",
        ),
    }
    model = MultitaskModel(backbone=backbone, heads=heads)

    task_datasets = {
        "task_a": DummyTaskDataset(
            "task_a",
            num_samples=num_samples,
            seq_len=3,
            hidden_size=hidden_size,
            level="per_protein",
            problem_type="classification",
        ),
    }

    results = predict(
        model=model,
        tokenizer=None,
        checkpoint="dummy",
        task_datasets=task_datasets,
        split="test",
        batch_size=2,
    )

    assert "task_a" in results
    assert len(results["task_a"]) == num_samples
    for row in results["task_a"]:
        assert "id" in row
        assert "prediction" in row
        assert "target" in row
        assert "sequence" in row
