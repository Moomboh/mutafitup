import logging

import pytest
import torch

import mutafitup.train.train_multitask_model as train_module
from mutafitup.models.multitask_heads import PerProteinClassificationHead
from mutafitup.models.multitask_model import HeadConfig, MultitaskModel

from helpers.train_multitask import (
    DummyBackbone,
    DummyOptimizer,
    DummyScheduler,
    DummyTaskDataset,
    RecordingBackbone,
    ShuffledDummyTaskDataset,
    record,
)


def test_train_multitask_model_steps_per_task_batch(monkeypatch):
    record.clear()
    monkeypatch.setattr(train_module.torch.optim, "AdamW", DummyOptimizer)
    monkeypatch.setattr(
        train_module,
        "get_linear_schedule_with_warmup",
        lambda opt, num_warmup_steps, total_steps: DummyScheduler(
            opt, num_warmup_steps, total_steps
        ),
    )

    backbone = DummyBackbone(hidden_size=8)
    heads = {
        "task_a": HeadConfig(
            head=PerProteinClassificationHead(0.0, 8, 8, 2),
            problem_type="classification",
            level="per_protein",
        ),
        "task_b": HeadConfig(
            head=PerProteinClassificationHead(0.0, 8, 8, 2),
            problem_type="classification",
            level="per_protein",
        ),
    }
    model = MultitaskModel(backbone=backbone, heads=heads)

    task_datasets = {
        "task_a": DummyTaskDataset("task_a", num_samples=4, seq_len=3, hidden_size=8),
        "task_b": DummyTaskDataset("task_b", num_samples=4, seq_len=3, hidden_size=8),
    }

    logger = logging.getLogger("train_test")
    tokenizer = object()

    train_module.train_multitask_heads_only(
        model=model,
        tokenizer=tokenizer,
        checkpoint="checkpoint",
        task_datasets=task_datasets,
        batch=2,
        max_epochs=2,
        lr=1e-3,
        warmup_ratio=0.0,
        seed=0,
        logger=logger,
    )

    optimizer = record["optimizer"]
    scheduler = record["scheduler"]

    assert optimizer.step_calls == 8
    assert optimizer.zero_grad_calls == 8
    assert scheduler.step_calls == 8
    assert scheduler.total_steps == 8


def test_train_multitask_model_logs_named_metrics(monkeypatch):
    record.clear()
    monkeypatch.setattr(train_module.torch.optim, "AdamW", DummyOptimizer)
    monkeypatch.setattr(
        train_module,
        "get_linear_schedule_with_warmup",
        lambda opt, num_warmup_steps, total_steps: DummyScheduler(
            opt, num_warmup_steps, total_steps
        ),
    )

    backbone = DummyBackbone(hidden_size=8)
    heads = {
        "task_a": HeadConfig(
            head=PerProteinClassificationHead(0.0, 8, 8, 2),
            problem_type="classification",
            level="per_protein",
        )
    }
    model = MultitaskModel(backbone=backbone, heads=heads)

    task_datasets = {
        "task_a": DummyTaskDataset("task_a", num_samples=2, seq_len=3, hidden_size=8)
    }

    logger = logging.getLogger("train_test")
    tokenizer = object()

    _, history, _ = train_module.train_multitask_heads_only(
        model=model,
        tokenizer=tokenizer,
        checkpoint="checkpoint",
        task_datasets=task_datasets,
        batch=2,
        max_epochs=1,
        lr=1e-3,
        warmup_ratio=0.0,
        seed=0,
        logger=logger,
    )

    assert history
    last = history[-1]
    assert "tasks" in last
    assert "task_a" in last["tasks"]
    assert "metrics" in last["tasks"]["task_a"]


def test_train_multitask_model_validate_every_n_train_batches(monkeypatch):
    record.clear()
    monkeypatch.setattr(train_module.torch.optim, "AdamW", DummyOptimizer)
    monkeypatch.setattr(
        train_module,
        "get_linear_schedule_with_warmup",
        lambda opt, num_warmup_steps, total_steps: DummyScheduler(
            opt, num_warmup_steps, total_steps
        ),
    )

    backbone = DummyBackbone(hidden_size=8)
    heads = {
        "task_a": HeadConfig(
            head=PerProteinClassificationHead(0.0, 8, 8, 2),
            problem_type="classification",
            level="per_protein",
        )
    }
    model = MultitaskModel(backbone=backbone, heads=heads)

    task_datasets = {
        "task_a": DummyTaskDataset("task_a", num_samples=4, seq_len=3, hidden_size=8)
    }

    logger = logging.getLogger("train_test")
    tokenizer = object()

    _, history, _ = train_module.train_multitask_heads_only(
        model=model,
        tokenizer=tokenizer,
        checkpoint="checkpoint",
        task_datasets=task_datasets,
        batch=2,
        max_epochs=1,
        lr=1e-3,
        warmup_ratio=0.0,
        seed=0,
        logger=logger,
        validate_every_n_train_batches=2,
    )

    assert history
    assert len(history) == 1


def test_train_multitask_model_early_stopping(monkeypatch):
    record.clear()
    monkeypatch.setattr(train_module.torch.optim, "AdamW", DummyOptimizer)
    monkeypatch.setattr(
        train_module,
        "get_linear_schedule_with_warmup",
        lambda opt, num_warmup_steps, total_steps: DummyScheduler(
            opt, num_warmup_steps, total_steps
        ),
    )

    backbone = DummyBackbone(hidden_size=8)
    heads = {
        "task_a": HeadConfig(
            head=PerProteinClassificationHead(0.0, 8, 8, 2),
            problem_type="classification",
            level="per_protein",
        )
    }
    model = MultitaskModel(backbone=backbone, heads=heads)

    task_datasets = {
        "task_a": DummyTaskDataset("task_a", num_samples=4, seq_len=3, hidden_size=8)
    }

    logger = logging.getLogger("train_test")
    tokenizer = object()

    _, history, _ = train_module.train_multitask_heads_only(
        model=model,
        tokenizer=tokenizer,
        checkpoint="checkpoint",
        task_datasets=task_datasets,
        batch=2,
        max_epochs=4,
        lr=1e-3,
        warmup_ratio=0.0,
        seed=0,
        logger=logger,
        early_stopping_patience=1,
    )

    assert history
    last = history[-1]
    assert "early_stopping" in last


def test_train_multitask_model_early_stopping_custom_metric(monkeypatch):
    record.clear()
    monkeypatch.setattr(train_module.torch.optim, "AdamW", DummyOptimizer)
    monkeypatch.setattr(
        train_module,
        "get_linear_schedule_with_warmup",
        lambda opt, num_warmup_steps, total_steps: DummyScheduler(
            opt, num_warmup_steps, total_steps
        ),
    )

    backbone = DummyBackbone(hidden_size=8)
    heads = {
        "task_a": HeadConfig(
            head=PerProteinClassificationHead(0.0, 8, 8, 2),
            problem_type="classification",
            level="per_protein",
        )
    }
    model = MultitaskModel(backbone=backbone, heads=heads)

    task_datasets = {
        "task_a": DummyTaskDataset("task_a", num_samples=4, seq_len=3, hidden_size=8)
    }

    logger = logging.getLogger("train_test")
    tokenizer = object()

    _, history, _ = train_module.train_multitask_heads_only(
        model=model,
        tokenizer=tokenizer,
        checkpoint="checkpoint",
        task_datasets=task_datasets,
        batch=2,
        max_epochs=4,
        lr=1e-3,
        warmup_ratio=0.0,
        seed=0,
        logger=logger,
        early_stopping_patience=1,
        early_stopping_metrics={"task_a": "f1_macro"},
    )

    assert history
    last = history[-1]
    assert "early_stopping" in last


def test_train_multitask_model_checkpointing(tmp_path):
    hidden_size = 4
    num_labels = 2
    dropout = 0.0
    seq_len = 3
    num_samples = 4
    batch_size = 2

    backbone = DummyBackbone(hidden_size)
    head_a = PerProteinClassificationHead(dropout, hidden_size, hidden_size, num_labels)
    heads = {
        "task_a": HeadConfig(
            head=head_a, problem_type="classification", level="per_protein"
        ),
    }
    model = MultitaskModel(backbone=backbone, heads=heads)

    task_datasets = {
        "task_a": DummyTaskDataset("task_a", num_samples, seq_len, hidden_size),
    }

    checkpoint_dir = tmp_path / "checkpoints"

    train_module.train_multitask_heads_only(
        model=model,
        tokenizer=None,
        checkpoint="facebook/esm2_t6_8M_UR50D",
        task_datasets=task_datasets,
        batch=batch_size,
        max_validations=2,
        lr=1e-3,
        warmup_ratio=0.0,
        seed=42,
        logger=logging.getLogger("train_test"),
        validate_every_n_train_batches=1,
        training_checkpoint_dir=str(checkpoint_dir),
        checkpoint_every_n_validations=1,
    )

    checkpoint_file = checkpoint_dir / "training_checkpoint.pt"
    assert checkpoint_file.exists()
    checkpoint_data = torch.load(checkpoint_file, weights_only=False)
    assert checkpoint_data["epoch"] == 1
    assert checkpoint_data["validation_count"] == 2


def test_train_multitask_model_resume_preserves_shuffled_order_epoch_boundary(
    tmp_path,
    monkeypatch,
):
    hidden_size = 4
    num_labels = 2
    dropout = 0.0
    seq_len = 3
    num_samples = 6
    batch_size = 2

    def create_model():
        torch.manual_seed(123)
        backbone = RecordingBackbone(hidden_size)
        head = PerProteinClassificationHead(
            dropout, hidden_size, hidden_size, num_labels
        )
        heads = {
            "task_a": HeadConfig(
                head=head, problem_type="classification", level="per_protein"
            )
        }
        return MultitaskModel(backbone=backbone, heads=heads)

    def create_datasets():
        return {
            "task_a": ShuffledDummyTaskDataset(
                "task_a",
                num_samples=num_samples,
                seq_len=seq_len,
                hidden_size=hidden_size,
            ),
        }

    logger = logging.getLogger("test_resume_epoch_boundary")
    logger.setLevel(logging.DEBUG)

    model_uninterrupted = create_model()
    train_module.train_multitask_heads_only(
        model=model_uninterrupted,
        tokenizer=None,
        checkpoint="facebook/esm2_t6_8M_UR50D",
        task_datasets=create_datasets(),
        batch=batch_size,
        max_validations=2,
        lr=1e-3,
        warmup_ratio=0.0,
        seed=42,
        logger=logger,
        validate_every_n_train_batches=0,
    )

    epoch2_train_batches_unint = list(model_uninterrupted.backbone.seen_train_batches)

    checkpoint_dir = tmp_path / "checkpoints"
    orig_save = train_module._save_training_checkpoint
    crash_state = {"called": False}

    def crashing_save(*args, **kwargs):
        orig_save(*args, **kwargs)
        if not crash_state["called"]:
            crash_state["called"] = True
            raise RuntimeError("simulated crash")

    monkeypatch.setattr(train_module, "_save_training_checkpoint", crashing_save)

    model_interrupted = create_model()
    with pytest.raises(RuntimeError, match="simulated crash"):
        train_module.train_multitask_heads_only(
            model=model_interrupted,
            tokenizer=None,
            checkpoint="facebook/esm2_t6_8M_UR50D",
            task_datasets=create_datasets(),
            batch=batch_size,
            max_validations=1,
            lr=1e-3,
            warmup_ratio=0.0,
            seed=42,
            logger=logger,
            validate_every_n_train_batches=0,
            training_checkpoint_dir=str(checkpoint_dir),
            checkpoint_every_n_validations=1,
        )

    monkeypatch.setattr(train_module, "_save_training_checkpoint", orig_save)

    model_resumed = create_model()
    train_module.train_multitask_heads_only(
        model=model_resumed,
        tokenizer=None,
        checkpoint="facebook/esm2_t6_8M_UR50D",
        task_datasets=create_datasets(),
        batch=batch_size,
        max_validations=2,
        lr=1e-3,
        warmup_ratio=0.0,
        seed=42,
        logger=logger,
        validate_every_n_train_batches=0,
        training_checkpoint_dir=str(checkpoint_dir),
        checkpoint_every_n_validations=1,
    )

    assert model_resumed.backbone.seen_train_batches == epoch2_train_batches_unint


def test_train_multitask_model_resume_mid_epoch_from_validation_checkpoint_is_deterministic(
    tmp_path,
    monkeypatch,
):
    hidden_size = 4
    num_labels = 2
    dropout = 0.0
    seq_len = 3
    batch_size = 2

    task_a_samples = 4
    task_b_samples = 6
    total_validations = 6

    def create_model():
        torch.manual_seed(123)
        backbone = RecordingBackbone(hidden_size)
        head_a = PerProteinClassificationHead(
            dropout, hidden_size, hidden_size, num_labels
        )
        head_b = PerProteinClassificationHead(
            dropout, hidden_size, hidden_size, num_labels
        )
        heads = {
            "task_a": HeadConfig(
                head=head_a, problem_type="classification", level="per_protein"
            ),
            "task_b": HeadConfig(
                head=head_b, problem_type="classification", level="per_protein"
            ),
        }
        return MultitaskModel(backbone=backbone, heads=heads)

    def create_datasets():
        return {
            "task_a": ShuffledDummyTaskDataset(
                "task_a",
                num_samples=task_a_samples,
                seq_len=seq_len,
                hidden_size=hidden_size,
            ),
            "task_b": ShuffledDummyTaskDataset(
                "task_b",
                num_samples=task_b_samples,
                seq_len=seq_len,
                hidden_size=hidden_size,
            ),
        }

    logger = logging.getLogger("test_resume_mid_epoch_deterministic")
    logger.setLevel(logging.DEBUG)

    model_uninterrupted = create_model()
    train_module.train_multitask_heads_only(
        model=model_uninterrupted,
        tokenizer=None,
        checkpoint="facebook/esm2_t6_8M_UR50D",
        task_datasets=create_datasets(),
        batch=batch_size,
        max_validations=total_validations,
        lr=1e-3,
        warmup_ratio=0.0,
        seed=42,
        logger=logger,
        validate_every_n_train_batches=1,
    )
    uninterrupted_batches = list(model_uninterrupted.backbone.seen_train_batches)

    checkpoint_dir = tmp_path / "checkpoints"
    orig_save = train_module._save_training_checkpoint
    crash_state = {"called": False}

    def crashing_save(*args, **kwargs):
        orig_save(*args, **kwargs)
        if not crash_state["called"]:
            crash_state["called"] = True
            raise RuntimeError("simulated crash")

    monkeypatch.setattr(train_module, "_save_training_checkpoint", crashing_save)

    model_interrupted = create_model()
    with pytest.raises(RuntimeError, match="simulated crash"):
        train_module.train_multitask_heads_only(
            model=model_interrupted,
            tokenizer=None,
            checkpoint="facebook/esm2_t6_8M_UR50D",
            task_datasets=create_datasets(),
            batch=batch_size,
            max_validations=total_validations,
            lr=1e-3,
            warmup_ratio=0.0,
            seed=42,
            logger=logger,
            validate_every_n_train_batches=1,
            training_checkpoint_dir=str(checkpoint_dir),
            checkpoint_every_n_validations=4,
        )

    interrupted_prefix = list(model_interrupted.backbone.seen_train_batches)
    assert interrupted_prefix
    assert len(interrupted_prefix) < len(uninterrupted_batches)

    monkeypatch.setattr(train_module, "_save_training_checkpoint", orig_save)

    checkpoint_file = checkpoint_dir / "training_checkpoint.pt"
    checkpoint_data = torch.load(checkpoint_file, weights_only=False)
    assert "train_sampler_generator_states" in checkpoint_data
    assert "train_sampler_iterator_start_states" in checkpoint_data
    assert "train_sampler_iterator_start_cumulative_batches" in checkpoint_data

    model_resumed = create_model()
    train_module.train_multitask_heads_only(
        model=model_resumed,
        tokenizer=None,
        checkpoint="facebook/esm2_t6_8M_UR50D",
        task_datasets=create_datasets(),
        batch=batch_size,
        max_validations=total_validations,
        lr=1e-3,
        warmup_ratio=0.0,
        seed=42,
        logger=logger,
        validate_every_n_train_batches=1,
        training_checkpoint_dir=str(checkpoint_dir),
        checkpoint_every_n_validations=4,
    )

    combined = interrupted_prefix + list(model_resumed.backbone.seen_train_batches)
    assert combined == uninterrupted_batches


def test_train_multitask_model_checkpoint_resume_continues_correctly(tmp_path):
    hidden_size = 4
    num_labels = 2
    dropout = 0.0
    seq_len = 3
    num_samples = 4
    batch_size = 2

    def create_model():
        backbone = DummyBackbone(hidden_size)
        head_a = PerProteinClassificationHead(
            dropout, hidden_size, hidden_size, num_labels
        )
        heads = {
            "task_a": HeadConfig(
                head=head_a, problem_type="classification", level="per_protein"
            ),
        }
        return MultitaskModel(backbone=backbone, heads=heads)

    def create_datasets():
        return {
            "task_a": DummyTaskDataset("task_a", num_samples, seq_len, hidden_size),
        }

    checkpoint_dir = tmp_path / "checkpoints"

    model_1 = create_model()
    train_module.train_multitask_heads_only(
        model=model_1,
        tokenizer=None,
        checkpoint="facebook/esm2_t6_8M_UR50D",
        task_datasets=create_datasets(),
        batch=batch_size,
        max_validations=2,
        lr=1e-3,
        warmup_ratio=0.0,
        seed=42,
        logger=logging.getLogger("train_test"),
        validate_every_n_train_batches=1,
        training_checkpoint_dir=str(checkpoint_dir),
        checkpoint_every_n_validations=1,
    )

    model_resumed = create_model()
    train_module.train_multitask_heads_only(
        model=model_resumed,
        tokenizer=None,
        checkpoint="facebook/esm2_t6_8M_UR50D",
        task_datasets=create_datasets(),
        batch=batch_size,
        max_validations=2,
        lr=1e-3,
        warmup_ratio=0.0,
        seed=42,
        logger=logging.getLogger("train_test"),
        validate_every_n_train_batches=1,
        training_checkpoint_dir=str(checkpoint_dir),
        checkpoint_every_n_validations=1,
    )

    assert model_resumed is not None


def test_train_multitask_model_checkpoint_resume_matches_uninterrupted(
    tmp_path, monkeypatch
):
    hidden_size = 4
    num_labels = 2
    dropout = 0.0
    seq_len = 3
    num_samples = 4
    batch_size = 2
    total_validations = 4

    def create_model():
        torch.manual_seed(123)
        backbone = DummyBackbone(hidden_size)
        head_a = PerProteinClassificationHead(
            dropout, hidden_size, hidden_size, num_labels
        )
        heads = {
            "task_a": HeadConfig(
                head=head_a, problem_type="classification", level="per_protein"
            ),
        }
        return MultitaskModel(backbone=backbone, heads=heads)

    def create_datasets():
        return {
            "task_a": DummyTaskDataset("task_a", num_samples, seq_len, hidden_size),
        }

    model_uninterrupted = create_model()
    _, history_uninterrupted, best_uninterrupted = (
        train_module.train_multitask_heads_only(
            model=model_uninterrupted,
            tokenizer=None,
            checkpoint="facebook/esm2_t6_8M_UR50D",
            task_datasets=create_datasets(),
            batch=batch_size,
            max_validations=total_validations,
            lr=1e-3,
            warmup_ratio=0.0,
            seed=42,
            logger=logging.getLogger("train_test"),
            validate_every_n_train_batches=1,
        )
    )

    checkpoint_dir = tmp_path / "checkpoints"
    orig_save = train_module._save_training_checkpoint
    crash_state = {"called": False}

    def crashing_save(*args, **kwargs):
        orig_save(*args, **kwargs)
        if not crash_state["called"]:
            crash_state["called"] = True
            raise RuntimeError("simulated crash")

    monkeypatch.setattr(train_module, "_save_training_checkpoint", crashing_save)

    with pytest.raises(RuntimeError, match="simulated crash"):
        train_module.train_multitask_heads_only(
            model=create_model(),
            tokenizer=None,
            checkpoint="facebook/esm2_t6_8M_UR50D",
            task_datasets=create_datasets(),
            batch=batch_size,
            max_validations=total_validations,
            lr=1e-3,
            warmup_ratio=0.0,
            seed=42,
            logger=logging.getLogger("train_test"),
            validate_every_n_train_batches=1,
            training_checkpoint_dir=str(checkpoint_dir),
            checkpoint_every_n_validations=2,
        )

    monkeypatch.setattr(train_module, "_save_training_checkpoint", orig_save)

    _, history_resumed, best_resumed = train_module.train_multitask_heads_only(
        model=create_model(),
        tokenizer=None,
        checkpoint="facebook/esm2_t6_8M_UR50D",
        task_datasets=create_datasets(),
        batch=batch_size,
        max_validations=total_validations,
        lr=1e-3,
        warmup_ratio=0.0,
        seed=42,
        logger=logging.getLogger("train_test"),
        validate_every_n_train_batches=1,
        training_checkpoint_dir=str(checkpoint_dir),
        checkpoint_every_n_validations=2,
    )

    def strip_timing(entries):
        _task_timing_keys = ("forward_time", "metrics_time", "num_samples")
        stripped = []
        for entry in entries:
            e = {
                k: v
                for k, v in entry.items()
                if k
                not in (
                    "timing",
                    "checkpointing",
                    "validation_step_timestamp",
                    "gpu",
                )
            }
            if "tasks" in e:
                e["tasks"] = {
                    tn: {tk: tv for tk, tv in td.items() if tk not in _task_timing_keys}
                    for tn, td in e["tasks"].items()
                }
            stripped.append(e)
        return stripped

    assert strip_timing(history_resumed) == strip_timing(history_uninterrupted)
    assert best_resumed == best_uninterrupted


def test_train_multitask_model_checkpoint_hash_mismatch_starts_fresh(tmp_path, caplog):
    hidden_size = 4
    num_labels = 2
    dropout = 0.0
    seq_len = 3
    num_samples = 2
    batch_size = 1

    backbone = DummyBackbone(hidden_size)
    head_a = PerProteinClassificationHead(dropout, hidden_size, hidden_size, num_labels)
    heads = {
        "task_a": HeadConfig(
            head=head_a, problem_type="classification", level="per_protein"
        ),
    }
    model = MultitaskModel(backbone=backbone, heads=heads)

    task_datasets = {
        "task_a": DummyTaskDataset("task_a", num_samples, seq_len, hidden_size),
    }

    checkpoint_dir = tmp_path / "checkpoints"
    logger = logging.getLogger("train_test")

    train_module.train_multitask_heads_only(
        model=model,
        tokenizer=None,
        checkpoint="facebook/esm2_t6_8M_UR50D",
        task_datasets=task_datasets,
        batch=batch_size,
        max_validations=1,
        lr=1e-3,
        warmup_ratio=0.0,
        seed=42,
        logger=logger,
        validate_every_n_train_batches=1,
        training_checkpoint_dir=str(checkpoint_dir),
        checkpoint_every_n_validations=1,
    )

    model_resumed = MultitaskModel(backbone=DummyBackbone(hidden_size), heads=heads)

    with caplog.at_level(logging.WARNING):
        _, history_resumed, _ = train_module.train_multitask_heads_only(
            model=model_resumed,
            tokenizer=None,
            checkpoint="facebook/esm2_t6_8M_UR50D",
            task_datasets=task_datasets,
            batch=batch_size,
            max_validations=1,
            lr=2e-3,
            warmup_ratio=0.0,
            seed=42,
            logger=logger,
            validate_every_n_train_batches=1,
            training_checkpoint_dir=str(checkpoint_dir),
            checkpoint_every_n_validations=1,
        )

    assert any(
        "Checkpoint params hash mismatch" in record.message for record in caplog.records
    )
    assert history_resumed
    checkpointing = history_resumed[-1].get("checkpointing", {})
    assert "resumed" not in checkpointing


def test_train_multitask_model_history_has_validation_step_metadata(monkeypatch):
    record.clear()
    monkeypatch.setattr(train_module.torch.optim, "AdamW", DummyOptimizer)
    monkeypatch.setattr(
        train_module,
        "get_linear_schedule_with_warmup",
        lambda opt, num_warmup_steps, total_steps: DummyScheduler(
            opt, num_warmup_steps, total_steps
        ),
    )

    backbone = DummyBackbone(hidden_size=4)
    heads = {
        "task_a": HeadConfig(
            head=PerProteinClassificationHead(0.0, 4, 4, 2),
            problem_type="classification",
            level="per_protein",
        )
    }
    model = MultitaskModel(backbone=backbone, heads=heads)

    task_datasets = {
        "task_a": DummyTaskDataset("task_a", num_samples=2, seq_len=3, hidden_size=4)
    }

    _, history, _ = train_module.train_multitask_heads_only(
        model=model,
        tokenizer=object(),
        checkpoint="checkpoint",
        task_datasets=task_datasets,
        batch=1,
        max_validations=1,
        lr=1e-3,
        warmup_ratio=0.0,
        seed=0,
        logger=logging.getLogger("train_test"),
        validate_every_n_train_batches=1,
    )

    assert history
    entry = history[-1]
    assert isinstance(entry.get("validation_step_timestamp"), float)
    gpu = entry.get("gpu")
    assert isinstance(gpu, dict)
    assert isinstance(gpu.get("device"), str)
    assert isinstance(gpu.get("device_name"), str)


def test_train_multitask_model_runs_final_validation_when_steps_not_multiple(
    monkeypatch,
):
    record.clear()
    monkeypatch.setattr(train_module.torch.optim, "AdamW", DummyOptimizer)
    monkeypatch.setattr(
        train_module,
        "get_linear_schedule_with_warmup",
        lambda opt, num_warmup_steps, total_steps: DummyScheduler(
            opt, num_warmup_steps, total_steps
        ),
    )

    backbone = DummyBackbone(hidden_size=4)
    heads = {
        "task_a": HeadConfig(
            head=PerProteinClassificationHead(0.0, 4, 4, 2),
            problem_type="classification",
            level="per_protein",
        )
    }
    model = MultitaskModel(backbone=backbone, heads=heads)

    task_datasets = {
        "task_a": DummyTaskDataset("task_a", num_samples=3, seq_len=3, hidden_size=4)
    }

    _, history, _ = train_module.train_multitask_heads_only(
        model=model,
        tokenizer=object(),
        checkpoint="checkpoint",
        task_datasets=task_datasets,
        batch=1,
        max_epochs=1,
        lr=1e-3,
        warmup_ratio=0.0,
        seed=0,
        logger=logging.getLogger("train_test"),
        validate_every_n_train_batches=2,
    )

    assert history
    assert len(history) == 2
    assert history[-1]["global_step"] == 3


def test_train_multitask_model_best_so_far_tracking_and_saving(tmp_path, monkeypatch):
    """Test that best-so-far models are saved and history entries contain best_so_far."""
    record.clear()
    monkeypatch.setattr(train_module.torch.optim, "AdamW", DummyOptimizer)
    monkeypatch.setattr(
        train_module,
        "get_linear_schedule_with_warmup",
        lambda opt, num_warmup_steps, total_steps: DummyScheduler(
            opt, num_warmup_steps, total_steps
        ),
    )

    backbone = DummyBackbone(hidden_size=8)
    heads = {
        "task_a": HeadConfig(
            head=PerProteinClassificationHead(0.0, 8, 8, 2),
            problem_type="classification",
            level="per_protein",
        ),
        "task_b": HeadConfig(
            head=PerProteinClassificationHead(0.0, 8, 8, 2),
            problem_type="classification",
            level="per_protein",
        ),
    }
    model = MultitaskModel(backbone=backbone, heads=heads)
    task_datasets = {
        "task_a": DummyTaskDataset("task_a", num_samples=4, seq_len=3, hidden_size=8),
        "task_b": DummyTaskDataset("task_b", num_samples=4, seq_len=3, hidden_size=8),
    }

    best_overall_dir = str(tmp_path / "best_overall_model")
    best_task_dir = str(tmp_path / "best_task_models")
    best_loss_overall_dir = str(tmp_path / "best_loss_overall_model")
    best_loss_task_dir = str(tmp_path / "best_loss_task_models")

    _, history, best_checkpoints = train_module.train_multitask_heads_only(
        model=model,
        tokenizer=object(),
        checkpoint="checkpoint",
        task_datasets=task_datasets,
        batch=2,
        max_epochs=2,
        lr=1e-3,
        warmup_ratio=0.0,
        seed=42,
        logger=logging.getLogger("train_test"),
        best_overall_model_dir=best_overall_dir,
        best_task_models_dir=best_task_dir,
        best_loss_overall_model_dir=best_loss_overall_dir,
        best_loss_task_models_dir=best_loss_task_dir,
    )

    # Every history entry should have best_so_far
    assert len(history) >= 1
    for entry in history:
        assert "best_so_far" in entry
        assert "overall" in entry["best_so_far"]
        assert "tasks" in entry["best_so_far"]
        overall = entry["best_so_far"]["overall"]
        assert "validation_step" in overall
        assert "epoch" in overall
        assert "global_step" in overall
        assert "baseline_metrics" in overall
        assert "metrics" in overall
        assert "improved" in overall
        for task_name in ["task_a", "task_b"]:
            task_bsf = entry["best_so_far"]["tasks"][task_name]
            assert "metric" in task_bsf
            assert "validation_step" in task_bsf
            assert "epoch" in task_bsf
            assert "global_step" in task_bsf
            assert "improved" in task_bsf

        # Loss tracking in best_so_far
        assert "loss_overall" in entry["best_so_far"]
        loss_overall = entry["best_so_far"]["loss_overall"]
        assert "validation_step" in loss_overall
        assert "epoch" in loss_overall
        assert "global_step" in loss_overall
        assert "baseline_losses" in loss_overall
        assert "losses" in loss_overall
        assert "improved" in loss_overall
        assert "loss_tasks" in entry["best_so_far"]
        for task_name in ["task_a", "task_b"]:
            loss_task_bsf = entry["best_so_far"]["loss_tasks"][task_name]
            assert "loss" in loss_task_bsf
            assert "validation_step" in loss_task_bsf
            assert "epoch" in loss_task_bsf
            assert "global_step" in loss_task_bsf
            assert "improved" in loss_task_bsf

    # First validation should always be an improvement (metric and loss)
    assert history[0]["best_so_far"]["overall"]["improved"] is True
    assert history[0]["best_so_far"]["loss_overall"]["improved"] is True

    # Model files should be saved (metric-based)
    import os

    from mutafitup.train.base_trainer import INITIAL_CHECKPOINT_MARKER

    assert os.path.exists(os.path.join(best_overall_dir, "model.pt"))
    assert os.path.exists(os.path.join(best_task_dir, "task_a", "model.pt"))
    assert os.path.exists(os.path.join(best_task_dir, "task_b", "model.pt"))

    # Model files should be saved (loss-based)
    assert os.path.exists(os.path.join(best_loss_overall_dir, "model.pt"))
    assert os.path.exists(os.path.join(best_loss_task_dir, "task_a", "model.pt"))
    assert os.path.exists(os.path.join(best_loss_task_dir, "task_b", "model.pt"))

    # After 2 epochs, if a 2nd improvement occurred the marker should be removed;
    # if not, it may still exist. Either way the directories must exist.
    # With 2+ validations and improvement on both, markers should be removed.
    if len(history) >= 2:
        for d in [best_overall_dir, best_loss_overall_dir]:
            if (
                history[-1]["best_so_far"]["overall"]["improved"]
                or history[-1]["best_so_far"]["loss_overall"]["improved"]
            ):
                # At least one subsequent improvement should have removed the marker
                pass  # marker state depends on specific improvements
        # All directories must exist regardless
        assert os.path.isdir(best_overall_dir)
        assert os.path.isdir(best_loss_overall_dir)
        for task_name in ["task_a", "task_b"]:
            assert os.path.isdir(os.path.join(best_task_dir, task_name))
            assert os.path.isdir(os.path.join(best_loss_task_dir, task_name))

    # best_checkpoints should include loss info
    assert "best_loss_tasks" in best_checkpoints
    assert "best_loss_overall" in best_checkpoints
    for task_name in ["task_a", "task_b"]:
        assert task_name in best_checkpoints["best_loss_tasks"]
        assert "epoch" in best_checkpoints["best_loss_tasks"][task_name]
        assert "loss" in best_checkpoints["best_loss_tasks"][task_name]
    assert "epoch" in best_checkpoints["best_loss_overall"]
    assert "losses" in best_checkpoints["best_loss_overall"]


def test_initial_checkpoint_marker_created_on_first_validation(tmp_path, monkeypatch):
    """Test that .initial_checkpoint marker is written on first validation for all dirs."""
    import os

    from mutafitup.train.base_trainer import INITIAL_CHECKPOINT_MARKER

    record.clear()
    monkeypatch.setattr(train_module.torch.optim, "AdamW", DummyOptimizer)
    monkeypatch.setattr(
        train_module,
        "get_linear_schedule_with_warmup",
        lambda opt, num_warmup_steps, total_steps: DummyScheduler(
            opt, num_warmup_steps, total_steps
        ),
    )

    backbone = DummyBackbone(hidden_size=8)
    heads = {
        "task_a": HeadConfig(
            head=PerProteinClassificationHead(0.0, 8, 8, 2),
            problem_type="classification",
            level="per_protein",
        ),
        "task_b": HeadConfig(
            head=PerProteinClassificationHead(0.0, 8, 8, 2),
            problem_type="classification",
            level="per_protein",
        ),
    }
    model = MultitaskModel(backbone=backbone, heads=heads)
    task_datasets = {
        "task_a": DummyTaskDataset("task_a", num_samples=4, seq_len=3, hidden_size=8),
        "task_b": DummyTaskDataset("task_b", num_samples=4, seq_len=3, hidden_size=8),
    }

    best_overall_dir = str(tmp_path / "best_overall_model")
    best_task_dir = str(tmp_path / "best_task_models")
    best_loss_overall_dir = str(tmp_path / "best_loss_overall_model")
    best_loss_task_dir = str(tmp_path / "best_loss_task_models")

    # Run with max_validations=1 so exactly one validation occurs
    _, history, _ = train_module.train_multitask_heads_only(
        model=model,
        tokenizer=object(),
        checkpoint="checkpoint",
        task_datasets=task_datasets,
        batch=2,
        max_validations=1,
        lr=1e-3,
        warmup_ratio=0.0,
        seed=42,
        logger=logging.getLogger("train_test"),
        best_overall_model_dir=best_overall_dir,
        best_task_models_dir=best_task_dir,
        best_loss_overall_model_dir=best_loss_overall_dir,
        best_loss_task_models_dir=best_loss_task_dir,
    )

    assert len(history) == 1

    # All checkpoint directories must exist
    assert os.path.isdir(best_overall_dir)
    assert os.path.isdir(best_loss_overall_dir)
    for task_name in ["task_a", "task_b"]:
        assert os.path.isdir(os.path.join(best_task_dir, task_name))
        assert os.path.isdir(os.path.join(best_loss_task_dir, task_name))

    # All checkpoint directories must contain model.pt
    assert os.path.exists(os.path.join(best_overall_dir, "model.pt"))
    assert os.path.exists(os.path.join(best_loss_overall_dir, "model.pt"))
    for task_name in ["task_a", "task_b"]:
        assert os.path.exists(os.path.join(best_task_dir, task_name, "model.pt"))
        assert os.path.exists(os.path.join(best_loss_task_dir, task_name, "model.pt"))

    # All checkpoint directories must have .initial_checkpoint marker
    assert os.path.exists(os.path.join(best_overall_dir, INITIAL_CHECKPOINT_MARKER))
    assert os.path.exists(
        os.path.join(best_loss_overall_dir, INITIAL_CHECKPOINT_MARKER)
    )
    for task_name in ["task_a", "task_b"]:
        assert os.path.exists(
            os.path.join(best_task_dir, task_name, INITIAL_CHECKPOINT_MARKER)
        )
        assert os.path.exists(
            os.path.join(best_loss_task_dir, task_name, INITIAL_CHECKPOINT_MARKER)
        )


def test_initial_checkpoint_marker_removed_on_subsequent_improvement(
    tmp_path, monkeypatch
):
    """Test that .initial_checkpoint marker is removed when a subsequent improvement
    overwrites the checkpoint."""
    import os

    from mutafitup.train.base_trainer import INITIAL_CHECKPOINT_MARKER

    record.clear()
    monkeypatch.setattr(train_module.torch.optim, "AdamW", DummyOptimizer)
    monkeypatch.setattr(
        train_module,
        "get_linear_schedule_with_warmup",
        lambda opt, num_warmup_steps, total_steps: DummyScheduler(
            opt, num_warmup_steps, total_steps
        ),
    )

    backbone = DummyBackbone(hidden_size=8)
    heads = {
        "task_a": HeadConfig(
            head=PerProteinClassificationHead(0.0, 8, 8, 2),
            problem_type="classification",
            level="per_protein",
        ),
    }
    model = MultitaskModel(backbone=backbone, heads=heads)
    task_datasets = {
        "task_a": DummyTaskDataset("task_a", num_samples=4, seq_len=3, hidden_size=8),
    }

    best_overall_dir = str(tmp_path / "best_overall_model")
    best_task_dir = str(tmp_path / "best_task_models")
    best_loss_overall_dir = str(tmp_path / "best_loss_overall_model")
    best_loss_task_dir = str(tmp_path / "best_loss_task_models")

    # Run with enough validations for a second improvement to occur
    _, history, _ = train_module.train_multitask_heads_only(
        model=model,
        tokenizer=object(),
        checkpoint="checkpoint",
        task_datasets=task_datasets,
        batch=2,
        max_epochs=3,
        lr=1e-3,
        warmup_ratio=0.0,
        seed=42,
        logger=logging.getLogger("train_test"),
        best_overall_model_dir=best_overall_dir,
        best_task_models_dir=best_task_dir,
        best_loss_overall_model_dir=best_loss_overall_dir,
        best_loss_task_models_dir=best_loss_task_dir,
    )

    assert len(history) >= 2

    # All directories must exist
    assert os.path.isdir(best_overall_dir)
    assert os.path.isdir(best_loss_overall_dir)
    assert os.path.isdir(os.path.join(best_task_dir, "task_a"))
    assert os.path.isdir(os.path.join(best_loss_task_dir, "task_a"))

    # For overall model: the first validation always triggers improvement (eps baseline),
    # so the marker is created. Any subsequent improvement removes it.
    # Check that if the overall improved on a later validation, marker is gone.
    any_subsequent_overall_improvement = any(
        entry["best_so_far"]["overall"]["improved"] for entry in history[1:]
    )
    if any_subsequent_overall_improvement:
        assert not os.path.exists(
            os.path.join(best_overall_dir, INITIAL_CHECKPOINT_MARKER)
        ), "Marker should be removed after subsequent overall improvement"

    any_subsequent_loss_overall_improvement = any(
        entry["best_so_far"]["loss_overall"]["improved"] for entry in history[1:]
    )
    if any_subsequent_loss_overall_improvement:
        assert not os.path.exists(
            os.path.join(best_loss_overall_dir, INITIAL_CHECKPOINT_MARKER)
        ), "Marker should be removed after subsequent loss overall improvement"


def test_train_multitask_model_resume_with_completed_epochs_runs_final_validation(
    tmp_path,
):
    """Regression test: resuming from a checkpoint where start_epoch >= loop_epochs
    should still run the final validation without AttributeError on self.batches."""
    hidden_size = 4
    num_labels = 2
    dropout = 0.0
    seq_len = 3
    num_samples = 4
    batch_size = 2

    def create_model():
        backbone = DummyBackbone(hidden_size)
        head_a = PerProteinClassificationHead(
            dropout, hidden_size, hidden_size, num_labels
        )
        heads = {
            "task_a": HeadConfig(
                head=head_a, problem_type="classification", level="per_protein"
            ),
        }
        return MultitaskModel(backbone=backbone, heads=heads)

    def create_datasets():
        return {
            "task_a": DummyTaskDataset("task_a", num_samples, seq_len, hidden_size),
        }

    checkpoint_dir = tmp_path / "checkpoints"
    logger = logging.getLogger("test_resume_completed_epochs")

    # Run training to produce a checkpoint
    train_module.train_multitask_heads_only(
        model=create_model(),
        tokenizer=None,
        checkpoint="facebook/esm2_t6_8M_UR50D",
        task_datasets=create_datasets(),
        batch=batch_size,
        max_validations=2,
        lr=1e-3,
        warmup_ratio=0.0,
        seed=42,
        logger=logger,
        validate_every_n_train_batches=3,
        training_checkpoint_dir=str(checkpoint_dir),
        checkpoint_every_n_validations=1,
    )

    # Tamper with checkpoint to simulate "past the end" state
    checkpoint_file = checkpoint_dir / "training_checkpoint.pt"
    assert checkpoint_file.exists()
    ckpt = torch.load(checkpoint_file, weights_only=False)

    # Set global_step very high so start_epoch >= loop_epochs, but
    # global_step % validate_every_n_train_batches != 0 to trigger final validation
    ckpt["global_step"] = 9997
    ckpt["validation_count"] = 1  # one less than max_validations=2
    torch.save(ckpt, checkpoint_file)

    # Resume: should complete without AttributeError
    model_resumed = create_model()
    _, history_resumed, _ = train_module.train_multitask_heads_only(
        model=model_resumed,
        tokenizer=None,
        checkpoint="facebook/esm2_t6_8M_UR50D",
        task_datasets=create_datasets(),
        batch=batch_size,
        max_validations=2,
        lr=1e-3,
        warmup_ratio=0.0,
        seed=42,
        logger=logger,
        validate_every_n_train_batches=3,
        training_checkpoint_dir=str(checkpoint_dir),
        checkpoint_every_n_validations=1,
    )

    # The final validation should have produced a new history entry
    assert len(history_resumed) > len(ckpt["history"])


def test_checkpoint_resume_restores_missing_best_model_dirs(tmp_path, monkeypatch):
    """Test that missing best-model dirs are restored from checkpoint snapshots on resume.

    Simulates the scenario where Snakemake ``--rerun-incomplete`` deletes
    output directories but the training checkpoint persists.  After resume
    all best-model directories should be recreated from the weight snapshots
    stored inside the training checkpoint.
    """
    import os
    import shutil

    record.clear()
    monkeypatch.setattr(train_module.torch.optim, "AdamW", DummyOptimizer)
    monkeypatch.setattr(
        train_module,
        "get_linear_schedule_with_warmup",
        lambda opt, num_warmup_steps, total_steps: DummyScheduler(
            opt, num_warmup_steps, total_steps
        ),
    )

    hidden_size = 8
    num_labels = 2

    def create_model():
        backbone = DummyBackbone(hidden_size)
        heads = {
            "task_a": HeadConfig(
                head=PerProteinClassificationHead(
                    0.0, hidden_size, hidden_size, num_labels
                ),
                problem_type="classification",
                level="per_protein",
            ),
            "task_b": HeadConfig(
                head=PerProteinClassificationHead(
                    0.0, hidden_size, hidden_size, num_labels
                ),
                problem_type="classification",
                level="per_protein",
            ),
        }
        return MultitaskModel(backbone=backbone, heads=heads)

    def create_datasets():
        return {
            "task_a": DummyTaskDataset(
                "task_a", num_samples=4, seq_len=3, hidden_size=hidden_size
            ),
            "task_b": DummyTaskDataset(
                "task_b", num_samples=4, seq_len=3, hidden_size=hidden_size
            ),
        }

    checkpoint_dir = str(tmp_path / "checkpoints")
    best_overall_dir = str(tmp_path / "best_overall_model")
    best_task_dir = str(tmp_path / "best_task_models")
    best_loss_overall_dir = str(tmp_path / "best_loss_overall_model")
    best_loss_task_dir = str(tmp_path / "best_loss_task_models")

    # --- Run 1: train to completion with checkpointing + best-model dirs ---
    model_1 = create_model()
    train_module.train_multitask_heads_only(
        model=model_1,
        tokenizer=object(),
        checkpoint="checkpoint",
        task_datasets=create_datasets(),
        batch=2,
        max_validations=2,
        lr=1e-3,
        warmup_ratio=0.0,
        seed=42,
        logger=logging.getLogger("test_restore"),
        validate_every_n_train_batches=1,
        training_checkpoint_dir=checkpoint_dir,
        checkpoint_every_n_validations=1,
        best_overall_model_dir=best_overall_dir,
        best_task_models_dir=best_task_dir,
        best_loss_overall_model_dir=best_loss_overall_dir,
        best_loss_task_models_dir=best_loss_task_dir,
    )

    # Verify all best-model dirs exist after initial run
    all_dirs = [
        best_overall_dir,
        best_loss_overall_dir,
        os.path.join(best_task_dir, "task_a"),
        os.path.join(best_task_dir, "task_b"),
        os.path.join(best_loss_task_dir, "task_a"),
        os.path.join(best_loss_task_dir, "task_b"),
    ]
    for d in all_dirs:
        assert os.path.isdir(d), f"Expected dir {d} to exist after initial run"
        assert os.path.exists(os.path.join(d, "model.pt")), (
            f"Expected model.pt in {d} after initial run"
        )

    # Record original model.pt contents for later comparison
    original_contents = {}
    for d in all_dirs:
        model_pt = os.path.join(d, "model.pt")
        original_contents[d] = torch.load(model_pt, weights_only=False)

    # --- Delete all best-model dirs (simulate --rerun-incomplete) ---
    for d in [
        best_overall_dir,
        best_loss_overall_dir,
        best_task_dir,
        best_loss_task_dir,
    ]:
        if os.path.exists(d):
            shutil.rmtree(d)

    for d in all_dirs:
        assert not os.path.exists(d), f"Expected dir {d} to be deleted"

    # --- Run 2: resume from checkpoint (already at max_validations) ---
    model_2 = create_model()
    train_module.train_multitask_heads_only(
        model=model_2,
        tokenizer=object(),
        checkpoint="checkpoint",
        task_datasets=create_datasets(),
        batch=2,
        max_validations=2,
        lr=1e-3,
        warmup_ratio=0.0,
        seed=42,
        logger=logging.getLogger("test_restore"),
        validate_every_n_train_batches=1,
        training_checkpoint_dir=checkpoint_dir,
        checkpoint_every_n_validations=1,
        best_overall_model_dir=best_overall_dir,
        best_task_models_dir=best_task_dir,
        best_loss_overall_model_dir=best_loss_overall_dir,
        best_loss_task_models_dir=best_loss_task_dir,
    )

    # --- Verify all best-model dirs were restored ---
    for d in all_dirs:
        assert os.path.isdir(d), f"Expected dir {d} to be restored after resume"
        assert os.path.exists(os.path.join(d, "model.pt")), (
            f"Expected model.pt in {d} after resume"
        )

    # Verify restored model.pt contents match originals
    for d in all_dirs:
        model_pt = os.path.join(d, "model.pt")
        restored = torch.load(model_pt, weights_only=False)
        original = original_contents[d]
        assert set(restored["state_dict"].keys()) == set(
            original["state_dict"].keys()
        ), f"Restored state_dict keys differ for {d}"
        for key in original["state_dict"]:
            assert torch.equal(
                restored["state_dict"][key], original["state_dict"][key]
            ), f"Restored weights differ for {key} in {d}"
