import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from torch.utils.data import DataLoader, Dataset

from mutafitup.datasets import BaseMultitaskDataset
from mutafitup.models.multitask_model import MultitaskBackbone, MultitaskForwardArgs


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


class RecordingBackbone(DummyBackbone):
    def __init__(self, hidden_size: int):
        super().__init__(hidden_size)
        self.seen_train_batches = []
        self.seen_eval_batches = []

    def forward(self, args: MultitaskForwardArgs) -> torch.Tensor:
        batch_ids = args.input_ids[:, 0, 0].detach().cpu().tolist()
        if self.training:
            self.seen_train_batches.append(batch_ids)
        else:
            self.seen_eval_batches.append(batch_ids)
        return super().forward(args)


class DummyBackboneWithLoraADefault(MultitaskBackbone):
    """Backbone that exposes a module path containing `lora_A.default` and uses it in forward."""

    def __init__(self, hidden_size: int, rank: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.rank = rank
        self._gradient_checkpointing = False

        self.lora_A = nn.ModuleDict(
            {"default": nn.Linear(hidden_size, rank, bias=False)}
        )
        self.out_proj = nn.Linear(rank, hidden_size, bias=False)

        with torch.no_grad():
            self.lora_A["default"].weight.zero_()
            if rank >= 2:
                self.lora_A["default"].weight[1, :].fill_(1.0)

    def _forward_block(self, x: torch.Tensor) -> torch.Tensor:
        """The core computation, separable for gradient checkpointing."""
        lora = self.lora_A["default"](x)
        return self.out_proj(lora)

    def forward(self, args: MultitaskForwardArgs) -> torch.Tensor:
        x = args.input_ids.float()
        if self._gradient_checkpointing and self.training:
            return torch_checkpoint(self._forward_block, x, use_reentrant=False)
        return self._forward_block(x)

    def enable_gradient_checkpointing(self) -> None:
        self._gradient_checkpointing = True

    def preprocess_sequences(self, sequences, checkpoint):
        return sequences

    def _inject_lora_config(self, lora_config=None) -> None:
        self._injected = lora_config


class RecordingBackboneWithLoraADefault(DummyBackboneWithLoraADefault):
    def __init__(self, hidden_size: int, rank: int = 2):
        super().__init__(hidden_size=hidden_size, rank=rank)
        self.seen_train_batches = []
        self.seen_eval_batches = []

    def forward(self, args: MultitaskForwardArgs) -> torch.Tensor:
        batch_ids = args.input_ids[:, 0, 0].detach().cpu().tolist()
        if self.training:
            self.seen_train_batches.append(batch_ids)
        else:
            self.seen_eval_batches.append(batch_ids)
        return super().forward(args)


class BatchDataset(Dataset):
    def __init__(self, num_samples: int, seq_len: int, hidden_size: int):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.hidden_size = hidden_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        attention_mask = torch.ones(self.seq_len, dtype=torch.long)
        input_ids = torch.ones(self.seq_len, self.hidden_size, dtype=torch.long)
        label = torch.zeros((), dtype=torch.long)
        return {
            "attention_mask": attention_mask,
            "input_ids": input_ids,
            "labels": label,
        }


class IndexedBatchDataset(Dataset):
    def __init__(self, num_samples: int, seq_len: int, hidden_size: int):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.hidden_size = hidden_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        attention_mask = torch.ones(self.seq_len, dtype=torch.long)
        input_ids = torch.full((self.seq_len, self.hidden_size), idx, dtype=torch.long)
        label = torch.tensor(idx % 2, dtype=torch.long)
        return {
            "attention_mask": attention_mask,
            "input_ids": input_ids,
            "labels": label,
        }


class OffsetIndexedBatchDataset(Dataset):
    def __init__(self, num_samples: int, seq_len: int, hidden_size: int, offset: int):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.offset = offset

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        attention_mask = torch.ones(self.seq_len, dtype=torch.long)
        value = idx + int(self.offset)
        input_ids = torch.full(
            (self.seq_len, self.hidden_size), value, dtype=torch.long
        )
        label = torch.tensor(0, dtype=torch.long)
        return {
            "attention_mask": attention_mask,
            "input_ids": input_ids,
            "labels": label,
        }


class DummyTaskDataset(BaseMultitaskDataset):
    def __init__(self, name: str, num_samples: int, seq_len: int, hidden_size: int):
        super().__init__(name)
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.hidden_size = hidden_size

    def get_dataloader(
        self,
        split: str,
        backbone: MultitaskBackbone,
        tokenizer,
        checkpoint: str,
        batch_size: int,
    ) -> DataLoader:
        if split == "train":
            num = self.num_samples
        else:
            num = self.num_samples
        dataset = BatchDataset(num, self.seq_len, self.hidden_size)
        return DataLoader(dataset, batch_size=batch_size)


class ShuffledDummyTaskDataset(DummyTaskDataset):
    def get_dataloader(
        self,
        split: str,
        backbone: MultitaskBackbone,
        tokenizer,
        checkpoint: str,
        batch_size: int,
    ) -> DataLoader:
        dataset = IndexedBatchDataset(self.num_samples, self.seq_len, self.hidden_size)
        shuffle = split == "train"
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class OffsetDummyTaskDataset(BaseMultitaskDataset):
    def __init__(
        self, name: str, num_samples: int, seq_len: int, hidden_size: int, offset: int
    ):
        super().__init__(name)
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.offset = offset

    def get_dataloader(
        self,
        split: str,
        backbone: MultitaskBackbone,
        tokenizer,
        checkpoint: str,
        batch_size: int,
    ) -> DataLoader:
        dataset = OffsetIndexedBatchDataset(
            self.num_samples, self.seq_len, self.hidden_size, offset=self.offset
        )
        return DataLoader(dataset, batch_size=batch_size)


record = {}


class DummyOptimizer:
    def __init__(self, params, lr: float):
        self.params = list(params)
        self.lr = lr
        self.step_calls = 0
        self.zero_grad_calls = 0
        record["optimizer"] = self

    def zero_grad(self) -> None:
        self.zero_grad_calls += 1
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def step(self) -> None:
        self.step_calls += 1

    def state_dict(self):
        return {
            "lr": self.lr,
            "step_calls": self.step_calls,
            "zero_grad_calls": self.zero_grad_calls,
        }

    def load_state_dict(self, state) -> None:
        if state is None:
            return
        self.lr = state.get("lr", self.lr)
        self.step_calls = state.get("step_calls", self.step_calls)
        self.zero_grad_calls = state.get("zero_grad_calls", self.zero_grad_calls)


class DummyScheduler:
    def __init__(self, optimizer, num_warmup_steps: int, total_steps: int):
        self.optimizer = optimizer
        self.step_calls = 0
        self.total_steps = total_steps
        record["scheduler"] = self

    def step(self) -> None:
        self.step_calls += 1

    def get_last_lr(self):
        return [self.optimizer.lr]

    def state_dict(self):
        return {"step_calls": self.step_calls, "total_steps": self.total_steps}

    def load_state_dict(self, state) -> None:
        if state is None:
            return
        self.step_calls = state.get("step_calls", self.step_calls)
        self.total_steps = state.get("total_steps", self.total_steps)
