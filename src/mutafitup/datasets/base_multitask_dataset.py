from abc import ABC, abstractmethod

from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from mutafitup.models.multitask_model import MultitaskBackbone


class BaseMultitaskDataset(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def get_dataloader(
        self,
        split: str,
        backbone: MultitaskBackbone,
        tokenizer: PreTrainedTokenizerBase,
        checkpoint: str,
        batch_size: int,
    ) -> DataLoader:
        raise NotImplementedError
