from dataclasses import dataclass
from typing import Optional, Union, cast

import torch
from transformers.data.data_collator import DataCollatorMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils.generic import PaddingStrategy


@dataclass
class DataCollator(DataCollatorMixin):
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def torch_call(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"

        labels = (
            [feature[label_name] for feature in features]
            if label_name in features[0].keys()
            else None
        )

        no_labels_features = [
            {k: v for k, v in feature.items() if k != label_name}
            for feature in features
        ]

        batch = self.tokenizer.pad(
            no_labels_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        if labels is None:
            return batch

        sequence_length = cast(torch.Tensor, batch["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side

        if not labels:
            return batch

        first_label = labels[0]

        def to_list(tensor_or_iterable):
            if isinstance(tensor_or_iterable, torch.Tensor):
                return tensor_or_iterable.tolist()
            return list(tensor_or_iterable)

        is_sequence_label = isinstance(first_label, (list, tuple)) or (
            isinstance(first_label, torch.Tensor) and first_label.dim() > 0
        )

        if is_sequence_label:
            if padding_side == "right":
                batch[label_name] = [
                    [self.label_pad_token_id]
                    + to_list(label)
                    + [self.label_pad_token_id] * (sequence_length - len(label) - 1)
                    for label in labels
                ]
            else:
                batch[label_name] = [
                    [self.label_pad_token_id] * (sequence_length - len(label))
                    + to_list(label)
                    for label in labels
                ]

            if isinstance(first_label, (list, tuple)) and len(first_label) > 0:
                if isinstance(first_label[0], int):
                    batch[label_name] = torch.tensor(
                        batch[label_name], dtype=torch.long
                    )
                else:
                    batch[label_name] = torch.tensor(
                        batch[label_name], dtype=torch.float
                    )
            elif isinstance(first_label, torch.Tensor):
                if first_label.dtype in (torch.int32, torch.int64):
                    batch[label_name] = torch.tensor(
                        batch[label_name], dtype=torch.long
                    )
                else:
                    batch[label_name] = torch.tensor(
                        batch[label_name], dtype=torch.float
                    )
            else:
                batch[label_name] = torch.tensor(batch[label_name], dtype=torch.long)
        else:
            if isinstance(first_label, torch.Tensor):
                if first_label.dtype in (torch.int32, torch.int64):
                    labels = [int(label) for label in labels]
                    batch[label_name] = torch.tensor(labels, dtype=torch.long)
                else:
                    labels = [float(label) for label in labels]
                    batch[label_name] = torch.tensor(labels, dtype=torch.float)
            elif isinstance(first_label, float):
                batch[label_name] = torch.tensor(labels, dtype=torch.float)
            else:
                batch[label_name] = torch.tensor(labels, dtype=torch.long)

        return batch
