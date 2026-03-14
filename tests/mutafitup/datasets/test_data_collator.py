import torch

from mutafitup.datasets import DataCollator


class DummyTokenizer:
    def __init__(self, padding_side: str = "right"):
        self.padding_side = padding_side

    def pad(
        self,
        features,
        padding=True,
        max_length=None,
        pad_to_multiple_of=None,
        return_tensors=None,
    ):
        max_len = max(len(f["input_ids"]) for f in features)
        padded = []
        for f in features:
            ids = list(f["input_ids"])
            if self.padding_side == "right":
                ids = ids + [0] * (max_len - len(ids))
            else:
                ids = [0] * (max_len - len(ids)) + ids
            padded.append(ids)
        return {"input_ids": torch.tensor(padded, dtype=torch.long)}


def test_data_collator_sequence_int_labels_right_padding():
    tokenizer = DummyTokenizer(padding_side="right")
    collator = DataCollator(tokenizer=tokenizer, label_pad_token_id=-100)

    features = [
        {"input_ids": [1, 2, 3, 4], "labels": [10, 11, 12]},
        {"input_ids": [5, 6, 7, 8], "labels": [20, 21, 22]},
    ]

    batch = collator.torch_call(features)

    assert batch["input_ids"].shape == (2, 4)
    assert batch["labels"].shape == (2, 4)
    assert batch["labels"].dtype == torch.long
    assert batch["labels"][0, 0].item() == -100
    assert batch["labels"][0, 1:].tolist() == [10, 11, 12]


def test_data_collator_sequence_float_labels_left_padding():
    tokenizer = DummyTokenizer(padding_side="left")
    collator = DataCollator(tokenizer=tokenizer, label_pad_token_id=-100)

    features = [
        {"input_ids": [1, 2, 3, 4], "labels": [1.0, 2.0, 3.0]},
        {"input_ids": [5, 6, 7, 8], "labels": [4.0, 5.0, 6.0]},
    ]

    batch = collator.torch_call(features)

    assert batch["input_ids"].shape == (2, 4)
    assert batch["labels"].shape == (2, 4)
    assert batch["labels"].dtype == torch.float
    assert batch["labels"][0].tolist() == [-100.0, 1.0, 2.0, 3.0]


def test_data_collator_scalar_labels_and_no_labels():
    tokenizer = DummyTokenizer()
    collator = DataCollator(tokenizer=tokenizer, label_pad_token_id=-100)

    scalar_features = [
        {"input_ids": [1, 2], "labels": 1},
        {"input_ids": [3, 4], "labels": 2},
    ]
    batch_scalar = collator.torch_call(scalar_features)

    assert batch_scalar["labels"].shape == (2,)
    assert batch_scalar["labels"].dtype == torch.long
    assert batch_scalar["labels"].tolist() == [1, 2]

    no_label_features = [
        {"input_ids": [1, 2]},
        {"input_ids": [3, 4]},
    ]
    batch_no_label = collator.torch_call(no_label_features)

    assert "labels" not in batch_no_label
    assert batch_no_label["input_ids"].shape == (2, 2)
