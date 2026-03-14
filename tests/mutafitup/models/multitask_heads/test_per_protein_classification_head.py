import torch

from mutafitup.models.multitask_heads import PerProteinClassificationHead


def test_per_protein_classification_head_output_shape():
    head = PerProteinClassificationHead(0.1, 4, 8, 3)
    inputs = torch.randn(2, 5, 4)
    outputs = head(inputs)

    assert outputs.shape == (2, 3)

