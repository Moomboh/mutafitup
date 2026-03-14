import torch

from mutafitup.models.multitask_heads import PerProteinRegressionHead


def test_per_protein_regression_head_output_shape():
    head = PerProteinRegressionHead(0.1, 4, 8)
    inputs = torch.randn(2, 5, 4)
    outputs = head(inputs)

    assert outputs.shape == (2, 1)

