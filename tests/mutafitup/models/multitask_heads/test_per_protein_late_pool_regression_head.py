import torch

from mutafitup.models.multitask_heads import PerProteinLatePoolRegressionHead


def test_per_protein_late_pool_regression_head_output_shape():
    head = PerProteinLatePoolRegressionHead(0.1, 4, 8)
    inputs = torch.randn(2, 5, 4)
    outputs = head(inputs)

    assert outputs.shape == (2, 1)


def test_per_protein_late_pool_regression_head_with_attention_mask():
    head = PerProteinLatePoolRegressionHead(0.0, 4, 8)
    inputs = torch.randn(2, 5, 4)
    attention_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]])
    outputs = head(inputs, attention_mask)

    assert outputs.shape == (2, 1)


def test_per_protein_late_pool_regression_head_masking_zeroes_out():
    head = PerProteinLatePoolRegressionHead(0.0, 4, 8)
    head.eval()

    inputs = torch.randn(2, 5, 4)
    mask = torch.tensor([[1, 0, 0, 0, 0], [1, 0, 0, 0, 0]])

    out_masked = head(inputs, mask)

    masked_inputs = inputs.clone()
    masked_inputs[:, 1:, :] = 0.0
    out_manual = head(masked_inputs, mask)

    assert torch.allclose(out_masked, out_manual, atol=1e-6)
