import torch

from mutafitup.models.multitask_heads import PerProteinLatePoolClassificationHead


def test_per_protein_late_pool_classification_head_output_shape():
    head = PerProteinLatePoolClassificationHead(0.1, 4, 8, 3)
    inputs = torch.randn(2, 5, 4)
    outputs = head(inputs)

    assert outputs.shape == (2, 3)


def test_per_protein_late_pool_classification_head_with_attention_mask():
    head = PerProteinLatePoolClassificationHead(0.0, 4, 8, 3)
    inputs = torch.randn(2, 5, 4)
    attention_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]])
    outputs = head(inputs, attention_mask)

    assert outputs.shape == (2, 3)


def test_per_protein_late_pool_classification_head_masking_zeroes_out():
    head = PerProteinLatePoolClassificationHead(0.0, 4, 8, 3)
    head.eval()

    inputs = torch.randn(2, 5, 4)
    mask = torch.tensor([[1, 0, 0, 0, 0], [1, 0, 0, 0, 0]])

    # With mask: only first residue contributes
    out_masked = head(inputs, mask)

    # Manually: zero out masked positions and compute
    masked_inputs = inputs.clone()
    masked_inputs[:, 1:, :] = 0.0
    out_manual = head(masked_inputs, mask)

    assert torch.allclose(out_masked, out_manual, atol=1e-6)
