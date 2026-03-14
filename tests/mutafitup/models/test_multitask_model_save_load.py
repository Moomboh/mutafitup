import torch
from peft import LoraConfig
from torch import nn

from mutafitup.models.multitask_heads import (
    PerProteinClassificationHead,
    PerResidueRegressionHead,
)
from mutafitup.models.multitask_model import (
    HeadConfig,
    MultitaskBackbone,
    MultitaskForwardArgs,
    MultitaskModel,
)


class DummyBackbone(MultitaskBackbone):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(hidden_size, hidden_size)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str,
        lora_config: LoraConfig | None = None,
    ):
        backbone = cls(hidden_size=8)
        backbone.apply_lora_config(lora_config)
        return backbone, None

    def _inject_lora_config(self, lora_config: LoraConfig | None) -> None:
        self._lora_config = lora_config

    def forward(self, args: MultitaskForwardArgs) -> torch.Tensor:
        return self.linear(args.input_ids.float())

    def preprocess_sequences(self, sequences, checkpoint):
        return sequences


def test_multitask_model_save_and_load_roundtrip(tmp_path):
    backbone = DummyBackbone(hidden_size=8)

    heads = {
        "per_protein_cls": HeadConfig(
            head=PerProteinClassificationHead(0.1, 8, 8, 3),
            problem_type="classification",
            level="per_protein",
        ),
        "per_residue_reg": HeadConfig(
            head=PerResidueRegressionHead(0.2, 8, 8),
            problem_type="regression",
            level="per_residue",
        ),
    }

    model = MultitaskModel(backbone=backbone, heads=heads)

    for _, param in model.named_parameters():
        with torch.no_grad():
            param.copy_(torch.arange(param.numel()).view_as(param).float())

    num_params = model.save_trainable_weights(tmp_path.as_posix(), "dummy_checkpoint")

    saved = torch.load(tmp_path / "model.pt", map_location="cpu", weights_only=False)
    saved_state = saved["state_dict"]

    assert num_params == sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert saved_state

    loaded = MultitaskModel.load_from_file(tmp_path / "model.pt")

    loaded_state = {
        name: p.detach().cpu()
        for name, p in loaded.named_parameters()
        if p.requires_grad
    }

    assert set(saved_state.keys()) == set(loaded_state.keys())

    for name, tensor in saved_state.items():
        assert torch.allclose(tensor, loaded_state[name])
