from typing import Optional

import torch
import torch.nn as nn


class PerProteinLatePoolRegressionHead(nn.Module):
    def __init__(self, dropout_prop: float, input_hidden_size: int, hidden_size: int):
        super().__init__()
        self.dropout = nn.Dropout(dropout_prop)
        self.linear = nn.Linear(input_hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, 1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        # Zero out masked/unresolved residues first
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1)  # [B, L, 1]
            hidden_states = hidden_states * mask

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.linear(hidden_states)
        hidden_states = torch.relu(hidden_states)

        # Masked mean pooling
        if attention_mask is not None:
            hidden_states = hidden_states.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            hidden_states = torch.mean(hidden_states, dim=1)

        logits = self.output(hidden_states)

        return logits
