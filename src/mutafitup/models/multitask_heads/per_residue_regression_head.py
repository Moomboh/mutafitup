import torch
import torch.nn as nn


class PerResidueRegressionHead(nn.Module):
    def __init__(self, dropout_prop: float, input_hidden_size: int, hidden_size: int):
        super().__init__()
        self.dropout = nn.Dropout(dropout_prop)
        self.linear = nn.Linear(input_hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.linear(hidden_states)
        hidden_states = torch.relu(hidden_states)
        logits = self.output(hidden_states)

        return logits
