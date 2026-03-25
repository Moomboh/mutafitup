"""Uncertainty-based loss weighting (Kendall et al., CVPR 2018).

Learns per-task homoscedastic uncertainty parameters that automatically
balance losses in multi-task training. The learnable parameter is the
log-variance s_i = log(sigma_i^2), initialized at 0 (equal weighting).

Regression:      L_weighted = 0.5 * exp(-s) * L + 0.5 * s
Classification:  L_weighted = exp(-s) * L + s
"""

from typing import Dict, List, Literal

import torch
import torch.nn as nn


class UncertaintyLossWeights(nn.Module):
    def __init__(
        self,
        task_names: List[str],
        problem_types: Dict[str, Literal["classification", "regression"]],
    ):
        super().__init__()
        self.log_vars = nn.ParameterDict(
            {name: nn.Parameter(torch.zeros(1)) for name in task_names}
        )
        self.problem_types = problem_types

    def forward(self, loss: torch.Tensor, task_name: str) -> torch.Tensor:
        log_var = self.log_vars[task_name]
        if self.problem_types[task_name] == "regression":
            return 0.5 * torch.exp(-log_var) * loss + 0.5 * log_var
        else:
            return torch.exp(-log_var) * loss + log_var
