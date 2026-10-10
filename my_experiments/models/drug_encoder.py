from __future__ import annotations

import torch
import torch.nn as nn


class DrugEncoder(nn.Module):
    def __init__(self, drug_dim: int = 2304, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(drug_dim, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, drug_feat: torch.Tensor) -> torch.Tensor:
        return self.net(torch.nan_to_num(drug_feat.float(), nan=0.0, posinf=0.0, neginf=0.0))
