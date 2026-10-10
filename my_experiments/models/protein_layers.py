from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, TransformerConv, global_mean_pool


class EdgeAwareGraphLayer(nn.Module):
    def __init__(self, hidden_dim: int = 256, edge_dim: int = 0, heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.edge_dim = int(edge_dim)
        out_channels = max(1, hidden_dim // heads)
        if self.edge_dim > 0:
            self.conv = TransformerConv(
                hidden_dim,
                out_channels,
                heads=heads,
                concat=True,
                edge_dim=self.edge_dim,
                dropout=dropout,
            )
        else:
            self.conv = GATConv(hidden_dim, out_channels, heads=heads, concat=True, dropout=dropout)
        conv_dim = out_channels * heads
        self.out_proj = nn.Linear(conv_dim, hidden_dim) if conv_dim != hidden_dim else nn.Identity()
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
        if x.ndim != 2:
            raise ValueError(f"EdgeAwareGraphLayer expected x [N,H], got {tuple(x.shape)}")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"EdgeAwareGraphLayer expected edge_index [2,E], got {tuple(edge_index.shape)}")
        if edge_index.numel() == 0:
            msg = x
        elif self.edge_dim > 0:
            if edge_attr is None or edge_attr.numel() == 0:
                edge_attr = torch.zeros(edge_index.shape[1], self.edge_dim, device=x.device, dtype=x.dtype)
            edge_attr = torch.nan_to_num(edge_attr.float(), nan=0.0, posinf=0.0, neginf=0.0)
            if edge_attr.ndim != 2 or edge_attr.shape[0] != edge_index.shape[1]:
                raise ValueError(
                    f"edge_attr shape {tuple(edge_attr.shape)} incompatible with edge_index E={edge_index.shape[1]}"
                )
            if edge_attr.shape[1] != self.edge_dim:
                raise ValueError(f"expected edge_dim={self.edge_dim}, got edge_attr dim={edge_attr.shape[1]}")
            msg = self.conv(x, edge_index, edge_attr)
        else:
            msg = self.conv(x, edge_index)
        msg = torch.nan_to_num(msg.float(), nan=0.0, posinf=0.0, neginf=0.0)
        msg = self.out_proj(msg)
        x = self.norm(x + self.dropout(msg))
        x = self.norm(x + self.dropout(self.ffn(x)))
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return x, {}


class GatedResidual(nn.Module):
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(self, x_old: torch.Tensor, x_new: torch.Tensor) -> tuple[torch.Tensor, dict]:
        gate = self.gate(torch.cat([x_old, x_new, x_old - x_new], dim=-1))
        out = gate * x_old + (1.0 - gate) * x_new
        return out, {
            "gate_mean": gate.mean().detach(),
            "gate_std": gate.std(unbiased=False).detach(),
        }


class ProteinMeanPooling(nn.Module):
    def forward(self, residue_emb: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        return global_mean_pool(residue_emb, batch)
