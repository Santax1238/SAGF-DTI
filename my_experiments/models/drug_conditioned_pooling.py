from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import softmax


class DrugConditionedPooling(nn.Module):
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.wq = nn.Linear(hidden_dim, hidden_dim)
        self.wk = nn.Linear(hidden_dim, hidden_dim)
        self.wv = nn.Linear(hidden_dim, hidden_dim)

    @staticmethod
    def attention_sums(attn: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
        sums = attn.new_zeros(num_graphs)
        if attn.numel() > 0:
            sums.index_add_(0, batch, attn)
        return sums

    def forward(
        self,
        drug_emb: torch.Tensor,
        residue_emb: torch.Tensor,
        batch: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        return_attention_sums: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if drug_emb.ndim != 2 or residue_emb.ndim != 2:
            raise ValueError(f"expected drug_emb [B,H] and residue_emb [N,H], got {tuple(drug_emb.shape)} and {tuple(residue_emb.shape)}")
        if drug_emb.shape[-1] != residue_emb.shape[-1]:
            raise ValueError(f"drug/residue hidden dim mismatch: {drug_emb.shape[-1]} vs {residue_emb.shape[-1]}")
        if batch.ndim != 1 or batch.numel() != residue_emb.shape[0]:
            raise ValueError(f"batch must be [N] aligned with residue_emb, got {tuple(batch.shape)} for N={residue_emb.shape[0]}")

        drug_emb = torch.nan_to_num(drug_emb, nan=0.0, posinf=0.0, neginf=0.0)
        residue_emb = torch.nan_to_num(residue_emb, nan=0.0, posinf=0.0, neginf=0.0)
        q = self.wq(drug_emb)
        k = self.wk(residue_emb)
        v = self.wv(residue_emb)

        if valid_mask is None:
            valid = torch.ones(residue_emb.shape[0], dtype=torch.bool, device=residue_emb.device)
        else:
            valid = valid_mask.to(device=residue_emb.device).view(-1).bool()
            if valid.numel() != residue_emb.shape[0]:
                raise ValueError(f"valid_mask must have N={residue_emb.shape[0]} entries, got {valid.numel()}")

        valid_counts = residue_emb.new_zeros(drug_emb.shape[0])
        valid_counts.index_add_(0, batch, valid.to(dtype=residue_emb.dtype))
        if (valid_counts <= 0).any():
            bad = torch.nonzero(valid_counts <= 0, as_tuple=False).view(-1).detach().cpu().tolist()
            raise ValueError(f"drug-conditioned attention found proteins with no valid residues: {bad[:10]}")

        attn = residue_emb.new_zeros(residue_emb.shape[0])
        if valid.any():
            valid_batch = batch[valid]
            score = (q[valid_batch] * k[valid]).sum(dim=-1) / math.sqrt(k.shape[-1])
            score = torch.nan_to_num(score, nan=0.0, posinf=20.0, neginf=-20.0)
            attn_valid = softmax(score, valid_batch, num_nodes=drug_emb.shape[0])
            attn[valid] = torch.nan_to_num(
                attn_valid, nan=0.0, posinf=0.0, neginf=0.0
            ).to(dtype=attn.dtype)

        pooled = global_add_pool(attn.unsqueeze(-1) * v, batch, size=drug_emb.shape[0])
        pooled = torch.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0)
        if return_attention_sums:
            return pooled, attn, self.attention_sums(attn, batch, drug_emb.shape[0])
        return pooled, attn
