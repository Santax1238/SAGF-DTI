from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch


def to_padded_sequence(node_emb: torch.Tensor, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if batch.numel() == 0:
        return node_emb.new_empty(0, 0, node_emb.shape[-1]), torch.empty(0, 0, dtype=torch.bool, device=node_emb.device), torch.empty(0, dtype=torch.long, device=node_emb.device)
    batch = batch.to(device=node_emb.device, dtype=torch.long)
    batch_size = int(batch.max().item()) + 1
    lengths = torch.bincount(batch, minlength=batch_size)
    padded, mask = to_dense_batch(node_emb, batch=batch, batch_size=batch_size)
    return padded, mask, lengths


def from_padded_sequence(padded: torch.Tensor, batch: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    if padded.numel() == 0:
        return padded.new_empty(0, padded.shape[-1])
    mask = torch.arange(padded.shape[1], device=padded.device).unsqueeze(0) < lengths.to(padded.device).unsqueeze(1)
    return padded[mask]


def _attention_entropy(weights: torch.Tensor, key_padding_mask: torch.Tensor | None) -> torch.Tensor:
    probs = weights.clamp_min(1e-8)
    entropy = -(probs * probs.log()).sum(dim=-1)
    if key_padding_mask is not None:
        valid_q = ~key_padding_mask
        entropy = entropy[valid_q]
    return entropy.mean().detach() if entropy.numel() else torch.tensor(0.0, device=weights.device)


class SequenceBlock(nn.Module):
    def __init__(self, hidden_dim: int = 256, heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 4), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim * 4, hidden_dim))
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        key_padding_mask = ~mask
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        attn_out = torch.nan_to_num(attn_out, nan=0.0, posinf=0.0, neginf=0.0)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return torch.nan_to_num(x * mask.unsqueeze(-1), nan=0.0, posinf=0.0, neginf=0.0)


class ProgressiveInjectionBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        heads: int = 4,
        dropout: float = 0.2,
        alpha_init: float = 0.1,
        log_attention_entropy: bool = False,
    ):
        super().__init__()
        self.log_attention_entropy = log_attention_entropy
        self.seq_block = SequenceBlock(hidden_dim, heads, dropout)
        self.seq_to_graph = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.graph_to_seq = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.seq_norm = nn.LayerNorm(hidden_dim)
        self.graph_norm = nn.LayerNorm(hidden_dim)
        self.alpha_seq = nn.Parameter(torch.tensor(float(alpha_init)))
        self.alpha_graph = nn.Parameter(torch.tensor(float(alpha_init)))
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h_seq: torch.Tensor,
        h_graph: torch.Tensor,
        batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        seq_pad, mask, lengths = to_padded_sequence(h_seq, batch)
        graph_pad, _, _ = to_padded_sequence(h_graph, batch)
        seq_pad = torch.nan_to_num(seq_pad, nan=0.0, posinf=0.0, neginf=0.0)
        graph_pad = torch.nan_to_num(graph_pad, nan=0.0, posinf=0.0, neginf=0.0)
        key_padding_mask = ~mask
        seq_pad = self.seq_block(seq_pad, mask)
        seq_update, seq_attn = self.seq_to_graph(
            seq_pad,
            graph_pad,
            graph_pad,
            key_padding_mask=key_padding_mask,
            need_weights=self.log_attention_entropy,
        )
        graph_update, graph_attn = self.graph_to_seq(
            graph_pad,
            seq_pad,
            seq_pad,
            key_padding_mask=key_padding_mask,
            need_weights=self.log_attention_entropy,
        )
        seq_update = torch.nan_to_num(seq_update, nan=0.0, posinf=0.0, neginf=0.0)
        graph_update = torch.nan_to_num(graph_update, nan=0.0, posinf=0.0, neginf=0.0)
        seq_pad = self.seq_norm(seq_pad + self.dropout(self.alpha_seq * seq_update)) * mask.unsqueeze(-1)
        graph_pad = self.graph_norm(graph_pad + self.dropout(self.alpha_graph * graph_update)) * mask.unsqueeze(-1)
        seq_pad = torch.nan_to_num(seq_pad, nan=0.0, posinf=0.0, neginf=0.0)
        graph_pad = torch.nan_to_num(graph_pad, nan=0.0, posinf=0.0, neginf=0.0)
        aux = {
            "alpha_seq": self.alpha_seq.detach(),
            "alpha_graph": self.alpha_graph.detach(),
        }
        if self.log_attention_entropy:
            aux["seq_to_graph_attn_entropy"] = _attention_entropy(seq_attn, key_padding_mask)
            aux["graph_to_seq_attn_entropy"] = _attention_entropy(graph_attn, key_padding_mask)
        return from_padded_sequence(seq_pad, batch, lengths), from_padded_sequence(graph_pad, batch, lengths), aux
