from __future__ import annotations

import torch
import torch.nn as nn

from .progressive_injection import ProgressiveInjectionBlock, SequenceBlock
from .protein_encoders import _edge_attr
from .protein_layers import EdgeAwareGraphLayer, ProteinMeanPooling


class ProteinSDIEncoder(nn.Module):
    def __init__(
        self,
        esm_dim: int = 1280,
        dyn_dim: int = 480,
        hidden_dim: int = 256,
        edge_dim: int = 36,
        layers: int = 1,
        dropout: float = 0.2,
        attn_heads: int = 4,
        log_attention_entropy: bool = False,
    ):
        super().__init__()
        self.esm_proj = nn.Sequential(nn.LayerNorm(esm_dim), nn.Linear(esm_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.dyn_proj = nn.Sequential(nn.LayerNorm(dyn_dim), nn.Linear(dyn_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.graph_layer = EdgeAwareGraphLayer(hidden_dim, edge_dim=edge_dim, dropout=dropout)
        self.seq_block = SequenceBlock(hidden_dim, heads=attn_heads, dropout=dropout)
        self.injection = nn.ModuleList([
            ProgressiveInjectionBlock(
                hidden_dim,
                heads=attn_heads,
                dropout=dropout,
                log_attention_entropy=log_attention_entropy,
            )
            for _ in range(layers)
        ])
        self.pool = ProteinMeanPooling()

    def forward(self, graph) -> tuple[torch.Tensor, torch.Tensor, dict]:
        x_esm = torch.nan_to_num(getattr(graph, "x_esm", graph.x).float(), nan=0.0, posinf=0.0, neginf=0.0)
        x_dyn = getattr(graph, "x_dyn", None)
        h_seq = self.esm_proj(x_esm)
        if x_dyn is None:
            h_dyn = torch.zeros_like(h_seq)
        else:
            x_dyn = torch.nan_to_num(x_dyn.float(), nan=0.0, posinf=0.0, neginf=0.0)
            h_dyn = self.dyn_proj(x_dyn)
        h_graph, _ = self.graph_layer(h_dyn, graph.edge_index.long(), _edge_attr(graph))
        aux = {}
        # SequenceBlock inside ProgressiveInjectionBlock already performs the sequence-only update.
        for block in self.injection:
            h_seq, h_graph, block_aux = block(h_seq, h_graph, graph.batch)
            aux.update(block_aux)
        residue_emb = torch.nan_to_num(0.5 * (h_seq + h_graph), nan=0.0, posinf=0.0, neginf=0.0)
        protein_emb = self.pool(residue_emb, graph.batch)
        return protein_emb, residue_emb, aux
