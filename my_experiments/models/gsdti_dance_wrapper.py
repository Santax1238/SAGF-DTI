from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn

from .protein_encoders import gaussian_poe_fusion


def _load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_origin_graphcnn():
    repo_root = Path(__file__).resolve().parents[2]
    origin_model_path = repo_root / "origin_model.py"
    origin_pool_path = repo_root / "origin_pool.py"
    if not origin_model_path.exists() or not origin_pool_path.exists():
        raise FileNotFoundError("origin_model.py and origin_pool.py are required for the origin GSDTI backend")
    old_pool = sys.modules.get("pool")
    try:
        _load_module_from_path("pool", origin_pool_path)
        origin_model = _load_module_from_path("origin_model_runtime_for_e4", origin_model_path)
        return origin_model.GraphCNN
    finally:
        if old_pool is not None:
            sys.modules["pool"] = old_pool
        else:
            sys.modules.pop("pool", None)


class DynamicPropertyProjector(nn.Module):
    """Encode ESMDance residue prediction-head outputs when the input is [L, 50]."""

    def __init__(self, dyn_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.dyn_dim = int(dyn_dim)
        motion_hidden = hidden_dim // 2
        conf_hidden = hidden_dim - motion_hidden
        self.motion_encoder = nn.Sequential(nn.LayerNorm(5), nn.Linear(5, motion_hidden), nn.ReLU(), nn.Dropout(dropout))
        self.conformation_encoder = nn.Sequential(nn.LayerNorm(45), nn.Linear(45, conf_hidden), nn.ReLU(), nn.Dropout(dropout))
        self.property_proj = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.legacy_proj = nn.Sequential(nn.LayerNorm(dyn_dim), nn.Linear(dyn_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))

    def forward(self, x_dyn: torch.Tensor) -> torch.Tensor:
        x_dyn = torch.nan_to_num(x_dyn.float(), nan=0.0, posinf=0.0, neginf=0.0)
        if x_dyn.shape[-1] == 50:
            motion = torch.cat([x_dyn[:, 2:3], x_dyn[:, 1:2], x_dyn[:, 3:6]], dim=-1)
            conformation = torch.cat([x_dyn[:, 0:1], x_dyn[:, 6:]], dim=-1)
            return self.property_proj(torch.cat([self.motion_encoder(motion), self.conformation_encoder(conformation)], dim=-1))
        return self.legacy_proj(x_dyn)


class GSDTIWithDanceNodeWrapper(nn.Module):
    """E11: original GSDTI backbone with gated ESM2/ESMDance-node fusion.

    This keeps the original GSDTI choices: GATConv x3, GraphMultisetTransformer
    pooling, bilinear fusion, and the original classifier style. It does not use
    edge_attr, ESMDance pair features, mean pooling, or progressive injection.
    """

    def __init__(
        self,
        drug_dim: int = 2304,
        esm_dim: int = 1280,
        dyn_dim: int = 480,
        hidden_dim: int = 256,
        output_dim: int = 2,
        dropout: float = 0.3,
        drop_edge: float = 0.1,
    ):
        super().__init__()
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root))
        from model import GraphCNN

        self.esm_proj = nn.Sequential(
            nn.LayerNorm(esm_dim),
            nn.Linear(esm_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.dyn_proj = DynamicPropertyProjector(dyn_dim, hidden_dim, dropout)
        self.fuse_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fuse_gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.gcn = GraphCNN(hidden_dim=hidden_dim, num_layers=3, heads=2, dropout=dropout, drop_edge=drop_edge, pooling="MTP")
        self.drug_mlp = nn.Sequential(
            nn.Linear(drug_dim, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, hidden_dim),
            nn.Dropout(dropout),
        )
        self.prot_proj = nn.Linear(hidden_dim, hidden_dim)
        self.bilinear = nn.Bilinear(hidden_dim, hidden_dim, hidden_dim)
        self.outlinear = nn.Sequential(nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, output_dim))

    def forward(self, batch: dict) -> tuple[torch.Tensor, dict]:
        graph = batch["protein_graph"]
        x_esm = getattr(graph, "x_esm", None)
        if x_esm is None:
            x_esm = graph.x
        x_dyn = getattr(graph, "x_dyn", None)
        if x_dyn is None:
            raise ValueError("E11 requires graph.x_dyn from ESMDance node features")
        if x_esm.shape[0] != x_dyn.shape[0]:
            raise ValueError(f"E11 node mismatch: x_esm={tuple(x_esm.shape)}, x_dyn={tuple(x_dyn.shape)}")
        x_esm = torch.nan_to_num(x_esm.float(), nan=0.0, posinf=0.0, neginf=0.0)
        x_dyn = torch.nan_to_num(x_dyn.float(), nan=0.0, posinf=0.0, neginf=0.0)
        h_esm = self.esm_proj(x_esm)
        h_dyn = self.dyn_proj(x_dyn)
        h_new = self.fuse_mlp(torch.cat([h_esm, h_dyn], dim=-1))
        gate = self.fuse_gate(torch.cat([h_esm, h_new, h_esm - h_new], dim=-1))
        node_feat = gate * h_esm + (1.0 - gate) * h_new
        node_feat = torch.nan_to_num(node_feat, nan=0.0, posinf=0.0, neginf=0.0)
        _, prot_g = self.gcn(node_feat, graph)
        prot_emb = self.prot_proj(prot_g)
        drug_emb = self.drug_mlp(torch.nan_to_num(batch["drug_feat"].float(), nan=0.0, posinf=0.0, neginf=0.0))
        pair_emb = self.bilinear(drug_emb, prot_emb)
        logits = self.outlinear(pair_emb)
        return torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0), {
            "drug_emb": drug_emb,
            "protein_emb": prot_emb,
            "node_fusion_gate": gate.detach(),
            "node_fusion_gate_mean": gate.mean().detach(),
            "node_fusion_gate_std": gate.std(unbiased=False).detach(),
        }


class GSDTIGaussianNodeFusionWrapper(nn.Module):
    """Original GS-DTI backbone with Gaussian ESM2/ESMDance node fusion.

    This is the controlled "E4 on GSDTI" variant:
      E0 original node input:      x = ESM2 -> Linear(1280, 512)
      This variant node input:     x = GaussianPoE(ESM2, ESMDance) in 512 dim

    Everything after the node feature is the original GS-DTI backbone:
    GCNConv x3, GraphMultisetTransformer pooling, Bilinear fusion.
    """

    def __init__(
        self,
        drug_dim: int = 2304,
        esm_dim: int = 1280,
        dyn_dim: int = 480,
        output_dim: int = 2,
        dropout: float = 0.2,
        logvar_min: float = -5.0,
        logvar_max: float = 5.0,
        gaussian_output_mode: str = "residual_gaussian",
        gamma_init: float = 0.1,
    ):
        super().__init__()
        GraphCNN = _load_origin_graphcnn()
        self.hidden_dim = 512
        self.emb_dim = 256
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)
        self.gaussian_output_mode = str(gaussian_output_mode)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

        self.esm_proj = nn.Linear(esm_dim, self.hidden_dim)
        self.dyn_proj = DynamicPropertyProjector(dyn_dim, self.hidden_dim, dropout)
        mid = self.hidden_dim // 2
        self.var_head_esm = nn.Sequential(nn.Linear(self.hidden_dim, mid), nn.ReLU(), nn.Linear(mid, 1))
        self.var_head_dyn = nn.Sequential(nn.Linear(self.hidden_dim, mid), nn.ReLU(), nn.Linear(mid, 1))
        self._init_var_head(self.var_head_esm)
        self._init_var_head(self.var_head_dyn)
        self.out_norm = nn.LayerNorm(self.hidden_dim)

        self.gcn = GraphCNN(pooling="MTP")
        self.drug_linear = nn.Linear(drug_dim, 1024)
        self.hidden2 = nn.Linear(1024, self.emb_dim)
        self.bilinear = nn.Bilinear(self.emb_dim, self.emb_dim, self.emb_dim)
        self.outlinear = nn.Linear(self.emb_dim, output_dim)
        self.dropout = float(dropout)

    @staticmethod
    def _init_var_head(head: nn.Sequential) -> None:
        last = head[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def _build_node_feature(self, graph) -> tuple[torch.Tensor, dict]:
        x_esm = getattr(graph, "x_esm", None)
        if x_esm is None:
            x_esm = graph.x
        x_dyn = getattr(graph, "x_dyn", None)
        if x_dyn is None:
            raise ValueError("E4_GSDTI_GAUSS requires graph.x_dyn from ESMDance node features")
        if x_esm.shape[0] != x_dyn.shape[0]:
            raise ValueError(f"node count mismatch: x_esm={tuple(x_esm.shape)} x_dyn={tuple(x_dyn.shape)}")
        h_esm = self.esm_proj(torch.nan_to_num(x_esm.float(), nan=0.0, posinf=0.0, neginf=0.0))
        h_dyn = self.dyn_proj(x_dyn)
        logvar_esm = self.var_head_esm(h_esm).clamp(min=self.logvar_min, max=self.logvar_max)
        logvar_dyn = self.var_head_dyn(h_dyn).clamp(min=self.logvar_min, max=self.logvar_max)
        mu_fused, logvar_fused, w_esm, w_dyn, _ = gaussian_poe_fusion(
            h_esm,
            logvar_esm,
            h_dyn,
            logvar_dyn,
            logvar_min=self.logvar_min,
            logvar_max=self.logvar_max,
        )
        if self.gaussian_output_mode == "strict_gaussian":
            node_feat = mu_fused
        elif self.gaussian_output_mode == "residual_gaussian":
            node_feat = self.out_norm(h_esm + self.gamma * (mu_fused - h_esm))
        else:
            raise ValueError(f"unknown gaussian_output_mode={self.gaussian_output_mode!r}")
        node_feat = torch.nan_to_num(node_feat, nan=0.0, posinf=0.0, neginf=0.0)
        var_esm = torch.exp(logvar_esm)
        var_dyn = torch.exp(logvar_dyn)
        var_fused = torch.exp(logvar_fused)
        entropy_const = logvar_esm.new_tensor(2.0 * torch.pi * torch.e).log()
        aux = {
            "weight_esm_mean": w_esm.detach().float().mean(),
            "weight_dyn_mean": w_dyn.detach().float().mean(),
            "esm_var_mean": var_esm.detach().float().mean(),
            "dyn_var_mean": var_dyn.detach().float().mean(),
            "fused_var_mean": var_fused.detach().float().mean(),
            "esm_entropy": (0.5 * (entropy_const + logvar_esm.detach().float())).mean(),
            "dyn_entropy": (0.5 * (entropy_const + logvar_dyn.detach().float())).mean(),
            "fused_entropy": (0.5 * (entropy_const + logvar_fused.detach().float())).mean(),
            "gamma": self.gamma.detach(),
            "h_esm_norm": h_esm.detach().float().norm(dim=-1).mean(),
            "h_dyn_norm": h_dyn.detach().float().norm(dim=-1).mean(),
            "h_out_norm": node_feat.detach().float().norm(dim=-1).mean(),
        }
        return node_feat, aux

    def forward(self, batch: dict) -> tuple[torch.Tensor, dict]:
        graph = batch["protein_graph"]
        node_feat, aux = self._build_node_feature(graph)
        _, prot_emb = self.gcn(node_feat, graph)
        drug_emb = self.drug_linear(torch.nan_to_num(batch["drug_feat"].float(), nan=0.0, posinf=0.0, neginf=0.0))
        drug_emb = torch.relu(drug_emb)
        drug_emb = torch.dropout(drug_emb, self.dropout, train=self.training)
        drug_emb = self.hidden2(drug_emb)
        pair_emb = self.bilinear(drug_emb, prot_emb)
        pair_emb = torch.relu(pair_emb)
        pair_emb = torch.dropout(pair_emb, self.dropout, train=self.training)
        logits = self.outlinear(pair_emb)
        aux.update({"drug_emb": drug_emb, "protein_emb": prot_emb})
        return torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0), aux
