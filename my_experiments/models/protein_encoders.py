from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch

from .drug_conditioned_pooling import DrugConditionedPooling
from .protein_layers import EdgeAwareGraphLayer, GatedResidual, ProteinMeanPooling


def _edge_attr(graph) -> torch.Tensor | None:
    attr = getattr(graph, "edge_attr", None)
    if isinstance(attr, torch.Tensor) and attr.numel() > 0 and attr.shape[-1] > 0:
        return torch.nan_to_num(attr.float(), nan=0.0, posinf=0.0, neginf=0.0)
    return None


def _dense_embedding_aux(graph, **tensors: torch.Tensor) -> dict:
    aux: dict[str, torch.Tensor] = {}
    mask = None
    for key, value in tensors.items():
        dense, mask = to_dense_batch(value, graph.batch)
        aux[key] = dense.detach()
    if mask is None:
        _, mask = to_dense_batch(torch.zeros(graph.num_nodes, 1, device=graph.batch.device), graph.batch)
    aux["protein_mask"] = mask.detach()
    return aux


def _node_valid_mask(graph) -> torch.Tensor | None:
    """Return an optional node-validity mask without treating missing ESMDance as padding."""
    for name in ("residue_valid_mask", "node_valid_mask", "valid_mask", "node_mask"):
        value = getattr(graph, name, None)
        if isinstance(value, torch.Tensor):
            return value
    return None


class BaseProteinEncoder(nn.Module):
    def __init__(
        self,
        esm_dim: int = 1280,
        dyn_dim: int = 50,
        esmdance35_dim: int = 480,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        edge_dim: int = 0,
        dyn_ablation_mode: str = "none",
        dyn_pollution_ratio: float = 0.0,
        dyn_conditioned_on_esm35: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dyn_dim = int(dyn_dim)
        self.esmdance35_dim = int(esmdance35_dim)
        self.dyn_conditioned_on_esm35 = bool(dyn_conditioned_on_esm35)
        self.dyn_ablation_mode = str(dyn_ablation_mode).lower()
        self.dyn_pollution_ratio = float(dyn_pollution_ratio)
        self.esm_proj = nn.Sequential(nn.LayerNorm(esm_dim), nn.Linear(esm_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.esmdance35_proj = nn.Sequential(
            nn.LayerNorm(self.esmdance35_dim),
            nn.Linear(self.esmdance35_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        motion_hidden = hidden_dim // 2
        conf_hidden = hidden_dim - motion_hidden
        self.motion_encoder = nn.Sequential(
            nn.LayerNorm(5),
            nn.Linear(5, motion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.conformation_encoder = nn.Sequential(
            nn.LayerNorm(45),
            nn.Linear(45, conf_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.dynamic_property_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.dynamic_condition_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.dynamic_condition_delta = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.dynamic_condition_norm = nn.LayerNorm(hidden_dim)
        self.dyn_proj = nn.Sequential(nn.LayerNorm(dyn_dim), nn.Linear(dyn_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.pool = ProteinMeanPooling()

    def project_esm(self, graph) -> torch.Tensor:
        x_esm = getattr(graph, "x_esm", None)
        if x_esm is None:
            x_esm = graph.x
        x_esm = torch.nan_to_num(x_esm.float(), nan=0.0, posinf=0.0, neginf=0.0)
        return self.esm_proj(x_esm)

    def _get_esmdance35(self, graph) -> torch.Tensor | None:
        for name in (
            "esmdance_35m_embedding",
            "esmdance35_embedding",
            "esmdance_encoder_embedding",
            "esmdance_base_embedding",
            "x_esmdance35",
            "x_esm35",
        ):
            value = getattr(graph, name, None)
            if isinstance(value, torch.Tensor):
                return value
        return None

    def _project_dynamic_properties(self, x_dyn: torch.Tensor) -> torch.Tensor:
        # Saved order: sasa_mean, sasa_std, rmsf_nor, nma_res1..3,
        # ss(8), chi(12), phi(12), psi(12).
        motion = torch.cat([x_dyn[:, 2:3], x_dyn[:, 1:2], x_dyn[:, 3:6]], dim=-1)
        conformation = torch.cat([x_dyn[:, 0:1], x_dyn[:, 6:]], dim=-1)
        return self.dynamic_property_proj(torch.cat([
            self.motion_encoder(motion),
            self.conformation_encoder(conformation),
        ], dim=-1))

    def project_dyn(self, graph, return_aux: bool = False):
        aux: dict[str, torch.Tensor] = {}
        x_dyn = getattr(graph, "x_dyn", None)
        if x_dyn is None:
            x_dyn = getattr(graph, "predicted_dynamic_residue_features", None)
        if x_dyn is None:
            x_dyn = getattr(graph, "esmdance_residue_properties", None)
        if x_dyn is None:
            h_zero = torch.zeros(graph.num_nodes, self.hidden_dim, device=graph.x.device)
            return (h_zero, aux) if return_aux else h_zero
        x_dyn = torch.nan_to_num(x_dyn.float(), nan=0.0, posinf=0.0, neginf=0.0)
        x_dyn = self._apply_dynamic_ablation(x_dyn, graph)
        if x_dyn.shape[-1] == 50:
            esm35_only = self.dyn_ablation_mode in {
                "esm35_only",
                "esm2_35m_only",
                "no_esmdance_properties",
            }
            h_prop = (
                torch.zeros(x_dyn.shape[0], self.hidden_dim, device=x_dyn.device, dtype=x_dyn.dtype)
                if esm35_only
                else self._project_dynamic_properties(x_dyn)
            )
            x_35 = self._get_esmdance35(graph)
            if esm35_only and not isinstance(x_35, torch.Tensor):
                raise ValueError("ESM2-35M-only ablation requires an ESM2-35M residue embedding")
            if self.dyn_conditioned_on_esm35 and isinstance(x_35, torch.Tensor):
                x_35 = torch.nan_to_num(x_35.float(), nan=0.0, posinf=0.0, neginf=0.0)
                if x_35.shape[0] != x_dyn.shape[0]:
                    raise ValueError(f"ESMDance-35M/dynamic50 node mismatch: {tuple(x_35.shape)} vs {tuple(x_dyn.shape)}")
                h_35 = self.esmdance35_proj(x_35)
                if esm35_only:
                    # Keep the carrier architecture but remove every ESMDance
                    # property from the auxiliary view.  This is the direct
                    # control for gains caused merely by a second PLM.
                    cond_gate = torch.zeros_like(h_35)
                    h_dyn = self.dynamic_condition_norm(h_35)
                    conditioned_shift = torch.zeros(h_35.shape[0], device=h_35.device, dtype=h_35.dtype)
                else:
                    cond_in = torch.cat([h_35, h_prop], dim=-1)
                    cond_gate = self.dynamic_condition_gate(cond_in)
                    cond_delta = self.dynamic_condition_delta(cond_in)
                    h_dyn = self.dynamic_condition_norm(h_35 + cond_gate * cond_delta)
                    conditioned_shift = (h_dyn - h_35).detach().float().norm(dim=-1)
                aux.update({
                    "esmdance_condition_gate_node": cond_gate.detach(),
                    "esmdance35_node_norm": h_35.detach().float().norm(dim=-1),
                    "dynamic_property_node_norm": h_prop.detach().float().norm(dim=-1),
                    "conditioned_dyn_shift_node": conditioned_shift,
                })
            else:
                h_dyn = h_prop
        else:
            h_dyn = self.dyn_proj(x_dyn)
        mask = getattr(graph, "esmdance_embedding_mask", None)
        if isinstance(mask, torch.Tensor):
            h_dyn = h_dyn * mask.to(device=h_dyn.device, dtype=h_dyn.dtype)
        return (h_dyn, aux) if return_aux else h_dyn

    def _shuffled_dynamic_features(self, x_dyn: torch.Tensor, graph) -> torch.Tensor:
        batch = getattr(graph, "batch", None)
        if not isinstance(batch, torch.Tensor) or batch.numel() != x_dyn.shape[0]:
            perm = torch.randperm(x_dyn.shape[0], device=x_dyn.device)
            return x_dyn[perm]
        out = x_dyn.clone()
        for graph_id in batch.unique(sorted=True):
            idx = (batch == graph_id).nonzero(as_tuple=False).view(-1)
            if idx.numel() > 1:
                out[idx] = x_dyn[idx[torch.randperm(idx.numel(), device=x_dyn.device)]]
        return out

    def _apply_dynamic_ablation(self, x_dyn: torch.Tensor, graph) -> torch.Tensor:
        mode = self.dyn_ablation_mode
        ratio = min(max(self.dyn_pollution_ratio, 0.0), 1.0)
        if mode in {"none", "full", ""} and ratio <= 0.0:
            return x_dyn
        if mode in {
            "zero",
            "zero_dynamic",
            "no_dynamic",
            "esm35_only",
            "esm2_35m_only",
            "no_esmdance_properties",
        }:
            return torch.zeros_like(x_dyn)
        shuffled = self._shuffled_dynamic_features(x_dyn, graph)
        if mode in {"shuffle", "shuffled", "shuffle_residue", "shuffled_dynamics"} and ratio <= 0.0:
            return shuffled
        if mode in {"pollute", "pollution", "shuffle_mix", "shuffled_mix"} or ratio > 0.0:
            return (1.0 - ratio) * x_dyn + ratio * shuffled
        raise ValueError(f"unknown dyn_ablation_mode={self.dyn_ablation_mode!r}")


class ESM2OnlyEncoder(BaseProteinEncoder):
    def forward(self, graph) -> tuple[torch.Tensor, torch.Tensor, dict]:
        h = self.project_esm(graph)
        return self.pool(h, graph.batch), h, {}


class ESMDanceOnlyEncoder(BaseProteinEncoder):
    def forward(self, graph) -> tuple[torch.Tensor, torch.Tensor, dict]:
        h = self.project_dyn(graph)
        return self.pool(h, graph.batch), h, {}


class ESM2ESMFoldEdgeIndexEncoder(BaseProteinEncoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layer = EdgeAwareGraphLayer(self.hidden_dim, edge_dim=0, dropout=kwargs.get("dropout", 0.2))

    def forward(self, graph) -> tuple[torch.Tensor, torch.Tensor, dict]:
        h = self.project_esm(graph)
        h, aux = self.layer(h, graph.edge_index.long(), None)
        return self.pool(h, graph.batch), h, aux


class ESM2ESMFoldEdgeAttrEncoder(BaseProteinEncoder):
    def __init__(self, edge_dim: int = 23, **kwargs):
        super().__init__(edge_dim=edge_dim, **kwargs)
        self.layer = EdgeAwareGraphLayer(self.hidden_dim, edge_dim=edge_dim, dropout=kwargs.get("dropout", 0.2))

    def forward(self, graph) -> tuple[torch.Tensor, torch.Tensor, dict]:
        h = self.project_esm(graph)
        h, aux = self.layer(h, graph.edge_index.long(), _edge_attr(graph))
        return self.pool(h, graph.batch), h, aux


class ESM2ESMDancePairEdgeEncoder(BaseProteinEncoder):
    """E3.2: ESM2 nodes with ESMDance-pair-derived edge_index and edge_attr."""

    def __init__(self, edge_dim: int = 13, **kwargs):
        super().__init__(edge_dim=edge_dim, **kwargs)
        self.layer = EdgeAwareGraphLayer(self.hidden_dim, edge_dim=edge_dim, dropout=kwargs.get("dropout", 0.2))

    def forward(self, graph) -> tuple[torch.Tensor, torch.Tensor, dict]:
        h = self.project_esm(graph)
        edge_attr = _edge_attr(graph)
        if edge_attr is None:
            raise ValueError("E3_2 requires ESMDance pair edge_attr")
        h, aux = self.layer(h, graph.edge_index.long(), edge_attr)
        return self.pool(h, graph.batch), h, aux


class ESM2ESMDanceNodeEncoder(BaseProteinEncoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fuse = nn.Sequential(nn.LayerNorm(self.hidden_dim * 2), nn.Linear(self.hidden_dim * 2, self.hidden_dim), nn.ReLU())
        self.gate = GatedResidual(self.hidden_dim)

    def forward(self, graph, return_embeddings: bool = False) -> tuple[torch.Tensor, torch.Tensor, dict]:
        h_esm = self.project_esm(graph)
        h_dyn = self.project_dyn(graph)
        h_new = self.fuse(torch.cat([h_esm, h_dyn], dim=-1))
        h, aux = self.gate(h_esm, h_new)
        if return_embeddings:
            aux = dict(aux)
            aux.update(_dense_embedding_aux(graph, h_esm=h_esm, h_dyn=h_dyn, h_new=h_new, h_fused=h))
        return self.pool(h, graph.batch), h, aux


def _diag_gaussian_entropy(var: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.log(2.0 * torch.pi * torch.e * var).mean()


def _diag_gaussian_kl_to_standard(mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    return 0.5 * (mu.square() + var - var.log() - 1.0).mean()


def gaussian_poe_fusion(
    mu_esm: torch.Tensor,
    logvar_esm: torch.Tensor,
    mu_dyn: torch.Tensor,
    logvar_dyn: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-8,
    gaussian_correlation_mode: str = "independent",
    rho: torch.Tensor | float | None = None,
    logvar_min: float = -5.0,
    logvar_max: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Closed-form fusion for two scalar Gaussian estimates.

    independent is the usual product-of-experts. global_learnable uses the
    minimum-variance unbiased fusion formula for two correlated estimates.
    """
    if mu_esm.shape != mu_dyn.shape:
        raise ValueError(f"mu shape mismatch: mu_esm={tuple(mu_esm.shape)} mu_dyn={tuple(mu_dyn.shape)}")
    if logvar_esm.shape != logvar_dyn.shape:
        raise ValueError(f"logvar shape mismatch: esm={tuple(logvar_esm.shape)} dyn={tuple(logvar_dyn.shape)}")
    if logvar_esm.shape[:-1] != mu_esm.shape[:-1] or logvar_esm.shape[-1] != 1:
        raise ValueError(f"expected scalar logvar [...,1] aligned with mu, got {tuple(logvar_esm.shape)} and {tuple(mu_esm.shape)}")

    if gaussian_correlation_mode == "independent":
        precision_esm = torch.exp(-logvar_esm)
        precision_dyn = torch.exp(-logvar_dyn)
        precision_sum = precision_esm + precision_dyn + eps
        w_esm = precision_esm / precision_sum
        w_dyn = precision_dyn / precision_sum
        mu_fused = w_esm * mu_esm + w_dyn * mu_dyn
        var_fused = 1.0 / precision_sum
        cov = None
    elif gaussian_correlation_mode == "global_learnable":
        if rho is None:
            raise ValueError("global_learnable gaussian fusion requires rho")
        rho_t = torch.as_tensor(rho, dtype=mu_esm.dtype, device=mu_esm.device)
        var_esm = torch.exp(logvar_esm).clamp_min(eps)
        var_dyn = torch.exp(logvar_dyn).clamp_min(eps)
        std_esm = torch.sqrt(var_esm + eps)
        std_dyn = torch.sqrt(var_dyn + eps)
        cov = rho_t * std_esm * std_dyn
        denom = (var_esm + var_dyn - 2.0 * cov).clamp_min(eps)
        w_esm = (var_dyn - cov) / denom
        w_dyn = (var_esm - cov) / denom
        mu_fused = w_esm * mu_esm + w_dyn * mu_dyn
        var_fused = ((var_esm * var_dyn - cov.square()) / denom).clamp_min(eps)
    else:
        raise ValueError(f"unknown gaussian_correlation_mode={gaussian_correlation_mode!r}")

    logvar_fused = torch.log(var_fused + eps).clamp(min=logvar_min, max=logvar_max)

    if mask is not None:
        m = mask.to(dtype=mu_fused.dtype, device=mu_fused.device).unsqueeze(-1)
        mu_fused = mu_fused * m
        logvar_fused = logvar_fused * m
        w_esm = w_esm * m
        w_dyn = w_dyn * m
        if cov is not None:
            cov = cov * m
    return mu_fused, logvar_fused, w_esm, w_dyn, cov


def _init_scalar_logvar_head(head: nn.Sequential) -> None:
    last = head[-1]
    if isinstance(last, nn.Linear):
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)


def _protein_gaussian_pool(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    batch: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    if mu.ndim != 2 or logvar.ndim != 2 or logvar.shape[-1] != 1:
        raise ValueError(f"expected mu [N,H] and logvar [N,1], got {tuple(mu.shape)} and {tuple(logvar.shape)}")
    if batch.numel() == 0:
        return mu.new_zeros(0, mu.shape[-1]), logvar.new_zeros(0, 1)
    n_graphs = int(batch.max().item()) + 1
    counts = torch.bincount(batch, minlength=n_graphs).to(mu.dtype).clamp_min(1.0).unsqueeze(-1)

    mu_sum = mu.new_zeros(n_graphs, mu.shape[-1])
    mu_sum.index_add_(0, batch, mu)
    mu_protein = mu_sum / counts

    var = torch.exp(logvar).clamp_min(eps)
    var_sum = var.new_zeros(n_graphs, 1)
    var_sum.index_add_(0, batch, var)
    var_protein = var_sum / counts.square()
    return mu_protein, torch.log(var_protein.clamp_min(eps))


def _scalar_gaussian_entropy_from_logvar(logvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * (torch.log(logvar.new_tensor(2.0 * torch.pi * torch.e)) + logvar)


class E4GaussianVarianceFusionEncoder(BaseProteinEncoder):
    """E4 variant: scalar-variance Gaussian PoE for ESM2/ESMDance node fusion."""

    def __init__(
        self,
        gaussian_output_mode: str = "residual_gaussian",
        gamma_init: float = 0.1,
        logvar_min: float = -5.0,
        logvar_max: float = 5.0,
        gaussian_correlation_mode: str = "independent",
        rho_init: float = 0.0,
        rho_max: float = 0.95,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.gaussian_output_mode = str(gaussian_output_mode)
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)
        self.gaussian_correlation_mode = str(gaussian_correlation_mode)
        self.rho_max = float(rho_max)
        if not 0.0 <= float(rho_init) < self.rho_max:
            raise ValueError(f"rho_init must be in [0, rho_max), got rho_init={rho_init} rho_max={rho_max}")
        rho_ratio = max(min(float(rho_init) / max(self.rho_max, 1e-8), 1.0 - 1e-6), 1e-6)
        self.rho_param = nn.Parameter(torch.tensor(torch.logit(torch.tensor(rho_ratio)).item()))
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        self.out_norm = nn.LayerNorm(self.hidden_dim)
        mid = max(1, self.hidden_dim // 2)
        self.var_head_esm = nn.Sequential(nn.Linear(self.hidden_dim, mid), nn.ReLU(), nn.Linear(mid, 1))
        self.var_head_dyn = nn.Sequential(nn.Linear(self.hidden_dim, mid), nn.ReLU(), nn.Linear(mid, 1))
        _init_scalar_logvar_head(self.var_head_esm)
        _init_scalar_logvar_head(self.var_head_dyn)

    def _valid_stats(self, graph, **values: torch.Tensor) -> dict[str, torch.Tensor]:
        aux: dict[str, torch.Tensor] = {}
        w_dyn = values["w_dyn"].detach().float().view(-1)
        w_esm = values["w_esm"].detach().float().view(-1)
        logvar_esm = values["logvar_esm"].detach().float().view(-1)
        logvar_dyn = values["logvar_dyn"].detach().float().view(-1)
        logvar_fused = values["logvar_fused"].detach().float().view(-1)
        var_esm = torch.exp(logvar_esm)
        var_dyn = torch.exp(logvar_dyn)
        var_fused = torch.exp(logvar_fused)
        entropy_esm = _scalar_gaussian_entropy_from_logvar(logvar_esm)
        entropy_dyn = _scalar_gaussian_entropy_from_logvar(logvar_dyn)
        entropy_fused = _scalar_gaussian_entropy_from_logvar(logvar_fused)

        aux.update({
            "w_esm_mean": w_esm.mean(),
            "w_dyn_mean": w_dyn.mean(),
            "w_esm_std": w_esm.std(unbiased=False),
            "w_dyn_std": w_dyn.std(unbiased=False),
            "w_dyn_q10": torch.quantile(w_dyn, 0.10),
            "w_dyn_q25": torch.quantile(w_dyn, 0.25),
            "w_dyn_q50": torch.quantile(w_dyn, 0.50),
            "w_dyn_q75": torch.quantile(w_dyn, 0.75),
            "w_dyn_q90": torch.quantile(w_dyn, 0.90),
            "logvar_esm_mean": logvar_esm.mean(),
            "logvar_dyn_mean": logvar_dyn.mean(),
            "logvar_fused_mean": logvar_fused.mean(),
            "var_esm_mean": var_esm.mean(),
            "var_dyn_mean": var_dyn.mean(),
            "var_fused_mean": var_fused.mean(),
            "entropy_esm_mean": entropy_esm.mean(),
            "entropy_dyn_mean": entropy_dyn.mean(),
            "entropy_fused_mean": entropy_fused.mean(),
            "precision_esm_mean": torch.exp(-logvar_esm).mean(),
            "precision_dyn_mean": torch.exp(-logvar_dyn).mean(),
            "gamma": self.gamma.detach().view(()),
            "h_esm_norm": values["h_esm"].detach().float().norm(dim=-1).mean(),
            "h_dyn_norm": values["h_dyn"].detach().float().norm(dim=-1).mean(),
            "h_out_norm": values["residue_emb"].detach().float().norm(dim=-1).mean(),
        })
        # Backward-compatible aliases used by existing logs/visualization code.
        aux["weight_esm_mean"] = aux["w_esm_mean"]
        aux["weight_dyn_mean"] = aux["w_dyn_mean"]
        aux["esm_var_mean"] = aux["var_esm_mean"]
        aux["dyn_var_mean"] = aux["var_dyn_mean"]
        aux["fused_var_mean"] = aux["var_fused_mean"]
        aux["esm_entropy"] = aux["entropy_esm_mean"]
        aux["dyn_entropy"] = aux["entropy_dyn_mean"]
        aux["fused_entropy"] = aux["entropy_fused_mean"]
        return aux

    def _rho(self) -> torch.Tensor:
        return self.rho_max * torch.sigmoid(self.rho_param)

    def forward(self, graph, return_embeddings: bool = False) -> tuple[torch.Tensor, torch.Tensor, dict]:
        h_esm = self.project_esm(graph)
        h_dyn = self.project_dyn(graph)
        if h_esm.shape != h_dyn.shape:
            raise ValueError(f"E4 Gaussian variance shape mismatch: h_esm={tuple(h_esm.shape)} h_dyn={tuple(h_dyn.shape)}")

        mu_esm = h_esm
        mu_dyn = h_dyn
        logvar_esm = self.var_head_esm(h_esm).clamp(min=self.logvar_min, max=self.logvar_max)
        logvar_dyn = self.var_head_dyn(h_dyn).clamp(min=self.logvar_min, max=self.logvar_max)
        rho = self._rho() if self.gaussian_correlation_mode == "global_learnable" else None
        mu_fused, logvar_fused, w_esm, w_dyn, cov = gaussian_poe_fusion(
            mu_esm,
            logvar_esm,
            mu_dyn,
            logvar_dyn,
            gaussian_correlation_mode=self.gaussian_correlation_mode,
            rho=rho,
            logvar_min=self.logvar_min,
            logvar_max=self.logvar_max,
        )

        if self.gaussian_output_mode == "strict_gaussian":
            residue_emb = mu_fused
        elif self.gaussian_output_mode == "residual_gaussian":
            residue_emb = self.out_norm(h_esm + self.gamma * (mu_fused - h_esm))
        else:
            raise ValueError(f"unknown gaussian_output_mode={self.gaussian_output_mode!r}")
        residue_emb = torch.nan_to_num(residue_emb, nan=0.0, posinf=0.0, neginf=0.0)
        protein_emb = self.pool(residue_emb, graph.batch)
        mu_protein, logvar_protein = _protein_gaussian_pool(mu_fused, logvar_fused, graph.batch)

        var_fused = torch.exp(logvar_fused).clamp_min(1e-8)
        kl = _diag_gaussian_kl_to_standard(mu_fused, var_fused)
        aux = {
            "kl_loss": kl,
            "gauss_kl": kl.detach(),
            "gaussian_output_mode": self.gaussian_output_mode,
            "correlation_mode": self.gaussian_correlation_mode,
            "rho": (rho.detach() if isinstance(rho, torch.Tensor) else h_esm.new_tensor(0.0)),
            "cov_mean": (cov.detach().float().mean() if isinstance(cov, torch.Tensor) else h_esm.new_tensor(0.0)),
        }
        aux.update(self._valid_stats(
            graph,
            h_esm=h_esm,
            h_dyn=h_dyn,
            logvar_esm=logvar_esm,
            logvar_dyn=logvar_dyn,
            logvar_fused=logvar_fused,
            w_esm=w_esm,
            w_dyn=w_dyn,
            residue_emb=residue_emb,
        ))
        if return_embeddings:
            aux.update(_dense_embedding_aux(
                graph,
                h_esm=h_esm,
                h_dyn=h_dyn,
                mu_esm=mu_esm,
                mu_dyn=mu_dyn,
                logvar_esm=logvar_esm,
                logvar_dyn=logvar_dyn,
                mu_fused=mu_fused,
                logvar_fused=logvar_fused,
                w_esm=w_esm,
                w_dyn=w_dyn,
                residue_emb=residue_emb,
                h_fused=residue_emb,
            ))
            aux["protein_emb"] = protein_emb.detach()
            aux["mu_protein"] = mu_protein.detach()
            aux["logvar_protein"] = logvar_protein.detach()
        return protein_emb, residue_emb, aux


class GaussianNodeFusionEncoder(BaseProteinEncoder):
    """E4.1/E4.2: uncertainty-aware Gaussian product-of-experts node fusion."""

    def __init__(
        self,
        min_var: float = 1e-4,
        max_var: float = 50.0,
        sample_fused: bool = False,
        pooling_type: str = "mean",
        fusion_ablation_mode: str = "full",
        tempered_poe: bool = False,
        semantic_anchor: bool = False,
        alpha_init: float = 0.1,
        alpha_hidden_dim: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.min_var = float(min_var)
        self.max_var = float(max_var)
        self.sample_fused = bool(sample_fused)
        self.pooling_type = str(pooling_type).lower()
        self.fusion_ablation_mode = str(fusion_ablation_mode).lower()
        self.tempered_poe = bool(tempered_poe)
        self.semantic_anchor = bool(semantic_anchor)
        self.alpha_init = float(alpha_init)
        if self.pooling_type not in {"mean", "drug_conditioned_attention"}:
            raise ValueError(
                "GaussianNodeFusionEncoder pooling_type must be 'mean' or "
                f"'drug_conditioned_attention', got {pooling_type!r}"
            )
        if self.fusion_ablation_mode not in {"full", "semantic_only", "concat"}:
            raise ValueError(
                "fusion_ablation_mode must be 'full', 'semantic_only', or "
                f"'concat', got {fusion_ablation_mode!r}"
            )
        self.esm_stats = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.dyn_stats = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.concat_fuse = (
            nn.Sequential(
                nn.LayerNorm(self.hidden_dim * 2),
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(float(kwargs.get("dropout", 0.2))),
            )
            if self.fusion_ablation_mode == "concat"
            else None
        )
        if self.tempered_poe:
            alpha_hidden = int(alpha_hidden_dim or max(32, self.hidden_dim // 2))
            self.alpha_head = nn.Sequential(
                nn.LayerNorm(self.hidden_dim * 2 + 2),
                nn.Linear(self.hidden_dim * 2 + 2, alpha_hidden),
                nn.ReLU(),
                nn.Linear(alpha_hidden, 1),
            )
            self._init_alpha_head(self.alpha_init)
        else:
            self.alpha_head = None
        self.drug_conditioned_pool = (
            DrugConditionedPooling(self.hidden_dim)
            if self.pooling_type == "drug_conditioned_attention"
            else None
        )

    def _init_alpha_head(self, alpha_init: float) -> None:
        alpha = min(max(float(alpha_init), 1e-4), 1.0 - 1e-4)
        last = self.alpha_head[-1] if self.alpha_head is not None else None
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.constant_(last.bias, float(torch.logit(torch.tensor(alpha))))

    def _stats(self, h: torch.Tensor, head: nn.Linear) -> tuple[torch.Tensor, torch.Tensor]:
        mu, raw_var = head(h).chunk(2, dim=-1)
        var = F.softplus(raw_var) + self.min_var
        var = var.clamp(min=self.min_var, max=self.max_var)
        return mu, var

    def _dynamic_coverage(self, graph, ref: torch.Tensor) -> torch.Tensor:
        mask = getattr(graph, "esmdance_embedding_mask", None)
        if isinstance(mask, torch.Tensor):
            if mask.ndim == 1:
                mask = mask.unsqueeze(-1)
            return mask.to(device=ref.device, dtype=ref.dtype).clamp(0.0, 1.0)
        return torch.ones(ref.shape[0], 1, device=ref.device, dtype=ref.dtype)

    def forward(
        self,
        graph,
        return_embeddings: bool = False,
        drug_emb: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        h_esm = self.project_esm(graph)
        h_dyn, dyn_aux = self.project_dyn(graph, return_aux=True)
        mu_esm, var_esm = self._stats(h_esm, self.esm_stats)
        mu_dyn, var_dyn = self._stats(h_dyn, self.dyn_stats)
        if self.semantic_anchor:
            mu_esm = h_esm

        precision_esm = 1.0 / var_esm
        precision_dyn = 1.0 / var_dyn
        dyn_coverage = self._dynamic_coverage(graph, h_dyn)
        if self.fusion_ablation_mode == "semantic_only":
            alpha_dyn = torch.zeros_like(dyn_coverage)
            tempered_precision_dyn = torch.zeros_like(precision_dyn)
            precision_sum = precision_esm.clamp_min(1e-8)
            var_fused = var_esm
            mu_fused = mu_esm
            weight_esm = torch.ones_like(precision_esm)
            weight_dyn = torch.zeros_like(precision_dyn)
            residue_emb = mu_esm
        elif self.fusion_ablation_mode == "concat":
            # Direct feature concatenation with the same input projections,
            # drug-conditioned pooling, and interaction head as the full model.
            alpha_dyn = torch.ones_like(dyn_coverage) * dyn_coverage
            tempered_precision_dyn = precision_dyn * dyn_coverage
            precision_sum = (precision_esm + tempered_precision_dyn).clamp_min(1e-8)
            var_fused = torch.ones_like(var_esm)
            if self.concat_fuse is None:
                raise RuntimeError("concat fusion module was not initialized")
            mu_fused = self.concat_fuse(torch.cat([h_esm, h_dyn], dim=-1))
            weight_esm = torch.full_like(precision_esm, float("nan"))
            weight_dyn = torch.full_like(precision_dyn, float("nan"))
            residue_emb = mu_fused
        else:
            if self.tempered_poe:
                logvar_esm_mean = torch.log(var_esm.clamp_min(1e-8)).mean(dim=-1, keepdim=True)
                logvar_dyn_mean = torch.log(var_dyn.clamp_min(1e-8)).mean(dim=-1, keepdim=True)
                alpha_in = torch.cat([h_esm, h_dyn, logvar_esm_mean, logvar_dyn_mean], dim=-1)
                alpha_dyn = torch.sigmoid(self.alpha_head(alpha_in)) * dyn_coverage
            else:
                # Standard Gaussian PoE: the auxiliary exponent is exactly one
                # on every covered residue and is not learnable.
                alpha_dyn = dyn_coverage
            tempered_precision_dyn = alpha_dyn * precision_dyn
            precision_sum = (precision_esm + tempered_precision_dyn).clamp_min(1e-8)
            # The precision-normalized mean must use the exact inverse precision.
            # Clipping is a numerical safeguard for the reported/sampled variance
            # only; using the clipped value in the mean would break PoE
            # normalization when an expert variance reaches a configured bound.
            inverse_precision_sum = 1.0 / precision_sum
            mu_fused = inverse_precision_sum * (precision_esm * mu_esm + tempered_precision_dyn * mu_dyn)
            var_fused = inverse_precision_sum.clamp(min=self.min_var, max=self.max_var)
            weight_esm = precision_esm / precision_sum
            weight_dyn = tempered_precision_dyn / precision_sum
            if self.training and self.sample_fused:
                residue_emb = mu_fused + torch.randn_like(mu_fused) * torch.sqrt(var_fused)
            else:
                residue_emb = mu_fused
        residue_emb = torch.nan_to_num(residue_emb, nan=0.0, posinf=0.0, neginf=0.0)

        kl = _diag_gaussian_kl_to_standard(mu_fused, var_fused)
        aux = {
            "kl_loss": kl,
            "gauss_kl": kl.detach(),
            "pooling_type": self.pooling_type,
            "fusion_ablation_mode": self.fusion_ablation_mode,
            "esm_var_mean": var_esm.mean().detach(),
            "dyn_var_mean": var_dyn.mean().detach(),
            "fused_var_mean": var_fused.mean().detach(),
            "esm_entropy": _diag_gaussian_entropy(var_esm).detach(),
            "dyn_entropy": _diag_gaussian_entropy(var_dyn).detach(),
            "fused_entropy": _diag_gaussian_entropy(var_fused).detach(),
            "weight_esm_mean": weight_esm.mean().detach(),
            "weight_dyn_mean": weight_dyn.mean().detach(),
            "alpha_dyn_mean": alpha_dyn.mean().detach(),
            "dynamic_coverage_mean": dyn_coverage.mean().detach(),
        }
        if dyn_aux:
            aux.update(dyn_aux)
        protein_emb_esm = self.pool(mu_esm, graph.batch)
        protein_emb_dyn = self.pool(mu_dyn * dyn_coverage, graph.batch)
        aux["protein_emb_esm"] = protein_emb_esm
        aux["protein_emb_dyn"] = protein_emb_dyn
        if self.fusion_ablation_mode != "concat":
            aux["weight_esm_sample"] = self.pool(weight_esm.mean(dim=-1, keepdim=True), graph.batch).squeeze(-1)
            aux["weight_dyn_sample"] = self.pool(weight_dyn.mean(dim=-1, keepdim=True), graph.batch).squeeze(-1)
        aux["alpha_dyn_sample"] = self.pool(alpha_dyn, graph.batch).squeeze(-1)
        aux["var_esm_sample"] = self.pool(var_esm.mean(dim=-1, keepdim=True), graph.batch).squeeze(-1)
        aux["var_dyn_sample"] = self.pool(var_dyn.mean(dim=-1, keepdim=True), graph.batch).squeeze(-1)
        aux["bias_dyn_sample"] = self.pool(((mu_dyn - mu_esm) ** 2).mean(dim=-1, keepdim=True), graph.batch).squeeze(-1)
        if "esmdance_condition_gate_node" in aux:
            aux["esmdance_condition_gate_sample"] = self.pool(
                aux["esmdance_condition_gate_node"].mean(dim=-1, keepdim=True),
                graph.batch,
            ).squeeze(-1)
        for node_key, sample_key in (
            ("esmdance35_node_norm", "esmdance35_norm_sample"),
            ("dynamic_property_node_norm", "dynamic_property_norm_sample"),
            ("conditioned_dyn_shift_node", "conditioned_dyn_shift_sample"),
        ):
            node_value = aux.get(node_key)
            if isinstance(node_value, torch.Tensor):
                aux[sample_key] = self.pool(node_value.unsqueeze(-1), graph.batch).squeeze(-1)
        if self.pooling_type == "mean":
            protein_emb = self.pool(residue_emb, graph.batch)
        else:
            if drug_emb is None:
                raise ValueError("drug_conditioned_attention pooling requires drug_emb")
            protein_emb, residue_attention, attention_sums = self.drug_conditioned_pool(
                drug_emb,
                residue_emb,
                graph.batch,
                valid_mask=_node_valid_mask(graph),
                return_attention_sums=True,
            )
            aux["residue_attention"] = residue_attention
            aux["residue_attention_sum_error"] = (attention_sums - 1.0).abs().max().detach()
            aux["residue_attention_min_sum"] = attention_sums.min().detach()
            aux["residue_attention_max_sum"] = attention_sums.max().detach()
        if return_embeddings:
            aux.update(_dense_embedding_aux(
                graph,
                h_esm=h_esm,
                h_dyn=h_dyn,
                h_fused=residue_emb,
                mu_fused=mu_fused,
                weight_esm=weight_esm,
                weight_dyn=weight_dyn,
                var_esm=var_esm,
                var_dyn=var_dyn,
            ))
        return protein_emb, residue_emb, aux


class JointGaussianNodeFusionEncoder(BaseProteinEncoder):
    """E4.3: keep E4's joint ESM2/ESMDance node fusion, then fuse uncertainties."""

    def __init__(
        self,
        min_var: float = 1e-4,
        max_var: float = 50.0,
        sample_fused: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.min_var = float(min_var)
        self.max_var = float(max_var)
        self.sample_fused = bool(sample_fused)
        self.fuse = nn.Sequential(nn.LayerNorm(self.hidden_dim * 2), nn.Linear(self.hidden_dim * 2, self.hidden_dim), nn.ReLU())
        self.esm_stats = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.joint_stats = nn.Linear(self.hidden_dim, self.hidden_dim * 2)

    def _stats(self, h: torch.Tensor, head: nn.Linear) -> tuple[torch.Tensor, torch.Tensor]:
        mu, raw_var = head(h).chunk(2, dim=-1)
        var = F.softplus(raw_var) + self.min_var
        var = var.clamp(min=self.min_var, max=self.max_var)
        return mu, var

    def forward(self, graph) -> tuple[torch.Tensor, torch.Tensor, dict]:
        h_esm = self.project_esm(graph)
        h_dyn = self.project_dyn(graph)
        h_joint = self.fuse(torch.cat([h_esm, h_dyn], dim=-1))
        if h_esm.shape != h_joint.shape:
            raise ValueError(f"E4_3 shape mismatch: h_esm={tuple(h_esm.shape)} h_joint={tuple(h_joint.shape)}")

        mu_esm, var_esm = self._stats(h_esm, self.esm_stats)
        mu_joint, var_joint = self._stats(h_joint, self.joint_stats)

        precision_esm = 1.0 / var_esm
        precision_joint = 1.0 / var_joint
        precision_sum = (precision_esm + precision_joint).clamp_min(1e-8)
        var_fused = (1.0 / precision_sum).clamp(min=self.min_var, max=self.max_var)
        mu_fused = var_fused * (precision_esm * mu_esm + precision_joint * mu_joint)
        weight_esm = precision_esm / precision_sum
        weight_joint = precision_joint / precision_sum

        if self.training and self.sample_fused:
            residue_emb = mu_fused + torch.randn_like(mu_fused) * torch.sqrt(var_fused)
        else:
            residue_emb = mu_fused
        residue_emb = torch.nan_to_num(residue_emb, nan=0.0, posinf=0.0, neginf=0.0)
        protein_emb = self.pool(residue_emb, graph.batch)

        kl = _diag_gaussian_kl_to_standard(mu_fused, var_fused)
        aux = {
            "kl_loss": kl,
            "gauss_kl": kl.detach(),
            "esm_var_mean": var_esm.mean().detach(),
            "dyn_var_mean": var_joint.mean().detach(),
            "joint_var_mean": var_joint.mean().detach(),
            "fused_var_mean": var_fused.mean().detach(),
            "esm_entropy": _diag_gaussian_entropy(var_esm).detach(),
            "dyn_entropy": _diag_gaussian_entropy(var_joint).detach(),
            "joint_entropy": _diag_gaussian_entropy(var_joint).detach(),
            "fused_entropy": _diag_gaussian_entropy(var_fused).detach(),
            "weight_esm_mean": weight_esm.mean().detach(),
            "weight_dyn_mean": weight_joint.mean().detach(),
            "weight_joint_mean": weight_joint.mean().detach(),
        }
        return protein_emb, residue_emb, aux


class UncertaintyResidualGateEncoder(BaseProteinEncoder):
    """E4.3: E4 joint node update with an interpretable uncertainty-based residual gate."""

    def __init__(
        self,
        logvar_min: float = -5.0,
        logvar_max: float = 5.0,
        temperature: float = 1.0,
        gamma_init: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)
        self.temperature = float(temperature)
        self.fuse = nn.Sequential(nn.LayerNorm(self.hidden_dim * 2), nn.Linear(self.hidden_dim * 2, self.hidden_dim), nn.ReLU())
        self.var_head_esm = nn.Linear(self.hidden_dim, 1)
        self.var_head_new = nn.Linear(self.hidden_dim, 1)
        self.bias = nn.Parameter(torch.zeros(1))
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        self.out_norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, graph, return_embeddings: bool = False) -> tuple[torch.Tensor, torch.Tensor, dict]:
        h_esm = self.project_esm(graph)
        h_dyn = self.project_dyn(graph)
        h_new = self.fuse(torch.cat([h_esm, h_dyn], dim=-1))
        if h_esm.shape != h_new.shape:
            raise ValueError(f"E4_3 shape mismatch: h_esm={tuple(h_esm.shape)} h_new={tuple(h_new.shape)}")

        logvar_esm = self.var_head_esm(h_esm).clamp(min=self.logvar_min, max=self.logvar_max)
        logvar_new = self.var_head_new(h_new).clamp(min=self.logvar_min, max=self.logvar_max)
        temp = max(self.temperature, 1e-6)
        w_new = torch.sigmoid((logvar_esm - logvar_new) / temp + self.bias)
        h_out = self.out_norm(h_esm + self.gamma * w_new * (h_new - h_esm))
        h_out = torch.nan_to_num(h_out, nan=0.0, posinf=0.0, neginf=0.0)
        protein_emb = self.pool(h_out, graph.batch)

        w_flat = w_new.detach().float().view(-1)
        aux = {
            "w_new_mean": w_flat.mean(),
            "w_new_std": w_flat.std(unbiased=False),
            "w_new_q10": torch.quantile(w_flat, 0.10),
            "w_new_q50": torch.quantile(w_flat, 0.50),
            "w_new_q90": torch.quantile(w_flat, 0.90),
            "logvar_esm_mean": logvar_esm.detach().mean(),
            "logvar_new_mean": logvar_new.detach().mean(),
            "precision_esm_mean": torch.exp(-logvar_esm.detach()).mean(),
            "precision_new_mean": torch.exp(-logvar_new.detach()).mean(),
            "gamma": self.gamma.detach().view(()),
            "h_esm_norm": h_esm.detach().float().norm(dim=-1).mean(),
            "h_new_norm": h_new.detach().float().norm(dim=-1).mean(),
            "h_out_norm": h_out.detach().float().norm(dim=-1).mean(),
        }
        if return_embeddings:
            aux.update(_dense_embedding_aux(
                graph,
                h_esm=h_esm,
                h_dyn=h_dyn,
                h_new=h_new,
                h_fused=h_out,
                h_out=h_out,
                w_new=w_new,
                logvar_esm=logvar_esm,
                logvar_new=logvar_new,
            ))
        return protein_emb, h_out, aux


class ESM2ESMDanceNodePairEncoder(BaseProteinEncoder):
    def __init__(self, edge_dim: int = 36, **kwargs):
        super().__init__(edge_dim=edge_dim, **kwargs)
        self.fuse = nn.Sequential(nn.LayerNorm(self.hidden_dim * 2), nn.Linear(self.hidden_dim * 2, self.hidden_dim), nn.ReLU())
        self.layer = EdgeAwareGraphLayer(self.hidden_dim, edge_dim=edge_dim, dropout=kwargs.get("dropout", 0.2))

    def forward(self, graph) -> tuple[torch.Tensor, torch.Tensor, dict]:
        h = self.fuse(torch.cat([self.project_esm(graph), self.project_dyn(graph)], dim=-1))
        h, aux = self.layer(h, graph.edge_index.long(), _edge_attr(graph))
        return self.pool(h, graph.batch), h, aux


class NaiveConcatEncoder(ESM2ESMDanceNodePairEncoder):
    pass


class GatedStructureDynamicFusionEncoder(BaseProteinEncoder):
    def __init__(self, edge_dim: int = 36, **kwargs):
        super().__init__(edge_dim=edge_dim, **kwargs)
        dropout = kwargs.get("dropout", 0.2)
        self.static_layer = EdgeAwareGraphLayer(self.hidden_dim, edge_dim=edge_dim, dropout=dropout)
        self.dynamic_layer = EdgeAwareGraphLayer(self.hidden_dim, edge_dim=edge_dim, dropout=dropout)
        self.beta = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Sigmoid(),
        )

    def forward(self, graph) -> tuple[torch.Tensor, torch.Tensor, dict]:
        edge_attr = _edge_attr(graph)
        h_static, _ = self.static_layer(self.project_esm(graph), graph.edge_index.long(), edge_attr)
        h_dynamic, _ = self.dynamic_layer(self.project_dyn(graph), graph.edge_index.long(), edge_attr)
        beta = self.beta(torch.cat([h_static, h_dynamic, h_static - h_dynamic], dim=-1))
        h = beta * h_static + (1.0 - beta) * h_dynamic
        aux = {
            "static_dynamic_beta_mean": beta.mean().detach(),
            "static_dynamic_beta_std": beta.std(unbiased=False).detach(),
        }
        return self.pool(h, graph.batch), h, aux


class FullModalityGraphEncoder(BaseProteinEncoder):
    """E10: ESM2 + ESMDance node with ESMFold edge attributes, no ESMDance pair edge feature."""

    def __init__(self, edge_dim: int = 23, node_fusion_type: str = "gated", **kwargs):
        super().__init__(edge_dim=edge_dim, **kwargs)
        self.node_fusion_type = node_fusion_type
        self.concat_fuse = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
        )
        self.node_gate = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Sigmoid(),
        )
        self.layer = EdgeAwareGraphLayer(self.hidden_dim, edge_dim=edge_dim, dropout=kwargs.get("dropout", 0.2))

    def forward(self, graph) -> tuple[torch.Tensor, torch.Tensor, dict]:
        h_esm = self.project_esm(graph)
        h_dyn = self.project_dyn(graph)
        if self.node_fusion_type == "concat":
            h = self.concat_fuse(torch.cat([h_esm, h_dyn], dim=-1))
            gate = None
        elif self.node_fusion_type == "gated":
            gate = self.node_gate(torch.cat([h_esm, h_dyn], dim=-1))
            h = gate * h_esm + (1.0 - gate) * h_dyn
        else:
            raise ValueError(f"unknown node_fusion_type={self.node_fusion_type!r}")
        h, layer_aux = self.layer(h, graph.edge_index.long(), _edge_attr(graph))
        aux = dict(layer_aux)
        if gate is not None:
            aux["node_fusion_gate"] = gate.detach()
            aux["node_fusion_gate_mean"] = gate.mean().detach()
        protein_emb = self.pool(h, graph.batch)
        aux["protein_emb"] = protein_emb.detach()
        return protein_emb, h, aux


def _position_encoding(batch: torch.Tensor, dim: int = 4) -> torch.Tensor:
    pieces = []
    device = batch.device
    for b in range(int(batch.max().item()) + 1 if batch.numel() else 0):
        idx = (batch == b).nonzero(as_tuple=False).view(-1)
        n = int(idx.numel())
        if n == 0:
            continue
        pos = torch.linspace(0.0, 1.0, n, device=device).unsqueeze(-1)
        enc = torch.cat([pos, pos.square(), torch.sin(2 * torch.pi * pos), torch.cos(2 * torch.pi * pos)], dim=-1)
        pieces.append((idx, enc[:, :dim]))
    out = torch.zeros(batch.shape[0], dim, device=device)
    for idx, enc in pieces:
        out[idx] = enc
    return out


class StructureDynamicPriorEncoder(BaseProteinEncoder):
    """E12: ESM2 semantic stream plus ESMFold/ESMDance structure-dynamic prior."""

    def __init__(
        self,
        edge_dim: int = 36,
        structure_fusion_type: str = "gated",
        posenc_dim: int = 4,
        **kwargs,
    ):
        super().__init__(edge_dim=edge_dim, **kwargs)
        self.structure_fusion_type = structure_fusion_type
        self.posenc_dim = posenc_dim
        self.plddt_proj = nn.Sequential(nn.LayerNorm(1), nn.Linear(1, self.hidden_dim), nn.ReLU())
        self.posenc_proj = nn.Sequential(nn.LayerNorm(posenc_dim), nn.Linear(posenc_dim, self.hidden_dim), nn.ReLU())
        self.struct_input = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 3),
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.ReLU(),
        )
        self.edge_gate = nn.Sequential(
            nn.LayerNorm(edge_dim),
            nn.Linear(edge_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, edge_dim),
            nn.Sigmoid(),
        )
        self.struct_layer = EdgeAwareGraphLayer(self.hidden_dim, edge_dim=edge_dim, dropout=kwargs.get("dropout", 0.2))
        self.fusion_gate = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Sigmoid(),
        )

    def forward(self, graph) -> tuple[torch.Tensor, torch.Tensor, dict]:
        h_seq = self.project_esm(graph)
        x_dyn = self.project_dyn(graph)
        plddt = getattr(graph, "plddt", None)
        if not isinstance(plddt, torch.Tensor):
            plddt = torch.zeros(h_seq.shape[0], 1, device=h_seq.device)
        plddt = torch.nan_to_num(plddt.float(), nan=0.0, posinf=1.0, neginf=0.0)
        posenc = _position_encoding(graph.batch, self.posenc_dim)
        h_struct_input = self.struct_input(torch.cat([x_dyn, self.plddt_proj(plddt), self.posenc_proj(posenc)], dim=-1))

        edge_attr = _edge_attr(graph)
        if edge_attr is None:
            edge_attr = torch.zeros(graph.edge_index.shape[1], self.struct_layer.edge_dim, device=h_seq.device)
        if edge_attr.shape[-1] != self.struct_layer.edge_dim:
            raise ValueError(f"E12 expected edge_dim={self.struct_layer.edge_dim}, got {edge_attr.shape[-1]}")
        edge_gate = self.edge_gate(edge_attr)
        h_struct, _ = self.struct_layer(h_struct_input, graph.edge_index.long(), edge_attr * edge_gate)

        if self.structure_fusion_type != "gated":
            raise ValueError("E12 first version supports structure_fusion_type='gated' only")
        gate = self.fusion_gate(torch.cat([h_seq, h_struct], dim=-1))
        h = gate * h_seq + (1.0 - gate) * h_struct
        h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
        protein_emb = self.pool(h, graph.batch)
        aux = {
            "edge_gate": edge_gate.detach(),
            "edge_gate_mean": edge_gate.mean().detach(),
            "structure_dynamic_gate": gate.detach(),
            "structure_dynamic_gate_mean": gate.mean().detach(),
            "h_seq": h_seq.detach(),
            "h_struct": h_struct.detach(),
        }
        return protein_emb, h, aux
