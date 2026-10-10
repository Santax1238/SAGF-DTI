from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .drug_conditioned_pooling import DrugConditionedPooling
from .drug_encoder import DrugEncoder
from .gsdti_dance_wrapper import GSDTIGaussianNodeFusionWrapper, GSDTIWithDanceNodeWrapper
from .protein_encoders import (
    ESM2ESMDanceNodeEncoder,
    ESM2ESMDanceNodePairEncoder,
    ESM2ESMFoldEdgeAttrEncoder,
    ESM2ESMFoldEdgeIndexEncoder,
    ESM2ESMDancePairEdgeEncoder,
    ESM2OnlyEncoder,
    E4GaussianVarianceFusionEncoder,
    ESMDanceOnlyEncoder,
    FullModalityGraphEncoder,
    GaussianNodeFusionEncoder,
    GatedStructureDynamicFusionEncoder,
    JointGaussianNodeFusionEncoder,
    NaiveConcatEncoder,
    StructureDynamicPriorEncoder,
    UncertaintyResidualGateEncoder,
)
from .protein_sdi_encoder import ProteinSDIEncoder


def _cfg(config: dict[str, Any] | None, key: str, default: Any) -> Any:
    if config is None:
        return default
    model_cfg = config.get("model", config)
    return model_cfg.get(key, default)


class OriginalGSDTIWrapper(nn.Module):
    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__()
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root))
        from model import GraphDTI_bi

        self.model = GraphDTI_bi(
            drug_dim=int(_cfg(config, "drug_dim", 2304)),
            prot_dim=int(_cfg(config, "esm_dim", 1280)),
            output_dim=int(_cfg(config, "output_dim", 2)),
            hidden_dim=int(_cfg(config, "hidden_dim", 256)),
            dropout=float(_cfg(config, "dropout", 0.3)),
            use_esmdance=False,
        )

    def forward(self, batch: dict) -> tuple[torch.Tensor, dict]:
        logits, drug_emb, prot_emb = self.model(batch["drug_feat"], batch["protein_graph"])
        return logits, {"drug_emb": drug_emb, "protein_emb": prot_emb}


class DTIModel(nn.Module):
    def __init__(self, experiment_id: str, config: dict[str, Any] | None = None):
        super().__init__()
        self.experiment_id = experiment_id.upper()
        self.config = config or {}
        self.fusion_type = str(self.config.get("fusion_type", _cfg(config, "fusion_type", "interaction_mlp")))
        self.hidden_dim = int(_cfg(config, "hidden_dim", 256))
        self.output_dim = int(_cfg(config, "output_dim", 2))
        self.edge_dim = int(_cfg(config, "edge_dim", 36))
        common = {
            "esm_dim": int(_cfg(config, "esm_dim", 1280)),
            "dyn_dim": int(_cfg(config, "dyn_dim", 50)),
            "esmdance35_dim": int(_cfg(config, "esmdance35_dim", 480)),
            "hidden_dim": self.hidden_dim,
            "dropout": float(_cfg(config, "dropout", 0.2)),
            "dyn_ablation_mode": str(_cfg(config, "dyn_ablation_mode", "none")),
            "dyn_pollution_ratio": float(_cfg(config, "dyn_pollution_ratio", 0.0)),
            "dyn_conditioned_on_esm35": bool(_cfg(config, "dyn_conditioned_on_esm35", False)),
        }
        if self.experiment_id == "E0":
            self.original = OriginalGSDTIWrapper(config)
            return
        if self.experiment_id == "E11":
            self.original_dance = GSDTIWithDanceNodeWrapper(
                drug_dim=int(_cfg(config, "drug_dim", 2304)),
                esm_dim=int(_cfg(config, "esm_dim", 1280)),
                dyn_dim=int(_cfg(config, "dyn_dim", 50)),
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim,
                dropout=float(_cfg(config, "dropout", 0.3)),
            )
            return
        if self.experiment_id == "E4_GSDTI_GAUSS":
            self.gsdti_gauss = GSDTIGaussianNodeFusionWrapper(
                drug_dim=int(_cfg(config, "drug_dim", 2304)),
                esm_dim=int(_cfg(config, "esm_dim", 1280)),
                dyn_dim=int(_cfg(config, "dyn_dim", 50)),
                output_dim=self.output_dim,
                dropout=float(_cfg(config, "dropout", 0.2)),
                logvar_min=float(_cfg(config, "logvar_min", -5.0)),
                logvar_max=float(_cfg(config, "logvar_max", 5.0)),
                gaussian_output_mode=str(_cfg(config, "gaussian_output_mode", "residual_gaussian")),
                gamma_init=float(_cfg(config, "gamma_init", 0.1)),
            )
            return
        self.drug_encoder = DrugEncoder(int(_cfg(config, "drug_dim", 2304)), self.hidden_dim, float(_cfg(config, "dropout", 0.2)))
        self.drug_pool = None
        if self.experiment_id == "E1":
            self.protein_encoder = ESM2OnlyEncoder(**common)
        elif self.experiment_id in {"E4_DYN", "E4DYN", "ESMDANCE_ONLY"}:
            self.protein_encoder = ESMDanceOnlyEncoder(**common)
        elif self.experiment_id == "E2":
            self.protein_encoder = ESM2ESMFoldEdgeIndexEncoder(**common)
        elif self.experiment_id == "E3":
            self.protein_encoder = ESM2ESMFoldEdgeAttrEncoder(edge_dim=int(_cfg(config, "esmfold_edge_dim", 23)), **common)
        elif self.experiment_id in {"E3_2", "E3.2"}:
            self.protein_encoder = ESM2ESMDancePairEdgeEncoder(edge_dim=int(_cfg(config, "pair_edge_dim", 13)), **common)
        elif self.experiment_id == "E4":
            self.protein_encoder = ESM2ESMDanceNodeEncoder(**common)
        elif self.experiment_id in {"E4_GAUSSIAN_VARIANCE", "E4_GAUSSIAN_VARIANCE_STRICT"}:
            self.protein_encoder = E4GaussianVarianceFusionEncoder(
                gaussian_output_mode=str(_cfg(config, "gaussian_output_mode", "residual_gaussian")),
                gamma_init=float(_cfg(config, "gamma_init", 0.1)),
                logvar_min=float(_cfg(config, "logvar_min", -5.0)),
                logvar_max=float(_cfg(config, "logvar_max", 5.0)),
                gaussian_correlation_mode=str(_cfg(config, "gaussian_correlation_mode", "independent")),
                rho_init=float(_cfg(config, "rho_init", 0.0)),
                rho_max=float(_cfg(config, "rho_max", 0.95)),
                **common,
            )
        elif self.experiment_id in {"E4_1", "E4.1", "E4_2", "E4.2", "E4_GAUSS_NODE_FUSION"}:
            self.protein_encoder = GaussianNodeFusionEncoder(
                min_var=float(_cfg(config, "gaussian_min_var", 1e-4)),
                max_var=float(_cfg(config, "gaussian_max_var", 50.0)),
                sample_fused=bool(_cfg(config, "sample_fused", False)),
                pooling_type=str(self.config.get("pooling_type", _cfg(config, "pooling_type", "mean"))),
                fusion_ablation_mode=str(_cfg(config, "fusion_ablation_mode", "full")),
                tempered_poe=bool(_cfg(config, "gaussian_tempered_poe", False)),
                semantic_anchor=bool(_cfg(config, "gaussian_semantic_anchor", False)),
                alpha_init=float(_cfg(config, "gaussian_alpha_init", 0.1)),
                alpha_hidden_dim=int(_cfg(config, "gaussian_alpha_hidden_dim", max(32, self.hidden_dim // 2))),
                **common,
            )
        elif self.experiment_id in {"E4_3", "E4.3"}:
            self.protein_encoder = UncertaintyResidualGateEncoder(
                logvar_min=float(_cfg(config, "logvar_min", -5.0)),
                logvar_max=float(_cfg(config, "logvar_max", 5.0)),
                temperature=float(_cfg(config, "uncertainty_temperature", 1.0)),
                gamma_init=float(_cfg(config, "gamma_init", 0.1)),
                **common,
            )
        elif self.experiment_id == "E5":
            self.protein_encoder = ESM2ESMDanceNodePairEncoder(edge_dim=self.edge_dim, **common)
        elif self.experiment_id == "E6":
            self.protein_encoder = NaiveConcatEncoder(edge_dim=int(_cfg(config, "esmfold_edge_dim", 23)), **common)
        elif self.experiment_id == "E7":
            self.protein_encoder = GatedStructureDynamicFusionEncoder(edge_dim=self.edge_dim, **common)
        elif self.experiment_id in {"E8", "E9"}:
            self.protein_encoder = ProteinSDIEncoder(
                edge_dim=self.edge_dim,
                layers=int(_cfg(config, "sdi_layers", 1)),
                attn_heads=int(_cfg(config, "attn_heads", 4)),
                log_attention_entropy=bool(_cfg(config, "log_attention_entropy", False)),
                **common,
            )
            if self.experiment_id == "E9":
                self.drug_pool = DrugConditionedPooling(self.hidden_dim)
        elif self.experiment_id == "E10":
            self.protein_encoder = FullModalityGraphEncoder(
                edge_dim=int(_cfg(config, "esmfold_edge_dim", 23)),
                node_fusion_type=str(self.config.get("node_fusion_type", _cfg(config, "node_fusion_type", "gated"))),
                **common,
            )
        elif self.experiment_id == "E12":
            self.protein_encoder = StructureDynamicPriorEncoder(
                edge_dim=self.edge_dim,
                structure_fusion_type=str(self.config.get("structure_fusion_type", _cfg(config, "structure_fusion_type", "gated"))),
                **common,
            )
        else:
            raise ValueError(f"Unknown experiment_id={experiment_id!r}")
        self.interaction_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(_cfg(config, "dropout", 0.2))),
            nn.Linear(self.hidden_dim, self.output_dim),
        )
        self.bilinear = nn.Bilinear(self.hidden_dim, self.hidden_dim, self.hidden_dim)
        self.bilinear_head = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(float(_cfg(config, "dropout", 0.2))),
            nn.Linear(self.hidden_dim, self.output_dim),
        )
        # Keep both decoder modules in the state dict for compatibility with
        # existing checkpoints, but optimize only the decoder used by forward().
        if self.fusion_type == "bilinear":
            self.interaction_head.requires_grad_(False)
        else:
            self.bilinear.requires_grad_(False)
            self.bilinear_head.requires_grad_(False)
        uni_cfg = self.config.get("unimodal_aux", {}) or {}
        self.unimodal_aux_enabled = bool(uni_cfg.get("enabled", False))
        if self.unimodal_aux_enabled:
            self.aux_head_esm = nn.Sequential(
                nn.Linear(self.hidden_dim * 4, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(float(_cfg(config, "dropout", 0.2))),
                nn.Linear(self.hidden_dim, self.output_dim),
            )
            self.aux_head_dyn = nn.Sequential(
                nn.Linear(self.hidden_dim * 4, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(float(_cfg(config, "dropout", 0.2))),
                nn.Linear(self.hidden_dim, self.output_dim),
            )

    def _pair_features(self, drug_emb: torch.Tensor, protein_emb: torch.Tensor) -> torch.Tensor:
        pair_feat = torch.cat([drug_emb, protein_emb, drug_emb * protein_emb, torch.abs(drug_emb - protein_emb)], dim=-1)
        return torch.nan_to_num(pair_feat, nan=0.0, posinf=0.0, neginf=0.0)

    def forward(self, batch: dict, return_aux: bool = True, return_embeddings: bool = False) -> tuple[torch.Tensor, dict]:
        if self.experiment_id == "E0":
            return self.original(batch)
        if self.experiment_id == "E11":
            return self.original_dance(batch)
        if self.experiment_id == "E4_GSDTI_GAUSS":
            return self.gsdti_gauss(batch)
        graph = batch["protein_graph"]
        drug_emb = self.drug_encoder(batch["drug_feat"])
        protein_kwargs = {}
        if isinstance(self.protein_encoder, GaussianNodeFusionEncoder):
            protein_kwargs["drug_emb"] = drug_emb
        if return_embeddings and self.experiment_id in {
            "E4",
            "E4_1",
            "E4.1",
            "E4_2",
            "E4.2",
            "E4_GAUSS_NODE_FUSION",
            "E4_3",
            "E4.3",
            "E4_GAUSSIAN_VARIANCE",
            "E4_GAUSSIAN_VARIANCE_STRICT",
        }:
            protein_emb, residue_emb, aux = self.protein_encoder(graph, return_embeddings=True, **protein_kwargs)
        else:
            protein_emb, residue_emb, aux = self.protein_encoder(graph, **protein_kwargs)
        if self.experiment_id == "E9":
            protein_emb, residue_attention = self.drug_pool(drug_emb, residue_emb, graph.batch)
            aux = dict(aux)
            aux["residue_attention"] = residue_attention
        drug_emb = torch.nan_to_num(drug_emb, nan=0.0, posinf=0.0, neginf=0.0)
        protein_emb = torch.nan_to_num(protein_emb, nan=0.0, posinf=0.0, neginf=0.0)
        if drug_emb.shape != protein_emb.shape:
            raise ValueError(f"drug/protein embedding shape mismatch: {tuple(drug_emb.shape)} vs {tuple(protein_emb.shape)}")
        if self.fusion_type == "bilinear":
            pair_emb = self.bilinear(drug_emb, protein_emb)
            logits = self.bilinear_head(pair_emb)
        else:
            logits = self.interaction_head(self._pair_features(drug_emb, protein_emb))
        logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
        if self.unimodal_aux_enabled:
            protein_emb_esm = aux.get("protein_emb_esm")
            protein_emb_dyn = aux.get("protein_emb_dyn")
            if isinstance(protein_emb_esm, torch.Tensor) and isinstance(protein_emb_dyn, torch.Tensor):
                protein_emb_esm = torch.nan_to_num(protein_emb_esm, nan=0.0, posinf=0.0, neginf=0.0)
                protein_emb_dyn = torch.nan_to_num(protein_emb_dyn, nan=0.0, posinf=0.0, neginf=0.0)
                aux["logits_esm"] = torch.nan_to_num(
                    self.aux_head_esm(self._pair_features(drug_emb, protein_emb_esm)),
                    nan=0.0,
                    posinf=20.0,
                    neginf=-20.0,
                )
                aux["logits_dyn"] = torch.nan_to_num(
                    self.aux_head_dyn(self._pair_features(drug_emb, protein_emb_dyn)),
                    nan=0.0,
                    posinf=20.0,
                    neginf=-20.0,
                )
        aux.setdefault("drug_emb", drug_emb)
        aux.setdefault("protein_emb", protein_emb)
        aux.setdefault("fusion_type", self.fusion_type)
        if return_embeddings:
            aux.setdefault("protein_id", batch.get("sequence_hash", []))
        return logits, aux
