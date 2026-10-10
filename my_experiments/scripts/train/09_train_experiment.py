#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import numpy as np
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from datasets import DTIDataset, collate_dti
from loss import NTXentContrastiveLoss
from models import DTIModel
from phase4_utils import compute_binary_metrics, set_seed


GRAPH_VERSION = {
    "E0": "esmfold_edge_index",
    "E1": "esm2_only",
    "E2": "esmfold_edge_index",
    "E3": "esmfold_edge_attr",
    "E3_2": "esmdance_pair_edge_attr",
    "E3.2": "esmdance_pair_edge_attr",
    "E4": "esmdance_node",
    "E4_1": "esmdance_node",
    "E4.1": "esmdance_node",
    "E4_2": "esmdance_node",
    "E4.2": "esmdance_node",
    "E4_GAUSS_NODE_FUSION": "esmdance_node",
    "E4_GSDTI_GAUSS": "esmdance_node",
    "E4_3": "esmdance_node",
    "E4.3": "esmdance_node",
    "E4_GAUSSIAN_VARIANCE": "esmdance_node",
    "E4_GAUSSIAN_VARIANCE_STRICT": "esmdance_node",
    "E4_DYN": "esmdance_node",
    "E4DYN": "esmdance_node",
    "ESMDANCE_ONLY": "esmdance_node",
    "E5": "esmdance_node_pair",
    "E6": "naive_concat",
    "E7": "sdi_full",
    "E8": "sdi_full",
    "E9": "sdi_full",
    "E10": "naive_concat",
    "E11": "esmdance_node",
    "E12": "sdi_full",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train one E0-E9 DTI experiment.")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--experiment", default=None)
    p.add_argument("--run-name", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-batches", type=int, default=None)
    return p.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return yaml.safe_load(f) or {}


def merge_config(base: dict[str, Any], exp: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in exp.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            merged = dict(out[key])
            merged.update(value)
            out[key] = merged
        else:
            out[key] = value
    return out


def experiment_config_path(experiment: str) -> Path:
    return ROOT / "configs" / "experiments" / "e0_e9" / f"{experiment.lower()}.yaml"


def load_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_yaml(args.config)
    exp_id = (args.experiment or cfg.get("experiment_id") or "").upper()
    if args.experiment and exp_id:
        exp_path = experiment_config_path(exp_id)
        if exp_path.exists():
            cfg = merge_config(cfg, load_yaml(exp_path))
    if args.experiment:
        cfg["experiment_id"] = args.experiment.upper()
    if args.seed is not None:
        cfg["seed"] = args.seed
    return cfg


def filter_existing_graphs(dataset: DTIDataset) -> DTIDataset:
    before = len(dataset.rows)
    graph_ok = dataset.rows["sequence_hash"].astype(str).map(dataset.has_graph)
    after_graph = int(graph_ok.sum())
    if after_graph < before:
        print(f"[data] filtered missing graphs: {before-after_graph} pairs removed from {dataset.split_csv}")
    rows = dataset.rows[graph_ok].copy()

    before_drug = len(rows)
    drug_ok = rows["drug_id"].astype(str).map(lambda d: d in dataset.drug_feats)
    rows = rows[drug_ok].reset_index(drop=True)
    after_drug = len(rows)
    if after_drug < before_drug:
        print(f"[data] filtered missing drug features: {before_drug-after_drug} pairs removed from {dataset.split_csv}")

    dataset.rows = rows
    return dataset


def make_loader(cfg: dict[str, Any], split: str, graph_version: str, batch_size: int, shuffle: bool) -> DataLoader:
    data_cfg = cfg.get("data", {})
    if split == "train":
        csv_path = data_cfg.get("train_csv", "my_experiments/processed/splits/bindingdb_train.csv")
        drug_npz = data_cfg.get("bindingdb_drug_npz", "my_experiments/data/bindingdb/drugs/kpgt.npz")
    elif split == "val":
        csv_path = data_cfg.get("val_csv", "my_experiments/processed/splits/bindingdb_val.csv")
        drug_npz = data_cfg.get("bindingdb_drug_npz", "my_experiments/data/bindingdb/drugs/kpgt.npz")
    elif split == "test":
        csv_path = data_cfg.get("test_csv", "my_experiments/processed/splits/davis_test.csv")
        drug_npz = data_cfg.get("davis_drug_npz", "my_experiments/data/davis/drugs/kpgt.npz")
    else:
        raise ValueError(split)
    ds = DTIDataset(csv_path, drug_npz, data_cfg.get("graph_root", "my_experiments/processed/graphs"), graph_version)
    ds = filter_existing_graphs(ds)
    num_workers = int(cfg.get("num_workers", 0))
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "collate_fn": collate_dti,
        "pin_memory": bool(cfg.get("pin_memory", True)),
    }
    # An isolated generator makes data order and worker seeds independent of
    # architecture-specific parameter initialization. This is especially
    # important for paired, same-seed ablations.
    loader_seed = cfg.get("loader_seed")
    if loader_seed is not None:
        split_offset = {"train": 0, "val": 100_000, "test": 200_000}[split]
        generator = torch.Generator()
        generator.manual_seed(int(loader_seed) + split_offset)
        kwargs["generator"] = generator
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(cfg.get("persistent_workers", True))
        kwargs["prefetch_factor"] = int(cfg.get("prefetch_factor", 2))
    return DataLoader(ds, **kwargs)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits.float(), labels.long(), reduction="none", label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


class FocalCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        class_weight: torch.Tensor | None = None,
        gamma: float = 1.0,
        eps: float = 1e-12,
    ):
        super().__init__()
        self.gamma = float(gamma)
        self.eps = float(eps)
        if class_weight is None:
            self.register_buffer("class_weight", None)
        else:
            self.register_buffer("class_weight", class_weight.float())

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2 or logits.shape[-1] != 2:
            raise ValueError(f"FocalCrossEntropyLoss expects logits [B,2], got {tuple(logits.shape)}")
        labels = labels.long()
        log_probs = F.log_softmax(logits.float(), dim=-1)
        log_pt = log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp().clamp(min=self.eps, max=1.0)
        if self.class_weight is None:
            alpha_t = torch.ones_like(pt)
        else:
            alpha_t = self.class_weight.to(logits.device)[labels]
        loss = -alpha_t * (1.0 - pt).pow(self.gamma) * log_pt
        return loss.sum() / alpha_t.sum().clamp_min(self.eps)


class BinaryBCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if logits.shape[-1] == 1:
            binary_logit = logits.view(-1)
        else:
            binary_logit = logits[:, 1] - logits[:, 0]
        return self.loss(binary_logit, labels.float())


def class_weights(positive_weight: float) -> torch.Tensor:
    return torch.tensor([1.0, float(positive_weight)], dtype=torch.float32)


def make_loss(cfg: dict[str, Any]) -> nn.Module:
    loss_cfg = cfg.get("loss", "bce")
    if isinstance(loss_cfg, dict):
        loss_name = str(loss_cfg.get("type", "bce")).lower()
        positive_weight = float(loss_cfg.get("positive_weight", cfg.get("positive_weight", 1.0)))
        focal_gamma = float(loss_cfg.get("gamma", loss_cfg.get("focal_gamma", cfg.get("focal_gamma", 2.0))))
    else:
        loss_name = str(loss_cfg).lower()
        positive_weight = float(cfg.get("positive_weight", 1.0))
        focal_gamma = float(cfg.get("focal_gamma", 2.0))
    if loss_name == "focal":
        return FocalCrossEntropyLoss(class_weights(positive_weight), focal_gamma)
    if loss_name in {"ce", "cross_entropy"}:
        return nn.CrossEntropyLoss(label_smoothing=float(cfg.get("label_smoothing", 0.0)))
    if loss_name in {"weighted_ce", "wce"}:
        return nn.CrossEntropyLoss(
            weight=class_weights(positive_weight),
            label_smoothing=float(cfg.get("label_smoothing", 0.0)),
        )
    return BinaryBCEWithLogitsLoss()


def _npz_matrix(path: str | Path) -> np.ndarray:
    z = np.load(path, allow_pickle=True)
    key = "arr_0" if "arr_0" in z.files else z.files[0]
    return z[key].astype("float32")


def make_contrastive_context(cfg: dict[str, Any], device: torch.device) -> dict[str, Any] | None:
    cl_cfg = cfg.get("contrastive", {})
    if not bool(cl_cfg.get("enabled", cfg.get("use_contrastive", False))):
        return None
    data_cfg = cfg.get("data", {})
    drugs_csv = Path(cl_cfg.get("drugs_csv", data_cfg.get("drugs_csv", "my_experiments/data/original_BindingDB/BindingDB/drugs/drugs.csv")))
    targets_csv = Path(cl_cfg.get("targets_csv", data_cfg.get("targets_csv", "my_experiments/data/original_BindingDB/BindingDB/targets/targets.csv")))
    drug_sim_npz = Path(cl_cfg.get("drug_sim_npz", data_cfg.get("drug_sim_npz", "my_experiments/data/original_BindingDB/BindingDB/drugs/drugsim_matrix.npz")))
    target_sim_npz = Path(cl_cfg.get("target_sim_npz", data_cfg.get("target_sim_npz", "my_experiments/data/original_BindingDB/BindingDB/targets/target_simmatrix.npz")))

    drug_df = pd.read_csv(drugs_csv)
    target_df = pd.read_csv(targets_csv)
    drug_ids = drug_df["Drug_ID"].astype(str).tolist()
    target_ids = target_df["Target_ID"].astype(str).tolist()
    drug_sim = _npz_matrix(drug_sim_npz)
    target_sim = _npz_matrix(target_sim_npz)
    if drug_sim.shape != (len(drug_ids), len(drug_ids)):
        raise ValueError(f"drug similarity shape {drug_sim.shape} does not match drugs_csv rows={len(drug_ids)}")
    if target_sim.shape != (len(target_ids), len(target_ids)):
        raise ValueError(f"target similarity shape {target_sim.shape} does not match targets_csv rows={len(target_ids)}")

    return {
        "drug_id_to_index": {d: i for i, d in enumerate(drug_ids)},
        "target_id_to_index": {t: i for i, t in enumerate(target_ids)},
        "drug_sim": drug_sim,
        "target_sim": target_sim,
        "drug_loss": NTXentContrastiveLoss(
            temperature=float(cl_cfg.get("temperature", 0.07)),
            sim_threshold=float(cl_cfg.get("drug_sim_threshold", 0.8)),
        ).to(device),
        "target_loss": NTXentContrastiveLoss(
            temperature=float(cl_cfg.get("temperature", 0.07)),
            sim_threshold=float(cl_cfg.get("target_sim_threshold", 0.5)),
        ).to(device),
        "drug_weight": float(cl_cfg.get("drug_weight", 0.05)),
        "target_weight": float(cl_cfg.get("target_weight", 0.05)),
    }


def contrastive_loss_from_batch(batch: dict, aux: dict, ctx: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    drug_emb = aux.get("drug_emb")
    protein_emb = aux.get("protein_emb")
    if not isinstance(drug_emb, torch.Tensor) or not isinstance(protein_emb, torch.Tensor):
        zero = torch.zeros((), device=device)
        return zero, {"loss_contrastive": zero.detach()}
    try:
        drug_idx = [ctx["drug_id_to_index"][str(x)] for x in batch["drug_id"]]
        target_idx = [ctx["target_id_to_index"][str(x)] for x in batch["sequence_hash"]]
    except KeyError as exc:
        raise KeyError(f"contrastive id not found in author similarity CSVs: {exc}") from exc
    drug_sim = torch.as_tensor(ctx["drug_sim"][np.ix_(drug_idx, drug_idx)], dtype=torch.float32, device=device)
    target_sim = torch.as_tensor(ctx["target_sim"][np.ix_(target_idx, target_idx)], dtype=torch.float32, device=device)
    drug_raw = ctx["drug_loss"](drug_emb, drug_sim)
    target_raw = ctx["target_loss"](protein_emb, target_sim)
    drug_loss = ctx["drug_weight"] * drug_raw
    target_loss = ctx["target_weight"] * target_raw
    total = drug_loss + target_loss
    stats = {
        "loss_contrastive": total.detach(),
        "loss_contrastive_drug": drug_loss.detach(),
        "loss_contrastive_target": target_loss.detach(),
        "loss_contrastive_drug_raw": drug_raw.detach(),
        "loss_contrastive_target_raw": target_raw.detach(),
    }
    return total, stats


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        "pair_id": batch["pair_id"],
        "drug_id": batch["drug_id"],
        "sequence_hash": batch["sequence_hash"],
        "drug_feat": batch["drug_feat"].to(device),
        "protein_graph": batch["protein_graph"].to(device),
        "label": batch["label"].to(device).long(),
    }


def logits_to_binary_score(logits: torch.Tensor) -> torch.Tensor:
    """Return the binary decision logit in float32 for stable ranking metrics."""
    if logits.shape[-1] == 1:
        return logits.view(-1).float()
    return logits[:, 1].float() - logits[:, 0].float()


def logits_to_prob(logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits_to_binary_score(logits))


def binary_bce_per_sample(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.shape[-1] == 1:
        binary_logit = logits.view(-1)
    else:
        binary_logit = logits[:, 1] - logits[:, 0]
    return F.binary_cross_entropy_with_logits(binary_logit.float(), labels.float(), reduction="none")


def gaussian_unimodal_aux_loss(aux: dict, labels: torch.Tensor, cfg: dict[str, Any]) -> tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
    uni_cfg = cfg.get("unimodal_aux", {}) or {}
    if not bool(uni_cfg.get("enabled", False)):
        return None, {}
    logits_esm = aux.get("logits_esm")
    logits_dyn = aux.get("logits_dyn")
    if not isinstance(logits_esm, torch.Tensor) or not isinstance(logits_dyn, torch.Tensor):
        return None, {}
    e_esm = binary_bce_per_sample(logits_esm, labels)
    e_dyn = binary_bce_per_sample(logits_dyn, labels)
    loss_uni = e_esm.mean() + e_dyn.mean()
    stats: dict[str, torch.Tensor] = {
        "loss_uni": loss_uni.detach(),
        "loss_esm_aux": e_esm.detach().mean(),
        "loss_dyn_aux": e_dyn.detach().mean(),
    }

    loss_rel = logits_esm.new_zeros(())
    w_esm = aux.get("weight_esm_sample")
    w_dyn = aux.get("weight_dyn_sample")
    if isinstance(w_esm, torch.Tensor) and isinstance(w_dyn, torch.Tensor):
        temp = float(uni_cfg.get("temperature", 1.0))
        target = torch.softmax(torch.stack([-e_esm.detach() / temp, -e_dyn.detach() / temp], dim=-1), dim=-1)
        pred = torch.stack([w_esm.float(), w_dyn.float()], dim=-1).clamp_min(1e-6)
        pred = pred / pred.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        loss_rel = F.kl_div(pred.log(), target, reduction="batchmean")
        stats.update({
            "loss_rel": loss_rel.detach(),
            "target_w_esm_mean": target[:, 0].detach().mean(),
            "target_w_dyn_mean": target[:, 1].detach().mean(),
            "sample_w_esm_mean": pred[:, 0].detach().mean(),
            "sample_w_dyn_mean": pred[:, 1].detach().mean(),
        })

    total = float(uni_cfg.get("lambda_uni", 0.2)) * loss_uni + float(uni_cfg.get("lambda_rel", 0.03)) * loss_rel
    stats["loss_aux_total"] = total.detach()
    return total, stats


def finite_pearson(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x.detach().float().view(-1)
    y = y.detach().float().view(-1)
    mask = torch.isfinite(x) & torch.isfinite(y)
    if int(mask.sum()) < 3:
        return x.new_tensor(float("nan"))
    x = x[mask]
    y = y[mask]
    x = x - x.mean()
    y = y - y.mean()
    denom = x.norm() * y.norm()
    if float(denom.detach().cpu()) <= 1e-12:
        return x.new_tensor(float("nan"))
    return (x * y).sum() / denom.clamp_min(1e-12)


def gaussian_training_diagnostics(aux: dict, labels: torch.Tensor, logits: torch.Tensor) -> dict[str, torch.Tensor]:
    stats: dict[str, torch.Tensor] = {}
    protein_emb = aux.get("protein_emb")
    protein_emb_esm = aux.get("protein_emb_esm")
    protein_emb_dyn = aux.get("protein_emb_dyn")
    if isinstance(protein_emb, torch.Tensor) and isinstance(protein_emb_esm, torch.Tensor):
        stats["semantic_drift_fused_esm"] = (1.0 - F.cosine_similarity(protein_emb.float(), protein_emb_esm.float(), dim=-1)).detach().mean()
    if isinstance(protein_emb, torch.Tensor) and isinstance(protein_emb_dyn, torch.Tensor):
        stats["semantic_drift_fused_dyn"] = (1.0 - F.cosine_similarity(protein_emb.float(), protein_emb_dyn.float(), dim=-1)).detach().mean()
    if isinstance(protein_emb_esm, torch.Tensor) and isinstance(protein_emb_dyn, torch.Tensor):
        stats["semantic_drift_esm_dyn"] = (1.0 - F.cosine_similarity(protein_emb_esm.float(), protein_emb_dyn.float(), dim=-1)).detach().mean()

    w_esm = aux.get("weight_esm_sample")
    w_dyn = aux.get("weight_dyn_sample")
    logits_esm = aux.get("logits_esm")
    logits_dyn = aux.get("logits_dyn")
    if isinstance(w_esm, torch.Tensor) and isinstance(logits_esm, torch.Tensor):
        loss_esm = binary_bce_per_sample(logits_esm, labels)
        stats["corr_w_esm_loss_esm"] = finite_pearson(w_esm, loss_esm)
    if isinstance(w_dyn, torch.Tensor) and isinstance(logits_dyn, torch.Tensor):
        loss_dyn = binary_bce_per_sample(logits_dyn, labels)
        stats["corr_w_dyn_loss_dyn"] = finite_pearson(w_dyn, loss_dyn)
    if isinstance(w_dyn, torch.Tensor):
        loss_fused = binary_bce_per_sample(logits, labels)
        stats["corr_w_dyn_loss_fused"] = finite_pearson(w_dyn, loss_fused)

    var_esm = aux.get("var_esm_sample")
    var_dyn = aux.get("var_dyn_sample")
    bias_dyn = aux.get("bias_dyn_sample")
    if all(isinstance(v, torch.Tensor) for v in (w_dyn, var_esm, var_dyn, bias_dyn)):
        bound = 2.0 * var_esm.float() / (var_esm.float() + var_dyn.float() + bias_dyn.float()).clamp_min(1e-8)
        margin = bound - w_dyn.float()
        ok = ((w_dyn.float() > 0.0) & (margin > 0.0)).float()
        stats["dynamic_safety_bound_mean"] = bound.detach().mean()
        stats["dynamic_safety_margin_mean"] = margin.detach().mean()
        stats["dynamic_safety_ok_rate"] = ok.detach().mean()
    alpha_sample = aux.get("alpha_dyn_sample")
    if isinstance(alpha_sample, torch.Tensor):
        stats["alpha_dyn_sample_mean"] = alpha_sample.detach().float().mean()
    extra_sample_keys = {
        "esmdance_condition_gate_sample": "esmdance_condition_gate_mean",
        "esmdance35_norm_sample": "esmdance35_norm_mean",
        "dynamic_property_norm_sample": "dynamic_property_norm_mean",
        "conditioned_dyn_shift_sample": "conditioned_dyn_shift_mean",
    }
    for sample_key, stat_key in extra_sample_keys.items():
        value = aux.get(sample_key)
        if isinstance(value, torch.Tensor):
            stats[stat_key] = value.detach().float().mean()
    return stats


def aux_to_float(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return float(value.detach().float().mean().cpu())
    try:
        return float(value)
    except Exception:
        return None


class CauchyHypergradientLR:
    """Cauchy-normalized hypergradient learning-rate modulation.

    The update uses the cosine between consecutive full-model gradients:

        cos(g_t, g_{t-1}) = <g_t, g_{t-1}> / (||g_t|| ||g_{t-1}||)

    The Cauchy-Schwarz normalization keeps the hypergradient signal bounded in
    [-1, 1], so it is much less sensitive to raw gradient scale.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        beta: float = 0.01,
        min_lr: float = 1e-6,
        max_lr: float = 2e-4,
        ema_alpha: float | None = None,
        missing_grad_policy: str = "skip",
        grad_dtype: str = "float32",
    ):
        self.optimizer = optimizer
        self.beta = float(beta)
        self.min_lr = float(min_lr)
        self.max_lr = float(max_lr)
        self.ema_alpha = None if ema_alpha is None else float(ema_alpha)
        self.missing_grad_policy = str(missing_grad_policy).lower()
        self.grad_dtype = str(grad_dtype).lower()
        if self.ema_alpha is not None and not 0.0 <= self.ema_alpha < 1.0:
            raise ValueError(f"ema_alpha must be in [0, 1), got {self.ema_alpha}")
        if self.missing_grad_policy not in {"skip", "zero"}:
            raise ValueError(f"missing_grad_policy must be 'skip' or 'zero', got {missing_grad_policy!r}")
        if self.grad_dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError(f"grad_dtype must be 'float32', 'float16', or 'bfloat16', got {grad_dtype!r}")
        self.prev_grads: list[torch.Tensor | None] | None = None
        self.ema_cos: float | None = None
        self.last_stats = {
            "hypergrad_cos": math.nan,
            "hypergrad_raw_cos": math.nan,
            "hypergrad_ema_cos": math.nan,
            "hypergrad_lr_scale": 1.0,
            "hypergrad_lr": optimizer.param_groups[0]["lr"],
        }

    def _current_grads(self, model: nn.Module) -> list[torch.Tensor | None]:
        dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[self.grad_dtype]
        grads: list[torch.Tensor | None] = []
        for param in model.parameters():
            if not param.requires_grad or param.grad is None:
                grads.append(None)
            else:
                grads.append(param.grad.detach().to(dtype=dtype, copy=True))
        return grads

    def _dot_and_norms(
        self,
        current: list[torch.Tensor | None],
        previous: list[torch.Tensor | None],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dot = torch.zeros((), device=device)
        cur_norm_sq = torch.zeros((), device=device)
        prev_norm_sq = torch.zeros((), device=device)
        for cur, prev in zip(current, previous):
            if self.missing_grad_policy == "skip":
                if cur is None:
                    continue
                cur_t = torch.nan_to_num(cur, nan=0.0, posinf=0.0, neginf=0.0).to(device)
                cur_norm_sq = cur_norm_sq + cur_t.pow(2).sum()
                if prev is not None:
                    prev_t = torch.nan_to_num(prev, nan=0.0, posinf=0.0, neginf=0.0).to(device)
                    dot = dot + (cur_t * prev_t).sum()
                    prev_norm_sq = prev_norm_sq + prev_t.pow(2).sum()
                continue

            if cur is not None:
                cur_t = torch.nan_to_num(cur, nan=0.0, posinf=0.0, neginf=0.0).to(device)
                cur_norm_sq = cur_norm_sq + cur_t.pow(2).sum()
            else:
                cur_t = None
            if prev is not None:
                prev_t = torch.nan_to_num(prev, nan=0.0, posinf=0.0, neginf=0.0).to(device)
                prev_norm_sq = prev_norm_sq + prev_t.pow(2).sum()
            else:
                prev_t = None
            if cur_t is not None and prev_t is not None:
                dot = dot + (cur_t * prev_t).sum()
        return dot, cur_norm_sq.sqrt(), prev_norm_sq.sqrt()

    def update(self, model: nn.Module, device: torch.device) -> dict[str, float]:
        current = self._current_grads(model)
        if self.prev_grads is None:
            self.prev_grads = current
            self.last_stats = {
                "hypergrad_cos": math.nan,
                "hypergrad_raw_cos": math.nan,
                "hypergrad_ema_cos": math.nan,
                "hypergrad_lr_scale": 1.0,
                "hypergrad_lr": self.optimizer.param_groups[0]["lr"],
            }
            return self.last_stats

        dot, cur_norm, prev_norm = self._dot_and_norms(current, self.prev_grads, device)
        denom = (cur_norm * prev_norm).clamp_min(1e-12)
        raw_cos = (dot / denom).clamp(min=-1.0, max=1.0)
        used_cos = raw_cos
        if torch.isfinite(raw_cos):
            raw_cos_value = float(raw_cos.detach().cpu())
            if self.ema_alpha is not None:
                if self.ema_cos is None or not math.isfinite(self.ema_cos):
                    self.ema_cos = raw_cos_value
                else:
                    self.ema_cos = self.ema_alpha * self.ema_cos + (1.0 - self.ema_alpha) * raw_cos_value
                used_cos = torch.tensor(self.ema_cos, device=device)
            scale = float(torch.exp(self.beta * used_cos).detach().cpu())
            for group in self.optimizer.param_groups:
                group["lr"] = min(self.max_lr, max(self.min_lr, float(group["lr"]) * scale))
        else:
            scale = 1.0
            raw_cos = torch.tensor(float("nan"), device=device)
            used_cos = torch.tensor(float("nan"), device=device)

        self.prev_grads = current
        self.last_stats = {
            "hypergrad_cos": float(used_cos.detach().cpu()),
            "hypergrad_raw_cos": float(raw_cos.detach().cpu()),
            "hypergrad_ema_cos": float(self.ema_cos) if self.ema_cos is not None else math.nan,
            "hypergrad_lr_scale": scale,
            "hypergrad_lr": self.optimizer.param_groups[0]["lr"],
        }
        return self.last_stats


def make_hypergradient(cfg: dict[str, Any], optimizer: torch.optim.Optimizer) -> CauchyHypergradientLR | None:
    hg_cfg = cfg.get("hypergradient", {})
    if not bool(hg_cfg.get("enabled", False)):
        return None
    mode = str(hg_cfg.get("mode", "cauchy_lr")).lower()
    if mode not in {"cauchy_lr", "cauchy_lr_ema"}:
        raise ValueError(f"unsupported hypergradient.mode={mode!r}; expected 'cauchy_lr' or 'cauchy_lr_ema'")
    ema_alpha = hg_cfg.get("ema_alpha")
    missing_grad_policy = str(hg_cfg.get("missing_grad_policy", "skip"))
    if mode == "cauchy_lr_ema":
        ema_alpha = float(0.9 if ema_alpha is None else ema_alpha)
        missing_grad_policy = str(hg_cfg.get("missing_grad_policy", "zero"))
    grad_dtype = str(hg_cfg.get("grad_dtype", "float32"))
    return CauchyHypergradientLR(
        optimizer,
        beta=float(hg_cfg.get("beta", 0.01)),
        min_lr=float(hg_cfg.get("min_lr", 1e-6)),
        max_lr=float(hg_cfg.get("max_lr", 2e-4)),
        ema_alpha=None if ema_alpha is None else float(ema_alpha),
        missing_grad_policy=missing_grad_policy,
        grad_dtype=grad_dtype,
    )


def make_scheduler(cfg: dict[str, Any], optimizer: torch.optim.Optimizer):
    name = str(cfg.get("lr_scheduler", cfg.get("scheduler", ""))).lower()
    if name not in {"reduce_on_plateau", "plateau", "reducelronplateau"}:
        return None
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=str(cfg.get("scheduler_mode", "min")),
        factor=float(cfg.get("scheduler_factor", 0.1)),
        patience=int(cfg.get("scheduler_patience", 5)),
        threshold=float(cfg.get("scheduler_threshold", 1e-4)),
        threshold_mode=str(cfg.get("scheduler_threshold_mode", "rel")),
        cooldown=int(cfg.get("scheduler_cooldown", 0)),
        min_lr=float(cfg.get("scheduler_min_lr", 1e-6)),
        eps=float(cfg.get("scheduler_eps", 1e-8)),
    )


def make_optimizer(cfg: dict[str, Any], model: nn.Module) -> torch.optim.Optimizer:
    name = str(cfg.get("optimizer", "adamw")).lower()
    lr = float(cfg.get("lr", 1e-4))
    weight_decay = float(cfg.get("weight_decay", 1e-4))
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=float(cfg.get("momentum", 0.9)))
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def scheduler_value(name: str, val_metrics: dict[str, float]) -> float:
    key = name.lower()
    if key in {"loss", "val_loss"}:
        return float(val_metrics["Loss"])
    metric_key = {
        "auroc": "AUROC",
        "val_auroc": "AUROC",
        "aupr": "AUPR",
        "val_aupr": "AUPR",
        "mcc": "MCC",
        "val_mcc": "MCC",
        "f1": "F1",
        "val_f1": "F1",
    }.get(key, name)
    return float(val_metrics[metric_key])


def residue_attention_entropy(attn: torch.Tensor | None) -> float | None:
    if attn is None or attn.numel() == 0:
        return None
    a = attn.detach().float().clamp_min(1e-8)
    return float((-(a * a.log())).mean().cpu())


def evaluate(
    model,
    loader,
    loss_fn,
    device,
    amp: bool = False,
    save_predictions: Path | None = None,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    y_true, y_prob, y_score, losses = [], [], [], []
    rows = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            batch = move_batch(batch, device)
            with torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                logits, _ = model(batch)
                loss = loss_fn(logits, batch["label"])
            binary_score = logits_to_binary_score(logits)
            prob = torch.sigmoid(binary_score)
            losses.append(float(loss.detach().cpu()))
            labels = batch["label"].detach().cpu().numpy().tolist()
            probs = prob.detach().cpu().numpy().tolist()
            scores = binary_score.detach().cpu().numpy().tolist()
            y_true.extend(labels)
            y_prob.extend(probs)
            y_score.extend(scores)
            if save_predictions is not None:
                for i, pair_id in enumerate(batch["pair_id"]):
                    rows.append({
                        "pair_id": pair_id,
                        "drug_id": batch["drug_id"][i],
                        "sequence_hash": batch["sequence_hash"][i],
                        "label": labels[i],
                        "pred_logit": scores[i],
                        "pred_prob": probs[i],
                    })
            if max_batches and batch_idx >= max_batches:
                break
    metrics = compute_binary_metrics(y_true, y_prob, y_score=y_score)
    metrics["Loss"] = float(sum(losses) / max(1, len(losses)))
    if save_predictions is not None:
        pd.DataFrame(rows).to_csv(save_predictions, index=False)
    return metrics


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    loss_fn,
    device,
    cfg,
    max_batches=None,
    contrastive_ctx=None,
    hypergradient: CauchyHypergradientLR | None = None,
) -> tuple[float, dict[str, float]]:
    model.train()
    total_loss, steps = 0.0, 0
    aux_acc: dict[str, list[float]] = {
        "alpha_seq": [],
        "alpha_graph": [],
        "static_dynamic_beta_mean": [],
        "residue_attention_entropy": [],
        "drug_emb_norm": [],
        "protein_emb_norm": [],
        "logits_abs_max": [],
        "gauss_kl": [],
        "esm_var_mean": [],
        "dyn_var_mean": [],
        "fused_var_mean": [],
        "esm_entropy": [],
        "dyn_entropy": [],
        "fused_entropy": [],
        "weight_esm_mean": [],
        "weight_dyn_mean": [],
        "alpha_dyn_mean": [],
        "dynamic_coverage_mean": [],
        "loss_uni": [],
        "loss_esm_aux": [],
        "loss_dyn_aux": [],
        "loss_rel": [],
        "loss_aux_total": [],
        "loss_contrastive": [],
        "loss_contrastive_drug": [],
        "loss_contrastive_target": [],
        "loss_contrastive_drug_raw": [],
        "loss_contrastive_target_raw": [],
        "semantic_drift_fused_esm": [],
        "semantic_drift_fused_dyn": [],
        "semantic_drift_esm_dyn": [],
        "corr_w_esm_loss_esm": [],
        "corr_w_dyn_loss_dyn": [],
        "corr_w_dyn_loss_fused": [],
        "dynamic_safety_bound_mean": [],
        "dynamic_safety_margin_mean": [],
        "dynamic_safety_ok_rate": [],
        "alpha_dyn_sample_mean": [],
        "esmdance_condition_gate_mean": [],
        "esmdance35_norm_mean": [],
        "dynamic_property_norm_mean": [],
        "conditioned_dyn_shift_mean": [],
        "target_w_esm_mean": [],
        "target_w_dyn_mean": [],
        "sample_w_esm_mean": [],
        "sample_w_dyn_mean": [],
        "joint_var_mean": [],
        "joint_entropy": [],
        "weight_joint_mean": [],
        "w_new_mean": [],
        "w_new_std": [],
        "w_new_q10": [],
        "w_new_q50": [],
        "w_new_q90": [],
        "w_esm_mean": [],
        "w_dyn_mean": [],
        "w_esm_std": [],
        "w_dyn_std": [],
        "w_dyn_q10": [],
        "w_dyn_q25": [],
        "w_dyn_q50": [],
        "w_dyn_q75": [],
        "w_dyn_q90": [],
        "logvar_esm_mean": [],
        "logvar_dyn_mean": [],
        "logvar_new_mean": [],
        "logvar_fused_mean": [],
        "var_esm_mean": [],
        "var_dyn_mean": [],
        "var_fused_mean": [],
        "entropy_esm_mean": [],
        "entropy_dyn_mean": [],
        "entropy_fused_mean": [],
        "precision_esm_mean": [],
        "precision_dyn_mean": [],
        "precision_new_mean": [],
        "gamma": [],
        "h_esm_norm": [],
        "h_dyn_norm": [],
        "h_new_norm": [],
        "h_out_norm": [],
        "rho": [],
        "cov_mean": [],
        "hypergrad_cos": [],
        "hypergrad_raw_cos": [],
        "hypergrad_ema_cos": [],
        "hypergrad_lr_scale": [],
        "hypergrad_lr": [],
    }
    amp = bool(cfg.get("amp", True))
    clip = float(cfg.get("grad_clip", 1.0))
    for batch_idx, batch in enumerate(loader, start=1):
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            logits, aux = model(batch)
            loss = loss_fn(logits, batch["label"])
            kl_weight = float(cfg.get("kl_weight", cfg.get("gaussian_kl_weight", 0.0)))
            use_gaussian_kl = bool(cfg.get("use_gaussian_kl", kl_weight > 0))
            if use_gaussian_kl and kl_weight > 0 and isinstance(aux.get("kl_loss"), torch.Tensor):
                loss = loss + kl_weight * aux["kl_loss"]
            aux_loss, aux_loss_stats = gaussian_unimodal_aux_loss(aux, batch["label"], cfg)
            if aux_loss is not None:
                loss = loss + aux_loss
                aux.update(aux_loss_stats)
            if contrastive_ctx is not None:
                cl_loss, cl_stats = contrastive_loss_from_batch(batch, aux, contrastive_ctx, device)
                loss = loss + cl_loss
                aux.update(cl_stats)
            aux.update(gaussian_training_diagnostics(aux, batch["label"], logits))
        scaler.scale(loss).backward()
        if clip > 0 or hypergradient is not None:
            scaler.unscale_(optimizer)
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        if hypergradient is not None:
            hg_stats = hypergradient.update(model, device)
            for key, value in hg_stats.items():
                if value is not None and math.isfinite(float(value)):
                    aux_acc[key].append(float(value))
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.detach().cpu())
        steps += 1
        for key in ("alpha_seq", "alpha_graph", "static_dynamic_beta_mean"):
            v = aux_to_float(aux.get(key))
            if v is not None:
                aux_acc[key].append(v)
        ent = residue_attention_entropy(aux.get("residue_attention"))
        if ent is not None:
            aux_acc["residue_attention_entropy"].append(ent)
        for key in ("drug_emb", "protein_emb"):
            v = aux.get(key)
            if isinstance(v, torch.Tensor):
                aux_acc[f"{key}_norm"].append(float(v.detach().float().norm(dim=-1).mean().cpu()))
        aux_acc["logits_abs_max"].append(float(logits.detach().abs().max().cpu()))
        for key in (
            "gauss_kl",
            "esm_var_mean",
            "dyn_var_mean",
            "fused_var_mean",
            "esm_entropy",
            "dyn_entropy",
            "fused_entropy",
            "weight_esm_mean",
            "weight_dyn_mean",
            "alpha_dyn_mean",
            "dynamic_coverage_mean",
            "loss_uni",
            "loss_esm_aux",
            "loss_dyn_aux",
            "loss_rel",
            "loss_aux_total",
            "loss_contrastive",
            "loss_contrastive_drug",
            "loss_contrastive_target",
            "loss_contrastive_drug_raw",
            "loss_contrastive_target_raw",
            "semantic_drift_fused_esm",
            "semantic_drift_fused_dyn",
            "semantic_drift_esm_dyn",
            "corr_w_esm_loss_esm",
            "corr_w_dyn_loss_dyn",
            "corr_w_dyn_loss_fused",
            "dynamic_safety_bound_mean",
            "dynamic_safety_margin_mean",
            "dynamic_safety_ok_rate",
            "alpha_dyn_sample_mean",
            "esmdance_condition_gate_mean",
            "esmdance35_norm_mean",
            "dynamic_property_norm_mean",
            "conditioned_dyn_shift_mean",
            "target_w_esm_mean",
            "target_w_dyn_mean",
            "sample_w_esm_mean",
            "sample_w_dyn_mean",
            "joint_var_mean",
            "joint_entropy",
            "weight_joint_mean",
            "w_new_mean",
            "w_new_std",
            "w_new_q10",
            "w_new_q50",
            "w_new_q90",
            "w_esm_mean",
            "w_dyn_mean",
            "w_esm_std",
            "w_dyn_std",
            "w_dyn_q10",
            "w_dyn_q25",
            "w_dyn_q50",
            "w_dyn_q75",
            "w_dyn_q90",
            "logvar_esm_mean",
            "logvar_dyn_mean",
            "logvar_new_mean",
            "logvar_fused_mean",
            "var_esm_mean",
            "var_dyn_mean",
            "var_fused_mean",
            "entropy_esm_mean",
            "entropy_dyn_mean",
            "entropy_fused_mean",
            "precision_esm_mean",
            "precision_dyn_mean",
            "precision_new_mean",
            "gamma",
            "h_esm_norm",
            "h_dyn_norm",
            "h_new_norm",
            "h_out_norm",
            "rho",
            "cov_mean",
        ):
            v = aux_to_float(aux.get(key))
            if v is not None:
                aux_acc.setdefault(key, []).append(v)
        if max_batches and batch_idx >= max_batches:
            break
    aux_mean = {k: (sum(v) / len(v) if v else math.nan) for k, v in aux_acc.items()}
    return total_loss / max(1, steps), aux_mean


def main() -> int:
    args = parse_args()
    cfg = load_config(args)
    experiment = str(cfg.get("experiment_id", args.experiment or "E1")).upper()
    graph_version = str(cfg.get("graph_version", GRAPH_VERSION[experiment]))
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    device = torch.device(args.device or cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    run_name = args.run_name or cfg.get("run_name") or f"{experiment.lower()}_seed{seed}"
    run_dir = Path(cfg.get("run_root", "my_experiments/runs")) / run_name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.config, run_dir / "base_config.yaml")
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    batch_size = int(cfg.get("batch_size", 8))
    train_loader = make_loader(cfg, "train", graph_version, batch_size, True)
    val_loader = make_loader(cfg, "val", graph_version, int(cfg.get("eval_batch_size", batch_size)), False)
    model = DTIModel(experiment, cfg).to(device)
    loss_fn = make_loss(cfg).to(device)
    contrastive_ctx = make_contrastive_context(cfg, device)
    if contrastive_ctx is not None:
        print("[train] contrastive enabled: drug_weight="
              f"{contrastive_ctx['drug_weight']} target_weight={contrastive_ctx['target_weight']}")
    optimizer = make_optimizer(cfg, model)
    scheduler = make_scheduler(cfg, optimizer)
    hypergradient = make_hypergradient(cfg, optimizer)
    if hypergradient is not None:
        hg_cfg = cfg.get("hypergradient", {})
        print("[train] cauchy hypergradient enabled: "
              f"mode={hg_cfg.get('mode', 'cauchy_lr')} beta={hg_cfg.get('beta', 0.01)} "
              f"min_lr={hg_cfg.get('min_lr', 1e-6)} max_lr={hg_cfg.get('max_lr', 2e-4)} "
              f"ema_alpha={hg_cfg.get('ema_alpha', '')} "
              f"missing_grad_policy={hg_cfg.get('missing_grad_policy', '')} "
              f"grad_dtype={hg_cfg.get('grad_dtype', 'float32')}")
    scaler = torch.amp.GradScaler(enabled=bool(cfg.get("amp", True)) and device.type == "cuda")

    if args.dry_run:
        train_loss, aux = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            loss_fn,
            device,
            cfg,
            max_batches=args.max_batches or 1,
            contrastive_ctx=contrastive_ctx,
            hypergradient=hypergradient,
        )
        val = evaluate(model, val_loader, loss_fn, device, amp=bool(cfg.get("amp", True)), max_batches=args.max_batches or 1)
        print({"dry_run": True, "train_loss": train_loss, "val": val, "aux": aux})
        return 0

    fields = ["epoch", "train_loss", "val_loss", "val_auroc", "val_aupr", "val_mcc", "val_f1", "val_acc", "val_bacc", "lr",
              "alpha_seq", "alpha_graph", "static_dynamic_beta_mean", "residue_attention_entropy",
              "drug_emb_norm", "protein_emb_norm", "logits_abs_max",
              "gauss_kl", "esm_var_mean", "dyn_var_mean", "fused_var_mean",
              "esm_entropy", "dyn_entropy", "fused_entropy", "weight_esm_mean", "weight_dyn_mean",
              "alpha_dyn_mean", "dynamic_coverage_mean",
              "loss_uni", "loss_esm_aux", "loss_dyn_aux", "loss_rel", "loss_aux_total",
              "loss_contrastive", "loss_contrastive_drug", "loss_contrastive_target",
              "loss_contrastive_drug_raw", "loss_contrastive_target_raw",
              "semantic_drift_fused_esm", "semantic_drift_fused_dyn", "semantic_drift_esm_dyn",
              "corr_w_esm_loss_esm", "corr_w_dyn_loss_dyn", "corr_w_dyn_loss_fused",
              "dynamic_safety_bound_mean", "dynamic_safety_margin_mean", "dynamic_safety_ok_rate",
              "alpha_dyn_sample_mean",
              "esmdance_condition_gate_mean", "esmdance35_norm_mean",
              "dynamic_property_norm_mean", "conditioned_dyn_shift_mean",
              "target_w_esm_mean", "target_w_dyn_mean", "sample_w_esm_mean", "sample_w_dyn_mean",
              "joint_var_mean", "joint_entropy", "weight_joint_mean",
              "w_new_mean", "w_new_std", "w_new_q10", "w_new_q50", "w_new_q90",
              "w_esm_mean", "w_dyn_mean", "w_esm_std", "w_dyn_std",
              "w_dyn_q10", "w_dyn_q25", "w_dyn_q50", "w_dyn_q75", "w_dyn_q90",
              "logvar_esm_mean", "logvar_dyn_mean", "logvar_new_mean", "logvar_fused_mean",
              "var_esm_mean", "var_dyn_mean", "var_fused_mean",
              "entropy_esm_mean", "entropy_dyn_mean", "entropy_fused_mean",
              "precision_esm_mean", "precision_dyn_mean", "precision_new_mean",
              "gamma", "h_esm_norm", "h_dyn_norm", "h_new_norm", "h_out_norm",
              "rho", "cov_mean",
              "hypergrad_cos", "hypergrad_raw_cos", "hypergrad_ema_cos",
              "hypergrad_lr_scale", "hypergrad_lr"]
    metrics_path = run_dir / "metrics.csv"
    log_path = run_dir / "train_log.txt"
    with metrics_path.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    best_metric_name = str(cfg.get("save_best_by", "AUPR"))
    best_metric_key = {
        "auroc": "AUROC",
        "aupr": "AUPR",
        "mcc": "MCC",
        "f1": "F1",
    }.get(best_metric_name.lower(), best_metric_name)
    best_score, best_epoch, patience_bad = -float("inf"), 0, 0
    epochs = int(cfg.get("epochs", 30))
    patience = int(cfg.get("patience", 8))
    max_batches = args.max_batches
    for epoch in range(1, epochs + 1):
        train_loss, aux = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            loss_fn,
            device,
            cfg,
            max_batches=max_batches,
            contrastive_ctx=contrastive_ctx,
            hypergradient=hypergradient,
        )
        val = evaluate(
            model,
            val_loader,
            loss_fn,
            device,
            amp=bool(cfg.get("amp", True)),
            save_predictions=run_dir / "val_predictions.csv",
            max_batches=max_batches,
        )
        if scheduler is not None:
            scheduler.step(scheduler_value(str(cfg.get("scheduler_metric", "val_loss")), val))
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val["Loss"],
            "val_auroc": val["AUROC"],
            "val_aupr": val["AUPR"],
            "val_mcc": val["MCC"],
            "val_f1": val["F1"],
            "val_acc": val["ACC"],
            "val_bacc": val.get("BACC", float("nan")),
            "lr": optimizer.param_groups[0]["lr"],
            **aux,
        }
        with metrics_path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)
        line = json.dumps(row, ensure_ascii=False)
        print(line)
        with log_path.open("a") as f:
            f.write(line + "\n")

        meta = {
            "experiment_id": experiment,
            "fusion_type": cfg.get("fusion_type", getattr(model, "fusion_type", "")),
            "protein_encoder_type": cfg.get("protein_encoder_type", type(getattr(model, "protein_encoder", model)).__name__),
            "pooling_type": cfg.get("pooling_type", ""),
            "node_fusion_type": cfg.get("node_fusion_type", ""),
            "structure_fusion_type": cfg.get("structure_fusion_type", ""),
            "gaussian_output_mode": cfg.get("gaussian_output_mode", ""),
            "variance_type": cfg.get("variance_type", ""),
            "gaussian_correlation_mode": cfg.get("gaussian_correlation_mode", ""),
        }
        val_with_meta = dict(val)
        val_with_meta.update(meta)
        state = {
            "model": model.state_dict(),
            "config": cfg,
            "experiment": experiment,
            "graph_version": graph_version,
            "epoch": epoch,
            "val_metrics": val_with_meta,
            "meta": meta,
        }
        torch.save(state, run_dir / "checkpoints" / "last.pt")
        current = float(val[best_metric_key]) if not math.isnan(float(val[best_metric_key])) else -float("inf")
        if current > best_score:
            best_score, best_epoch, patience_bad = current, epoch, 0
            torch.save(state, run_dir / "checkpoints" / "best.pt")
            (run_dir / "val_metrics.json").write_text(json.dumps(val_with_meta, indent=2))
        else:
            patience_bad += 1
        if patience_bad >= patience:
            print(f"[early-stop] best_epoch={best_epoch} best_val_{best_metric_key.lower()}={best_score:.6f}")
            break

    data_cfg = cfg.get("data", {})
    if data_cfg.get("test_csv"):
        best_path = run_dir / "checkpoints" / "best.pt"
        if best_path.exists():
            state = torch.load(best_path, map_location=device, weights_only=False)
            model.load_state_dict(state["model"])
        test_loader = make_loader(
            cfg,
            "test",
            graph_version,
            int(cfg.get("eval_batch_size", batch_size)),
            False,
        )
        test_metrics = evaluate(
            model,
            test_loader,
            loss_fn,
            device,
            amp=bool(cfg.get("amp", True)),
            save_predictions=run_dir / "test_predictions.csv",
            max_batches=max_batches,
        )
        test_metrics.update({
            "best_epoch": best_epoch,
            "best_val_metric": best_metric_key,
            "best_val_score": best_score,
        })
        (run_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))
        print("[test] " + json.dumps(test_metrics, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
