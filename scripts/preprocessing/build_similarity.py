#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build BioFusionDTI kNN similarity matrices for contrastive loss.")
    p.add_argument("--feature-root", type=Path, default=Path("my_experiments/data/Bio/features"))
    p.add_argument("--view", default="biofusion1500_front")
    p.add_argument("--graph-version", default="esmdance_node_dyn50")
    p.add_argument("--datasets", nargs="*", default=["DRH", "Kinase", "SNAP"])
    p.add_argument("--top-frac", type=float, default=0.05)
    p.add_argument("--min-k", type=int, default=20)
    p.add_argument("--max-k", type=int, default=300)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x.astype("float32"), nan=0.0, posinf=0.0, neginf=0.0)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norm, 1e-8, None)


def knn_binary_similarity(x: np.ndarray, top_frac: float, min_k: int, max_k: int) -> np.ndarray:
    n = int(x.shape[0])
    if n <= 1:
        return np.eye(n, dtype=np.uint8)
    x = l2_normalize(x)
    sim = x @ x.T
    np.fill_diagonal(sim, -np.inf)
    k = min(n - 1, max(int(round(top_frac * (n - 1))), min_k), max_k)
    idx = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
    out = np.zeros((n, n), dtype=np.uint8)
    rows = np.arange(n)[:, None]
    out[rows, idx] = 1
    out = np.maximum(out, out.T)
    np.fill_diagonal(out, 1)
    return out


def tensor_from_graph(graph) -> torch.Tensor:
    for key in ("x_esm", "x", "esm2_embedding", "embedding"):
        value = getattr(graph, key, None)
        if isinstance(value, torch.Tensor):
            return value.float()
    if isinstance(graph, dict):
        for key in ("x_esm", "x", "esm2_embedding", "embedding"):
            value = graph.get(key)
            if isinstance(value, torch.Tensor):
                return value.float()
    raise ValueError(f"cannot find ESM2 tensor in graph object {type(graph)}")


def target_embeddings(targets_csv: Path, graph_dir: Path) -> np.ndarray:
    targets = pd.read_csv(targets_csv).drop_duplicates("Target_ID").reset_index(drop=True)
    embs: list[np.ndarray] = []
    missing: list[str] = []
    for tid in targets["Target_ID"].astype(str):
        path = graph_dir / f"{tid}.pt"
        if not path.exists():
            missing.append(tid)
            continue
        graph = torch.load(path, map_location="cpu", weights_only=False)
        x = torch.nan_to_num(tensor_from_graph(graph), nan=0.0, posinf=0.0, neginf=0.0)
        embs.append(x.mean(dim=0).cpu().numpy().astype("float32"))
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} target graphs under {graph_dir}; first={missing[:5]}")
    return np.stack(embs, axis=0)


def main() -> int:
    args = parse_args()
    summaries = []
    for dataset in args.datasets:
        base = args.feature_root / dataset / args.view
        out_dir = base / "contrastive"
        out_dir.mkdir(parents=True, exist_ok=True)
        drug_out = out_dir / "drug_knn_sim.npz"
        target_out = out_dir / "target_knn_sim.npz"
        if drug_out.exists() and target_out.exists() and not args.force:
            print(f"[skip] {dataset}: existing {out_dir}")
            continue

        kpgt = np.load(base / "kpgt" / "kpgt.npz", allow_pickle=True)
        drug_sim = knn_binary_similarity(kpgt["fps"], args.top_frac, args.min_k, args.max_k)
        np.savez_compressed(drug_out, drug_sim)

        target_x = target_embeddings(base / "targets.csv", base / "graphs" / args.graph_version)
        target_sim = knn_binary_similarity(target_x, args.top_frac, args.min_k, args.max_k)
        np.savez_compressed(target_out, target_sim)

        summary = {
            "dataset": dataset,
            "drugs": int(drug_sim.shape[0]),
            "targets": int(target_sim.shape[0]),
            "drug_positive_rate": float((drug_sim.sum() - drug_sim.shape[0]) / max(1, drug_sim.size - drug_sim.shape[0])),
            "target_positive_rate": float((target_sim.sum() - target_sim.shape[0]) / max(1, target_sim.size - target_sim.shape[0])),
            "output_dir": str(out_dir),
        }
        (out_dir / "contrastive_summary.json").write_text(json.dumps(summary, indent=2))
        summaries.append(summary)
        print(json.dumps(summary, indent=2))
    print(json.dumps({"datasets": summaries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
