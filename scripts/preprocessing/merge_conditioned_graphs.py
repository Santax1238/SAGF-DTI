#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge BioFusion dynamic50 graphs with ESMDance/ESM2-35M residue embeddings."
    )
    p.add_argument("--feature-root", type=Path, default=Path("my_experiments/data/Bio/features"))
    p.add_argument("--view", default="biofusion1500_front")
    p.add_argument("--datasets", nargs="*", default=["DRH", "Kinase", "SNAP"])
    p.add_argument("--dynamic-version", default="esmdance_node_dyn50")
    p.add_argument("--esmdance35-version", default="esmdance_node")
    p.add_argument("--output-version", default="esmdance_node_cond35_dyn50")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def node_tensor(obj, names: tuple[str, ...]) -> torch.Tensor:
    for name in names:
        value = getattr(obj, name, None)
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().float()
        if isinstance(obj, dict):
            value = obj.get(name)
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().float()
    raise KeyError(f"none of {names} found in {type(obj)}")


def slice_node_aligned(data, n_old: int, n_new: int):
    if n_old == n_new:
        return data
    keep_edge = None
    if isinstance(getattr(data, "edge_index", None), torch.Tensor):
        edge_index = data.edge_index
        keep_edge = (edge_index[0] < n_new) & (edge_index[1] < n_new)
        data.edge_index = edge_index[:, keep_edge]
    for key in list(data.keys()):
        if key == "edge_index":
            continue
        value = getattr(data, key)
        if not isinstance(value, torch.Tensor):
            continue
        if value.ndim > 0 and value.shape[0] == n_old:
            setattr(data, key, value[:n_new])
        elif keep_edge is not None and value.ndim > 0 and value.shape[0] == keep_edge.numel():
            setattr(data, key, value[keep_edge])
    data.num_nodes = n_new
    if hasattr(data, "effective_len"):
        data.effective_len = min(int(data.effective_len), n_new)
    if hasattr(data, "full_sequence_len"):
        data.full_sequence_len = min(int(data.full_sequence_len), n_new)
    return data


def build_one(dynamic_path: Path, esmdance35_path: Path, out_path: Path, output_version: str) -> dict:
    dyn_graph = torch.load(dynamic_path, map_location="cpu", weights_only=False)
    old_graph = torch.load(esmdance35_path, map_location="cpu", weights_only=False)

    dyn50 = node_tensor(dyn_graph, ("predicted_dynamic_residue_features", "esmdance_residue_properties", "x_dyn"))
    x35 = node_tensor(old_graph, ("esmdance_embedding", "x_dyn", "embedding", "x"))
    x_esm = node_tensor(dyn_graph, ("x_esm", "x"))
    n_old = int(x_esm.shape[0])
    dyn50_old = int(dyn50.shape[0])
    x35_old = int(x35.shape[0])
    n = min(int(x_esm.shape[0]), int(dyn50.shape[0]), int(x35.shape[0]))
    dyn_graph = slice_node_aligned(dyn_graph, n_old, n)

    dyn50 = dyn50[:n].contiguous()
    x35 = x35[:n].contiguous()
    x_esm = x_esm[:n].contiguous()

    dyn_graph.x = x_esm
    dyn_graph.x_esm = x_esm
    dyn_graph.x_dyn = dyn50
    dyn_graph.predicted_dynamic_residue_features = dyn50
    dyn_graph.esmdance_residue_properties = dyn50
    dyn_graph.esmdance_35m_embedding = x35
    dyn_graph.esmdance35_embedding = x35
    dyn_graph.esmdance_encoder_embedding = x35
    dyn_graph.graph_version = output_version
    meta = getattr(dyn_graph, "esmdance_meta", {}) or {}
    if not isinstance(meta, dict):
        meta = {}
    meta.update({
        "conditioned_dynamic_graph": True,
        "dynamic_version": str(dynamic_path.parent.name),
        "esmdance35_version": str(esmdance35_path.parent.name),
        "dynamic50_dim": int(dyn50.shape[-1]),
        "esmdance35_dim": int(x35.shape[-1]),
    })
    dyn_graph.esmdance_meta = meta
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dyn_graph, out_path)
    return {
        "target": dynamic_path.stem,
        "nodes": n,
        "dynamic50_dim": int(dyn50.shape[-1]),
        "esmdance35_dim": int(x35.shape[-1]),
        "cropped": n != n_old or n != dyn50_old or n != x35_old,
    }


def main() -> int:
    args = parse_args()
    all_summary = []
    for dataset in args.datasets:
        base = args.feature_root / dataset / args.view / "graphs"
        dyn_dir = base / args.dynamic_version
        old_dir = base / args.esmdance35_version
        out_dir = base / args.output_version
        if not dyn_dir.exists():
            raise FileNotFoundError(dyn_dir)
        if not old_dir.exists():
            raise FileNotFoundError(old_dir)
        built = 0
        skipped = 0
        missing_35m = []
        cropped = 0
        for dynamic_path in sorted(dyn_dir.glob("*.pt")):
            out_path = out_dir / dynamic_path.name
            if out_path.exists() and not args.force:
                skipped += 1
                continue
            old_path = old_dir / dynamic_path.name
            if not old_path.exists():
                missing_35m.append(dynamic_path.stem)
                continue
            info = build_one(dynamic_path, old_path, out_path, args.output_version)
            built += 1
            cropped += int(bool(info["cropped"]))
        summary = {
            "dataset": dataset,
            "built": built,
            "skipped": skipped,
            "missing_35m": len(missing_35m),
            "cropped": cropped,
            "output_dir": str(out_dir),
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "conditioned_dynamic_summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        all_summary.append(summary)
        if missing_35m:
            print(json.dumps({"dataset": dataset, "missing_35m_first": missing_35m[:10]}, indent=2))
    print(json.dumps({"datasets": all_summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
