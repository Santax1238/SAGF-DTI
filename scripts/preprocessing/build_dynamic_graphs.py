#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import Data


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build Bio E4 esmdance_node PyG graphs from ESM2 and ESMDance residue features."
    )
    p.add_argument("--targets-csv", type=Path, required=True)
    p.add_argument("--esm2-dir", type=Path, required=True)
    p.add_argument("--esmdance-dir", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--graph-version", default="esmdance_node")
    p.add_argument("--max-len", type=int, default=1500)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def tensor_from_obj(obj, keys: tuple[str, ...]) -> torch.Tensor:
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, dict):
        for key in keys:
            value = obj.get(key)
            if isinstance(value, torch.Tensor):
                return value
    raise ValueError(f"cannot find tensor keys={keys} in object type {type(obj)}")


def load_esm(path: Path) -> torch.Tensor:
    return tensor_from_obj(torch.load(path, map_location="cpu", weights_only=False), ("x_esm", "embedding", "esm2_embedding", "x")).float()


def load_dyn(path: Path) -> tuple[torch.Tensor, torch.Tensor | None, dict]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    dyn = tensor_from_obj(
        obj,
        (
            "predicted_dynamic_residue_features",
            "esmdance_residue_properties",
            "x_dyn",
            "embedding",
            "esmdance_embedding",
        ),
    ).float()
    mask = None
    meta = {}
    if isinstance(obj, dict):
        raw_mask = obj.get("esmdance_embedding_mask")
        if isinstance(raw_mask, torch.Tensor):
            mask = raw_mask.float()
        meta = {
            k: obj.get(k)
            for k in ("raw_len", "effective_len", "covered_len", "is_partial", "feature_names", "feature_source")
            if k in obj
        }
    return dyn, mask, meta


def sequence_edges(n: int) -> torch.Tensor:
    if n <= 1:
        return torch.empty(2, 0, dtype=torch.long)
    src = torch.arange(n - 1, dtype=torch.long)
    dst = src + 1
    return torch.cat([torch.stack([src, dst]), torch.stack([dst, src])], dim=1)


def main() -> int:
    args = parse_args()
    out_dir = args.output_root / args.graph_version
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = pd.read_csv(args.targets_csv).drop_duplicates("Target_ID").reset_index(drop=True)

    written = skipped = missing_esm = missing_dyn = failed = partial = 0
    failures: list[tuple[str, str]] = []
    for row in targets.itertuples(index=False):
        tid = str(getattr(row, "Target_ID"))
        seq = str(getattr(row, "Target"))
        out_path = out_dir / f"{tid}.pt"
        if out_path.exists() and not args.force:
            skipped += 1
            continue
        esm_path = args.esm2_dir / f"{tid}.pt"
        dyn_path = args.esmdance_dir / f"{tid}.pt"
        if not esm_path.exists():
            missing_esm += 1
            continue
        if not dyn_path.exists():
            missing_dyn += 1
            continue
        try:
            x_esm = torch.nan_to_num(load_esm(esm_path), nan=0.0, posinf=0.0, neginf=0.0)
            x_dyn, mask, dyn_meta = load_dyn(dyn_path)
            x_dyn = torch.nan_to_num(x_dyn, nan=0.0, posinf=0.0, neginf=0.0)
            n = min(len(seq), args.max_len, int(x_esm.shape[0]), int(x_dyn.shape[0]))
            if n <= 0:
                raise ValueError("empty graph")
            x_esm = x_esm[:n].contiguous()
            x_dyn = x_dyn[:n].contiguous()
            if mask is None:
                mask = torch.ones(n, 1, dtype=torch.float32)
            else:
                mask = mask[:n].contiguous().float()
            if float(mask.mean()) < 1.0:
                partial += 1

            graph = Data(
                x=x_esm,
                x_esm=x_esm,
                x_dyn=x_dyn,
                predicted_dynamic_residue_features=x_dyn,
                esmdance_residue_properties=x_dyn,
                esmdance_embedding_mask=mask,
                edge_index=sequence_edges(n),
            )
            graph.target_id = tid
            graph.sequence_hash = tid
            graph.effective_len = int(n)
            graph.full_sequence_len = int(len(seq))
            graph.graph_version = args.graph_version
            graph.esmdance_meta = dyn_meta
            torch.save(graph, out_path)
            written += 1
        except Exception as exc:
            failed += 1
            failures.append((tid, f"{type(exc).__name__}: {exc}"))
            print(f"[WARN] {tid}: {exc}")

    summary = {
        "targets": int(len(targets)),
        "written": written,
        "skipped": skipped,
        "missing_esm2": missing_esm,
        "missing_esmdance": missing_dyn,
        "partial_esmdance_mask": partial,
        "failed": failed,
        "output_dir": str(out_dir),
    }
    (out_dir / "_summary.json").write_text(json.dumps(summary, indent=2))
    if failures:
        (out_dir / "_failures.tsv").write_text("\n".join(f"{tid}\t{msg}" for tid, msg in failures))
    print(json.dumps(summary, indent=2))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
