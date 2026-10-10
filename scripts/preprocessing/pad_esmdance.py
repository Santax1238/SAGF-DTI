#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pad Bio ESMDance predicted residue dynamic properties to a fixed graph/ESM2 length with a mask."
    )
    p.add_argument("--targets-csv", type=Path, required=True)
    p.add_argument("--input-dir", type=Path, required=True, help="Raw ESMDance .pt dir, usually <=1022 residues.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-len", type=int, default=1500)
    p.add_argument("--target-id-col", default="Target_ID")
    p.add_argument("--sequence-col", default="Target")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def tensor_from_obj(obj) -> torch.Tensor:
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, dict):
        for key in (
            "predicted_dynamic_residue_features",
            "esmdance_residue_properties",
            "x_dyn",
            "res_emb",
            "embedding",
            "esmdance_embedding",
        ):
            value = obj.get(key)
            if isinstance(value, torch.Tensor):
                return value
    raise ValueError(f"cannot find ESMDance tensor in object type {type(obj)}")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    targets = pd.read_csv(args.targets_csv).drop_duplicates(args.target_id_col).reset_index(drop=True)

    written = skipped = missing = failed = partial = 0
    failures: list[tuple[str, str]] = []
    for row in targets.itertuples(index=False):
        tid = str(getattr(row, args.target_id_col))
        seq = str(getattr(row, args.sequence_col)).strip()
        effective_len = min(len(seq), args.max_len)
        in_path = args.input_dir / f"{tid}.pt"
        out_path = args.output_dir / f"{tid}.pt"
        if out_path.exists() and not args.force:
            skipped += 1
            continue
        if not in_path.exists():
            missing += 1
            continue
        try:
            raw_obj = torch.load(in_path, map_location="cpu", weights_only=False)
            dyn = tensor_from_obj(raw_obj).detach().cpu().float()
            if dyn.ndim != 2:
                raise ValueError(f"expected [L,D], got {tuple(dyn.shape)}")
            d = int(dyn.shape[-1])
            padded = torch.zeros(effective_len, d, dtype=torch.float32)
            mask = torch.zeros(effective_len, 1, dtype=torch.float32)
            take = min(effective_len, int(dyn.shape[0]))
            if take:
                padded[:take] = torch.nan_to_num(dyn[:take], nan=0.0, posinf=0.0, neginf=0.0)
                mask[:take] = 1.0
            if take < effective_len:
                partial += 1
            torch.save(
                {
                    "target_id": tid,
                    "x_dyn": padded,
                    "predicted_dynamic_residue_features": padded,
                    "esmdance_residue_properties": padded,
                    "feature_names": raw_obj.get("feature_names") if isinstance(raw_obj, dict) else None,
                    "feature_source": (
                        raw_obj.get("feature_source", "ESMDance residue prediction head")
                        if isinstance(raw_obj, dict)
                        else "ESMDance residue prediction head"
                    ),
                    "esmdance_embedding_mask": mask,
                    "raw_len": int(dyn.shape[0]),
                    "effective_len": int(effective_len),
                    "covered_len": int(take),
                    "is_partial": bool(take < effective_len),
                },
                out_path,
            )
            written += 1
        except Exception as exc:
            failed += 1
            failures.append((tid, f"{type(exc).__name__}: {exc}"))
            print(f"[WARN] {tid}: {exc}")

    summary = {
        "targets": int(len(targets)),
        "written": written,
        "skipped": skipped,
        "missing_raw_esmdance": missing,
        "partial_padded": partial,
        "failed": failed,
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "max_len": args.max_len,
    }
    (args.output_dir / "_summary.json").write_text(json.dumps(summary, indent=2))
    if failures:
        (args.output_dir / "_failures.tsv").write_text("\n".join(f"{tid}\t{msg}" for tid, msg in failures))
    print(json.dumps(summary, indent=2))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
