#!/usr/bin/env python3
"""
Generate ESMDance predicted residue dynamic properties (single-process, fast).

Unlike the subprocess-isolated version, this loads the model once and
processes all proteins sequentially. Use this when torchvision issues
are fixed and CUDA context is stable.

Example:
  python scripts/generate_esmdance_embeddings_fast.py \
      --targets scripts/data/my_sequnce/unique_proteins.csv \
      --output-dir data/esmdance_residue_properties \
      --seqdance-root SeqDance \
      --skip-existing
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

FEATURE_NAMES = (
    ["sasa_mean", "sasa_std", "rmsf_nor", "nma_res1", "nma_res2", "nma_res3"]
    + [f"ss_{i}" for i in range(8)]
    + [f"chi_{i}" for i in range(12)]
    + [f"phi_{i}" for i in range(12)]
    + [f"psi_{i}" for i in range(12)]
)


def build_dynamic_features(out: dict[str, torch.Tensor]) -> torch.Tensor:
    scalar_features = torch.stack(
        [
            out["sasa_mean"],
            out["sasa_std"],
            out["rmsf_nor"],
            out["nma_res1"],
            out["nma_res2"],
            out["nma_res3"],
        ],
        dim=-1,
    )
    dynamic_features = torch.cat(
        [scalar_features, out["ss"], out["chi"], out["phi"], out["psi"]],
        dim=-1,
    )
    if dynamic_features.shape[-1] != 50:
        raise RuntimeError(f"expected 50 dynamic features, got {dynamic_features.shape[-1]}")
    return dynamic_features


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--targets", type=Path,
                   default=Path("scripts/data/my_sequnce/unique_proteins.csv"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("data/esmdance_residue_properties"))
    p.add_argument("--seqdance-root", type=Path,
                   default=Path(os.environ.get("SEQDANCE_ROOT", "SeqDance")))
    p.add_argument("--weights-dir", type=Path, default=None)
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max-len", type=int, default=1022)
    p.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp32")
    p.add_argument("--min-original-len", type=int, default=None)
    p.add_argument("--max-original-len", type=int, default=None)
    p.add_argument("--sort-by-length", choices=("none", "ascending", "descending"), default="none")
    p.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds after each protein; useful on display GPUs.")
    p.add_argument("--limit", type=int, default=None, help="Process at most this many rows after filtering.")
    p.add_argument("--num-threads", type=int, default=None, help="Limit torch CPU threads.")
    args = p.parse_args()
    if args.num_threads is not None:
        torch.set_num_threads(int(args.num_threads))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load labels CSV (supports both schemas)
    df = pd.read_csv(args.targets)
    if "Target_ID" in df.columns and "Target" in df.columns:
        id_col, seq_col = "Target_ID", "Target"
    elif "target_id" in df.columns and "sequence" in df.columns:
        id_col, seq_col = "target_id", "sequence"
    else:
        raise ValueError("CSV must have (Target_ID, Target) or (target_id, sequence) columns")
    df = df.drop_duplicates(id_col).reset_index(drop=True)
    df["_original_len"] = df[seq_col].astype(str).str.len()
    if args.min_original_len is not None:
        df = df[df["_original_len"] >= int(args.min_original_len)].copy()
    if args.max_original_len is not None:
        df = df[df["_original_len"] <= int(args.max_original_len)].copy()
    if args.sort_by_length == "ascending":
        df = df.sort_values(["_original_len", id_col]).reset_index(drop=True)
    elif args.sort_by_length == "descending":
        df = df.sort_values(["_original_len", id_col], ascending=[False, True]).reset_index(drop=True)
    if args.limit is not None:
        df = df.head(int(args.limit)).copy()

    # Load ESMDance once
    weights_dir = args.weights_dir or (args.seqdance_root / "esmdance_weights")
    seqdance_model_dir = str(args.seqdance_root / "model")
    if not Path(seqdance_model_dir, "model.py").exists():
        raise FileNotFoundError(
            f"SeqDance model.py not found at {Path(seqdance_model_dir, 'model.py')}. "
            "Use --seqdance-root /path/to/SeqDance."
        )
    if not Path(weights_dir).exists():
        raise FileNotFoundError(
            f"ESMDance weights dir not found at {weights_dir}. "
            "Use --weights-dir /path/to/esmdance_weights."
        )
    sys.path.insert(0, seqdance_model_dir)
    if "" in sys.path:
        sys.path.remove("")

    try:
        from model import ESMwrap
    except ImportError as e:
        raise ImportError(f"Could not import ESMwrap from {seqdance_model_dir}: {e}") from e
    from transformers import AutoTokenizer

    print(f"Loading ESMDance from {weights_dir}...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D", local_files_only=True)
    dance_model = ESMwrap("model_35M", "esmdance")
    dance_model = dance_model.from_pretrained(str(weights_dir))
    dance_model = dance_model.to(args.device).eval()
    device_obj = torch.device(args.device)
    if device_obj.type == "cuda" and args.precision == "fp16":
        dance_model = dance_model.half()
    elif device_obj.type == "cuda" and args.precision == "bf16":
        dance_model = dance_model.bfloat16()
    print("Model loaded.")

    total = len(df)
    done = skipped = failed = 0

    for i, row in df.iterrows():
        tid = str(row[id_col])
        seq = str(row[seq_col])
        out_path = args.output_dir / f"{tid}.pt"

        if args.skip_existing and out_path.exists():
            skipped += 1
            continue

        try:
            s = seq[:args.max_len]
            inputs = tokenizer([s], return_tensors="pt", padding=False)
            inputs = {k: v.to(args.device) for k, v in inputs.items()}
            with torch.inference_mode():
                out = dance_model(
                    inputs,
                    return_res_emb=False,
                    return_attention_map=False,
                    return_res_pred=True,
                    return_pair_pred=False,
                )
            emb = build_dynamic_features(out)[0, 1:-1].cpu()   # [L, 50]
            payload = {
                "predicted_dynamic_residue_features": emb,
                "esmdance_residue_properties": emb,
                "feature_names": FEATURE_NAMES,
                "feature_source": "ESMDance residue prediction head",
                "original_len": int(row["_original_len"]),
                "max_len": int(args.max_len),
            }
            tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
            torch.save(payload, tmp_path)
            tmp_path.replace(out_path)
            done += 1
            if done == 1 or done % 100 == 0:
                print(f"  [{i+1}/{total}] {tid}: {tuple(emb.shape)}")
            del inputs, out, emb, payload
            if torch.device(args.device).type == "cuda":
                torch.cuda.empty_cache()
            if args.sleep > 0:
                time.sleep(float(args.sleep))
        except Exception as e:
            failed += 1
            print(f"  [FAILED] {tid}: {e}")

    print(f"\nDone. Total={total}, Generated={done}, Skipped={skipped}, Failed={failed}")
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
