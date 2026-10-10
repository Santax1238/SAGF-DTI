#!/usr/bin/env python3
"""
Generate ESMDance per-residue embeddings (single-process, fast).

Unlike the subprocess-isolated version, this loads the model once and
processes all proteins sequentially. Use this when torchvision issues
are fixed and CUDA context is stable.

Example:
  python scripts/generate_esmdance_embeddings_fast.py \
      --targets scripts/data/my_sequnce/unique_proteins.csv \
      --output-dir data/esmdance_embeddings_new \
      --seqdance-root SeqDance \
      --skip-existing
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--targets", type=Path,
                   default=Path("scripts/data/my_sequnce/unique_proteins.csv"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("data/esmdance_embeddings_new"))
    p.add_argument("--seqdance-root", type=Path,
                   default=Path(os.environ.get("SEQDANCE_ROOT", "SeqDance")))
    p.add_argument("--weights-dir", type=Path, default=None)
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max-len", type=int, default=1022)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load labels CSV (supports both schemas)
    df = pd.read_csv(args.targets)
    if "Target_ID" in df.columns and "Target" in df.columns:
        id_col, seq_col = "Target_ID", "Target"
    elif "target_id" in df.columns and "sequence" in df.columns:
        id_col, seq_col = "target_id", "sequence"
    else:
        raise ValueError("CSV must have (Target_ID, Target) or (target_id, sequence) columns")

    # Load ESMDance once
    weights_dir = args.weights_dir or (args.seqdance_root / "esmdance_weights")
    seqdance_model_dir = str(args.seqdance_root / "model")
    sys.path.insert(0, seqdance_model_dir)

    from model import ESMwrap
    from transformers import AutoTokenizer

    print(f"Loading ESMDance from {weights_dir}...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
    dance_model = ESMwrap("model_35M", "esmdance")
    dance_model = dance_model.from_pretrained(str(weights_dir))
    dance_model = dance_model.to(args.device).eval()
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
            with torch.no_grad():
                out = dance_model(
                    inputs,
                    return_res_emb=True,
                    return_attention_map=False,
                    return_res_pred=False,
                    return_pair_pred=False,
                )
            emb = out["res_emb"][0, 1:-1].cpu()   # [L, 480]
            torch.save(emb, out_path)
            done += 1
            if done == 1 or done % 100 == 0:
                print(f"  [{i+1}/{total}] {tid}: {tuple(emb.shape)}")
        except Exception as e:
            failed += 1
            print(f"  [FAILED] {tid}: {e}")

    print(f"\nDone. Total={total}, Generated={done}, Skipped={skipped}, Failed={failed}")
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
