#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import pandas as pd
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate ESM2 per-residue embeddings for Bio datasets. "
            "Outputs one <Target_ID>.pt file with x_esm/embedding [L, D]. "
            "Sequences are truncated to --max-len, and long inputs are embedded "
            "in chunks so a 1500-residue BioFusion-style view is usable with "
            "ESM2 checkpoints that only accept ~1022 residues per pass."
        )
    )
    p.add_argument("--targets-csv", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model-name-or-path", default="facebook/esm2_t33_650M_UR50D")
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-len", type=int, default=1500)
    p.add_argument("--chunk-len", type=int, default=1022)
    p.add_argument("--local-files-only", action="store_true", default=True)
    p.add_argument("--force", action="store_true")
    p.add_argument("--target-id-col", default="Target_ID")
    p.add_argument("--sequence-col", default="Target")
    return p.parse_args()


def infer_hidden(model, tokenizer, seq: str, device: torch.device) -> torch.Tensor:
    inputs = tokenizer([seq], return_tensors="pt", padding=False, add_special_tokens=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        out = model(**inputs)
    return out.last_hidden_state[0, 1 : 1 + len(seq)].detach().cpu().float()


def embed_sequence(model, tokenizer, seq: str, device: torch.device, chunk_len: int) -> torch.Tensor:
    if len(seq) <= chunk_len:
        return infer_hidden(model, tokenizer, seq, device)
    chunks = []
    for start in range(0, len(seq), chunk_len):
        part = seq[start : start + chunk_len]
        chunks.append(infer_hidden(model, tokenizer, part, device))
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    return torch.cat(chunks, dim=0)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    targets = pd.read_csv(args.targets_csv).drop_duplicates(args.target_id_col).reset_index(drop=True)
    if args.target_id_col not in targets.columns or args.sequence_col not in targets.columns:
        raise ValueError(f"{args.targets_csv} must contain {args.target_id_col} and {args.sequence_col}")

    from transformers import EsmModel, EsmTokenizer

    cache_dir = str(args.cache_dir) if args.cache_dir else None
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    print(f"[load] {args.model_name_or_path} device={device}")
    tokenizer = EsmTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=cache_dir,
        local_files_only=args.local_files_only,
    )
    model = EsmModel.from_pretrained(
        args.model_name_or_path,
        cache_dir=cache_dir,
        local_files_only=args.local_files_only,
    ).to(device).eval()

    written = skipped = failed = 0
    failures: list[tuple[str, str]] = []
    t0 = time.time()
    for i, row in targets.iterrows():
        tid = str(row[args.target_id_col])
        seq_full = str(row[args.sequence_col]).strip().upper()
        seq = seq_full[: args.max_len]
        out_path = args.output_dir / f"{tid}.pt"
        if out_path.exists() and not args.force:
            skipped += 1
            continue
        try:
            emb = embed_sequence(model, tokenizer, seq, device, args.chunk_len)
            if emb.shape[0] != len(seq):
                raise ValueError(f"embedding length {emb.shape[0]} != sequence length {len(seq)}")
            payload = {
                "target_id": tid,
                "x_esm": emb,
                "embedding": emb,
                "model": args.model_name_or_path,
                "full_len": int(len(seq_full)),
                "effective_len": int(len(seq)),
                "max_len": int(args.max_len),
                "chunk_len": int(args.chunk_len),
                "is_truncated": bool(len(seq_full) > args.max_len),
                "is_chunked": bool(len(seq) > args.chunk_len),
            }
            torch.save(payload, out_path)
            written += 1
            if written == 1 or written % 50 == 0:
                print(f"  [{i+1}/{len(targets)}] {tid}: {tuple(emb.shape)}")
        except Exception as exc:
            failed += 1
            failures.append((tid, f"{type(exc).__name__}: {exc}"))
            print(f"[WARN] {tid}: {exc}")
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

    summary = {
        "targets": int(len(targets)),
        "written": written,
        "skipped": skipped,
        "failed": failed,
        "elapsed_min": round((time.time() - t0) / 60.0, 3),
        "output_dir": str(args.output_dir),
        "model": args.model_name_or_path,
        "max_len": args.max_len,
        "chunk_len": args.chunk_len,
    }
    (args.output_dir / "_summary.json").write_text(json.dumps(summary, indent=2))
    if failures:
        (args.output_dir / "_failures.tsv").write_text("\n".join(f"{tid}\t{msg}" for tid, msg in failures))
    print(json.dumps(summary, indent=2))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
