#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Standardize a KPGT npz for Bio datasets so it contains drug_ids and fps."
    )
    p.add_argument("--drugs-csv", type=Path, required=True)
    p.add_argument("--kpgt-npz", type=Path, required=True, help="KPGT output, e.g. kpgt_base.npz.")
    p.add_argument("--output-npz", type=Path, required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    drugs = pd.read_csv(args.drugs_csv)
    if "Drug_ID" not in drugs.columns:
        raise ValueError(f"{args.drugs_csv} must contain Drug_ID")
    z = np.load(args.kpgt_npz, allow_pickle=True)
    if "fps" not in z.files:
        raise ValueError(f"{args.kpgt_npz} must contain fps; keys={z.files}")
    fps = np.asarray(z["fps"], dtype=np.float32)
    if "drug_ids" in z.files:
        drug_ids = np.asarray([str(x) for x in z["drug_ids"]], dtype=object)
    else:
        drug_ids = drugs["Drug_ID"].astype(str).to_numpy(dtype=object)
        if len(fps) != len(drug_ids):
            raise ValueError(
                f"fps rows ({len(fps)}) != drugs.csv rows ({len(drug_ids)}); "
                "cannot infer drug_ids safely."
            )

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, drug_ids=drug_ids, fps=fps)
    summary = {
        "drugs_csv": str(args.drugs_csv),
        "source_kpgt_npz": str(args.kpgt_npz),
        "output_npz": str(args.output_npz),
        "drugs": int(len(drug_ids)),
        "fps_shape": list(fps.shape),
    }
    (args.output_npz.parent / "_kpgt_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
