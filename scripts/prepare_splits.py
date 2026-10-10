#!/usr/bin/env python3
"""Convert original official CSVs without regenerating split memberships."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd


def _normalizer():
    path = Path(__file__).resolve().parent / "preprocessing/56_make_biofusion_official_split_configs.py"
    spec = importlib.util.spec_from_file_location("_sagf_frozen_official_splits", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.normalize_official_split


def prepare_splits(official_root: Path, data_root: Path, *, datasets=("SNAP", "DRH", "Kinase"),
                   settings=("random", "drug", "prot"), seeds=(42, 1234, 3407, 5678, 8888), force=False) -> list[dict]:
    """Use the frozen script56 transformation, retaining order and all rows."""
    normalize = _normalizer()
    planned = []
    for dataset in datasets:
        if dataset not in ("SNAP", "DRH", "Kinase"):
            raise ValueError(f"unsupported dataset: {dataset}")
        for setting in settings:
            if setting not in ("random", "drug", "prot"):
                raise ValueError(f"unsupported split setting: {setting}")
            for seed in seeds:
                for split in ("train", "val", "test"):
                    source = Path(official_root) / dataset / f"{setting}_split_results/seed_{seed}/{setting}_{split}.csv"
                    destination = Path(data_root) / dataset / f"biofusion1500_front/biofusion_official_splits/{setting}/seed_{seed}/{setting}_{split}.csv"
                    if not source.is_file():
                        raise FileNotFoundError(source)
                    if destination.exists() and not force:
                        raise FileExistsError(f"destination already exists: {destination}; pass --force explicitly to replace")
                    if source.resolve() == destination.resolve():
                        raise ValueError(f"source and destination must differ: {source}")
                    raw = pd.read_csv(source)
                    label_columns = [column for column in ("Label", "label", "Y") if column in raw]
                    for column in label_columns:
                        labels = pd.to_numeric(raw[column], errors="coerce")
                        if labels.isna().any() or not labels.isin((0, 1)).all():
                            raise ValueError(f"official split labels must be finite binary 0/1 before conversion: {source}")
                    id_columns = [column for column in ("DRUG_ID", "Drug_ID", "drug_id", "UNIPROT_ID", "Target_ID", "target_id", "sequence_hash") if column in raw]
                    if any(raw[column].isna().any() for column in id_columns):
                        raise ValueError(f"official split contains null identifiers: {source}")
                    converted = normalize(raw, dataset)
                    if len(raw) != len(converted):
                        raise ValueError(f"normalization changed row count: {source}")
                    planned.append((source, destination, converted, dataset, setting, seed, split))
    result = []
    for source, destination, converted, dataset, setting, seed, split in planned:
        destination.parent.mkdir(parents=True, exist_ok=True)
        converted.to_csv(destination, index=False)
        result.append({"dataset": dataset, "setting": setting, "seed": seed, "split": split, "rows": len(converted),
                       "positive": int((converted["Label"] == 1).sum()), "negative": int((converted["Label"] == 0).sum()),
                       "source": str(source), "output": str(destination),
                       "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                       "output_sha256": hashlib.sha256(destination.read_bytes()).hexdigest()})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=["SNAP", "DRH", "Kinase"])
    parser.add_argument("--settings", nargs="+", choices=["random", "drug", "prot"], default=["random", "drug", "prot"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 1234, 3407, 5678, 8888])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = prepare_splits(args.official_root, args.data_root, datasets=args.datasets, settings=args.settings, seeds=args.seeds, force=args.force)
    payload = json.dumps(report, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload)
    print(json.dumps({"converted_split_files": len(report), "rows": sum(item["rows"] for item in report), "data_root": str(args.data_root.resolve())}))


if __name__ == "__main__":
    main()
