#!/usr/bin/env python3
"""Run preserved feature utilities with explicit paths and complete-target checks.

Run ``python scripts/prepare_features.py STAGE --help`` for the original flags.
This wrapper keeps pretrained models external and forbids implicit downloads.
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

STAGES = {
    "kpgt-input": None,
    "esm2": "esm2.py", "carrier": "esmdance_carrier.py", "dynamic": "esmdance_dynamic.py",
    "kpgt": "standardize_kpgt.py", "pad": "pad_esmdance.py", "graphs": "build_dynamic_graphs.py",
    "merge": "merge_conditioned_graphs.py", "similarity": "build_similarity.py",
}


def option(arguments: list[str], flag: str, default: str | None = None) -> str | None:
    for i, argument in enumerate(arguments):
        if argument == flag:
            if i + 1 >= len(arguments) or arguments[i + 1].startswith("--"):
                raise ValueError(f"{flag} needs a value")
            return arguments[i + 1]
        if argument.startswith(flag + "="):
            return argument.split("=", 1)[1]
    return default


def required_path(arguments: list[str], flag: str, kind: str = "exists") -> Path:
    value = option(arguments, flag)
    if not value:
        raise ValueError(f"an explicit {flag} is required by the portable wrapper")
    path = Path(value).expanduser().resolve()
    if kind != "output" and not path.exists():
        raise FileNotFoundError(path)
    return path


def target_ids(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not {"Target_ID", "Target"}.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: expected Target_ID,Target columns")
        rows = list(reader)
    ids = [row["Target_ID"] for row in rows]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"{path}: Target_ID values must be nonempty and unique")
    for row in rows:
        identifier = row["Target_ID"]
        if not identifier or identifier in {".", ".."} or "/" in identifier or "\\" in identifier:
            raise ValueError(f"unsafe Target_ID: {identifier!r}")
        if not row["Target"].strip():
            raise ValueError(f"{path}: empty sequence for {identifier}")
    return ids


def require_targets(directory: Path, ids: list[str]) -> None:
    missing = [identifier for identifier in ids if not (directory / f"{identifier}.pt").is_file()]
    if missing:
        raise FileNotFoundError(f"{directory}: missing {len(missing)} target files; first={missing[:5]}")


def dataset_bases(arguments: list[str]) -> list[Path]:
    root = required_path(arguments, "--feature-root")
    if "--datasets" not in arguments:
        raise ValueError("an explicit --datasets DATASET [DATASET ...] is required")
    start = arguments.index("--datasets") + 1
    names = []
    for value in arguments[start:]:
        if value.startswith("--"):
            break
        if value not in {"SNAP", "DRH", "Kinase"}:
            raise ValueError(f"unknown dataset: {value}")
        names.append(value)
    if not names:
        raise ValueError("--datasets needs at least one dataset")
    view = option(arguments, "--view", "biofusion1500_front")
    if view != "biofusion1500_front":
        raise ValueError("the release wrapper supports the paper biofusion1500_front view")
    return [root / name / view for name in names]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, add_help=False)
    parser.add_argument("stage", choices=STAGES)
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        parser.print_help()
        return 0
    args, forwarded = parser.parse_known_args()
    if args.stage == "kpgt-input":
        stage_parser = argparse.ArgumentParser(description="Export Drug SMILES in Drug_ID row order for external KPGT preprocessing.")
        stage_parser.add_argument("--drugs-csv", type=Path, required=True)
        stage_parser.add_argument("--output-csv", type=Path, required=True)
        stage_args = stage_parser.parse_args(forwarded)
        with stage_args.drugs_csv.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if not {"Drug_ID", "Drug"}.issubset(reader.fieldnames or []):
                raise ValueError("drugs.csv must contain Drug_ID,Drug (SMILES)")
            rows = list(reader)
        ids = [row["Drug_ID"] for row in rows]
        if not ids or any(not value for value in ids) or len(set(ids)) != len(ids):
            raise ValueError("Drug_ID values must be nonempty and unique")
        if any(not row["Drug"].strip() for row in rows):
            raise ValueError("empty SMILES")
        if stage_args.output_csv.exists():
            raise FileExistsError(f"refusing to overwrite {stage_args.output_csv}")
        stage_args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with stage_args.output_csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["smiles", "label"])
            writer.writeheader()
            writer.writerows({"smiles": row["Drug"], "label": 0} for row in rows)
        print(f"[complete] exported {len(rows)} SMILES in drugs.csv row order; label is a KPGT input placeholder")
        return 0
    script = Path(__file__).resolve().parent / "preprocessing" / STAGES[args.stage]
    if not script.is_file():
        raise FileNotFoundError(script)
    if any(flag in forwarded for flag in ("--help", "-h")):
        return subprocess.call([sys.executable, str(script), "--help"])
    checks: list[tuple[Path, list[str]]] = []
    if args.stage in {"esm2", "carrier", "dynamic", "pad", "graphs"}:
        targets = required_path(forwarded, "--targets" if args.stage in {"carrier", "dynamic"} else "--targets-csv")
        ids = target_ids(targets)
        if args.stage in {"carrier", "dynamic"}:
            root = required_path(forwarded, "--seqdance-root")
            required_path(forwarded, "--weights-dir")
            if not (root / "model" / "model.py").is_file():
                raise FileNotFoundError(root / "model" / "model.py")
        elif args.stage == "esm2":
            required_path(forwarded, "--model-name-or-path")
        elif args.stage == "pad":
            require_targets(required_path(forwarded, "--input-dir"), ids)
        elif args.stage == "graphs":
            require_targets(required_path(forwarded, "--esm2-dir"), ids)
            require_targets(required_path(forwarded, "--esmdance-dir"), ids)
        if args.stage == "graphs":
            output = required_path(forwarded, "--output-root", "output") / option(forwarded, "--graph-version", "esmdance_node")
        else:
            output = required_path(forwarded, "--output-dir", "output")
        checks.append((output, ids))
    elif args.stage == "kpgt":
        required_path(forwarded, "--drugs-csv")
        required_path(forwarded, "--kpgt-npz")
        required_path(forwarded, "--output-npz", "output")
    else:
        for base in dataset_bases(forwarded):
            ids = target_ids(base / "targets.csv")
            if args.stage == "merge":
                require_targets(base / "graphs" / option(forwarded, "--dynamic-version", "esmdance_node_dyn50"), ids)
                require_targets(base / "graphs" / option(forwarded, "--esmdance35-version", "esmdance_node"), ids)
                checks.append((base / "graphs" / option(forwarded, "--output-version", "esmdance_node_cond35_dyn50"), ids))
            else:
                require_targets(base / "graphs" / option(forwarded, "--graph-version", "esmdance_node_dyn50"), ids)
                if not (base / "kpgt" / "kpgt.npz").is_file():
                    raise FileNotFoundError(base / "kpgt" / "kpgt.npz")
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    completed = subprocess.run([sys.executable, str(script), *forwarded], env=env, check=False)
    if completed.returncode:
        return completed.returncode
    for directory, ids in checks:
        require_targets(directory, ids)
    if args.stage == "kpgt":
        import numpy as np
        with required_path(forwarded, "--drugs-csv").open(newline="", encoding="utf-8") as stream:
            expected_ids = [row["Drug_ID"] for row in csv.DictReader(stream)]
        with np.load(required_path(forwarded, "--output-npz"), allow_pickle=True) as archive:
            ids, fps = archive["drug_ids"], archive["fps"]
            if fps.ndim != 2 or fps.shape != (len(ids), 2304):
                raise ValueError(f"expected KPGT fps[N,2304], found {fps.shape}")
            if len(set(map(str, ids))) != len(ids) or not np.isfinite(fps).all():
                raise ValueError("KPGT IDs must be unique and features finite")
            if list(map(str, ids)) != expected_ids:
                raise ValueError("KPGT drug_ids must exactly match drugs.csv row order")
    elif args.stage == "similarity":
        for base in dataset_bases(forwarded):
            for name in ("drug_knn_sim.npz", "target_knn_sim.npz"):
                if not (base / "contrastive" / name).is_file():
                    raise FileNotFoundError(base / "contrastive" / name)
    print("[complete] artifacts present for every requested target; run validate-data before training")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(1) from None
