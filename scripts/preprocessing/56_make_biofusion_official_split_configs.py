#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


DEFAULT_DATASETS = ["DRH", "Kinase", "SNAP"]
DEFAULT_SETTINGS = ["random", "drug", "prot"]
DEFAULT_SEEDS = [42, 1234, 3407, 5678, 8888]
DEFAULT_VARIANTS = ["hypergrad"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Convert BioFusionDTI official split files and create final dynamic-50 "
            "configs for within-dataset and Table 5 cross-dataset experiments."
        )
    )
    p.add_argument("--official-root", type=Path, default=Path("my_experiments/data/Bio/datasets"))
    p.add_argument("--feature-root", type=Path, default=Path("my_experiments/data/Bio/features"))
    p.add_argument("--configs-dir", type=Path, default=Path("my_experiments/configs/experiments/biofusion_officialsplit_final"))
    p.add_argument("--view", default="biofusion1500_front")
    p.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    p.add_argument("--settings", nargs="*", default=DEFAULT_SETTINGS, choices=DEFAULT_SETTINGS)
    p.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--variants", nargs="*", default=DEFAULT_VARIANTS, choices=["base", "hypergrad"])
    p.add_argument("--directions", nargs="*", default=["SNAP:DRH", "DRH:SNAP"])
    p.add_argument("--split-name", default="biofusion_official_splits")
    p.add_argument("--run-root-protocol", default="my_experiments/runs/biofusion_protocol_officialsplit_final")
    p.add_argument("--run-root-table5", default="my_experiments/runs/biofusion_table5_officialsplit_final")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--experiment-id", default="E4_GAUSS_NODE_FUSION")
    p.add_argument("--graph-version", default="esmdance_node_dyn50")
    p.add_argument("--dyn-dim", type=int, default=50)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--alpha-init", type=float, default=0.1)
    p.add_argument("--lambda-uni", type=float, default=0.2)
    p.add_argument("--lambda-rel", type=float, default=0.03)
    p.add_argument("--rel-temperature", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hypergrad-beta", type=float, default=0.01)
    p.add_argument("--hypergrad-min-lr", type=float, default=1e-6)
    p.add_argument("--hypergrad-max-lr", type=float, default=2e-4)
    p.add_argument("--run-prefix", default="bio_e4_final")
    p.add_argument("--hypergrad-prefix", default="hypergrad_bio_e4_final")
    return p.parse_args()


def official_split_path(args: argparse.Namespace, dataset: str, setting: str, seed: int, part: str) -> Path:
    return args.official_root / dataset / f"{setting}_split_results" / f"seed_{seed}" / f"{setting}_{part}.csv"


def converted_split_path(args: argparse.Namespace, dataset: str, setting: str, seed: int, part: str) -> Path:
    return (
        args.feature_root
        / dataset
        / args.view
        / args.split_name
        / setting
        / f"seed_{seed}"
        / f"{setting}_{part}.csv"
    )


def normalize_official_split(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    colmap = {
        "DRUG_ID": "Drug_ID",
        "UNIPROT_ID": "Target_ID",
        "drug_id": "Drug_ID",
        "target_id": "Target_ID",
        "sequence_hash": "Target_ID",
        "label": "Label",
        "Y": "Label",
    }
    out = df.rename(columns={k: v for k, v in colmap.items() if k in df.columns}).copy()
    missing = {"Drug_ID", "Target_ID", "Label"} - set(out.columns)
    if missing:
        raise ValueError(f"official split missing required columns {sorted(missing)}; got {list(df.columns)}")
    out = out[["Drug_ID", "Target_ID", "Label"]].copy()
    out["Drug_ID"] = out["Drug_ID"].astype(str)
    out["Target_ID"] = out["Target_ID"].astype(str)
    out["Label"] = out["Label"].astype(int)
    out["source_dataset"] = dataset
    return out


def convert_official_splits(args: argparse.Namespace, datasets: set[str]) -> list[dict]:
    rows: list[dict] = []
    for dataset in sorted(datasets):
        for setting in args.settings:
            for seed in args.seeds:
                for part in ("train", "val", "test"):
                    src = official_split_path(args, dataset, setting, seed, part)
                    dst = converted_split_path(args, dataset, setting, seed, part)
                    if not src.exists():
                        raise FileNotFoundError(src)
                    df = normalize_official_split(pd.read_csv(src), dataset)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    df.to_csv(dst, index=False)
                    labels = df["Label"].astype(int)
                    rows.append({
                        "dataset": dataset,
                        "setting": setting,
                        "seed": int(seed),
                        "part": part,
                        "path": str(dst),
                        "pairs": int(len(df)),
                        "pos": int(labels.sum()),
                        "neg": int((labels == 0).sum()),
                    })
    return rows


def graph_root(args: argparse.Namespace, dataset: str) -> Path:
    return args.feature_root / dataset / args.view / "graphs"


def drug_npz(args: argparse.Namespace, dataset: str) -> Path:
    return args.feature_root / dataset / args.view / "kpgt" / "kpgt.npz"


def add_final_model_config(args: argparse.Namespace, cfg: dict) -> None:
    cfg["model"] = {
        "drug_dim": 2304,
        "esm_dim": 1280,
        "dyn_dim": int(args.dyn_dim),
        "hidden_dim": int(args.hidden_dim),
        "edge_dim": 36,
        "output_dim": 2,
        "dropout": float(args.dropout),
        "gaussian_min_var": 1e-4,
        "gaussian_max_var": 50.0,
        "sample_fused": False,
        "gaussian_tempered_poe": True,
        "gaussian_semantic_anchor": True,
        "gaussian_alpha_init": float(args.alpha_init),
        "gaussian_alpha_hidden_dim": max(32, int(args.hidden_dim) // 2),
    }
    cfg["unimodal_aux"] = {
        "enabled": True,
        "lambda_uni": float(args.lambda_uni),
        "lambda_rel": float(args.lambda_rel),
        "temperature": float(args.rel_temperature),
    }


def make_config(
    args: argparse.Namespace,
    *,
    protocol: str,
    run_name: str,
    setting: str,
    seed: int,
    variant: str,
    run_root: str,
    train_csv: Path,
    val_csv: Path,
    test_csv: Path,
    graph_roots: list[Path],
    drug_npzs: list[Path],
    extra: dict,
) -> dict:
    cfg = {
        "experiment_id": args.experiment_id,
        "protocol": protocol,
        "biofusion_setting": setting,
        "run_name": run_name,
        "graph_version": args.graph_version,
        "fusion_type": "interaction_mlp",
        "seed": int(seed),
        "device": args.device,
        "run_root": run_root,
        "batch_size": int(args.batch_size),
        "eval_batch_size": int(args.eval_batch_size),
        "epochs": int(args.epochs),
        "patience": 8,
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "loss": "bce",
        "save_best_by": "AUROC",
        "kl_weight": 0.0,
        "amp": True,
        "grad_clip": 1.0,
        "num_workers": int(args.num_workers),
        "pin_memory": False,
        "data": {
            "train_csv": str(train_csv),
            "val_csv": str(val_csv),
            "test_csv": str(test_csv),
            "graph_root": [str(path) for path in graph_roots],
            "bindingdb_drug_npz": [str(path) for path in drug_npzs],
            "davis_drug_npz": [str(path) for path in drug_npzs],
        },
        **extra,
    }
    add_final_model_config(args, cfg)
    if variant == "hypergrad":
        cfg["hypergradient"] = {
            "enabled": True,
            "mode": "cauchy_lr",
            "beta": float(args.hypergrad_beta),
            "min_lr": float(args.hypergrad_min_lr),
            "max_lr": float(args.hypergrad_max_lr),
        }
    return cfg


def command(config_path: Path) -> str:
    return f'"$PYTHON_BIN" -u my_experiments/scripts/train/09_train_experiment.py --config {config_path}'


def write_run_script(path: Path, commands: list[str]) -> None:
    header = (
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        'REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"\n'
        'cd "${REPO_ROOT}"\n'
        'PYTHON_BIN="${PYTHON:-python}"\n\n'
    )
    path.write_text(header + "\n\n".join(commands) + "\n")
    path.chmod(0o755)


def main() -> int:
    args = parse_args()
    directions: list[tuple[str, str]] = []
    direction_datasets: set[str] = set()
    for item in args.directions:
        source, target = item.split(":", 1)
        directions.append((source, target))
        direction_datasets.update([source, target])
    needed = set(args.datasets) | direction_datasets
    split_summary = convert_official_splits(args, needed)

    protocol_dir = args.configs_dir / "protocol"
    table5_dir = args.configs_dir / "table5_cross_dataset"
    protocol_dir.mkdir(parents=True, exist_ok=True)
    table5_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict] = []
    protocol_commands: list[str] = []
    table5_commands: list[str] = []

    for dataset in args.datasets:
        base = args.feature_root / dataset / args.view / args.split_name
        for setting in args.settings:
            for seed in args.seeds:
                for variant in args.variants:
                    prefix = args.hypergrad_prefix if variant == "hypergrad" else args.run_prefix
                    run_name = f"{prefix}_{dataset.lower()}_{setting}_{args.view}_seed{seed}_officialsplit"
                    cfg = make_config(
                        args,
                        protocol="biofusiondti_officialsplit",
                        run_name=run_name,
                        setting=setting,
                        seed=seed,
                        variant=variant,
                        run_root=args.run_root_protocol,
                        train_csv=base / setting / f"seed_{seed}" / f"{setting}_train.csv",
                        val_csv=base / setting / f"seed_{seed}" / f"{setting}_val.csv",
                        test_csv=base / setting / f"seed_{seed}" / f"{setting}_test.csv",
                        graph_roots=[graph_root(args, dataset)],
                        drug_npzs=[drug_npz(args, dataset)],
                        extra={"dataset": dataset},
                    )
                    cfg_path = protocol_dir / f"{dataset.lower()}_{setting}_seed{seed}_{variant}.yaml"
                    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
                    protocol_commands.append(command(cfg_path))
                    summaries.append({
                        "table": "official_protocol",
                        "dataset": dataset,
                        "setting": setting,
                        "seed": int(seed),
                        "variant": variant,
                    })

    for source, target in directions:
        source_base = args.feature_root / source / args.view / args.split_name
        target_base = args.feature_root / target / args.view / args.split_name
        for setting in args.settings:
            for seed in args.seeds:
                for variant in args.variants:
                    prefix = args.hypergrad_prefix if variant == "hypergrad" else args.run_prefix
                    run_name = (
                        f"{prefix}_table5_{source.lower()}_to_{target.lower()}_"
                        f"{setting}_{args.view}_seed{seed}_officialsplit"
                    )
                    cfg = make_config(
                        args,
                        protocol="biofusiondti_table5_cross_dataset_officialsplit",
                        run_name=run_name,
                        setting=setting,
                        seed=seed,
                        variant=variant,
                        run_root=args.run_root_table5,
                        train_csv=source_base / setting / f"seed_{seed}" / f"{setting}_train.csv",
                        val_csv=source_base / setting / f"seed_{seed}" / f"{setting}_val.csv",
                        test_csv=target_base / setting / f"seed_{seed}" / f"{setting}_test.csv",
                        graph_roots=[graph_root(args, source), graph_root(args, target)],
                        drug_npzs=[drug_npz(args, source), drug_npz(args, target)],
                        extra={"source_dataset": source, "target_dataset": target},
                    )
                    cfg_path = table5_dir / f"{source.lower()}_to_{target.lower()}_{setting}_seed{seed}_{variant}.yaml"
                    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
                    table5_commands.append(command(cfg_path))
                    summaries.append({
                        "table": "table5_officialsplit",
                        "source_dataset": source,
                        "target_dataset": target,
                        "setting": setting,
                        "seed": int(seed),
                        "variant": variant,
                    })

    write_run_script(protocol_dir / "run_protocol.sh", protocol_commands)
    write_run_script(table5_dir / "run_table5_cross_dataset.sh", table5_commands)
    args.configs_dir.mkdir(parents=True, exist_ok=True)
    (args.configs_dir / "summary.json").write_text(json.dumps(summaries, indent=2))
    (args.configs_dir / "split_summary.json").write_text(json.dumps(split_summary, indent=2))
    print(json.dumps({
        "configs_dir": str(args.configs_dir),
        "converted_split_rows": len(split_summary),
        "protocol_configs": len(protocol_commands),
        "table5_configs": len(table5_commands),
        "protocol_run_script": str(protocol_dir / "run_protocol.sh"),
        "table5_run_script": str(table5_dir / "run_table5_cross_dataset.sh"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
