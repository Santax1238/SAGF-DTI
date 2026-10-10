"""Portable resolution and explicit comparison against the archived paper recipe."""
from __future__ import annotations

from copy import deepcopy
import json
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml


FEATURE_PREFIX = "my_experiments/data/Bio/features/"
PATH_FIELDS = {
    "data": ("train_csv", "val_csv", "test_csv", "graph_root", "bindingdb_drug_npz",
             "davis_drug_npz", "drugs_csv", "targets_csv", "drug_sim_npz", "target_sim_npz"),
    "contrastive": ("drugs_csv", "targets_csv", "drug_sim_npz", "target_sim_npz"),
}


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("experiment_id") != "E4_GAUSS_NODE_FUSION":
        raise ValueError("experiment_id must be E4_GAUSS_NODE_FUSION for the SAGF-DTI Gaussian API")
    if config.get("graph_version") != "esmdance_node_cond35_dyn50":
        raise ValueError("graph_version must be esmdance_node_cond35_dyn50")
    name = config.get("run_name")
    if not isinstance(name, str) or not name or name in (".", "..") or Path(name).name != name or "\\" in name:
        raise ValueError("run_name must be a single nonempty directory name")
    for key in ("batch_size", "eval_batch_size", "epochs", "patience"):
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
    data = config.get("data")
    if not isinstance(data, dict):
        raise ValueError("data must be a mapping")
    for key in ("train_csv", "val_csv", "test_csv", "graph_root", "bindingdb_drug_npz", "davis_drug_npz"):
        if not data.get(key):
            raise ValueError(f"data.{key} must be specified; legacy defaults are not portable")
    if not isinstance(config.get("model"), dict):
        raise ValueError("model must be a mapping")
    cl = config.get("contrastive", {})
    if not isinstance(cl, dict):
        raise ValueError("contrastive must be a mapping")
    if bool(cl.get("enabled", config.get("use_contrastive", False))):
        for key in ("drugs_csv", "targets_csv", "drug_sim_npz", "target_sim_npz"):
            if not cl.get(key, data.get(key)):
                raise ValueError(f"contrastive.{key} must be specified when contrastive is enabled")


def resolve_config(config_path: Path, data_root: Path, run_root: Path | None = None) -> dict[str, Any]:
    """Resolve feature paths beneath DATA_ROOT without changing recipe fields.

    Relative run_root is resolved beside the supplied configuration. Pass an
    explicit run_root for a stable output location. Edited numerical settings
    remain usable; call assert_paper_config to require archived paper settings.
    """
    config_path = Path(config_path).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text())
    if not isinstance(config, dict):
        raise ValueError("configuration YAML must contain a mapping")
    _validate_config(config)
    data_root = Path(data_root).expanduser().resolve()

    def resolve_path(value: Any, field: str) -> Any:
        if isinstance(value, list):
            if not value:
                raise ValueError(f"{field} must not be empty")
            return [resolve_path(item, field) for item in value]
        if not isinstance(value, str) or not value:
            raise ValueError(f"{field} must contain nonempty path strings")
        if value.startswith(FEATURE_PREFIX):
            value = value[len(FEATURE_PREFIX):]
        path = Path(value).expanduser()
        return str((path if path.is_absolute() else data_root / path).resolve())

    for section, keys in PATH_FIELDS.items():
        for key in keys:
            if key in config.get(section, {}):
                config[section][key] = resolve_path(config[section][key], f"{section}.{key}")
    if run_root is None:
        output = Path(config.get("run_root", "runs")).expanduser()
        if not output.is_absolute():
            output = config_path.parent / output
    else:
        output = Path(run_root).expanduser()
    config["run_root"] = str(output.resolve())
    return config


def _non_path_config(config: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(config)
    result.pop("run_root", None)
    for section, keys in PATH_FIELDS.items():
        for key in keys:
            result.get(section, {}).pop(key, None)
    return result


def _differences(actual: Any, expected: Any, prefix: str = "") -> list[str]:
    if isinstance(actual, dict) and isinstance(expected, dict):
        result = []
        for key in sorted(actual.keys() | expected.keys()):
            field = f"{prefix}.{key}" if prefix else key
            if key not in actual or key not in expected:
                result.append(field)
            else:
                result.extend(_differences(actual[key], expected[key], field))
        return result
    return [] if actual == expected else [prefix]


def compare_paper_config(config: dict[str, Any]) -> dict[str, Any]:
    """Identify recipe/path differences; a match does not assert feature hashes."""
    manifest = json.loads(files("sagf_dti").joinpath("paper_manifest.json").read_text())
    dataset, setting, seed = config.get("dataset"), config.get("biofusion_setting"), config.get("seed")
    member = f"configs/paper/{dataset}/{setting}/seed_{seed}.yaml"
    recipe = _non_path_config(config)
    archived = manifest["configs"].get(member)
    if archived is None:
        return {"matches": False, "config": member, "differences": ["dataset/biofusion_setting/seed"],
                "non_path_config": recipe}
    changed = _differences(recipe, archived["non_path_config"])
    for field, expected in archived["data_paths"].items():
        section, key = field.split(".")
        actual = config.get(section, {}).get(key)
        expected_list = expected if isinstance(expected, list) else [expected]
        actual_list = actual if isinstance(actual, list) else [actual]
        if isinstance(actual, list) != isinstance(expected, list) or len(actual_list) != len(expected_list):
            changed.append(field)
            continue
        for value, suffix in zip(actual_list, expected_list):
            if not isinstance(value, str) or tuple(Path(value).parts[-len(Path(suffix).parts):]) != Path(suffix).parts:
                changed.append(field)
                break
    return {"matches": not changed, "config": member, "differences": sorted(set(changed)),
            "source_sha256": archived["source_sha256"], "portable_sha256": archived["portable_sha256"],
            "non_path_config": recipe}


def assert_paper_config(config: dict[str, Any]) -> dict[str, Any]:
    """Raise if an edited configuration is being claimed as an original recipe."""
    comparison = compare_paper_config(config)
    if not comparison["matches"]:
        raise ValueError("configuration differs from archived paper settings: " + ", ".join(comparison["differences"]))
    return comparison
