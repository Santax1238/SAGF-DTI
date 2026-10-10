"""Read-only validation of complete cached inputs, without sample filtering."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _paths(value: Any) -> list[Path]:
    if value is None:
        return []
    values = value if isinstance(value, (list, tuple)) else [value]
    return [Path(item) for item in values]


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def validate_data(config: dict, *, splits=("train", "val"), inspect_features=False) -> dict:
    """Return all errors and counts for the explicitly selected splits.

    The default reads training and validation only. Test CSVs and the test
    drug NPZ are accessed only if ``test`` is explicitly included in splits.
    ``inspect_features`` loads every distinct selected protein graph and
    verifies auxiliary fields, shapes, finite values, identity and masks.
    Cached .pt files and object-ID NPZ arrays must come from trusted sources.
    """
    if isinstance(splits, str):
        splits = (splits,)
    splits = tuple(splits)
    if not splits or len(set(splits)) != len(splits) or any(item not in ("train", "val", "test") for item in splits):
        raise ValueError("splits must contain distinct train, val and/or test names")
    report: dict[str, Any] = {
        "valid": False, "splits_requested": list(splits), "inspect_features": bool(inspect_features),
        "errors": [], "warnings": [], "missing_paths": [], "splits": {},
        "features": {"drug_npz": {}, "graphs": {}, "contrastive": {}},
    }
    errors, warnings = report["errors"], report["warnings"]

    def exists(path: Path, role: str) -> bool:
        if path.is_file():
            return True
        report["missing_paths"].append(str(path))
        errors.append(f"missing {role}: {path}")
        return False

    def read_csv(path: Path, role: str) -> pd.DataFrame | None:
        if not exists(path, role):
            return None
        try:
            return pd.read_csv(path)
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            errors.append(f"cannot read {role} {path}: {exc}")
            return None

    data = config.get("data", {})
    model = config.get("model", {})
    graph_version = config.get("graph_version", "esmdance_node_cond35_dyn50")
    graph_dirs = [path / graph_version for path in _paths(data.get("graph_root"))]
    if not graph_dirs:
        errors.append("data.graph_root must be specified")
    rows_by_split: dict[str, pd.DataFrame] = {}
    all_drugs, all_targets = set(), set()
    aliases = {"drug_id": ("Drug_ID",), "sequence_hash": ("Target_ID",), "label": ("Label", "Y")}
    for split in splits:
        summary = {"rows": 0, "positive": 0, "negative": 0, "unique_drugs": 0, "unique_targets": 0,
                   "missing_drug_ids": [], "missing_graph_ids": []}
        report["splits"][split] = summary
        csv_path = data.get(f"{split}_csv")
        if not isinstance(csv_path, str) or not csv_path:
            errors.append(f"data.{split}_csv must be specified")
            continue
        path = Path(csv_path)
        summary["path"] = str(path)
        raw = read_csv(path, f"{split} CSV")
        if raw is None:
            continue
        summary["rows"] = len(raw)
        if not len(raw):
            errors.append(f"{split} CSV is empty: {path}")
        normalized = raw.copy()
        for canonical, alternatives in aliases.items():
            present = [key for key in (canonical, *alternatives) if key in raw]
            if not present:
                errors.append(f"{split} CSV missing {canonical} column (aliases {alternatives}): {path}")
                continue
            source = present[0]
            for other in present[1:]:
                if canonical == "label":
                    equal = pd.to_numeric(raw[source], errors="coerce").equals(pd.to_numeric(raw[other], errors="coerce"))
                else:
                    equal = raw[source].astype(str).equals(raw[other].astype(str))
                if not equal:
                    errors.append(f"{split} CSV conflicting alias columns {source}/{other}: {path}")
            if raw[source].isna().any():
                errors.append(f"{split} CSV null {canonical} values: {path}")
            normalized[canonical] = raw[source]
        if not {"drug_id", "sequence_hash", "label"}.issubset(normalized.columns):
            continue
        normalized["drug_id"] = normalized["drug_id"].astype(str)
        normalized["sequence_hash"] = normalized["sequence_hash"].astype(str)
        labels = pd.to_numeric(normalized["label"], errors="coerce")
        if labels.isna().any() or not labels.isin((0, 1)).all():
            errors.append(f"{split} CSV labels must be finite binary 0/1: {path}")
        normalized["label"] = labels
        if "pair_id" not in normalized:
            normalized["pair_id"] = [f"pair:{i}:{drug}:{target}" for i, (drug, target) in enumerate(zip(normalized["drug_id"], normalized["sequence_hash"]))]
        elif normalized["pair_id"].isna().any():
            errors.append(f"{split} CSV null pair_id values: {path}")
        summary.update(positive=int((labels == 1).sum()), negative=int((labels == 0).sum()),
                       unique_drugs=int(normalized["drug_id"].nunique()), unique_targets=int(normalized["sequence_hash"].nunique()),
                       duplicate_pair_ids=int(normalized["pair_id"].duplicated().sum()),
                       duplicate_pairs=int(normalized[["drug_id", "sequence_hash", "label"]].duplicated().sum()),
                       sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        if summary["duplicate_pair_ids"]:
            errors.append(f"{split} CSV has duplicate pair_id values: {path}")
        rows_by_split[split] = normalized
        all_drugs.update(normalized["drug_id"])
        all_targets.update(normalized["sequence_hash"])

    # Match the original trainer's train/validation versus test NPZ selection.
    required_npz = {}
    for split in splits:
        key = "davis_drug_npz" if split == "test" else "bindingdb_drug_npz"
        paths = _paths(data.get(key))
        required_npz[split] = paths
        if not paths:
            errors.append(f"data.{key} must be specified for {split}")
    drug_ids_by_path = {}
    for path in sorted({path for paths in required_npz.values() for path in paths}):
        ids = set()
        drug_ids_by_path[path] = ids
        if not exists(path, "KPGT NPZ"):
            continue
        try:
            with np.load(path, allow_pickle=True) as archive:
                if not {"drug_ids", "fps"}.issubset(archive.files):
                    errors.append(f"KPGT NPZ must contain drug_ids and fps: {path}")
                    continue
                raw_ids, matrix = archive["drug_ids"], archive["fps"]
                if raw_ids.ndim != 1 or matrix.ndim != 2 or matrix.shape[0] != len(raw_ids):
                    errors.append(f"KPGT drug_ids/fps shape mismatch: {raw_ids.shape}/{matrix.shape} in {path}")
                    continue
                ordered_ids = [str(value) for value in raw_ids]
                ids.update(ordered_ids)
                duplicates = sorted(key for key, count in Counter(ordered_ids).items() if count > 1)
                if duplicates:
                    errors.append(f"duplicate KPGT drug_ids in {path}: {duplicates}")
                expected_dim = int(model.get("drug_dim", 2304))
                if matrix.shape[1] != expected_dim:
                    errors.append(f"KPGT fps shape {matrix.shape} requires drug_dim={expected_dim}: {path}")
                # DTIDataset calls torch.as_tensor(..., dtype=torch.float32).
                # NumPy longdouble and non-native byte order are unsupported,
                # and complex conversion would silently discard information.
                dtype = matrix.dtype
                supported = dtype.isnative and (
                    (dtype.kind == "f" and dtype.itemsize in (2, 4, 8)) or
                    (dtype.kind in ("i", "u") and dtype.itemsize in (1, 2, 4, 8)) or
                    (dtype.kind == "b" and dtype.itemsize == 1)
                )
                if not supported:
                    errors.append(f"KPGT fps dtype {dtype} is not a supported real Torch-convertible dtype: {path}")
                elif not np.isfinite(matrix).all():
                    errors.append(f"KPGT fps must contain finite numeric values: {path}")
                else:
                    with np.errstate(over="ignore", invalid="ignore"):
                        converted = matrix.astype(np.float32, copy=False)
                    if not np.isfinite(converted).all():
                        errors.append(f"KPGT fps values overflow the frozen loader's float32 conversion: {path}")
                report["features"]["drug_npz"][str(path)] = {"rows": len(raw_ids), "shape": list(matrix.shape),
                                                           "dtype": str(matrix.dtype), "duplicate_ids": duplicates}
        except (OSError, ValueError, TypeError) as exc:
            errors.append(f"cannot inspect KPGT NPZ {path}: {exc}")
    graph_paths = {}
    for target in sorted(all_targets):
        if not target or Path(target).name != target or target in (".", "..") or "\\" in target:
            errors.append(f"invalid graph target identifier: {target!r}")
            continue
        candidates = [directory / f"{target}.pt" for directory in graph_dirs]
        path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0] if candidates else None)
        if path is not None and path.is_file():
            graph_paths[target] = path
        elif path is not None:
            exists(path, f"protein graph for {target}")
    for split, rows in rows_by_split.items():
        ordered_ids = [str(value) for path in required_npz[split] for value in drug_ids_by_path.get(path, set())]
        duplicates = sorted(key for key, count in Counter(ordered_ids).items() if count > 1)
        if duplicates:
            errors.append(f"{split} KPGT sources contain duplicate IDs that the legacy loader would overwrite: {duplicates}")
        missing_drugs = sorted(set(rows["drug_id"]) - set(ordered_ids))
        missing_targets = sorted(set(rows["sequence_hash"]) - set(graph_paths))
        report["splits"][split].update(missing_drug_ids=missing_drugs, missing_graph_ids=missing_targets)
        if missing_drugs:
            errors.append(f"{split}: missing KPGT features for {len(missing_drugs)} drug IDs: {missing_drugs}")
        if missing_targets:
            errors.append(f"{split}: missing protein graphs for {len(missing_targets)} target IDs: {missing_targets}")

    cl = config.get("contrastive", {})
    if bool(cl.get("enabled", config.get("use_contrastive", False))):
        for entity, csv_key, matrix_key, id_column, used_ids in (
            ("drug", "drugs_csv", "drug_sim_npz", "Drug_ID", all_drugs),
            ("target", "targets_csv", "target_sim_npz", "Target_ID", all_targets),
        ):
            csv_value = cl.get(csv_key, data.get(csv_key))
            matrix_value = cl.get(matrix_key, data.get(matrix_key))
            if not csv_value or not matrix_value:
                errors.append(f"contrastive {entity} CSV and similarity NPZ paths must be specified")
                continue
            csv_path, matrix_path = Path(csv_value), Path(matrix_value)
            table = read_csv(csv_path, f"contrastive {entity} CSV")
            has_matrix = exists(matrix_path, f"contrastive {entity} NPZ")
            if table is None or id_column not in table:
                if table is not None:
                    errors.append(f"contrastive {entity} CSV missing {id_column}: {csv_path}")
                continue
            if table[id_column].isna().any():
                errors.append(f"contrastive {entity} CSV contains null identifiers: {csv_path}")
            ids = table[id_column].astype(str).tolist()
            if len(set(ids)) != len(ids):
                errors.append(f"contrastive {entity} CSV contains duplicate IDs: {csv_path}")
            missing = sorted(used_ids - set(ids))
            if missing:
                errors.append(f"contrastive {entity} CSV lacks selected split identifiers: {missing}")
            summary = {"csv_path": str(csv_path), "npz_path": str(matrix_path), "rows": len(ids), "id_order": ids,
                       "csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
                       "id_order_sha256": hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
                       "embedded_id_order_verified": False}
            report["features"]["contrastive"][entity] = summary
            if not has_matrix:
                continue
            try:
                with np.load(matrix_path, allow_pickle=True) as archive:
                    if not archive.files:
                        errors.append(f"contrastive {entity} NPZ has no arrays: {matrix_path}")
                        continue
                    # This priority is identical to the frozen trainer.
                    key = "arr_0" if "arr_0" in archive.files else archive.files[0]
                    matrix = archive[key]
                    summary.update(matrix_key=key, shape=list(matrix.shape), dtype=str(matrix.dtype))
                    if matrix.shape != (len(ids), len(ids)):
                        errors.append(f"contrastive {entity} similarity shape {matrix.shape} does not match CSV rows={len(ids)}: {matrix_path}")
                    if not np.issubdtype(matrix.dtype, np.number) or not np.isfinite(matrix).all():
                        errors.append(f"contrastive {entity} matrix must contain finite numeric values: {matrix_path}")
                    for id_key in (f"{entity}_ids", "ids"):
                        if id_key in archive.files:
                            summary["embedded_id_order_verified"] = True
                            if archive[id_key].ndim != 1 or [str(value) for value in archive[id_key]] != ids:
                                errors.append(f"contrastive {entity} NPZ embedded identifiers disagree with CSV row order: {matrix_path}")
            except (OSError, ValueError, TypeError) as exc:
                errors.append(f"cannot inspect contrastive {entity} NPZ {matrix_path}: {exc}")

    if inspect_features:
        import torch
        from torch_geometric.data import Data

        def get(graph, key):
            return getattr(graph, key, None)

        for target, path in graph_paths.items():
            entry = {"path": str(path)}
            report["features"]["graphs"][target] = entry
            try:
                graph = torch.load(path, map_location="cpu", weights_only=False)
                if not isinstance(graph, Data):
                    errors.append(f"graph {target} must be torch_geometric.data.Data for the frozen collator: {path}")
                    continue
                tensors = {}
                for key, dimension in (("x", model.get("esm_dim", 1280)), ("x_esm", model.get("esm_dim", 1280)), ("x_dyn", model.get("dyn_dim", 50)),
                                       ("predicted_dynamic_residue_features", model.get("dyn_dim", 50)),
                                       ("esmdance_35m_embedding", model.get("esmdance35_dim", 480))):
                    tensor = get(graph, key)
                    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2 or tensor.shape[1] != int(dimension):
                        errors.append(f"graph {target} missing/invalid {key}; expected [N,{dimension}]: {path}")
                        continue
                    tensors[key] = tensor
                    if tensor.is_complex():
                        errors.append(f"graph {target} {key} must contain real values: {path}")
                    elif not bool(torch.isfinite(tensor).all()):
                        errors.append(f"graph {target} {key} must contain finite values: {path}")
                    elif not bool(torch.isfinite(tensor.float()).all()):
                        errors.append(f"graph {target} {key} values overflow the frozen model's float32 conversion: {path}")
                nodes = int(tensors["x_esm"].shape[0]) if "x_esm" in tensors else None
                entry["nodes"] = nodes
                if nodes is not None:
                    authoritative_nodes = graph.num_nodes if "x" in tensors else None
                    entry["num_nodes"] = _json_value(authoritative_nodes)
                    if authoritative_nodes != nodes:
                        errors.append(f"graph {target} num_nodes={authoritative_nodes} differs from x_esm={nodes}: {path}")
                    if nodes <= 0:
                        errors.append(f"graph {target} contains zero nodes: {path}")
                    if nodes > 1500:
                        errors.append(f"graph {target} has {nodes} nodes beyond historical front1500: {path}")
                    for key, tensor in tensors.items():
                        if tensor.shape[0] != nodes:
                            errors.append(f"graph {target} {key} residue count {tensor.shape[0]} differs from x_esm={nodes}: {path}")
                    for canonical, alias_keys in (("x_esm", ("x",)),
                                                  ("predicted_dynamic_residue_features", ("x_dyn", "esmdance_residue_properties")),
                                                  ("esmdance_35m_embedding", ("esmdance35_embedding", "esmdance_encoder_embedding"))):
                        canonical_tensor = tensors.get(canonical)
                        for alias in alias_keys:
                            alias_tensor = get(graph, alias)
                            if alias_tensor is not None and (not isinstance(alias_tensor, torch.Tensor) or canonical_tensor is None or not torch.equal(alias_tensor, canonical_tensor)):
                                errors.append(f"graph {target} alias {alias} differs from {canonical}: {path}")
                mask = get(graph, "esmdance_embedding_mask")
                if not isinstance(mask, torch.Tensor) or mask.ndim != 2 or mask.shape[1] != 1 or (nodes is not None and mask.shape[0] != nodes):
                    errors.append(f"graph {target} missing/invalid esmdance_embedding_mask; expected [N,1]: {path}")
                elif mask.is_complex() or not bool(torch.isfinite(mask).all()) or not bool(((mask == 0) | (mask == 1)).all()):
                    errors.append(f"graph {target} mask must contain finite binary 0/1 values: {path}")
                else:
                    covered = mask[:, 0].bool()
                    entry["covered_nodes"] = int(covered.sum())
                    zeros = (~covered).nonzero().flatten()
                    prefix = int(zeros[0]) if len(zeros) else len(covered)
                    entry.update(covered_prefix=prefix, coverage=float(covered.float().mean()), partial=not bool(covered.all()))
                    if not bool(covered.any()):
                        errors.append(f"graph {target} auxiliary mask is all zero; missing auxiliary fallback is forbidden: {path}")
                    if bool(covered[prefix:].any()):
                        errors.append(f"graph {target} auxiliary mask must describe a contiguous covered prefix: {path}")
                for alias in ("residue_valid_mask", "node_valid_mask", "valid_mask", "node_mask"):
                    attention_mask = get(graph, alias)
                    if attention_mask is None:
                        continue
                    if not isinstance(attention_mask, torch.Tensor) or nodes is None or attention_mask.numel() != nodes:
                        errors.append(f"graph {target} {alias} must have N={nodes} node entries: {path}")
                    elif attention_mask.is_complex() or not bool(torch.isfinite(attention_mask).all()) or not bool(((attention_mask == 0) | (attention_mask == 1)).all()):
                        errors.append(f"graph {target} {alias} must contain finite binary real 0/1 values: {path}")
                    elif not bool(attention_mask.bool().any()):
                        errors.append(f"graph {target} {alias} must leave at least one valid residue per protein: {path}")
                for key in ("sequence_hash", "target_id"):
                    value = get(graph, key)
                    if value is None or str(value) != target:
                        errors.append(f"graph {target} {key} identity mismatch ({value!r}): {path}")
                version = get(graph, "graph_version")
                if version != graph_version:
                    errors.append(f"graph {target} graph_version mismatch ({version!r}): {path}")
                edges = get(graph, "edge_index")
                if not isinstance(edges, torch.Tensor) or edges.ndim != 2 or edges.shape[0] != 2 or edges.dtype not in (torch.int32, torch.int64):
                    errors.append(f"graph {target} invalid edge_index; expected integer [2,E]: {path}")
                elif nodes is not None and edges.numel() and (int(edges.min()) < 0 or int(edges.max()) >= nodes):
                    errors.append(f"graph {target} edge_index exceeds node bounds: {path}")
                metadata = get(graph, "esmdance_meta")
                entry["metadata"] = _json_value(metadata)
                entry["effective_len"] = _json_value(get(graph, "effective_len"))
                entry["full_sequence_len"] = _json_value(get(graph, "full_sequence_len"))
                if isinstance(metadata, dict) and "covered_len" in metadata and "covered_nodes" in entry and metadata["covered_len"] != entry["covered_nodes"]:
                    warnings.append(f"graph {target} metadata covered_len={metadata['covered_len']} differs from true mask coverage={entry['covered_nodes']}; cache preserved")
                effective = get(graph, "effective_len")
                if nodes is not None and effective is not None and int(effective) != nodes:
                    warnings.append(f"graph {target} effective_len={effective} differs from tensor nodes={nodes}; cache preserved")
            except Exception as exc:
                errors.append(f"cannot inspect graph {target} {path}: {type(exc).__name__}: {exc}")
    else:
        report["features"]["graphs"] = {target: {"path": str(path), "inspected": False} for target, path in graph_paths.items()}
    entries = report["features"]["graphs"].values()
    report["features"]["graph_summary"] = {
        "referenced_targets": len(all_targets), "present_graphs": len(graph_paths),
        "inspected_graphs": len(graph_paths) if inspect_features else 0,
        "max_nodes": max((entry.get("nodes") or 0 for entry in entries), default=0),
        "partial_auxiliary_graphs": sum(bool(entry.get("partial")) for entry in report["features"]["graphs"].values()),
    }
    report["missing_paths"] = sorted(set(report["missing_paths"]))
    report["valid"] = not errors
    return report
