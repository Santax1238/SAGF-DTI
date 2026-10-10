"""Small artifact and identity helpers for auditable standalone runs."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def recipe_identity(config, *, include_seed=True):
    """Numerical recipe identity, excluding input/output file locations."""
    from .config import PATH_FIELDS
    result = deepcopy(config)
    result.pop('run_root', None)
    result.pop('run_name', None)
    result.pop('device', None)
    if not include_seed:
        result.pop('seed', None)
    for section, keys in PATH_FIELDS.items():
        for key in keys:
            result.get(section, {}).pop(key, None)
    return result


def recipe_sha256(config, *, include_seed=False):
    return hashlib.sha256(json.dumps(recipe_identity(config, include_seed=include_seed),
                         sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def environment_record(device='cpu'):
    import importlib.metadata
    import torch
    packages = {}
    for name in ('torch', 'torch-geometric', 'numpy', 'pandas', 'scikit-learn', 'PyYAML'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result = {'python': platform.python_version(), 'packages': packages,
              'cuda_runtime': torch.version.cuda, 'device': str(device)}
    if torch.device(device).type == 'cuda':
        result['gpu'] = torch.cuda.get_device_name(device)
        result['gpu_memory_gib'] = torch.cuda.get_device_properties(device).total_memory / 1024**3
    return result


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    files = list((root / 'sagf_dti').glob('*.py'))
    files.extend((root / 'my_experiments').rglob('*.py'))
    files.extend((root / 'provenance').glob('*.json'))
    return {str(path.relative_to(root)): sha256(path) for path in sorted(files)}


def input_files(config, splits=('train', 'val')):
    """Identify used caches; callers control when the test split is opened."""
    import pandas as pd
    from my_experiments.datasets import DTIDataset
    files, targets = set(), set()
    data = config['data']
    for split in splits:
        csv_path = Path(data[f'{split}_csv'])
        files.add(csv_path)
        rows = DTIDataset._normalize_rows(pd.read_csv(csv_path))
        targets.update(rows['sequence_hash'].astype(str))
        key = 'davis_drug_npz' if split == 'test' else 'bindingdb_drug_npz'
        paths = data[key] if isinstance(data[key], list) else [data[key]]
        files.update(Path(path) for path in paths)
    graph_roots = data['graph_root']
    graph_roots = graph_roots if isinstance(graph_roots, list) else [graph_roots]
    for target in sorted(targets):
        candidates = [Path(root) / config['graph_version'] / f'{target}.pt' for root in graph_roots]
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            raise FileNotFoundError(f'Missing graph for target {target}')
        files.add(path)
    contrastive = config.get('contrastive', {})
    if contrastive.get('enabled', False):
        for key in ('drugs_csv', 'targets_csv', 'drug_sim_npz', 'target_sim_npz'):
            files.add(Path(contrastive.get(key, data.get(key))))
    return sorted(files)


def input_hashes(config, splits=('train', 'val')):
    return {str(path.resolve()): sha256(path) for path in input_files(config, splits)}


def verified_files(folder, names):
    return {name: sha256(Path(folder) / name) for name in names}
