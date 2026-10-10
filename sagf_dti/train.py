"""Portable execution of the frozen paper trainer, with run evidence."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import re
import time

from .artifacts import (atomic_json, environment_record, input_hashes, recipe_sha256,
                        sha256, source_hashes, utc_now, verified_files)


def claim_run_directory(output, run_name):
    if not isinstance(run_name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', run_name):
        raise ValueError('run_name must be a single directory name')
    path = Path(output).expanduser().resolve() / run_name
    path.mkdir(parents=True, exist_ok=False)
    return path


@contextmanager
def patched_trainer(trainer, **overrides):
    previous = {name: getattr(trainer, name) for name in overrides}
    try:
        for name, value in overrides.items():
            setattr(trainer, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(trainer, name, value)


def train(config_path, data_root, output, *, device=None, dry_run=False,
          max_batches=None, require_paper_config=False, disable_host_pinning=False):
    import torch
    import yaml
    from .config import assert_paper_config, compare_paper_config, resolve_config
    from .data import validate_data
    from .legacy import load_trainer

    if max_batches is not None and (not dry_run or max_batches < 1):
        raise ValueError('--max-batches requires --dry-run and a positive value')
    config = resolve_config(config_path, data_root, output)
    comparison = assert_paper_config(config) if require_paper_config else compare_paper_config(config)
    validation = validate_data(config, splits=('train', 'val'), inspect_features=True)
    if not validation['valid']:
        raise ValueError('Input validation failed: ' + '; '.join(validation['errors']))
    trainer = load_trainer()
    selected_device = device or config.get('device', 'cpu')
    folder = claim_run_directory(output, config['run_name'])
    launch_config = folder / 'launch_config.yaml'
    launch_config.write_text(yaml.safe_dump(config, sort_keys=False))
    runtime_overrides = {'pin_memory': False} if disable_host_pinning else {}
    manifest = {'started_utc': utc_now(), 'seed': config['seed'],
                'dataset': config.get('dataset'), 'setting': config.get('biofusion_setting'),
                'original_paper_recipe': comparison['matches'],
                'recipe_differences': comparison['differences'], 'diagnostic': bool(dry_run),
                'max_batches': max_batches, 'recipe_sha256': recipe_sha256(config),
                'runtime_overrides': runtime_overrides,
                'source_sha256': source_hashes(), 'input_sha256': input_hashes(config),
                'environment': environment_record(selected_device), 'input_validation': validation,
                'test_opened_during_selection': False}
    atomic_json(folder / 'training_manifest.json', manifest)
    epoch_records, test_inputs = [], {}
    original_train, original_evaluate, original_loader = trainer.train_one_epoch, trainer.evaluate, trainer.make_loader

    def measured_train(*args, **kwargs):
        device_arg = args[5]
        if device_arg.type == 'cuda':
            torch.cuda.synchronize(device_arg)
            torch.cuda.reset_peak_memory_stats(device_arg)
        start = time.monotonic()
        result = original_train(*args, **kwargs)
        if device_arg.type == 'cuda':
            torch.cuda.synchronize(device_arg)
        loader = args[1]
        count = len(loader) if max_batches is None else min(len(loader), max_batches)
        record = {'epoch': len(epoch_records) + 1, 'train_seconds': time.monotonic() - start,
                  'train_pairs': len(loader.dataset), 'train_batches': count,
                  'peak_reserved_gib': torch.cuda.max_memory_reserved(device_arg) / 1024**3
                    if device_arg.type == 'cuda' else 0.}
        epoch_records.append(record)
        atomic_json(folder / 'runtime.json', {'epochs': epoch_records, 'updated_utc': utc_now()})
        return result

    def guarded_loader(cfg, split, graph_version, batch_size, shuffle):
        if split == 'test':
            best = folder / 'checkpoints' / 'best.pt'
            if not best.is_file():
                raise RuntimeError('A selected checkpoint is required before opening test data')
            state = torch.load(best, map_location='cpu', weights_only=False)
            lock = {'checkpoint_sha256': sha256(best), 'best_epoch': state['epoch'],
                    'selection_metric': cfg['save_best_by'],
                    'validation_metrics': state['val_metrics'], 'locked_utc': utc_now(),
                    'test_used_for_selection': False}
            atomic_json(folder / 'checkpoint_lock.json', lock)
            check = validate_data(cfg, splits=('test',), inspect_features=True)
            if not check['valid']:
                raise ValueError('Test input validation failed: ' + '; '.join(check['errors']))
            test_inputs.update(input_hashes(cfg, splits=('test',)))
            atomic_json(folder / 'test_input_manifest.json', test_inputs)
        # Preserve the archived recipe in checkpoints. This explicit transfer-only
        # override handles WSL host pinning limits without changing worker RNG,
        # batches, forward precision, optimizer or any loss calculation.
        loader = original_loader({**cfg, **runtime_overrides}, split, graph_version, batch_size, shuffle)
        expected = validation['splits'].get(split, {}).get('rows')
        if expected is not None and len(loader.dataset) != expected:
            raise ValueError(f'Legacy loader changed {split} membership: {len(loader.dataset)} != {expected}')
        return loader

    def measured_evaluate(*args, **kwargs):
        start = time.monotonic()
        result = original_evaluate(*args, **kwargs)
        path = kwargs.get('save_predictions')
        if epoch_records and (path is None or Path(path).name == 'val_predictions.csv'):
            epoch_records[-1]['validation_seconds'] = time.monotonic() - start
            atomic_json(folder / 'runtime.json', {'epochs': epoch_records, 'updated_utc': utc_now()})
        return result

    arguments = argparse.Namespace(config=launch_config, experiment=None, run_name=None,
                                   device=selected_device, seed=None, dry_run=dry_run,
                                   max_batches=max_batches)
    started = time.monotonic()
    try:
        with patched_trainer(trainer, parse_args=lambda: arguments,
                             train_one_epoch=measured_train, evaluate=measured_evaluate,
                             make_loader=guarded_loader):
            return_code = trainer.main()
        if return_code != 0:
            raise RuntimeError(f'Frozen trainer returned {return_code}')
        if source_hashes() != manifest['source_sha256']:
            raise ValueError('Training source changed during this run')
        for path, digest in {**manifest['input_sha256'], **test_inputs}.items():
            if sha256(path) != digest:
                raise ValueError(f'Training input changed: {path}')
        if dry_run:
            record = {'complete': True, 'diagnostic': True, 'training_seconds': time.monotonic()-started,
                      'epochs': epoch_records, 'test_evaluated': False,
                      'runtime_overrides': runtime_overrides}
            atomic_json(folder / 'diagnostic.json', record)
            return record
        required = ('checkpoints/best.pt', 'checkpoints/last.pt', 'metrics.csv', 'config.yaml',
                    'test_metrics.json', 'test_predictions.csv', 'checkpoint_lock.json',
                    'training_manifest.json', 'test_input_manifest.json', 'runtime.json')
        metrics = json.loads((folder / 'test_metrics.json').read_text())
        if not all(math.isfinite(float(metrics[key])) for key in ('AUROC', 'AUPR', 'ACC', 'Loss')):
            raise ValueError('Completed test metrics must be finite')
        record = {'complete': True, 'diagnostic': False, 'seed': config['seed'],
                  'dataset': config.get('dataset'), 'setting': config.get('biofusion_setting'),
                  'original_paper_recipe': comparison['matches'],
                  'recipe_sha256': manifest['recipe_sha256'], 'test_metrics': metrics,
                  'runtime_overrides': runtime_overrides,
                  'epochs_run': len(epoch_records), 'training_seconds': time.monotonic()-started,
                  'completed_utc': utc_now(), 'verified_files': verified_files(folder, required)}
        atomic_json(folder / 'completed.json', record)
        return record
    except BaseException as error:
        atomic_json(folder / 'failure.json', {'complete': False, 'error': str(error),
                                             'type': type(error).__name__, 'failed_utc': utc_now()})
        raise
