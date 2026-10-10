"""Summarize verified completed runs without promoting partial seed groups."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path, PurePosixPath
import re
import statistics

from .artifacts import atomic_json, recipe_identity, recipe_sha256, sha256


SEEDS = (42, 1234, 3407, 5678, 8888)
REQUIRED_FILES = frozenset({'checkpoints/best.pt', 'checkpoints/last.pt', 'metrics.csv', 'config.yaml',
                  'test_metrics.json', 'test_predictions.csv', 'checkpoint_lock.json',
                  'training_manifest.json', 'test_input_manifest.json', 'runtime.json'})
METRICS = ('AUROC', 'AUPR', 'ACC', 'Loss')
CSV_FIELDS = ('run', 'dataset', 'setting', 'seed', *METRICS)


def read_completed_run(folder):
    """Verify completion metadata against all ten hashed training artifacts.

    External source/input identity checks belong to the reproduction runner:
    a transferred run can be summarized without its original feature caches.
    Checkpoints are loaded with the restricted weights-only deserializer.
    """
    import torch
    import yaml
    from .config import compare_paper_config

    folder = Path(folder).resolve()

    def mapping(name):
        value = json.loads((folder / name).read_text())
        if not isinstance(value, dict):
            raise ValueError(f'{name} must contain an object')
        return value

    def finite(value, field, *, nonnegative=False):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or (nonnegative and value < 0):
            raise ValueError(f'{field} must be finite' + (' and nonnegative' if nonnegative else ''))
        return float(value)

    def positive_int(value, field):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f'{field} must be a positive integer')
        return value

    def hash_map(value, field):
        if not isinstance(value, dict) or not value:
            raise ValueError(f'{field} must contain input/source hash evidence')
        for name, digest in value.items():
            if not isinstance(name, str) or not name or not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
                raise ValueError(f'{field} contains invalid hash evidence')

    record = mapping('completed.json')
    if record.get('complete') is not True or record.get('diagnostic') is not False:
        raise ValueError('Partial or diagnostic completion cannot be summarized')
    hashes = record.get('verified_files')
    if not isinstance(hashes, dict) or not REQUIRED_FILES.issubset(hashes):
        raise ValueError('Incomplete completion evidence file set: all ten training artifacts are required')
    hash_map(hashes, 'verified_files')
    for name, digest in hashes.items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or '\\' in name:
            raise ValueError(f'Invalid artifact path: {name}')
        candidate = (folder / name).resolve()
        if not candidate.is_relative_to(folder):
            raise ValueError(f'Artifact path escapes run folder: {name}')
        if not candidate.is_file() or sha256(candidate) != digest:
            raise ValueError(f'Completed artifact changed: {name}')

    config = yaml.safe_load((folder / 'config.yaml').read_text())
    if not isinstance(config, dict):
        raise ValueError('config.yaml must contain a configuration mapping')
    manifest, lock = mapping('training_manifest.json'), mapping('checkpoint_lock.json')
    metrics, runtime = mapping('test_metrics.json'), mapping('runtime.json')
    test_inputs = mapping('test_input_manifest.json')
    hash_map(manifest.get('source_sha256'), 'training source manifest')
    hash_map(manifest.get('input_sha256'), 'training input manifest')
    hash_map(test_inputs, 'test input manifest')
    if manifest.get('diagnostic') is not False or manifest.get('max_batches') is not None or config.get('dry_run', False):
        raise ValueError('Diagnostic or bounded-batch training cannot count as complete')
    overrides = record.get('runtime_overrides', {})
    if not isinstance(overrides, dict) or overrides != manifest.get('runtime_overrides', {}):
        raise ValueError('Completion runtime_overrides differ from training manifest')
    if overrides and (set(overrides) != {'pin_memory'} or overrides['pin_memory'] is not False):
        raise ValueError('Completion has unsupported runtime_overrides')
    if manifest.get('test_opened_during_selection') is not False or lock.get('test_used_for_selection') is not False:
        raise ValueError('Completion lacks validation-only checkpoint selection evidence')

    seed = positive_int(config.get('seed'), 'configuration seed')
    for name, expected in (('seed', seed), ('dataset', config.get('dataset')), ('setting', config.get('biofusion_setting'))):
        if expected is None or record.get(name) != expected or manifest.get(name) != expected:
            raise ValueError(f'Completion {name} identity differs from config/training manifest')
    recipe = recipe_sha256(config)
    if record.get('recipe_sha256') != recipe or manifest.get('recipe_sha256') != recipe:
        raise ValueError('Completion recipe identity differs from configuration')
    paper = compare_paper_config(config)['matches']
    if record.get('original_paper_recipe') is not paper or manifest.get('original_paper_recipe') is not paper:
        raise ValueError('Completion original_paper_recipe identity differs from configuration')
    if record.get('test_metrics') != metrics:
        raise ValueError('Completion test_metrics differ from hashed test_metrics.json')
    for name in METRICS:
        finite(metrics.get(name), f'test {name}')
    for name, value in metrics.items():
        if isinstance(value, (int, float)):
            finite(value, f'test {name}')
    finite(record.get('training_seconds'), 'training_seconds', nonnegative=True)

    try:
        best = torch.load(folder / 'checkpoints/best.pt', map_location='cpu', weights_only=True)
        last = torch.load(folder / 'checkpoints/last.pt', map_location='cpu', weights_only=True)
    except Exception as error:
        raise ValueError(f'Cannot read completion checkpoint evidence: {error}') from error
    for name, state in (('best', best), ('last', last)):
        if not isinstance(state, dict) or state.get('experiment') != 'E4_GAUSS_NODE_FUSION' or not isinstance(state.get('config'), dict):
            raise ValueError(f'{name} checkpoint has invalid Gaussian metadata')
        if recipe_identity(state['config']) != recipe_identity(config):
            raise ValueError(f'{name} checkpoint recipe/seed identity differs from configuration')
        if state.get('graph_version') != config.get('graph_version'):
            raise ValueError(f'{name} checkpoint graph_version differs from configuration')
        positive_int(state.get('epoch'), f'{name} checkpoint epoch')
    if lock.get('checkpoint_sha256') != hashes['checkpoints/best.pt'] or lock.get('best_epoch') != best['epoch']:
        raise ValueError('Checkpoint lock hash/best_epoch differs from actual best checkpoint')
    if lock.get('validation_metrics') != best.get('val_metrics'):
        raise ValueError('Checkpoint lock validation_metrics differ from actual best checkpoint')
    selection = str(config.get('save_best_by', 'AUPR'))
    metric = {'auroc': 'AUROC', 'aupr': 'AUPR', 'mcc': 'MCC', 'f1': 'F1'}.get(selection.lower(), selection)
    if lock.get('selection_metric') != selection or metrics.get('best_val_metric') != metric:
        raise ValueError('Checkpoint lock/test selection metric differs from configuration')
    best_score = finite(best.get('val_metrics', {}).get(metric), 'selected validation metric')
    if metrics.get('best_epoch') != best['epoch'] or metrics.get('best_val_score') != best_score:
        raise ValueError('Test best_epoch/best_val_score differs from selected checkpoint')

    with (folder / 'metrics.csv').open(newline='') as stream:
        history = list(csv.DictReader(stream))
    epochs = runtime.get('epochs')
    count = positive_int(record.get('epochs_run'), 'epochs_run')
    if not isinstance(epochs, list) or len(epochs) != count or len(history) != count or last['epoch'] != count:
        raise ValueError('Completion epoch count differs from runtime/history/last checkpoint')
    if count > config.get('epochs', 0) or best['epoch'] > count:
        raise ValueError('Completion epoch count exceeds configured training')
    selected_column = 'val_' + metric.lower()
    validation = manifest.get('input_validation', {})
    if not isinstance(validation, dict) or not isinstance(validation.get('splits', {}), dict):
        raise ValueError('Invalid input validation evidence')
    train_validation = validation.get('splits', {}).get('train', {})
    if not isinstance(train_validation, dict):
        raise ValueError('Invalid training input validation evidence')
    expected_pairs = train_validation.get('rows')
    if expected_pairs is not None:
        positive_int(expected_pairs, 'validated train rows')
    scores = []
    for index, (row, timing) in enumerate(zip(history, epochs), start=1):
        if not isinstance(timing, dict) or int(row.get('epoch', 0)) != index or timing.get('epoch') != index:
            raise ValueError('Completion history/runtime epochs must be contiguous')
        for field in ('train_seconds', 'validation_seconds', 'peak_reserved_gib'):
            finite(timing.get(field), f'runtime {field}', nonnegative=True)
        pairs = positive_int(timing.get('train_pairs'), 'runtime train_pairs')
        if expected_pairs is not None and pairs != expected_pairs:
            raise ValueError('Runtime train_pairs differ from validated training membership')
        batches = positive_int(timing.get('train_batches'), 'runtime train_batches')
        batch_size = positive_int(config.get('batch_size'), 'configuration batch_size')
        if batches != math.ceil(pairs / batch_size):
            raise ValueError('Completion runtime records bounded/incomplete training batches')
        for field in ('train_loss', 'val_loss', 'val_auroc', 'val_aupr', 'val_acc'):
            finite(float(row.get(field, 'nan')), f'history {field}')
        scores.append(finite(float(row.get(selected_column, 'nan')), 'history selection metric'))
    selected_epoch = max(range(count), key=lambda index: scores[index]) + 1
    if selected_epoch != best['epoch'] or not math.isclose(scores[selected_epoch - 1], best_score, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError('Selected checkpoint is inconsistent with validation history')
    return record


def summarize_runs(runs, output):
    runs, output = Path(runs), Path(output)
    records, rejected, grouped = [], [], {}
    for path in sorted(runs.glob('*/completed.json')):
        try:
            value = read_completed_run(path.parent)
            key = (value['dataset'], value['setting'], value['recipe_sha256'])
            if value['seed'] in grouped.setdefault(key, {}):
                raise ValueError('Duplicate completed seed in a recipe group')
            grouped[key][value['seed']] = value
            records.append({'run': path.parent.name, 'dataset': value['dataset'], 'setting': value['setting'],
                            'seed': value['seed'], **value['test_metrics']})
        except (OSError, ValueError, KeyError, TypeError, csv.Error) as error:
            rejected.append({'run': path.parent.name, 'error': str(error)})
    groups = []
    for (dataset, setting, recipe), by_seed in grouped.items():
        seeds = sorted(by_seed)
        group = {'dataset': dataset, 'setting': setting, 'recipe_sha256': recipe,
                 'seeds': seeds, 'n_completed': len(seeds), 'missing_seeds': [s for s in SEEDS if s not in seeds],
                 'complete': set(seeds) == set(SEEDS)}
        if group['complete']:
            group['std_ddof'] = 1
            metrics = ('AUROC', 'AUPR', 'ACC', 'Loss')
            group['mean'] = {name: statistics.mean(float(by_seed[s]['test_metrics'][name]) for s in SEEDS)
                             for name in metrics}
            group['sample_std'] = {name: statistics.stdev(float(by_seed[s]['test_metrics'][name]) for s in SEEDS)
                                   for name in metrics}
        groups.append(group)
    result = {'groups': groups, 'completed_runs': records, 'rejected_runs': rejected}
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / 'summary.json', result)
    fields = sorted(set(CSV_FIELDS).union(*(row.keys() for row in records)))
    with (output / 'seed_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    lines = ['# SAGF-DTI reproduction results', '',
             'Five-seed means and sample standard deviations (ddof=1) are reported only after all five runs have verified completion evidence.', '']
    for group in groups:
        lines.append(f"{group['dataset']} {group['setting']}: {group['n_completed']}/5 seeds complete.")
        if group['complete']:
            lines.extend(f"- {key}: {group['mean'][key]:.4f} ± {group['sample_std'][key]:.4f}"
                         for key in ('AUROC', 'AUPR', 'ACC', 'Loss'))
        else:
            lines.append('Missing seeds: ' + ', '.join(map(str, group['missing_seeds'])))
        lines.append('')
    if rejected:
        lines.append(f'{len(rejected)} runs have missing or changed evidence; see summary.json.')
    (output / 'results.md').write_text('\n'.join(lines) + '\n')
    return result
