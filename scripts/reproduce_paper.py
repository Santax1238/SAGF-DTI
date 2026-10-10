#!/usr/bin/env python3
"""Sequential, identity-locked original SNAP cold-protein five-seed reproduction."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import fcntl
from importlib.resources import files
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from sagf_dti.artifacts import atomic_json, input_hashes, recipe_sha256, sha256, source_hashes, utc_now


SEEDS = (42, 1234, 3407, 5678, 8888)
TRAIN_COUNTS = {42: 19242, 1234: 19290, 3407: 19021, 5678: 18983, 8888: 19653}


@contextmanager
def runner_lock(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'runner.lock').open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'Reproduction is already running: {output}') from error
        stream.seek(0)
        stream.truncate()
        stream.write(json.dumps({'pid': os.getpid(), 'started_utc': utc_now()}) + '\n')
        stream.flush()
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def bind_identity(output, identity):
    path = Path(output) / 'reproduction_identity.json'
    if path.exists():
        if json.loads(path.read_text()) != identity:
            raise ValueError('Reproduction identity changed: source, protocol, configuration or input')
    else:
        atomic_json(path, identity)


def verify_completed_run(run, identity):
    from sagf_dti.report import read_completed_run
    run = Path(run)
    if not run.exists():
        return None
    completed = run / 'completed.json'
    if not completed.is_file():
        raise ValueError(f'Partial run requires an explicit new output root; refusing overwrite: {run}')
    record = read_completed_run(run)
    expected = identity.get('expected_runs', {}).get(run.name)
    if expected is not None and any(record.get(key) != value for key, value in expected.items()):
        raise ValueError(f'Completed recipe/seed identity differs from the requested paper run: {run}')
    manifest = json.loads((run / 'training_manifest.json').read_text())
    if manifest.get('runtime_overrides', {}) != identity.get('runtime_overrides', {}):
        raise ValueError(f'Completed runtime overrides changed: {run}')
    if manifest['source_sha256'] != identity['training_source_sha256']:
        raise ValueError(f'Completed source identity changed: {run}')
    for path, digest in manifest['input_sha256'].items():
        if identity['input_sha256'].get(path) != digest or sha256(path) != digest:
            raise ValueError(f'Completed input changed: {path}')
    test_manifest = run / 'test_input_manifest.json'
    if test_manifest.is_file():
        for path, digest in json.loads(test_manifest.read_text()).items():
            if sha256(path) != digest:
                raise ValueError(f'Completed test input changed: {path}')
    return record


def compare_reload(original, reloaded, metrics_original, metrics_reloaded, *, metric_tolerance=5e-5):
    import numpy as np
    import pandas as pd
    left, right = pd.read_csv(original), pd.read_csv(reloaded)
    identity_columns = [key for key in ('pair_id', 'drug_id', 'sequence_hash', 'label') if key in left]
    if len(left) != len(right) or any(key not in right or not left[key].equals(right[key]) for key in identity_columns):
        raise ValueError('Reloaded predictions changed pair membership/order/labels')
    differences = {}
    changed = np.zeros(len(left), dtype=bool)
    for key in ('pred_logit', 'pred_prob'):
        values = np.abs(left[key].to_numpy(float) - right[key].to_numpy(float))
        if not np.isfinite(values).all():
            raise ValueError('Reloaded predictions are non-finite')
        changed |= values != 0
        differences[key] = {'max_abs': float(values.max(initial=0)), 'mean_abs': float(values.mean())}
    metric_differences = {key: abs(float(metrics_original[key])-float(metrics_reloaded[key]))
                          for key in ('AUROC', 'AUPR')}
    within = all(math.isfinite(value) and value <= metric_tolerance for value in metric_differences.values())
    result = {'prediction_rows_differ': int(changed.sum()), 'n_pairs': len(left),
              'bitwise_prediction_values_equal': not bool(changed.any()),
              'prediction_differences': differences, 'metric_differences': metric_differences,
              'metric_tolerance': metric_tolerance, 'metrics_within_tolerance': within,
              'note': 'GPU roundoff may change predictions; exact equality is recorded, not assumed.'}
    if not within:
        raise ValueError('Checkpoint reload exceeded metric tolerance: ' + json.dumps(result))
    return result


def verify_reload(run, completed):
    run = Path(run)
    reload = run / 'checkpoint_reload_test'
    evaluation_path, comparison_path = reload / 'evaluation.json', run / 'reload_comparison.json'
    if not evaluation_path.is_file() or not comparison_path.is_file():
        raise ValueError(f'Partial checkpoint reload cannot be overwritten: {reload}')
    evaluation = json.loads(evaluation_path.read_text())
    comparison = json.loads(comparison_path.read_text())
    evidence = {'evaluation_sha256': evaluation_path,
                'original_predictions_sha256': run / 'test_predictions.csv',
                'reloaded_predictions_sha256': reload / 'test_predictions.csv'}
    for key, path in evidence.items():
        if comparison.get(key) != sha256(path):
            raise ValueError(f'Checkpoint reload evidence changed: {path}')
    if (not evaluation.get('complete') or evaluation.get('split') != 'test' or
        evaluation['checkpoint_sha256'] != sha256(run / 'checkpoints' / 'best.pt') or
        evaluation['predictions_sha256'] != sha256(reload / 'test_predictions.csv')):
        raise ValueError(f'Checkpoint reload evidence changed: {run}')
    verified = compare_reload(run / 'test_predictions.csv', reload / 'test_predictions.csv',
                              completed['test_metrics'], evaluation['metrics'])
    if any(comparison.get(key) != value for key, value in verified.items()):
        raise ValueError(f'Checkpoint reload comparison changed: {run}')
    return evaluation, comparison


def update_progress(status, status_path, seed, run):
    record = status['runs'][str(seed)]
    metrics = Path(run) / 'metrics.csv'
    if metrics.is_file():
        try:
            with metrics.open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            if rows and rows[-1].get('epoch'):
                record.update(latest_epoch=int(rows[-1]['epoch']), latest_metrics=rows[-1])
        except (OSError, ValueError, csv.Error):
            pass  # An incomplete line can be observed while the trainer appends.
    runtime = Path(run) / 'runtime.json'
    if runtime.is_file():
        try:
            epochs = json.loads(runtime.read_text()).get('epochs', [])
            if epochs:
                record['latest_runtime'] = epochs[-1]
        except (OSError, ValueError):
            pass
    status['updated_utc'] = utc_now()
    atomic_json(status_path, status)


def run_child(command, logfile, status, status_path, seed, run, *, poll_seconds=5):
    env = {**os.environ, 'PYTHONUNBUFFERED': '1'}
    with Path(logfile).open('a') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        status['runs'][str(seed)]['pid'] = process.pid
        try:
            while process.poll() is None:
                update_progress(status, status_path, seed, run)
                time.sleep(poll_seconds)
            update_progress(status, status_path, seed, run)
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    if process.returncode:
        status['runs'][str(seed)].update(status='failed', returncode=process.returncode,
                                        failed_utc=utc_now())
        update_progress(status, status_path, seed, run)
        raise RuntimeError(f'Seed {seed} subprocess exited {process.returncode}; see {logfile}')


def reproduction_identity(configs, data_root, output):
    from sagf_dti.config import assert_paper_config, resolve_config
    from sagf_dti.data import validate_data
    identity = {'protocol': 'original_SNAP_official_cold_protein_five_seed',
                'seeds': list(SEEDS), 'expected_train_counts': TRAIN_COUNTS,
                'training_source_sha256': source_hashes(),
                'runner_sha256': sha256(__file__), 'config_sha256': {}, 'input_sha256': {},
                'expected_runs': {}}
    # JSON object keys must agree before/after deserialization.
    identity['expected_train_counts'] = {str(seed): count for seed, count in TRAIN_COUNTS.items()}
    resolved = {}
    for seed, path in configs.items():
        cfg = resolve_config(path, data_root, output)
        assert_paper_config(cfg)
        check = validate_data(cfg, splits=('train', 'val'))
        if not check['valid']:
            raise ValueError('Input validation failed: ' + '; '.join(check['errors']))
        if check['splits']['train']['rows'] != TRAIN_COUNTS[seed]:
            raise ValueError(f'Original seed{seed} training count changed')
        resolved[seed] = cfg
        identity['expected_runs'][cfg['run_name']] = {
            'seed': seed, 'dataset': 'SNAP', 'setting': 'prot', 'original_paper_recipe': True,
            'recipe_sha256': recipe_sha256(cfg)}
        identity['config_sha256'][str(seed)] = sha256(path)
        identity['input_sha256'].update(input_hashes(cfg))
    return identity, resolved


def reproduce(data_root, output, profile_path, *, device='cuda', disable_host_pinning=False):
    import torch
    from sagf_dti.benchmark import assert_benchmark_ready
    from sagf_dti.report import summarize_runs
    if torch.device(device).type != 'cuda':
        raise ValueError('Original FP16 AMP paper reproduction requires CUDA')
    output = Path(output).expanduser().resolve()
    profile_path = Path(profile_path).expanduser().resolve()
    configs = {seed: Path(str(files('sagf_dti').joinpath(f'configs/paper/SNAP/prot/seed_{seed}.yaml')))
               for seed in SEEDS}
    with runner_lock(output):
        profile = json.loads(profile_path.read_text())
        assert_benchmark_ready(profile)
        overrides = {'pin_memory': False} if disable_host_pinning else {}
        if profile.get('runtime_overrides', {}) != overrides:
            raise ValueError('Runner runtime overrides must match the GPU benchmark')
        identity, resolved = reproduction_identity(configs, data_root, output)
        identity['runtime_overrides'] = overrides
        # Architectural/legacy hashes bind profiling to the exact numerical code.
        numeric = lambda hashes: {key: digest for key, digest in hashes.items()
                                  if key.startswith('my_experiments/') or key == 'sagf_dti/model.py'}
        if numeric(profile['source_sha256']) != numeric(identity['training_source_sha256']):
            raise ValueError('Model/trainer source changed since the GPU benchmark')
        for path, digest in profile['input_sha256'].items():
            if identity['input_sha256'].get(path) != digest:
                raise ValueError(f'Profiled input changed: {path}')
        identity['benchmark_sha256'] = sha256(profile_path)
        bind_identity(output, identity)
        status_path = output / 'reproduction_status.json'
        status = {'complete': False, 'started_utc': utc_now(), 'pid': os.getpid(),
                  'profile': str(profile_path), 'runs': {str(s): {'status': 'pending'} for s in SEEDS}}
        atomic_json(status_path, status)
        try:
            for seed in SEEDS:
                run = output / resolved[seed]['run_name']
                previous = verify_completed_run(run, identity)
                reused_training = previous is not None
                state = status['runs'][str(seed)]
                if previous is None:
                    state.update(status='training', started_utc=utc_now(), run=str(run))
                    command = [sys.executable, '-m', 'sagf_dti', 'train', '--config', str(configs[seed]),
                               '--data-root', str(data_root), '--output', str(output), '--device', device,
                               '--require-paper-config']
                    if disable_host_pinning:
                        command.append('--disable-host-pinning')
                    run_child(command, output / f'seed_{seed}_train.log', status, status_path, seed, run)
                    previous = verify_completed_run(run, identity)
                reload_dir = run / 'checkpoint_reload_test'
                reload_file = reload_dir / 'evaluation.json'
                if reload_dir.exists():
                    evaluation, comparison = verify_reload(run, previous)
                else:
                    state['status'] = 'checkpoint_reload_test'
                    command = [sys.executable, '-m', 'sagf_dti', 'evaluate', '--config', str(configs[seed]),
                               '--data-root', str(data_root), '--checkpoint', str(run / 'checkpoints' / 'best.pt'),
                               '--output', str(reload_dir), '--split', 'test', '--device', device]
                    if disable_host_pinning:
                        command.append('--disable-host-pinning')
                    run_child(command, output / f'seed_{seed}_reload.log', status, status_path, seed, run)
                    evaluation = json.loads(reload_file.read_text())
                    comparison = compare_reload(run / 'test_predictions.csv', reload_dir / 'test_predictions.csv',
                                                previous['test_metrics'], evaluation['metrics'])
                    comparison.update(evaluation_sha256=sha256(reload_file),
                                      original_predictions_sha256=sha256(run / 'test_predictions.csv'),
                                      reloaded_predictions_sha256=sha256(reload_dir / 'test_predictions.csv'))
                    atomic_json(run / 'reload_comparison.json', comparison)
                    evaluation, comparison = verify_reload(run, previous)
                state.update(status='complete', completed_utc=utc_now(),
                             checkpoint_sha256=evaluation['checkpoint_sha256'], test_metrics=evaluation['metrics'],
                             reload_comparison=comparison, reused_completed_training=reused_training)
                update_progress(status, status_path, seed, run)
            summary = summarize_runs(output, output / 'report')
            if not any(group['complete'] for group in summary['groups']) or summary['rejected_runs']:
                raise ValueError('All five verified runs are required for the reproduction summary')
            status.update(complete=True, completed_utc=utc_now(), summary=str(output / 'report' / 'summary.json'))
            atomic_json(status_path, status)
            return status
        except BaseException as error:
            for run_status in status['runs'].values():
                if run_status['status'] not in ('pending', 'complete'):
                    run_status.update(status='failed', error=str(error), failed_utc=utc_now())
            status.update(complete=False, error=str(error), failed_utc=utc_now())
            atomic_json(status_path, status)
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True, help='Passing full batch32 benchmark.json')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--background', action='store_true', help='Detach one locked coordinator process')
    parser.add_argument('--disable-host-pinning', action='store_true', help='Match the documented local benchmark runtime override')
    args = parser.parse_args(argv)
    if args.background:
        args.output.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(Path(__file__).resolve()), '--data-root', str(args.data_root.resolve()),
                   '--output', str(args.output.resolve()), '--profile', str(args.profile.resolve()),
                   '--device', args.device]
        if args.disable_host_pinning:
            command.append('--disable-host-pinning')
        with (args.output / 'coordinator.log').open('a') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                       stdin=subprocess.DEVNULL, env={**os.environ, 'PYTHONUNBUFFERED': '1'})
        print(json.dumps({'coordinator_pid': process.pid, 'status_file': str(args.output / 'reproduction_status.json')}))
        return 0
    result = reproduce(args.data_root.resolve(), args.output, args.profile, device=args.device,
                       disable_host_pinning=args.disable_host_pinning)
    print(json.dumps({'complete': result['complete'], 'summary': result['summary']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
