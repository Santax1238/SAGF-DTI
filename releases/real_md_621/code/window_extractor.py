#!/usr/bin/env python3
"""Apply saved-frame windows to the unchanged original MD feature extractor.

No model, graph, normalization, raw coordinate, pair table or label is changed.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
ROOT = EXPERIMENT.parents[1]
EXTRACTOR = ROOT / 'research_revision/real_md_residue_features_20261109/build_md_features.py'
EXTRACTOR_SHA = 'ded4268a84c9f3f8acb8906fc3f433bfe22fae0a48e581faf137971d05160063'


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for part in iter(lambda: handle.read(8 << 20), b''):
            h.update(part)
    return h.hexdigest()


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def original():
    if sha(EXTRACTOR) != EXTRACTOR_SHA:
        raise ValueError('Historical feature extractor changed')
    spec = importlib.util.spec_from_file_location('original_block_feature_extractor', EXTRACTOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def slice_chunks(chunks, start, stop):
    """Slice the saved-frame stream exactly, preserving frame order and metadata."""
    if not 0 <= start < stop:
        raise ValueError('Invalid half-open frame window')
    offset = emitted = 0
    for chunk in chunks:
        length = len(chunk)
        a, b = max(0, start - offset), min(length, stop - offset)
        if b > a:
            emitted += b - a
            yield chunk[a:b]
        offset += length
        if offset >= stop:
            break
    if emitted != stop - start:
        raise ValueError(f'Window has {emitted} frames; expected {stop-start}')


def prepare():
    module = original()
    _, _, tasks, _ = module.discover()
    old_root = EXTRACTOR.parent / 'trajectory_features'
    for task in tasks:
        task['historical_npz'] = str(old_root / (task['id'] + '.npz'))
    other = ROOT / 'research_revision/strict_real_md_20261114/supplement'
    specs = [
        (other/'audit/mdcath_feature_tasks.json', other/'features_mdcath/trajectory_features'),
        (other/'audit/misato_extraction_tasks.json', other/'features_misato/trajectory_features'),
        (EXPERIMENT/'supplement_mdcath_round2/audit/mdcath_feature_tasks.json', EXPERIMENT/'supplement_mdcath_round2/features_mdcath/trajectory_features'),
        (EXPERIMENT/'supplement_misato_round2/audit/misato_extraction_tasks.json', EXPERIMENT/'supplement_misato_round2/features_misato/trajectory_features'),
        (EXPERIMENT/'supplement_dynamicpdb/feature_tasks.json', EXPERIMENT/'supplement_dynamicpdb/features/trajectory_features'),
        (EXPERIMENT/'supplement_mdposit_round3/feature_tasks.json', EXPERIMENT/'supplement_mdposit_round3/features_mdposit/trajectory_features'),
    ]
    for manifest, directory in specs:
        for task in json.loads(manifest.read_text()):
            task['historical_npz'] = str(directory / (task['id'] + '.npz'))
            tasks.append(task)
    strata_path = EXPERIMENT/'augmented621_evaluation/locked_target_strata.csv'
    covered = {r['Target_ID'] for r in csv.DictReader(strata_path.open()) if r['full621_covered'] == '1'}
    tasks = sorted((t for t in tasks if t['Target_ID'] in covered), key=lambda t: t['id'])
    assert len(covered) == 621 and {t['Target_ID'] for t in tasks} == covered
    assert len({t['id'] for t in tasks}) == len(tasks) == 775
    inputs = {}
    for task in tasks:
        assert task['frames'] >= 4
        assert Path(task['historical_npz']).is_file()
        task['historical_npz_sha256'] = sha(task['historical_npz'])
        for key in ('path', 'topology_path'):
            if key not in task:
                continue
            path = Path(task[key]); stat = path.stat()
            inputs[str(path)] = {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
    out = HERE/'tasks.json'
    if out.exists():
        raise ValueError('Prepared inventory already exists; do not silently reselect inputs')
    write(out, tasks)
    write(HERE/'PROTOCOL.json', {
        'locked_utc': now(), 'status': 'locked_before_window_extraction_or_model_evaluation',
        'targets': 621, 'trajectory_target_tasks': len(tasks), 'windows_per_task': 2,
        'windows': {'first_half': '[0,floor(N/2))', 'second_half': '[floor(N/2),N)'},
        'selection': 'All source trajectories underlying all621 frozen covered targets; no labels or errors used',
        'feature_definition': 'Unchanged original10 features, per-trajectory extraction, original unweighted per-valid-feature aggregation',
        'reference': 'Own first retained frame per window, as in original extractor per-trajectory definition',
        'time': 'Saved-frame ordering only; no fabricated physical time; source-supported relative durations scaled by retained first-last cadence',
        'graph_policy': 'Only external x_dyn patches; original semantics/edges/q/mask/scalers remain unchanged',
        'invalid_policy': 'Before inference, require all original covered residues to have all10 finite features in both windows; otherwise mark target unavailable for this paired block analysis, never silently zero-fill or drop baseline targets',
        'normalization': 'Frozen historical median/IQR and clipping unchanged; never refit',
        'interpretation': 'Within-trajectory sampling-window sensitivity, not independent replicas or proof of physical uncertainty calibration',
        'known_topology_exclusions': 'Apply original normalize_cached_receipt, including MISATO1A28 DSSP limitation',
        'extractor_path': str(EXTRACTOR), 'extractor_sha256': EXTRACTOR_SHA,
        'tasks_sha256': sha(out), 'target_strata_sha256': sha(strata_path),
        'sources': dict(Counter(t['row']['source'] for t in tasks)),
        'inputs_stat': inputs, 'input_raw_sha_policy': 'Fresh SHA for files under10GiB; original whole-HDF download receipt size/mtime for large archive plus freshly SHA-hashed selected trajectory coordinate groups',
        'DTI_labels_read': False, 'training_or_test_inference_performed': False,
    })
    print(json.dumps({'prepared_targets': 621, 'tasks': len(tasks), 'sources': dict(Counter(t['row']['source'] for t in tasks))}), flush=True)


def verify_inputs(tasks, protocol):
    import h5py
    receipts = {}
    for name, stat in protocol['inputs_stat'].items():
        p = Path(name); actual = p.stat()
        assert actual.st_size == stat['size'] and actual.st_mtime_ns == stat['mtime_ns'], name
        if actual.st_size < 10 * (1 << 30):
            digest = sha(p)
            expected = {t['sha256'] for t in tasks if t['path'] == name}
            if expected and expected != {digest}:
                raise ValueError('Raw source digest differs: ' + name)
            receipts[name] = dict(stat, sha256=digest, fresh_whole_file_hash=True)
        else:
            receipt_path = p.with_name(p.name + '.download.json')
            receipt = json.loads(receipt_path.read_text())
            assert actual.st_size == receipt['file_stat']['size']
            assert actual.st_mtime_ns == receipt['file_stat']['mtime_ns']
            groups = {}
            with h5py.File(p, 'r') as handle:
                for group_name in sorted({t['group'] for t in tasks if t['path'] == name}):
                    group = handle[group_name] if group_name in handle else handle[group_name.lower()]
                    coords = group['trajectory_coordinates']
                    digest = hashlib.sha256()
                    for start in range(0, len(coords), 16):
                        digest.update(coords[start:start+16].tobytes())
                    groups[group_name] = {'frames': len(coords), 'coordinate_sha256': digest.hexdigest()}
            receipts[name] = dict(stat, fresh_whole_file_hash=False, historical_download_receipt_sha256=sha(receipt_path), selected_groups=groups)
    write(HERE/'input_integrity_receipt.json', {'utc': now(), 'inputs': receipts, 'protocol_sha256': sha(HERE/'PROTOCOL.json')})


def process_window(task, window):
    module = original()
    frame_count = int(task['frames'])
    start, stop = {'full_validation': (0, frame_count), 'first_half': (0, frame_count//2), 'second_half': (frame_count//2, frame_count)}[window]
    output = HERE/window
    (output/'trajectory_features').mkdir(parents=True, exist_ok=True)
    module.OUT = output
    base_chunks = module.chunks
    module.chunks = lambda edited, topology, chunk: slice_chunks(base_chunks(task, topology, chunk), start, stop)
    edited = copy.deepcopy(task)
    edited['frames'] = stop-start
    if task.get('duration_ns') is not None:
        edited['duration_ns'] = task['duration_ns']*(stop-start-1)/(frame_count-1)
        edited['duration_basis'] = task['duration_basis'] + '; retained_window_first_last_span'
    receipt_path = output/'trajectory_features'/(task['id']+'.json')
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt.get('window') != window or receipt.get('protocol_sha256') != sha(HERE/'PROTOCOL.json'):
            raise ValueError('Existing output belongs to another window/protocol')
        if receipt['status'] == 'failed' or sha(output/receipt['features_path']) != receipt['features_sha256']:
            raise ValueError('Existing failed/corrupt output requires diagnosis, not automatic retry')
        return receipt
    original_sasa = module.md.shrake_rupley
    cached_position = None
    if task['id'] == 'P02788__MoDEL__MD-A00098__R1':
        # This historical trajectory already has exact per-frame SASA caches.
        # Read only after the original coordinate/topology/selection checks;
        # the feature process and all cumulative reductions stay unchanged.
        recovery_path = EXTRACTOR.parent/'recover_last_trajectory.py'
        sys.path.insert(0, str(EXTRACTOR.parent))
        spec = importlib.util.spec_from_file_location('verified_sasa_cache_reader', recovery_path)
        recovery = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(recovery)
        config = recovery.prepared(task)
        chunk_size = config[-1]
        assert start % chunk_size == 0 and stop % chunk_size == 0
        cached_position = {'index': start//chunk_size, 'frame': start}
        def cached_sasa(tr, probe_radius, n_sphere_points, mode, atom_indices):
            import numpy as np
            assert probe_radius == .14 and n_sphere_points == 960 and mode == 'residue'
            np.testing.assert_array_equal(atom_indices, config[4])
            assert recovery.topdigest(tr.topology) == recovery.topdigest(config[2])
            values = recovery.cached_block(cached_position['index'], cached_position['frame'], tr, config)
            if values is None:
                raise ValueError('Missing exact historical SASA cache')
            cached_position['index'] += 1
            cached_position['frame'] += len(tr)
            return values
        module.md.shrake_rupley = cached_sasa
    try:
        receipt = module.process(edited)
    finally:
        module.md.shrake_rupley = original_sasa
    if cached_position is not None:
        assert cached_position['frame'] == stop, 'Not all cached frames were verified'
        receipt['sasa_cache_reuse'] = {
            'cache': str(recovery.CACHE), 'reader_sha256': sha(recovery_path),
            'verified_saved_frame_start': start, 'verified_saved_frame_stop': stop,
            'coordinate_topology_atom_selection_and_order_verified_per_chunk': True,
            'original_per_frame_SASA_values_and_accumulation_order_unchanged': True}
    if receipt['status'] != 'failed':
        receipt = module.normalize_cached_receipt(edited, receipt)
        receipt['features_sha256'] = sha(output/receipt['features_path'])
    receipt.update(window=window, frame_start=start, frame_stop_exclusive=stop,
                   original_frames=frame_count, protocol_sha256=sha(HERE/'PROTOCOL.json'), extractor_sha256=EXTRACTOR_SHA)
    write(receipt_path, receipt)
    return receipt


def validate(tasks):
    import numpy as np
    # Cover actual reader/mapping branches, not redundant source-name aliases.
    # Scientific extraction still includes every775 mapping/all621 targets.
    representatives = {}
    for task in tasks:
        prior = json.loads(Path(task['historical_npz']).with_suffix('.json').read_text())
        key = ('MISATO_HDF' if task['row']['source'] == 'MISATO' else
               'ATLAS_mapping' if task['row']['source'] == 'ATLAS' else
               'generic_' + Path(task['path']).suffix.lower())
        if task['id'] == 'P02788__MoDEL__MD-A00098__R1':
            key = 'exact_checked_SASA_cache'
        cost = (prior.get('wall_seconds', float('inf')), task['id'])
        if key not in representatives or cost < representatives[key][0]:
            representatives[key] = (cost, task)
    proof = []
    for source, (_, task) in sorted(representatives.items()):
        full = process_window(task, 'full_validation')
        assert full['status'] != 'failed', full
        with np.load(task['historical_npz']) as old, np.load(HERE/'full_validation'/full['features_path']) as new:
            for key in ['target_positions', 'topology_residue_indices', 'feature_valid_mask', 'dssp_valid_frames']:
                np.testing.assert_array_equal(old[key], new[key])
            np.testing.assert_allclose(old['features'], new['features'], rtol=1e-5, atol=1e-4, equal_nan=True)
        first, second = process_window(task, 'first_half'), process_window(task, 'second_half')
        assert first['status'] != 'failed' and second['status'] != 'failed'
        assert first['frames'] + second['frames'] == task['frames']
        proof.append({'reader_mapping_branch': source, 'source': task['row']['source'], 'task': task['id'], 'full_feature_reproduction_passed': True,
                      'window_frames': [first['frames'], second['frames']], 'original_frames': task['frames']})
        print(json.dumps(proof[-1]), flush=True)
    write(HERE/'validation_receipt.json', {'utc': now(), 'passed': True, 'sources': proof, 'branch_coverage': 'MISATO_HDF,ATLAS_mapping,generic_XTC,generic_DCD,exact_checked_SASA_cache', 'all621_study_targets_unchanged': True, 'protocol_sha256': sha(HERE/'PROTOCOL.json'), 'wrapper_sha256': sha(__file__)})


def run(tasks, workers):
    import fcntl
    if not 1 <= workers <= 8:
        raise ValueError('At most8 local feature workers')
    validation = json.loads((HERE/'validation_receipt.json').read_text())
    assert validation['passed'] and validation['protocol_sha256'] == sha(HERE/'PROTOCOL.json')
    assert validation['wrapper_sha256'] == sha(__file__), 'Window wrapper changed after validation'
    baseline = json.loads((HERE/'baseline_aggregation_verification.json').read_text())
    assert baseline['status'] == 'all621_historical_trajectory_features_reproduce_frozen_graphs'
    assert baseline['protocol_sha256'] == sha(HERE/'PROTOCOL.json')
    protocol = json.loads((HERE/'PROTOCOL.json').read_text())
    for name, expected in protocol['inputs_stat'].items():
        stat = Path(name).stat()
        assert stat.st_size == expected['size'] and stat.st_mtime_ns == expected['mtime_ns'], name
    lock = (HERE/'extraction.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    jobs = [(t, w) for t in tasks for w in ('first_half', 'second_half')]
    status = {'status': 'extracting', 'started_utc': now(), 'pid': os.getpid(), 'workers': workers, 'total_jobs': len(jobs), 'completed': 0, 'failed': [], 'wrapper_sha256': sha(__file__), 'protocol_sha256': sha(HERE/'PROTOCOL.json')}
    write(HERE/'progress.json', status)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_window, t, w): (t['id'], w) for t, w in jobs}
        for future in as_completed(futures):
            name, window = futures[future]
            try:
                receipt = future.result()
                if receipt['status'] == 'failed':
                    raise ValueError(receipt.get('exception', 'extraction failed'))
                status['completed'] += 1
            except Exception as exc:
                status['failed'].append({'task': name, 'window': window, 'error': repr(exc)})
            status['utc'] = now()
            write(HERE/'progress.json', status)
            print(json.dumps({'completed': status['completed'], 'failed': len(status['failed']), 'total': len(jobs), 'last': [name, window]}), flush=True)
    status.update(status='failed_requires_diagnosis' if status['failed'] else 'all_window_features_extracted_pending_target_aggregation', finished_utc=now())
    write(HERE/'progress.json', status)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'verify-inputs', 'validate', 'run'])
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    if args.action == 'prepare':
        return prepare()
    protocol = json.loads((HERE/'PROTOCOL.json').read_text())
    assert sha(HERE/'tasks.json') == protocol['tasks_sha256']
    tasks = json.loads((HERE/'tasks.json').read_text())
    if args.action == 'verify-inputs':
        return verify_inputs(tasks, protocol)
    if args.action == 'validate':
        return validate(tasks)
    integrity = json.loads((HERE/'input_integrity_receipt.json').read_text())
    assert integrity['protocol_sha256'] == sha(HERE/'PROTOCOL.json')
    return run(tasks, args.workers)


if __name__ == '__main__':
    main()
