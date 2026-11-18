#!/usr/bin/env python3
"""Build additive feature-only intervention patches; never rewrite graph inputs."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import csv
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import numpy as np
import torch
torch.set_num_threads(1)
from window_extractor import HERE, EXPERIMENT, original, now, sha, write


def normalize(values, scaler):
    values = torch.from_numpy(values).double()
    median = torch.tensor(scaler['median'], dtype=torch.float64)
    scale = torch.tensor(scaler['scale'], dtype=torch.float64)
    return ((values-median)/scale).clamp(-5, 5).float().numpy()


def verify_baseline():
    module = original()
    protocol = json.loads((HERE/'PROTOCOL.json').read_text())
    assert sha(HERE/'tasks.json') == protocol['tasks_sha256']
    tasks = json.loads((HERE/'tasks.json').read_text())
    by_target = {}
    for task in tasks:
        by_target.setdefault(task['Target_ID'], []).append(task)
    scaler_path = EXPERIMENT/'augmented621_src_snapshot/research_revision/md_quality_fusion_20261109/md_feature_scaler.json'
    scaler = json.loads(scaler_path.read_text())
    residues = 0
    for target, sources in sorted(by_target.items()):
        graph = torch.load(EXPERIMENT/'data/verified_621_source_conditioned/graphs/md_v1_real'/(target+'.pt'), map_location='cpu', weights_only=False)
        mask = graph.esmdance_embedding_mask.reshape(-1).bool().numpy()
        items = []
        for task in sources:
            assert sha(task['historical_npz']) == task['historical_npz_sha256']
            with np.load(task['historical_npz']) as data:
                items.append((data['target_positions'].copy(), data['features'].copy()))
        features, valid, _ = module.aggregate_features(items)
        assert valid[:len(mask)][mask].all(), target
        np.testing.assert_allclose(normalize(features[:len(mask)], scaler)[mask], graph.x_dyn.numpy()[mask], rtol=1e-5, atol=1e-5, err_msg=target)
        residues += int(mask.sum())
    assert len(by_target) == 621 and residues == 138660
    receipt = {'utc': now(), 'status': 'all621_historical_trajectory_features_reproduce_frozen_graphs',
               'targets': 621, 'original_covered_residues': residues, 'source_npz_files': len(tasks),
               'protocol_sha256': sha(HERE/'PROTOCOL.json'), 'scaler_sha256': sha(scaler_path),
               'raw_coordinates_recomputed': False, 'DTI_labels_read': False}
    write(HERE/'baseline_aggregation_verification.json', receipt)
    print(json.dumps(receipt))


def main():
    protocol = json.loads((HERE/'PROTOCOL.json').read_text())
    assert sha(HERE/'tasks.json') == protocol['tasks_sha256']
    progress = json.loads((HERE/'progress.json').read_text())
    assert progress['status'] == 'all_window_features_extracted_pending_target_aggregation'
    assert progress['completed'] == progress['total_jobs'] == 1550 and not progress['failed']
    module = original()
    tasks = json.loads((HERE/'tasks.json').read_text())
    by_target = {}
    for task in tasks:
        by_target.setdefault(task['Target_ID'], []).append(task)
    assert len(by_target) == 621
    scaler_path = EXPERIMENT/'augmented621_src_snapshot/research_revision/md_quality_fusion_20261109/md_feature_scaler.json'
    scaler = json.loads(scaler_path.read_text())
    target_rows = {r['Target_ID']: r for r in csv.DictReader((EXPERIMENT/'augmented621_evaluation/locked_target_strata.csv').open())}
    output = HERE/'patches'
    if output.exists():
        raise ValueError('Published patches already exist; do not overwrite')
    stage = HERE/'patches.partial'
    if stage.exists():
        raise ValueError('Previous incomplete aggregation requires diagnosis')
    stage.mkdir()
    rows = []
    for target, source_tasks in sorted(by_target.items()):
        graph_path = EXPERIMENT/'data/verified_621_source_conditioned/graphs/md_v1_real'/(target+'.pt')
        graph_sha = sha(graph_path)
        graph = torch.load(graph_path, map_location='cpu', weights_only=False)
        mask = graph.esmdance_embedding_mask.reshape(-1).bool().numpy()
        before = graph.x_dyn.numpy().copy()
        windows = {}
        valid = True
        details = []
        for window in ('historical_full', 'first_half', 'second_half'):
            items = []
            for task in source_tasks:
                if window == 'historical_full':
                    path = Path(task['historical_npz'])
                    expected = task['historical_npz_sha256']
                else:
                    receipt = json.loads((HERE/window/'trajectory_features'/(task['id']+'.json')).read_text())
                    assert receipt['status'] != 'failed' and receipt['window'] == window
                    assert receipt['protocol_sha256'] == sha(HERE/'PROTOCOL.json')
                    path = HERE/window/receipt['features_path']
                    expected = receipt['features_sha256']
                assert sha(path) == expected, str(path)
                with np.load(path) as data:
                    items.append((data['target_positions'].copy(), data['features'].copy()))
            features, feature_valid, counts = module.aggregate_features(items)
            length = len(mask)
            features, feature_valid = features[:length], feature_valid[:length]
            current = normalize(features, scaler)
            missing = int((~feature_valid[mask]).sum())
            details.append({'window': window, 'missing_features_on_original_covered_positions': missing})
            if window == 'historical_full':
                assert missing == 0
                np.testing.assert_allclose(current[mask], before[mask], rtol=1e-5, atol=1e-5)
            else:
                valid &= missing == 0 and np.isfinite(current[mask]).all()
                patch = before.copy()
                patch[mask] = current[mask]
                np.testing.assert_array_equal(patch[~mask], before[~mask])
                windows[window] = patch
        row = {'Target_ID': target, 'sequence_hash': target_rows[target]['full_sequence_sha256'],
               'source': target_rows[target]['source'], 'original_covered_residues': int(mask.sum()),
               'eligible_both_windows': bool(valid), 'historical_full_features_reproduce_graph': True,
               'original_graph_sha256': graph_sha, 'window_checks': details, 'trajectories': len(source_tasks)}
        if valid:
            path = stage/(target+'.npz')
            np.savez_compressed(path, first_half=windows['first_half'], second_half=windows['second_half'],
                                original_coverage_mask=mask, target_id=np.array(target),
                                full_sequence_sha256=np.array(row['sequence_hash']))
            row.update(patch_file=target+'.npz', patch_sha256=sha(path),
                       first_vs_second_rmse_on_original_covered=float(np.sqrt(np.mean((windows['first_half'][mask].astype(float)-windows['second_half'][mask].astype(float))**2))))
        rows.append(row)
    write(stage/'PATCH_MANIFEST.json', {'utc': now(), 'status': 'feature_patches_ready_for_frozen_model_intervention',
          'targets_total': 621, 'targets_eligible': sum(r['eligible_both_windows'] for r in rows),
          'protocol_sha256': sha(HERE/'PROTOCOL.json'), 'tasks_sha256': sha(HERE/'tasks.json'),
          'scaler_sha256': sha(scaler_path), 'scaler_refitted': False, 'DTI_labels_read': False,
          'graph_or_model_changed': False, 'quality_and_mask_unchanged': True,
          'applied_policy': 'Replace x_dyn only on original mask==1 for targets eligible in BOTH windows; every other input stays original. Analyse the same eligible-target test pairs for original/first/second conditions.',
          'rows': rows})
    stage.rename(output)
    progress.update(status='trajectory_window_patches_complete', finished_aggregation_utc=now(),
                    eligible_targets=sum(r['eligible_both_windows'] for r in rows),
                    patch_manifest=str(output/'PATCH_MANIFEST.json'), patch_manifest_sha256=sha(output/'PATCH_MANIFEST.json'))
    write(HERE/'progress.json', progress)
    print(json.dumps({k:progress[k] for k in ['status','completed','eligible_targets','patch_manifest']},ensure_ascii=False))


if __name__ == '__main__':
    if sys.argv[1:] == ['--verify-baseline']:
        verify_baseline()
    elif len(sys.argv) == 1:
        main()
    else:
        raise ValueError('Only optional --verify-baseline is supported')
