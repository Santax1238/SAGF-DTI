"""Completion summaries verify metadata across the actual saved artifacts."""
from copy import deepcopy
import csv
import json
from pathlib import Path
import statistics

import pytest
import torch
import yaml

from sagf_dti.artifacts import recipe_sha256, sha256
from sagf_dti.report import summarize_runs


EVIDENCE_FILES = (
    'checkpoints/best.pt', 'checkpoints/last.pt', 'metrics.csv', 'config.yaml',
    'test_metrics.json', 'test_predictions.csv', 'checkpoint_lock.json',
    'training_manifest.json', 'test_input_manifest.json', 'runtime.json',
)


def make_completed_run(folder, seed=42):
    """Write the same artifact schemas as train(), with real torch checkpoints."""
    folder = Path(folder)
    (folder / 'checkpoints').mkdir(parents=True)
    config = yaml.safe_load((Path(__file__).resolve().parents[1] / 'configs/paper/SNAP/prot/seed_42.yaml').read_text())
    config.update(seed=seed, run_name=folder.name, epochs=2, patience=1, device='cpu')
    config['model'].update(drug_dim=4, esm_dim=3, esmdance35_dim=5, hidden_dim=8)
    (folder / 'config.yaml').write_text(yaml.safe_dump(config))
    validation = {'AUROC': .91, 'AUPR': .92, 'ACC': .8, 'Loss': .3}
    state = {'model': {'weight': torch.ones(2, 4)}, 'config': config,
             'experiment': 'E4_GAUSS_NODE_FUSION', 'graph_version': config['graph_version'],
             'epoch': 1, 'val_metrics': validation, 'meta': {}}
    torch.save(state, folder / 'checkpoints/best.pt')
    last = deepcopy(state)
    last.update(epoch=2, val_metrics={**validation, 'AUROC': .9})
    torch.save(last, folder / 'checkpoints/last.pt')
    (folder / 'metrics.csv').write_text('epoch,train_loss,val_loss,val_auroc,val_aupr,val_acc\n1,0.4,0.3,0.91,0.92,0.8\n2,0.3,0.3,0.9,0.92,0.8\n')
    metrics = {'AUROC': .88, 'AUPR': .9, 'ACC': .8, 'Loss': .3,
               'best_epoch': 1, 'best_val_metric': 'AUROC', 'best_val_score': .91}
    (folder / 'test_metrics.json').write_text(json.dumps(metrics))
    (folder / 'test_predictions.csv').write_text('pair_id,drug_id,sequence_hash,label,pred_logit,pred_prob\na,d1,p1,0,-1.0,0.2689414214\nb,d2,p2,1,1.0,0.7310585786\n')
    lock = {'checkpoint_sha256': sha256(folder / 'checkpoints/best.pt'), 'best_epoch': 1,
            'selection_metric': 'AUROC', 'validation_metrics': validation,
            'test_used_for_selection': False}
    (folder / 'checkpoint_lock.json').write_text(json.dumps(lock))
    manifest = {'seed': seed, 'dataset': 'SNAP', 'setting': 'prot',
                'original_paper_recipe': False, 'recipe_sha256': recipe_sha256(config),
                'diagnostic': False, 'max_batches': None, 'test_opened_during_selection': False,
                'source_sha256': {'model.py': 'a' * 64}, 'input_sha256': {'/inputs/train.csv': 'b' * 64}}
    (folder / 'training_manifest.json').write_text(json.dumps(manifest))
    (folder / 'test_input_manifest.json').write_text(json.dumps({'/inputs/test.csv': 'c' * 64}))
    runtime = {'epochs': [{'epoch': epoch, 'train_seconds': 1.0, 'validation_seconds': .1,
                          'train_pairs': 32, 'train_batches': 1, 'peak_reserved_gib': 0.}
                         for epoch in (1, 2)]}
    (folder / 'runtime.json').write_text(json.dumps(runtime))
    record = {'complete': True, 'diagnostic': False, 'seed': seed, 'dataset': 'SNAP', 'setting': 'prot',
              'original_paper_recipe': False, 'recipe_sha256': recipe_sha256(config), 'test_metrics': metrics,
              'epochs_run': 2, 'training_seconds': 2.2,
              'verified_files': {name: sha256(folder / name) for name in EVIDENCE_FILES}}
    (folder / 'completed.json').write_text(json.dumps(record))
    return record


def rewrite_artifact(folder, name, value):
    (folder / name).write_text(json.dumps(value))
    record = json.loads((folder / 'completed.json').read_text())
    record['verified_files'][name] = sha256(folder / name)
    (folder / 'completed.json').write_text(json.dumps(record))


@pytest.mark.parametrize('field,value', [
    ('seed', 1234), ('dataset', 'DRH'), ('setting', 'drug'), ('recipe_sha256', '0' * 64),
    ('original_paper_recipe', True),
    ('test_metrics', {'AUROC': .99, 'AUPR': .9, 'ACC': .8, 'Loss': .3}),
])
def test_summary_rejects_edited_completion_fields(tmp_path, field, value):
    folder = tmp_path / 'runs/seed42'
    record = make_completed_run(folder)
    record[field] = value
    (folder / 'completed.json').write_text(json.dumps(record))
    report = summarize_runs(tmp_path / 'runs', tmp_path / 'report')
    assert not report['completed_runs']
    assert len(report['rejected_runs']) == 1


@pytest.mark.parametrize('missing', ['checkpoints/last.pt', 'runtime.json', 'test_input_manifest.json'])
def test_summary_requires_all_ten_completion_artifacts(tmp_path, missing):
    folder = tmp_path / 'runs/seed42'
    record = make_completed_run(folder)
    (folder / missing).unlink()
    del record['verified_files'][missing]
    (folder / 'completed.json').write_text(json.dumps(record))
    report = summarize_runs(tmp_path / 'runs', tmp_path / 'report')
    assert not report['completed_runs']
    assert len(report['rejected_runs']) == 1


@pytest.mark.parametrize('change', ['epoch', 'checkpoint_hash', 'validation_metrics', 'test_used'])
def test_summary_crosschecks_checkpoint_selection_lock(tmp_path, change):
    folder = tmp_path / 'runs/seed42'
    make_completed_run(folder)
    lock = json.loads((folder / 'checkpoint_lock.json').read_text())
    if change == 'epoch':
        lock['best_epoch'] = 2
    elif change == 'checkpoint_hash':
        lock['checkpoint_sha256'] = '0' * 64
    elif change == 'validation_metrics':
        lock['validation_metrics']['AUROC'] = .5
    else:
        lock['test_used_for_selection'] = True
    rewrite_artifact(folder, 'checkpoint_lock.json', lock)
    report = summarize_runs(tmp_path / 'runs', tmp_path / 'report')
    assert not report['completed_runs']
    assert len(report['rejected_runs']) == 1


@pytest.mark.parametrize('change', ['diagnostic', 'max_batches', 'nonfinite', 'epochs_run'])
def test_summary_rejects_invalid_completion_semantics(tmp_path, change):
    folder = tmp_path / 'runs/seed42'
    record = make_completed_run(folder)
    if change in ('diagnostic', 'max_batches'):
        manifest = json.loads((folder / 'training_manifest.json').read_text())
        manifest[change] = True if change == 'diagnostic' else 1
        rewrite_artifact(folder, 'training_manifest.json', manifest)
    elif change == 'nonfinite':
        record['test_metrics']['AUROC'] = float('nan')
        (folder / 'test_metrics.json').write_text(json.dumps(record['test_metrics']))
        record['verified_files']['test_metrics.json'] = sha256(folder / 'test_metrics.json')
        (folder / 'completed.json').write_text(json.dumps(record))
    else:
        record['epochs_run'] = 50
        (folder / 'completed.json').write_text(json.dumps(record))
    report = summarize_runs(tmp_path / 'runs', tmp_path / 'report')
    assert not report['completed_runs']
    assert len(report['rejected_runs']) == 1


def test_rejected_summary_clears_stale_seed_csv(tmp_path):
    folder = tmp_path / 'runs/seed42'
    make_completed_run(folder)
    output = tmp_path / 'report'
    summarize_runs(tmp_path / 'runs', output)
    (folder / 'metrics.csv').write_text('changed')
    report = summarize_runs(tmp_path / 'runs', output)
    assert not report['completed_runs']
    with (output / 'seed_metrics.csv').open(newline='') as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames
        assert list(reader) == []


def test_shared_reader_preserves_valid_partial_group_and_path_guard(tmp_path):
    from sagf_dti.report import read_completed_run
    folder = tmp_path / 'runs/seed42'
    record = make_completed_run(folder)
    assert read_completed_run(folder)['seed'] == 42
    summary = summarize_runs(tmp_path / 'runs', tmp_path / 'report')
    assert summary['groups'][0]['n_completed'] == 1
    assert not summary['groups'][0]['complete']
    assert 'mean' not in summary['groups'][0]
    external = tmp_path / 'outside.json'
    external.write_text('external')
    record['verified_files']['../outside.json'] = sha256(external)
    (folder / 'completed.json').write_text(json.dumps(record))
    with pytest.raises(ValueError, match='path'):
        read_completed_run(folder)


def test_five_verified_seeds_report_sample_standard_deviation(tmp_path):
    values = [.80, .82, .84, .86, .88]
    for seed, value in zip((42, 1234, 3407, 5678, 8888), values):
        folder = tmp_path / f'runs/seed{seed}'
        record = make_completed_run(folder, seed)
        record['test_metrics']['AUROC'] = value
        (folder / 'test_metrics.json').write_text(json.dumps(record['test_metrics']))
        record['verified_files']['test_metrics.json'] = sha256(folder / 'test_metrics.json')
        (folder / 'completed.json').write_text(json.dumps(record))
    report = summarize_runs(tmp_path / 'runs', tmp_path / 'report')
    assert not report['rejected_runs']
    group = report['groups'][0]
    assert group['complete'] and group['n_completed'] == 5
    assert group['mean']['AUROC'] == pytest.approx(statistics.mean(values))
    assert group['sample_std']['AUROC'] == pytest.approx(statistics.stdev(values))
    assert group['sample_std']['AUROC'] != pytest.approx(statistics.pstdev(values))
    assert group['std_ddof'] == 1
    assert 'sample standard deviations (ddof=1)' in (tmp_path / 'report/results.md').read_text()


@pytest.mark.parametrize('override,manifest_override', [
    ({'pin_memory': False}, {}),
    ({'pin_memory': True}, {'pin_memory': True}),
    ({'batch_size': 16}, {'batch_size': 16}),
])
def test_summary_rejects_changed_or_unsupported_runtime_overrides(tmp_path, override, manifest_override):
    folder = tmp_path / 'runs/seed42'
    make_completed_run(folder)
    manifest = json.loads((folder / 'training_manifest.json').read_text())
    manifest['runtime_overrides'] = manifest_override
    rewrite_artifact(folder, 'training_manifest.json', manifest)
    record = json.loads((folder / 'completed.json').read_text())
    record['runtime_overrides'] = override
    (folder / 'completed.json').write_text(json.dumps(record))
    report = summarize_runs(tmp_path / 'runs', tmp_path / 'report')
    assert not report['completed_runs']
    assert len(report['rejected_runs']) == 1


def test_summary_checks_runtime_pairs_against_input_validation(tmp_path):
    folder = tmp_path / 'runs/seed42'
    make_completed_run(folder)
    manifest = json.loads((folder / 'training_manifest.json').read_text())
    manifest['input_validation'] = {'splits': {'train': {'rows': 64}}}
    rewrite_artifact(folder, 'training_manifest.json', manifest)
    report = summarize_runs(tmp_path / 'runs', tmp_path / 'report')
    assert not report['completed_runs']
    assert len(report['rejected_runs']) == 1


def test_summary_accepts_bound_pin_memory_override(tmp_path):
    from sagf_dti.report import read_completed_run
    folder = tmp_path / 'runs/seed42'
    make_completed_run(folder)
    manifest = json.loads((folder / 'training_manifest.json').read_text())
    manifest['runtime_overrides'] = {'pin_memory': False}
    manifest['input_validation'] = {'splits': {'train': {'rows': 32}}}
    rewrite_artifact(folder, 'training_manifest.json', manifest)
    record = json.loads((folder / 'completed.json').read_text())
    record['runtime_overrides'] = {'pin_memory': False}
    (folder / 'completed.json').write_text(json.dumps(record))
    assert read_completed_run(folder)['runtime_overrides'] == {'pin_memory': False}
