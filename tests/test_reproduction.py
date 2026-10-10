"""Evidence gates for full-batch profiling and restart-safe reproduction."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]


def benchmark_module():
    assert importlib.util.find_spec('sagf_dti.benchmark'), 'benchmark API is missing'
    from sagf_dti import benchmark
    return benchmark


def runner_module():
    path = ROOT / 'scripts' / 'reproduce_paper.py'
    assert path.is_file(), 'paper runner is missing'
    spec = importlib.util.spec_from_file_location('paper_runner_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def passing_profile():
    return {'complete': True, 'passed': True, 'device': 'cuda', 'batch_size': 32,
            'eval_batch_size': 64, 'amp_dtype': 'float16', 'grad_scaler': True,
            'original_paper_recipe': True, 'optimizer_steps': 3,
            'finite_losses': True, 'finite_gradients': True,
            'gradient_groups_nonzero': {'dynamic': True, 'alpha': True, 'variance': True},
            'contrastive_enabled': True, 'auxiliary_enabled': True, 'hypergradient_enabled': True,
            'hypergradient_updated': True, 'peak_reserved_gib': 9.0,
            'max_reserved_gib': 10.5, 'longest_pairs_verified': True}


def test_cuda_allocator_device_resolves_explicit_index():
    module = benchmark_module()
    assert str(module.indexed_cuda_device('cuda', current_index=2)) == 'cuda:2'
    assert str(module.indexed_cuda_device('cuda:1', current_index=2)) == 'cuda:1'
    with pytest.raises(ValueError, match='CUDA'):
        module.indexed_cuda_device('cpu', current_index=0)


def test_paper_runner_rejects_cpu_before_opening_inputs(tmp_path):
    module = runner_module()
    with pytest.raises(ValueError, match='CUDA'):
        module.reproduce(tmp_path / 'absent_data', tmp_path / 'runs', tmp_path / 'absent_profile.json',
                         device='cpu')


def test_host_pinning_override_keeps_workers_prefetch_and_actual_batch(tmp_path):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    module = benchmark_module()
    original = DataLoader(TensorDataset(torch.arange(32)), batch_size=32, num_workers=2,
                          pin_memory=True, persistent_workers=True, prefetch_factor=2)
    options = module.profile_loader_options(original, disable_host_pinning=True)
    assert options['pin_memory'] is False
    assert options['num_workers'] == 2
    assert options['prefetch_factor'] == 2
    assert options['persistent_workers'] is True
    # Worker count is verified above; avoid forking pytest's existing torch threads.
    options.update(num_workers=0)
    options.pop('prefetch_factor')
    options.pop('persistent_workers')
    selected = DataLoader(original.dataset, batch_sampler=[list(range(32))], **options)
    assert next(iter(selected))[0].tolist() == list(range(32))
    assert module.profile_loader_options(original)['pin_memory'] is True


def test_variance_gradient_observation_uses_second_half_of_original_stats_heads():
    import torch
    module = benchmark_module()
    grad = torch.tensor([[3., 3.], [3., 3.], [0., 0.], [0., 0.]])
    assert module.gradient_group_totals([('protein_encoder.esm_stats.weight', grad)])['variance'] == 0
    grad[2:] = 1.
    assert module.gradient_group_totals([('protein_encoder.esm_stats.weight', grad)])['variance'] == 4.


def test_longest_selection_uses_actual_graph_lengths_and_stable_row_indices():
    module = benchmark_module()
    rows = pd.DataFrame({'sequence_hash': ['short', 'long', 'long', 'medium'],
                         'pair_id': ['a', 'b', 'c', 'd']})
    validation = {'valid': True, 'inspect_features': True,
                  'features': {'graphs': {'short': {'nodes': 3}, 'long': {'nodes': 9},
                                         'medium': {'nodes': 6}}}}
    assert module.longest_pair_indices(rows, validation, 3) == [1, 2, 3]
    validation['features']['graphs'].pop('long')
    with pytest.raises(ValueError, match='inspected graph'):
        module.longest_pair_indices(rows, validation, 3)


@pytest.mark.parametrize('field,value', [('batch_size', 16), ('amp_dtype', 'bfloat16'),
    ('optimizer_steps', 0), ('finite_gradients', False), ('peak_reserved_gib', 10.6),
    ('longest_pairs_verified', False), ('hypergradient_updated', False)])
def test_benchmark_gate_rejects_weakened_or_failed_profiles(field, value):
    module = benchmark_module()
    profile = passing_profile()
    module.assert_benchmark_ready(profile)
    profile[field] = value
    with pytest.raises(ValueError):
        module.assert_benchmark_ready(profile)


def test_runner_lock_excludes_second_process_and_releases(tmp_path):
    module = runner_module()
    with module.runner_lock(tmp_path):
        with pytest.raises(RuntimeError, match='already running'):
            with module.runner_lock(tmp_path):
                pass
    with module.runner_lock(tmp_path):
        pass


def test_resume_rejects_changed_identity_and_never_overwrites_partial_run(tmp_path):
    module = runner_module()
    identity = {'source_sha256': {'model.py': 'one'}, 'config_sha256': {'seed42': 'two'},
                'input_sha256': {'data.csv': 'three'}}
    module.bind_identity(tmp_path, identity)
    module.bind_identity(tmp_path, deepcopy(identity))
    changed = deepcopy(identity)
    changed['source_sha256']['model.py'] = 'edited'
    with pytest.raises(ValueError, match='identity changed'):
        module.bind_identity(tmp_path, changed)
    partial = tmp_path / 'seed42'
    partial.mkdir()
    (partial / 'metrics.csv').write_text('epoch\n1\n')
    with pytest.raises(ValueError, match='Partial'):
        module.verify_completed_run(partial, identity)


def test_completed_run_skipped_only_when_evidence_and_inputs_match(tmp_path):
    module = runner_module()
    from sagf_dti.artifacts import sha256
    data = tmp_path / 'input.csv'
    data.write_text('immutable input')
    source = {'model.py': sha256(data)}
    run = tmp_path / 'run'
    spec = importlib.util.spec_from_file_location('completed_fixture', ROOT / 'tests' / 'test_report.py')
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    record = fixture.make_completed_run(run)
    manifest = json.loads((run / 'training_manifest.json').read_text())
    manifest.update(source_sha256=source, input_sha256={str(data): sha256(data)})
    fixture.rewrite_artifact(run, 'training_manifest.json', manifest)
    fixture.rewrite_artifact(run, 'test_input_manifest.json', {str(data): sha256(data)})
    record = json.loads((run / 'completed.json').read_text())
    evidence = run / 'metrics.csv'
    identity = {'training_source_sha256': source, 'input_sha256': manifest['input_sha256']}
    assert module.verify_completed_run(run, identity)['complete']
    with pytest.raises(ValueError, match='runtime overrides'):
        module.verify_completed_run(run, {**identity, 'runtime_overrides': {'pin_memory': False}})
    (run / 'completed.json').write_text(json.dumps({**record, 'seed': 1234}))
    with pytest.raises(ValueError, match='identity'):
        module.verify_completed_run(run, identity)
    (run / 'completed.json').write_text(json.dumps(record))
    evidence.write_text('epoch\n2\n')
    with pytest.raises(ValueError, match='changed'):
        module.verify_completed_run(run, identity)


def test_completion_marker_without_required_evidence_is_rejected(tmp_path):
    module = runner_module()
    (tmp_path / 'completed.json').write_text(json.dumps(
        {'complete': True, 'diagnostic': False, 'verified_files': {}}))
    with pytest.raises(ValueError, match='evidence'):
        module.verify_completed_run(tmp_path, {'training_source_sha256': {}, 'input_sha256': {}})


def test_reload_comparison_reports_roundoff_and_checks_metric_tolerance(tmp_path):
    module = runner_module()
    original, reloaded = tmp_path / 'old.csv', tmp_path / 'new.csv'
    pd.DataFrame({'pair_id': ['a', 'b'], 'label': [0, 1],
                  'pred_logit': [-1., 1.], 'pred_prob': [.25, .75]}).to_csv(original, index=False)
    pd.DataFrame({'pair_id': ['a', 'b'], 'label': [0, 1],
                  'pred_logit': [-.99999, 1.], 'pred_prob': [.250001, .75]}).to_csv(reloaded, index=False)
    result = module.compare_reload(original, reloaded, {'AUROC': 1., 'AUPR': 1.},
                                   {'AUROC': 1., 'AUPR': 1.})
    assert result['prediction_rows_differ'] == 1
    assert result['metrics_within_tolerance']
    with pytest.raises(ValueError, match='metric tolerance'):
        module.compare_reload(original, reloaded, {'AUROC': 1., 'AUPR': 1.},
                              {'AUROC': .99, 'AUPR': 1.})


def test_subprocess_status_reads_epochs_and_keeps_stdout_in_log(tmp_path):
    module = runner_module()
    run = tmp_path / 'run'
    run.mkdir()
    script = tmp_path / 'child.py'
    script.write_text("import pathlib,sys\np=pathlib.Path(sys.argv[1])\n"
                      "(p/'metrics.csv').write_text('epoch,val_auroc\\n2,0.8\\n')\n"
                      "print('training output')\n")
    status_path = tmp_path / 'status.json'
    status = {'runs': {'42': {'status': 'training'}}}
    module.run_child([sys.executable, str(script), str(run)], tmp_path / 'train.log',
                     status, status_path, 42, run, poll_seconds=.01)
    saved = json.loads(status_path.read_text())
    assert saved['runs']['42']['latest_epoch'] == 2
    assert saved['runs']['42']['latest_metrics']['val_auroc'] == '0.8'
    assert 'training output' in (tmp_path / 'train.log').read_text()


def test_failed_subprocess_records_failed_status_and_exit_code(tmp_path):
    module = runner_module()
    run = tmp_path / 'run'
    run.mkdir()
    status_path = tmp_path / 'status.json'
    status = {'runs': {'42': {'status': 'training'}}}
    with pytest.raises(RuntimeError, match='exited 3'):
        module.run_child([sys.executable, '-c', 'print("failure output"); raise SystemExit(3)'],
                         tmp_path / 'train.log', status, status_path, 42, run, poll_seconds=.01)
    saved = json.loads(status_path.read_text())
    assert saved['runs']['42']['status'] == 'failed'
    assert saved['runs']['42']['returncode'] == 3
    assert 'failure output' in (tmp_path / 'train.log').read_text()


def test_reloaded_evaluation_metadata_hash_is_verified_on_resume(tmp_path):
    module = runner_module()
    from sagf_dti.artifacts import sha256
    run = tmp_path / 'run'
    reload = run / 'checkpoint_reload_test'
    reload.mkdir(parents=True)
    checkpoint = run / 'checkpoints' / 'best.pt'
    checkpoint.parent.mkdir()
    checkpoint.write_text('selected checkpoint')
    values = {'pair_id': ['a', 'b'], 'label': [0, 1], 'pred_logit': [-1., 1.], 'pred_prob': [.25, .75]}
    pd.DataFrame(values).to_csv(run / 'test_predictions.csv', index=False)
    pd.DataFrame(values).to_csv(reload / 'test_predictions.csv', index=False)
    metrics = {'AUROC': 1., 'AUPR': 1.}
    evaluation = {'complete': True, 'split': 'test', 'metrics': metrics,
                  'checkpoint_sha256': sha256(checkpoint),
                  'predictions_sha256': sha256(reload / 'test_predictions.csv')}
    path = reload / 'evaluation.json'
    path.write_text(json.dumps(evaluation))
    comparison = module.compare_reload(run / 'test_predictions.csv', reload / 'test_predictions.csv', metrics, metrics)
    comparison.update(evaluation_sha256=sha256(path), original_predictions_sha256=sha256(run / 'test_predictions.csv'),
                      reloaded_predictions_sha256=sha256(reload / 'test_predictions.csv'))
    (run / 'reload_comparison.json').write_text(json.dumps(comparison))
    assert module.verify_reload(run, {'test_metrics': metrics})[0]['complete']
    evaluation['metrics']['AUROC'] = .8
    path.write_text(json.dumps(evaluation))
    with pytest.raises(ValueError, match='changed'):
        module.verify_reload(run, {'test_metrics': metrics})
