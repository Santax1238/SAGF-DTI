"""User-facing portability, run protection, and completion reporting."""
from pathlib import Path
import importlib
import importlib.util
import hashlib
import json
import os
import subprocess
import sys

import pytest


RELEASE = Path(__file__).resolve().parents[1]


def test_module_help_runs_outside_checkout(tmp_path):
    env = dict(os.environ)
    env['PYTHONPATH'] = str(RELEASE) + os.pathsep + env.get('PYTHONPATH', '')
    result = subprocess.run([sys.executable, '-m', 'sagf_dti', '--help'],
                            cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert all(word in result.stdout for word in ('train', 'evaluate', 'validate-data', 'summarize'))


def test_run_guard_preserves_existing_artifacts(tmp_path):
    try:
        module = importlib.import_module('sagf_dti.train')
    except ImportError as error:
        pytest.fail(f'Portable training command is missing: {error}')
    run = tmp_path / 'seed42'
    run.mkdir()
    checkpoint = run / 'best.pt'
    checkpoint.write_bytes(b'existing-weights')
    with pytest.raises(FileExistsError):
        module.claim_run_directory(tmp_path, 'seed42')
    assert checkpoint.read_bytes() == b'existing-weights'
    with pytest.raises(ValueError):
        module.claim_run_directory(tmp_path, '../outside')


def test_summary_labels_partial_seed_results(tmp_path):
    try:
        module = importlib.import_module('sagf_dti.report')
    except ImportError as error:
        pytest.fail(f'Completed-run summary is missing: {error}')
    run = tmp_path / 'runs' / 'seed42'
    fixture_spec = importlib.util.spec_from_file_location('_report_fixture', Path(__file__).with_name('test_report.py'))
    fixture = importlib.util.module_from_spec(fixture_spec)
    fixture_spec.loader.exec_module(fixture)
    fixture.make_completed_run(run)
    result = module.summarize_runs(tmp_path / 'runs', tmp_path / 'report')
    assert result['groups'][0]['complete'] is False
    assert result['groups'][0]['n_completed'] == 1
    assert result['groups'][0]['missing_seeds'] == [1234, 3407, 5678, 8888]
    assert 'mean' not in result['groups'][0]
