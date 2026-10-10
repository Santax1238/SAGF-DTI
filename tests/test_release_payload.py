"""Ensure the repository excludes generated assets without dropping source."""
from pathlib import Path
import subprocess


def test_ignore_rules_include_frozen_dataset_source(tmp_path):
    release = Path(__file__).resolve().parents[1]
    (tmp_path / '.gitignore').write_text((release / '.gitignore').read_text())
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    required = ['my_experiments/datasets/__init__.py',
                'my_experiments/datasets/dti_dataset.py',
                'my_experiments/datasets/collate.py']
    for name in required:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# preserved source\n')
    check = subprocess.run(['git', 'check-ignore', '--stdin'], input='\n'.join(required) + '\n',
                            cwd=tmp_path, text=True, capture_output=True)
    assert check.returncode == 1, check.stdout
    assert not check.stdout
    for name in ('data/cache.csv', 'datasets/pairs.csv', 'runs/report.csv', 'weights/model.pt'):
        assert subprocess.run(['git', 'check-ignore', name], cwd=tmp_path,
                              capture_output=True).returncode == 0
