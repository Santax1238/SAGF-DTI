from __future__ import annotations

import importlib
import os
from pathlib import Path
import subprocess
import sys
import types


ROOT = Path(__file__).resolve().parents[1]


def test_trainer_loads_without_flat_import_shadowing(monkeypatch):
    # A real foreign module must remain usable before and after loading the trainer.
    foreign = {}
    for name in ("datasets", "models", "loss", "phase4_utils"):
        module = types.ModuleType(name)
        module.marker = object()
        monkeypatch.setitem(sys.modules, name, module)
        foreign[name] = module
    initial_path = list(sys.path)
    legacy = importlib.import_module("sagf_dti.legacy")
    assert hasattr(legacy, "load_trainer"), "the portable frozen trainer loader is missing"
    trainer = legacy.load_trainer()
    assert trainer.DTIModel.__module__ == "my_experiments.models.dti_models"
    assert trainer.DTIDataset.__module__ == "my_experiments.datasets.dti_dataset"
    assert callable(trainer.train_one_epoch) and callable(trainer.evaluate)
    assert trainer.GRAPH_VERSION["E4_GAUSS_NODE_FUSION"] == "esmdance_node"
    assert trainer.__file__.endswith("09_train_experiment.py")
    assert legacy.load_trainer() is trainer
    assert sys.path == initial_path
    assert all(sys.modules[name] is module for name, module in foreign.items())
    assert "rdkit" not in sys.modules


def test_regular_wheel_is_usable_outside_source_directory(tmp_path):
    wheel_dir = tmp_path / "wheel"
    install_dir = tmp_path / "installed"
    build = subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-build-isolation", "--no-deps",
                            "--wheel-dir", str(wheel_dir), str(ROOT)], text=True, capture_output=True)
    assert build.returncode == 0, build.stdout + build.stderr
    wheel = next(wheel_dir.glob("*.whl"))
    install = subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--target", str(install_dir),
                              str(wheel)], text=True, capture_output=True)
    assert install.returncode == 0, install.stdout + install.stderr
    env = dict(os.environ)
    env["PYTHONPATH"] = str(install_dir) + os.pathsep + env.get("PYTHONPATH", "")
    code = """
import sys
import sagf_dti
from sagf_dti.legacy import load_trainer
assert sagf_dti.SAGFDTI().experiment_id == 'E4_GAUSS_NODE_FUSION'
assert load_trainer().DTIModel.__module__ == 'my_experiments.models.dti_models'
assert callable(load_trainer().train_one_epoch)
assert 'rdkit' not in sys.modules
print(sagf_dti.__file__)
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert str(install_dir) in result.stdout
