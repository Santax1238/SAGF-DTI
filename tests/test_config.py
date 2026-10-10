from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from sagf_dti.config import assert_paper_config, compare_paper_config, resolve_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/paper/SNAP/prot/seed_42.yaml"


def test_resolves_data_paths_independently_of_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = resolve_config(CONFIG, tmp_path / "inputs", tmp_path / "outputs")
    assert config["run_root"] == str(tmp_path / "outputs")
    assert config["data"]["train_csv"] == str(
        tmp_path / "inputs/SNAP/biofusion1500_front/biofusion_official_splits/prot/seed_42/prot_train.csv"
    )
    assert config["data"]["graph_root"] == [str(tmp_path / "inputs/SNAP/biofusion1500_front/graphs")]
    assert config["batch_size"] == 32
    assert config["model"]["dyn_dim"] == 50
    assert_paper_config(config)


def test_all_paper_configs_match_manifest_and_preserve_nonpath_fields(tmp_path):
    manifest = json.loads((ROOT / "provenance/config_manifest.json").read_text())
    paths = sorted((ROOT / "configs/paper").glob("*/*/*.yaml"))
    assert len(paths) == len(manifest["configs"]) == 45
    for path in paths:
        config = resolve_config(path, tmp_path / "data", tmp_path / "runs")
        status = compare_paper_config(config)
        assert status["matches"], status
        assert json.loads(json.dumps(status["non_path_config"])) == manifest["configs"][str(path.relative_to(ROOT))]["non_path_config"]
        assert (ROOT / "sagf_dti" / path.relative_to(ROOT)).read_bytes() == path.read_bytes()


def test_edited_recipe_is_explicitly_usable_but_rejected_as_paper(tmp_path):
    config = yaml.safe_load(CONFIG.read_text())
    config["lr"] = 0.001
    path = tmp_path / "edited.yaml"
    path.write_text(yaml.safe_dump(config))
    resolved = resolve_config(path, tmp_path / "data")
    assert resolved["lr"] == 0.001
    assert compare_paper_config(resolved)["matches"] is False
    with pytest.raises(ValueError, match="lr"):
        assert_paper_config(resolved)


def test_historical_prefix_is_portably_resolved(tmp_path):
    config = yaml.safe_load(CONFIG.read_text())
    config["data"]["train_csv"] = "my_experiments/data/Bio/features/" + config["data"]["train_csv"]
    path = tmp_path / "historical.yaml"
    path.write_text(yaml.safe_dump(config))
    resolved = resolve_config(path, tmp_path / "data")
    assert resolved["data"]["train_csv"].startswith(str(tmp_path / "data/SNAP"))


@pytest.mark.parametrize("mutation,match", [
    ({"batch_size": 0}, "batch_size"),
    ({"experiment_id": "E4_CLS_RESEARCH"}, "experiment_id"),
    ({"run_name": "../other"}, "run_name"),
])
def test_invalid_configs_fail_before_loading_inputs(tmp_path, mutation, match):
    config = yaml.safe_load(CONFIG.read_text())
    config.update(mutation)
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match=match):
        resolve_config(path, tmp_path / "data")


def test_recipe_comparison_detects_changed_data_location(tmp_path):
    config = resolve_config(CONFIG, tmp_path / "data")
    config["data"]["train_csv"] = str(tmp_path / "data/cleaned/train.csv")
    status = compare_paper_config(config)
    assert not status["matches"]
    assert "data.train_csv" in status["differences"]
