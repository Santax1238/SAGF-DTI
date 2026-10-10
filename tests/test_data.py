from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Data

from sagf_dti.data import validate_data


@pytest.fixture
def inputs(tmp_path):
    graph_dir = tmp_path / "graphs/esmdance_node_cond35_dyn50"
    graph_dir.mkdir(parents=True)
    graph = Data(**{
        "x": torch.ones(4, 3), "x_esm": torch.ones(4, 3), "x_dyn": torch.ones(4, 2),
        "predicted_dynamic_residue_features": torch.ones(4, 2),
        "esmdance_35m_embedding": torch.ones(4, 5),
        "esmdance_embedding_mask": torch.tensor([[1.], [1.], [0.], [0.]]),
        "edge_index": torch.tensor([[0, 1, 2], [1, 2, 3]]),
        "target_id": "p1", "sequence_hash": "p1",
        "graph_version": "esmdance_node_cond35_dyn50", "effective_len": 4,
        "esmdance_meta": {"covered_len": 1, "raw_len": 2, "effective_len": 4},
    })
    torch.save(graph, graph_dir / "p1.pt")
    for split in ("train", "val"):
        pd.DataFrame({"Drug_ID": ["d1", "d2"], "Target_ID": ["p1", "p1"], "Y": [0, 1]}).to_csv(tmp_path / f"{split}.csv", index=False)
    np.savez(tmp_path / "kpgt.npz", drug_ids=np.array(["d1", "d2"], dtype=object), fps=np.ones((2, 4), dtype=np.float32))
    pd.DataFrame({"Drug_ID": ["d1", "d2"], "SMILES": ["C", "CC"]}).to_csv(tmp_path / "drugs.csv", index=False)
    pd.DataFrame({"Target_ID": ["p1"], "Target": ["AAAA"]}).to_csv(tmp_path / "targets.csv", index=False)
    np.savez(tmp_path / "drug_sim.npz", np.eye(2))
    np.savez(tmp_path / "target_sim.npz", np.eye(1))
    return {
        "graph_version": "esmdance_node_cond35_dyn50",
        "model": {"drug_dim": 4, "esm_dim": 3, "dyn_dim": 2, "esmdance35_dim": 5, "edge_dim": 36},
        "data": {"train_csv": str(tmp_path / "train.csv"), "val_csv": str(tmp_path / "val.csv"),
                 "test_csv": str(tmp_path / "unopened_test.csv"), "graph_root": [str(tmp_path / "graphs")],
                 "bindingdb_drug_npz": [str(tmp_path / "kpgt.npz")], "davis_drug_npz": [str(tmp_path / "missing_test_npz.npz")]},
        "contrastive": {"enabled": True, "drugs_csv": str(tmp_path / "drugs.csv"),
                        "targets_csv": str(tmp_path / "targets.csv"), "drug_sim_npz": str(tmp_path / "drug_sim.npz"),
                        "target_sim_npz": str(tmp_path / "target_sim.npz")},
    }


def test_validation_counts_every_row_without_opening_test(inputs):
    report = validate_data(inputs, inspect_features=True)
    assert report["valid"], report
    assert set(report["splits"]) == {"train", "val"}
    assert report["splits"]["train"]["rows"] == 2
    assert report["splits"]["train"]["positive"] == report["splits"]["train"]["negative"] == 1
    graph = report["features"]["graphs"]["p1"]
    assert graph["nodes"] == 4
    assert graph["covered_nodes"] == graph["covered_prefix"] == 2
    assert graph["metadata"]["covered_len"] == 1
    assert any("covered_len" in item for item in report["warnings"])
    assert report["features"]["contrastive"]["drug"]["id_order"] == ["d1", "d2"]
    assert report["features"]["contrastive"]["drug"]["embedded_id_order_verified"] is False


def test_missing_graph_and_drug_are_reported_without_dropping_rows(inputs):
    train = Path(inputs["data"]["train_csv"])
    pd.DataFrame({"drug_id": ["unknown"], "sequence_hash": ["missing"], "label": [1]}).to_csv(train, index=False)
    report = validate_data(inputs)
    assert not report["valid"]
    assert report["splits"]["train"]["rows"] == 1
    assert report["splits"]["train"]["missing_drug_ids"] == ["unknown"]
    assert report["splits"]["train"]["missing_graph_ids"] == ["missing"]
    assert any(path.endswith("missing.pt") for path in report["missing_paths"])


def test_test_inputs_are_validated_only_when_explicitly_selected(inputs):
    report = validate_data(inputs, splits=("test",))
    assert not report["valid"]
    assert inputs["data"]["test_csv"] in report["missing_paths"]
    assert inputs["data"]["davis_drug_npz"][0] in report["missing_paths"]
    assert inputs["data"]["train_csv"] not in report["missing_paths"]


@pytest.mark.parametrize("change,match", [
    (lambda graph: setattr(graph, "esmdance_embedding_mask", None), "mask"),
    (lambda graph: setattr(graph, "esmdance_embedding_mask", torch.zeros(4, 1)), "zero"),
    (lambda graph: setattr(graph, "x_dyn", torch.ones(4, 3)), "x_dyn"),
    (lambda graph: setattr(graph, "esmdance_35m_embedding", torch.full((4, 5), float("nan"))), "finite"),
    (lambda graph: setattr(graph, "sequence_hash", "different"), "sequence_hash"),
    (lambda graph: setattr(graph, "predicted_dynamic_residue_features", torch.zeros(4, 2)), "alias"),
    (lambda graph: setattr(graph, "esmdance_embedding_mask", torch.ones(4)), "mask"),
    (lambda graph: setattr(graph, "esmdance_embedding_mask", torch.tensor([[1.], [0.], [1.], [0.]])), "prefix"),
])
def test_feature_inspection_rejects_auxiliary_corruption(inputs, change, match):
    path = Path(inputs["data"]["graph_root"][0]) / inputs["graph_version"] / "p1.pt"
    graph = torch.load(path, weights_only=False)
    change(graph)
    torch.save(graph, path)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    report = validate_data(inputs, inspect_features=True)
    assert not report["valid"]
    assert any(match in error for error in report["errors"]), report
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_similarity_matrix_dimensions_and_embedded_order_are_checked(inputs):
    path = Path(inputs["contrastive"]["drug_sim_npz"])
    np.savez(path, arr_0=np.eye(2), drug_ids=np.array(["d2", "d1"]))
    report = validate_data(inputs)
    assert not report["valid"]
    assert any("row order" in error for error in report["errors"])
    np.savez(path, arr_0=np.eye(3))
    report = validate_data(inputs)
    assert any("shape" in error for error in report["errors"])


def test_duplicate_feature_ids_do_not_silently_overwrite(inputs):
    path = Path(inputs["data"]["bindingdb_drug_npz"][0])
    np.savez(path, drug_ids=np.array(["d1", "d1"]), fps=np.ones((2, 4)))
    report = validate_data(inputs)
    assert not report["valid"]
    assert any("duplicate" in error for error in report["errors"])


def test_invalid_labels_and_conflicting_aliases_fail(inputs):
    pd.DataFrame({"Drug_ID": ["d1"], "drug_id": ["d2"], "Target_ID": ["p1"], "Y": [0.5]}).to_csv(inputs["data"]["train_csv"], index=False)
    report = validate_data(inputs)
    assert not report["valid"]
    assert any("binary" in error for error in report["errors"])
    assert any("alias" in error for error in report["errors"])


def test_invalid_split_argument_is_a_programmer_error(inputs):
    with pytest.raises(ValueError, match="split"):
        validate_data(inputs, splits=("validation",))


@pytest.mark.parametrize("kind", ["dict", "missing_x", "wrong_num_nodes"])
def test_graph_must_support_real_pyg_batching(inputs, kind):
    path = Path(inputs["data"]["graph_root"][0]) / inputs["graph_version"] / "p1.pt"
    graph = torch.load(path, weights_only=False)
    if kind == "dict":
        graph = graph.to_dict()
    elif kind == "missing_x":
        del graph["x"]
    else:
        graph.num_nodes = 10
    torch.save(graph, path)
    report = validate_data(inputs, inspect_features=True)
    assert not report["valid"], report
    assert any(word in error for error in report["errors"] for word in ("Data", " x", "num_nodes"))


@pytest.mark.parametrize("fps", [
    np.ones((2, 4), dtype=np.complex64) * (1 + 1j),
    np.ones((2, 4), dtype=np.longdouble),
    np.full((2, 4), 1e300, dtype=np.float64),
    np.ones((2, 4), dtype=">f4"),
])
def test_kpgt_must_survive_frozen_torch_float32_conversion(inputs, fps):
    path = Path(inputs["data"]["bindingdb_drug_npz"][0])
    np.savez(path, drug_ids=np.array(["d1", "d2"]), fps=fps)
    report = validate_data(inputs)
    assert not report["valid"], report
    assert any("KPGT" in error and ("dtype" in error or "float32" in error) for error in report["errors"])


def test_carrier_dimension_uses_nested_model_default(inputs):
    config = deepcopy(inputs)
    config["esmdance35_dim"] = config["model"].pop("esmdance35_dim")
    report = validate_data(config, inspect_features=True)
    assert not report["valid"], report
    assert any("480" in error and "esmdance_35m_embedding" in error for error in report["errors"])


@pytest.mark.parametrize("alias", ["residue_valid_mask", "node_valid_mask", "valid_mask", "node_mask"])
@pytest.mark.parametrize("mask", [torch.ones(2), torch.zeros(4), torch.tensor([1., 0.5, 0., 0.])])
def test_optional_attention_masks_are_validated_per_protein(inputs, alias, mask):
    path = Path(inputs["data"]["graph_root"][0]) / inputs["graph_version"] / "p1.pt"
    graph = torch.load(path, weights_only=False)
    setattr(graph, alias, mask)
    torch.save(graph, path)
    report = validate_data(inputs, inspect_features=True)
    assert not report["valid"], report
    assert any(alias in error for error in report["errors"])


def test_accepted_inputs_execute_frozen_dataset_collation_and_model(inputs):
    from my_experiments.datasets.dti_dataset import DTIDataset
    from my_experiments.datasets.collate import collate_dti
    from sagf_dti.model import SAGFDTI

    config = deepcopy(inputs)
    config["experiment_id"] = "E4_GAUSS_NODE_FUSION"
    config["model"].update(hidden_dim=8, pooling_type="drug_conditioned_attention", dyn_conditioned_on_esm35=True)
    path = Path(config["data"]["graph_root"][0]) / config["graph_version"] / "p1.pt"
    graph = torch.load(path, weights_only=False)
    graph.node_valid_mask = torch.tensor([1, 1, 0, 0], dtype=torch.bool)
    torch.save(graph, path)
    report = validate_data(config, inspect_features=True)
    assert report["valid"], report
    dataset = DTIDataset(config["data"]["train_csv"], config["data"]["bindingdb_drug_npz"],
                         config["data"]["graph_root"], config["graph_version"])
    batch = collate_dti([dataset[0], dataset[1]])
    assert batch["protein_graph"].num_nodes == len(batch["protein_graph"].batch) == 8
    model = SAGFDTI(config).eval()
    with torch.no_grad():
        output, _ = model(batch)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("dtype,value", [(torch.complex64, 1 + 1j), (torch.float64, 1e300)])
def test_graph_embeddings_must_survive_real_float32_conversion(inputs, dtype, value):
    path = Path(inputs["data"]["graph_root"][0]) / inputs["graph_version"] / "p1.pt"
    graph = torch.load(path, weights_only=False)
    graph.x = graph.x_esm = torch.full((4, 3), value, dtype=dtype)
    torch.save(graph, path)
    report = validate_data(inputs, inspect_features=True)
    assert not report["valid"], report
    assert any("float32" in error or "real" in error for error in report["errors"])


def test_portable_split_conversion_preserves_original_order_and_labels(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/prepare_splits.py"
    spec = importlib.util.spec_from_file_location("prepare_splits", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    official = tmp_path / "official/SNAP/prot_split_results/seed_42"
    official.mkdir(parents=True)
    source = pd.DataFrame({"DRUG_ID": ["d2", "d1", "d2"], "UNIPROT_ID": ["p1", "p2", "p1"], "Label": [1, 0, 1], "SEQUENCE": ["AAAA", "BBBB", "AAAA"]})
    for split in ("train", "val", "test"):
        source.to_csv(official / f"prot_{split}.csv", index=False)
    result = module.prepare_splits(tmp_path / "official", tmp_path / "inputs", datasets=("SNAP",), settings=("prot",), seeds=(42,))
    assert len(result) == 3
    converted = pd.read_csv(tmp_path / "inputs/SNAP/biofusion1500_front/biofusion_official_splits/prot/seed_42/prot_train.csv")
    assert converted["Drug_ID"].tolist() == ["d2", "d1", "d2"]
    assert converted["Target_ID"].tolist() == ["p1", "p2", "p1"]
    assert converted["Label"].tolist() == [1, 0, 1]
    assert converted["source_dataset"].tolist() == ["SNAP"] * 3
    assert source.equals(pd.read_csv(official / "prot_train.csv"))
    with pytest.raises(FileExistsError):
        module.prepare_splits(tmp_path / "official", tmp_path / "inputs", datasets=("SNAP",), settings=("prot",), seeds=(42,))


def test_split_conversion_rejects_fractional_labels_before_writing(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/prepare_splits.py"
    spec = importlib.util.spec_from_file_location("prepare_splits_fractional", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    official = tmp_path / "official/SNAP/prot_split_results/seed_42"
    official.mkdir(parents=True)
    for split in ("train", "val", "test"):
        pd.DataFrame({"DRUG_ID": ["d1"], "UNIPROT_ID": ["p1"], "Label": [0.5]}).to_csv(official / f"prot_{split}.csv", index=False)
    with pytest.raises(ValueError, match="binary"):
        module.prepare_splits(tmp_path / "official", tmp_path / "inputs", datasets=("SNAP",), settings=("prot",), seeds=(42,))
    assert not (tmp_path / "inputs").exists()
