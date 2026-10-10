"""Check the public API against an independently loaded frozen implementation."""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import types

import pytest
import torch
from torch_geometric.data import Batch, Data


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def reference_model(tmp_path_factory):
    """Extract the original archive when available; installed tests use hashed bytes."""
    manifest = json.loads((ROOT / "provenance/source_manifest.json").read_text())
    archive = Path(os.environ.get("SAGF_DTI_SOURCE_ARCHIVE", ROOT.parents[1] / manifest["archive_path"]))
    destination = tmp_path_factory.mktemp("frozen_reference")
    if archive.is_file():
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == manifest["archive_sha256"]
        with tarfile.open(archive) as source:
            for member in manifest["members"]:
                path = destination / member["path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(source.extractfile(member["path"]).read())
    else:
        for member in manifest["members"]:
            path = destination / member["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / member["path"], path)
    for member in manifest["members"]:
        assert hashlib.sha256((destination / member["path"]).read_bytes()).hexdigest() == member["sha256"]
    name = "_sagf_independent_frozen_reference"
    namespace = types.ModuleType(name)
    namespace.__path__ = [str(destination / "my_experiments")]
    sys.modules[name] = namespace
    yield importlib.import_module(name + ".models.dti_models").DTIModel
    for key in list(sys.modules):
        if key == name or key.startswith(name + "."):
            del sys.modules[key]


def graph_batch(auxiliary="present"):
    """Use real PyG graphs with unequal protein lengths and actual feature widths."""
    generator = torch.Generator().manual_seed(810)
    graphs = []
    for nodes in (5, 3):
        graph = Data(x=torch.randn(nodes, 1280, generator=generator),
                     edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long))
        if auxiliary != "missing":
            graph.predicted_dynamic_residue_features = torch.randn(nodes, 50, generator=generator)
            graph.esmdance_35m_embedding = torch.randn(nodes, 480, generator=generator)
            graph.esmdance_embedding_mask = torch.ones(nodes, 1)
            if auxiliary == "masked":
                graph.esmdance_embedding_mask.zero_()
            elif auxiliary == "partial":
                graph.esmdance_embedding_mask[-2:] = 0
        graphs.append(graph)
    return {"drug_feat": torch.randn(2, 2304, generator=generator),
            "protein_graph": Batch.from_data_list(graphs), "label": torch.tensor([0., 1.])}


def config(sample_fused=False):
    return {"model": {"hidden_dim": 16, "dropout": 0.2, "dyn_conditioned_on_esm35": True,
                      "sample_fused": sample_fused},
            "pooling_type": "drug_conditioned_attention", "fusion_type": "interaction_mlp",
            "unimodal_aux": {"enabled": True}}


def public_model():
    module = importlib.import_module("sagf_dti")
    assert hasattr(module, "SAGFDTI"), "the public frozen SAGFDTI API is missing"
    return module.SAGFDTI


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("sample_fused", [False, True])
def test_initial_state_forward_and_gradients_match_archive(reference_model, training, sample_fused):
    api = public_model()
    torch.manual_seed(42)
    expected = reference_model("E4_GAUSS_NODE_FUSION", config(sample_fused))
    torch.manual_seed(42)
    actual = api(config(sample_fused))
    assert type(actual).__mro__[1].__name__ == "DTIModel"
    assert actual.experiment_id == "E4_GAUSS_NODE_FUSION"
    assert list(actual.state_dict()) == list(expected.state_dict())
    for key, tensor in actual.state_dict().items():
        torch.testing.assert_close(tensor, expected.state_dict()[key], rtol=0, atol=0)
    actual_flags = {key: value.requires_grad for key, value in actual.named_parameters()}
    assert actual_flags == {key: value.requires_grad for key, value in expected.named_parameters()}
    assert "bilinear.weight" in actual_flags and not actual_flags["bilinear.weight"]
    batch = graph_batch("partial")
    expected.train(training)
    actual.train(training)
    torch.manual_seed(103)
    expected_logits, expected_aux = expected(batch, return_embeddings=True)
    torch.manual_seed(103)
    logits, aux = actual(batch, return_embeddings=True)
    torch.testing.assert_close(logits, expected_logits, rtol=0, atol=0)
    assert set(aux) == set(expected_aux)
    for key, value in aux.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, expected_aux[key], rtol=0, atol=0, equal_nan=True)
        else:
            assert value == expected_aux[key]
    loss = torch.nn.functional.cross_entropy(logits, batch["label"].long()) + aux["kl_loss"] * 0.001
    loss.backward()
    gradients = [parameter.grad for parameter in actual.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)


@pytest.mark.parametrize("auxiliary", ["masked", "missing"])
def test_mask_and_missing_auxiliary_preserve_legacy_behavior(reference_model, auxiliary):
    torch.manual_seed(42)
    expected = reference_model("E4_GAUSS_NODE_FUSION", config()).eval()
    torch.manual_seed(42)
    actual = public_model()(config()).eval()
    batch = graph_batch(auxiliary)
    expected_logits, expected_aux = expected(batch)
    logits, aux = actual(batch)
    torch.testing.assert_close(logits, expected_logits, rtol=0, atol=0)
    torch.testing.assert_close(aux["dynamic_coverage_mean"], expected_aux["dynamic_coverage_mean"], rtol=0, atol=0)
    assert torch.isfinite(logits).all()
    assert float(aux["dynamic_coverage_mean"]) == (0.0 if auxiliary == "masked" else 1.0)


def test_public_model_defaults_to_gaussian_experiment():
    assert public_model()().experiment_id == "E4_GAUSS_NODE_FUSION"
