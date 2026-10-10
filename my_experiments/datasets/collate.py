from __future__ import annotations

import torch
from torch_geometric.data import Batch


METADATA_ATTRS = (
    "target_id",
    "sequence_hash",
    "full_sequence_len",
    "pdb_len",
    "effective_len",
    "is_truncated",
    "graph_meta",
)


def clean_graph_for_batch(graph):
    for attr in METADATA_ATTRS:
        if hasattr(graph, attr):
            delattr(graph, attr)
    return graph


def collate_dti(samples: list[dict]) -> dict:
    return {
        "pair_id": [s["pair_id"] for s in samples],
        "drug_id": [s["drug_id"] for s in samples],
        "sequence_hash": [s["sequence_hash"] for s in samples],
        "drug_feat": torch.stack([s["drug_feat"] for s in samples], dim=0),
        "protein_graph": Batch.from_data_list([clean_graph_for_batch(s["protein_graph"]) for s in samples]),
        "label": torch.stack([s["label"] for s in samples], dim=0).view(-1),
    }
