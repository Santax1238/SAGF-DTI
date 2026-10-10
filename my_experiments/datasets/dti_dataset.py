from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class DTIDataset(Dataset):
    def __init__(
        self,
        split_csv: str | Path,
        drug_npz: str | Path | list[str | Path],
        graph_root: str | Path | list[str | Path] = "my_experiments/processed/graphs",
        graph_version: str = "esm2_only",
    ) -> None:
        self.split_csv = Path(split_csv)
        self.rows = self._normalize_rows(pd.read_csv(self.split_csv))
        self.graph_dirs = [Path(root) / graph_version for root in self._as_list(graph_root)]
        self.graph_dir = self.graph_dirs[0]
        self.graph_version = graph_version
        self.drug_feats = self._load_drugs(drug_npz)

    @staticmethod
    def _as_list(paths: str | Path | Iterable[str | Path]) -> list[str | Path]:
        if isinstance(paths, (str, Path)):
            return [paths]
        return list(paths)

    @staticmethod
    def _normalize_rows(rows: pd.DataFrame) -> pd.DataFrame:
        rows = rows.copy()
        if "drug_id" not in rows.columns and "Drug_ID" in rows.columns:
            rows["drug_id"] = rows["Drug_ID"].astype(str)
        if "sequence_hash" not in rows.columns and "Target_ID" in rows.columns:
            rows["sequence_hash"] = rows["Target_ID"].astype(str)
        if "label" not in rows.columns:
            if "Label" in rows.columns:
                rows["label"] = rows["Label"]
            elif "Y" in rows.columns:
                rows["label"] = rows["Y"]
        if "pair_id" not in rows.columns:
            rows["pair_id"] = [
                f"pair:{i}:{drug}:{target}"
                for i, (drug, target) in enumerate(zip(rows.get("drug_id", []), rows.get("sequence_hash", [])))
            ]
        required = {"pair_id", "drug_id", "sequence_hash", "label"}
        missing = required - set(rows.columns)
        if missing:
            raise ValueError(f"split CSV missing required columns after normalization: {sorted(missing)}")
        rows["drug_id"] = rows["drug_id"].astype(str)
        rows["sequence_hash"] = rows["sequence_hash"].astype(str)
        rows["label"] = rows["label"].astype(float)
        return rows

    @staticmethod
    def _load_drugs(paths: str | Path | list[str | Path]) -> dict[str, torch.Tensor]:
        if isinstance(paths, (str, Path)):
            paths = [paths]
        out: dict[str, torch.Tensor] = {}
        for path_like in paths:
            path = Path(path_like)
            if not path.exists():
                raise FileNotFoundError(path)
            z = np.load(path, allow_pickle=True)
            if "drug_ids" not in z.files or "fps" not in z.files:
                raise ValueError(f"{path} must contain drug_ids and fps arrays")
            for drug_id, feat in zip(z["drug_ids"], z["fps"]):
                out[str(drug_id)] = torch.as_tensor(feat, dtype=torch.float32)
        return out

    def graph_path(self, sequence_hash: str) -> Path:
        for graph_dir in self.graph_dirs:
            path = graph_dir / f"{sequence_hash}.pt"
            if path.exists():
                return path
        return self.graph_dirs[0] / f"{sequence_hash}.pt"

    def has_graph(self, sequence_hash: str) -> bool:
        return self.graph_path(sequence_hash).exists()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows.iloc[idx]
        pair_id = str(row["pair_id"])
        drug_id = str(row["drug_id"])
        sequence_hash = str(row["sequence_hash"])
        if drug_id not in self.drug_feats:
            raise FileNotFoundError(f"Missing drug feature for {drug_id}")
        graph_path = self.graph_path(sequence_hash)
        if not graph_path.exists():
            raise FileNotFoundError(graph_path)
        return {
            "pair_id": pair_id,
            "drug_id": drug_id,
            "sequence_hash": sequence_hash,
            "drug_feat": self.drug_feats[drug_id],
            "protein_graph": torch.load(graph_path, map_location="cpu", weights_only=False),
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
        }
