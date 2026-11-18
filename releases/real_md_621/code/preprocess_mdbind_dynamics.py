import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch_geometric.data import Data


STANDARD_PROTEIN_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "CYX", "GLN", "GLU", "GLY",
    "HIS", "HID", "HIE", "HIP", "ILE", "LEU", "LYS", "MET", "PHE",
    "PRO", "SER", "THR", "TRP", "TYR", "VAL", "SEC", "PYL",
}

WATER_RESIDUES = {"HOH", "WAT", "SOL", "TIP3", "TIP3P"}
ION_RESIDUES = {
    "LI", "NA", "K", "RB", "CS", "MG", "CA", "SR", "BA", "ZN", "MN",
    "FE", "CO", "NI", "CU", "CD", "HG", "AL", "CR", "V", "MO",
    "CL", "BR", "I",
}


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert MDBind local trajectories into residue-level dynamic graph features."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/mdbind_50_locals"),
        help="Directory containing MDBind local folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/mdbind_processed"),
        help="Directory to store processed graph feature files.",
    )
    parser.add_argument(
        "--locals-file",
        type=Path,
        default=None,
        help="Optional file listing locals to process, one id per line.",
    )
    parser.add_argument(
        "--only-local",
        type=str,
        default=None,
        help="Process a single local id.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Frame stride when reading trajectories.",
    )
    parser.add_argument(
        "--max-frames-per-traj",
        type=int,
        default=None,
        help="Optional cap on frames retained from each trajectory after striding.",
    )
    parser.add_argument(
        "--contact-cutoff",
        type=float,
        default=0.8,
        help="Protein residue contact cutoff in nm for static contacts and occupancy.",
    )
    parser.add_argument(
        "--ligand-contact-cutoff",
        type=float,
        default=0.45,
        help="Protein-ligand closest-heavy contact cutoff in nm.",
    )
    parser.add_argument(
        "--pocket-threshold",
        type=float,
        default=0.1,
        help="Pocket flag threshold applied to residue-ligand contact frequency.",
    )
    parser.add_argument(
        "--dynamic-edge-threshold",
        type=float,
        default=0.1,
        help="Residue pair occupancy threshold used to keep dynamic-only edges.",
    )
    parser.add_argument(
        "--drug-edge-cutoff",
        type=float,
        default=0.20,
        help=(
            "Distance cutoff in nm for inferring drug atom edges when the "
            "input topology does not contain ligand bonds."
        ),
    )
    parser.add_argument(
        "--esm2-dir",
        type=Path,
        default=None,
        help="Optional directory containing per-local ESM-2 embeddings.",
    )
    parser.add_argument(
        "--esmdance-dir",
        type=Path,
        default=None,
        help="Optional directory containing per-local ESMDance embeddings.",
    )
    parser.add_argument(
        "--store-ca-xyz",
        action="store_true",
        default=True,
        help="Store per-replicate full CA trajectory [T, N, 3] for next-frame prediction.",
    )
    parser.add_argument(
        "--store-frame-contacts",
        action="store_true",
        default=True,
        help="Store per-frame pairwise CA distances [T, E] for contact open/close prediction.",
    )
    parser.add_argument(
        "--output-format",
        choices=["pt"],
        default="pt",
        help="Serialized output format. The first version writes torch .pt files.",
    )
    return parser


def require_mdtraj():
    try:
        import mdtraj as md
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "This script requires `mdtraj`. Install it first, for example with "
            "`pip install mdtraj` or add it to your conda environment."
        ) from exc
    return md


def discover_local_ids(input_dir: Path, locals_file: Optional[Path], only_local: Optional[str]) -> List[str]:
    if only_local:
        return [only_local]
    if locals_file is None:
        default_file = input_dir / "locals.txt"
        locals_file = default_file if default_file.exists() else None
    if locals_file and locals_file.exists():
        return [line.strip() for line in locals_file.read_text().splitlines() if line.strip()]
    return sorted(path.name for path in input_dir.iterdir() if path.is_dir())


def parse_replicate_index(path: Path) -> int:
    match = re.search(r"\.(\d+)_", path.stem)
    if not match:
        raise ValueError(
            f"Cannot parse replicate index from {path.name}; expected pattern like '<id>.<rep>_structure.pdb'"
        )
    return int(match.group(1))


def collect_local_files(local_dir: Path) -> Dict[str, object]:
    prmtops = sorted(local_dir.glob("*_topology.prmtop"))
    if len(prmtops) != 1:
        raise FileNotFoundError(f"Expected exactly one topology file in {local_dir}, found {len(prmtops)}")

    pdbs = sorted(local_dir.glob("*_structure.pdb"), key=parse_replicate_index)
    xtcs = sorted(local_dir.glob("*_trajectory.xtc"), key=parse_replicate_index)
    if not pdbs or not xtcs or len(pdbs) != len(xtcs):
        raise FileNotFoundError(
            f"Expected matching pdb/xtc replicate pairs in {local_dir}; found {len(pdbs)} pdb and {len(xtcs)} xtc files"
        )

    pairs: List[Tuple[int, Path, Path]] = []
    xtc_by_rep = {parse_replicate_index(path): path for path in xtcs}
    for pdb_path in pdbs:
        rep = parse_replicate_index(pdb_path)
        xtc_path = xtc_by_rep.get(rep)
        if xtc_path is None:
            raise FileNotFoundError(f"Missing xtc for replicate {rep} in {local_dir}")
        pairs.append((rep, pdb_path, xtc_path))

    return {"topology": prmtops[0], "replicates": pairs}


def is_protein_residue(residue_name: str) -> bool:
    return residue_name.upper() in STANDARD_PROTEIN_RESIDUES


def is_water_residue(residue_name: str) -> bool:
    return residue_name.upper() in WATER_RESIDUES


def is_ion_residue(residue_name: str) -> bool:
    return residue_name.upper() in ION_RESIDUES


def classify_residues(topology) -> Dict[str, List[int]]:
    protein = []
    ligand = []
    ions = []
    waters = []
    other = []
    for residue in topology.residues:
        name = residue.name.upper()
        if is_protein_residue(name):
            protein.append(residue.index)
        elif is_water_residue(name):
            waters.append(residue.index)
        elif is_ion_residue(name):
            ions.append(residue.index)
        else:
            ligand.append(residue.index)
            other.append(residue.index)
    return {
        "protein_residue_indices": protein,
        "ligand_residue_indices": ligand,
        "ion_residue_indices": ions,
        "water_residue_indices": waters,
        "other_non_protein_residue_indices": other,
    }


def get_residue_ca_atom_indices(topology, protein_residue_indices: Sequence[int]) -> List[int]:
    ca_atom_indices = []
    for residue_index in protein_residue_indices:
        residue = topology.residue(residue_index)
        ca_atom = next((atom for atom in residue.atoms if atom.name == "CA"), None)
        if ca_atom is None:
            raise ValueError(f"Residue {residue} does not contain a CA atom")
        ca_atom_indices.append(ca_atom.index)
    return ca_atom_indices


def get_residue_atom_indices(topology, residue_indices: Sequence[int]) -> List[int]:
    atom_indices = []
    for residue_index in residue_indices:
        residue = topology.residue(residue_index)
        atom_indices.extend(atom.index for atom in residue.atoms)
    return atom_indices


def build_residue_pairs(n_residues: int) -> np.ndarray:
    return np.array([(i, j) for i in range(n_residues) for j in range(i + 1, n_residues)], dtype=np.int32)


def to_topology_residue_pairs(
    protein_residue_indices: Sequence[int], pair_indices: np.ndarray
) -> List[Tuple[int, int]]:
    return [
        (protein_residue_indices[i], protein_residue_indices[j])
        for i, j in pair_indices
    ]


def maybe_trim_frames(traj, max_frames_per_traj: Optional[int]):
    if max_frames_per_traj is not None and traj.n_frames > max_frames_per_traj:
        traj = traj[:max_frames_per_traj]
    return traj


def compute_rmsf(ca_xyz: np.ndarray) -> np.ndarray:
    mean_coords = ca_xyz.mean(axis=0, keepdims=True)
    sq_disp = np.sum((ca_xyz - mean_coords) ** 2, axis=2)
    return np.sqrt(sq_disp.mean(axis=0))


def compute_dccm(ca_xyz: np.ndarray) -> np.ndarray:
    centered = ca_xyz - ca_xyz.mean(axis=0, keepdims=True)
    numerator = np.einsum("tik,tjk->ij", centered, centered)
    denom_i = np.sqrt(np.einsum("tik,tik->i", centered, centered))
    denom = np.outer(denom_i, denom_i)
    with np.errstate(divide="ignore", invalid="ignore"):
        dccm = np.divide(numerator, denom, out=np.zeros_like(numerator), where=denom > 0)
    np.fill_diagonal(dccm, 1.0)
    return dccm


def stack_edge_features(
    edge_pairs: np.ndarray,
    pair_lookup: Dict[Tuple[int, int], int],
    static_contact: np.ndarray,
    occupancy: np.ndarray,
    dccm: np.ndarray,
    mean_distance: np.ndarray,
    std_distance: np.ndarray,
) -> Tuple[torch.Tensor, torch.Tensor]:
    src = []
    dst = []
    attrs = []
    for i, j in edge_pairs:
        pair_idx = pair_lookup[(int(i), int(j))]
        features = [
            float(static_contact[pair_idx]),
            float(occupancy[pair_idx]),
            float(dccm[int(i), int(j)]),
            float(mean_distance[pair_idx]),
            float(std_distance[pair_idx]),
        ]
        src.extend([int(i), int(j)])
        dst.extend([int(j), int(i)])
        attrs.extend([features, features])

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(attrs, dtype=torch.float32)
    return edge_index, edge_attr


def load_optional_embedding(embedding_dir: Optional[Path], local_id: str, num_residues: int) -> Optional[torch.Tensor]:
    if embedding_dir is None:
        return None

    candidates = [
        embedding_dir / f"{local_id}.pt",
        embedding_dir / f"{local_id}.pth",
        embedding_dir / f"{local_id}.npy",
        embedding_dir / f"{local_id}.npz",
    ]
    existing = next((path for path in candidates if path.exists()), None)
    if existing is None:
        return None

    if existing.suffix in {".pt", ".pth"}:
        obj = torch.load(existing, map_location="cpu")
        if isinstance(obj, dict):
            for key in ("embedding", "embeddings", local_id):
                if key in obj:
                    obj = obj[key]
                    break
        array = np.asarray(obj)
    elif existing.suffix == ".npy":
        array = np.load(existing)
    else:
        npz = np.load(existing)
        key = "embedding" if "embedding" in npz else list(npz.keys())[0]
        array = npz[key]

    if array.shape[0] != num_residues:
        raise ValueError(
            f"Embedding shape mismatch for {existing}: expected first dimension {num_residues}, got {array.shape}"
        )
    return torch.tensor(array, dtype=torch.float32)


def _element_attr(atom, name: str, default):
    element = getattr(atom, "element", None)
    value = getattr(element, name, default) if element is not None else default
    return default if value is None else value


def build_drug_graph_and_metadata(
    topology,
    ligand_residue_indices: Sequence[int],
    reference_xyz: np.ndarray,
    mean_xyz: Optional[np.ndarray] = None,
    edge_cutoff_nm: float = 0.20,
) -> Tuple[Data, Dict[str, object]]:
    """Build an atom-level graph for non-protein ligand/drug residues."""
    ligand_atom_indices = get_residue_atom_indices(topology, ligand_residue_indices)

    if not ligand_atom_indices:
        empty_graph = Data(
            x=torch.empty((0, 3), dtype=torch.float32),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.empty((0, 1), dtype=torch.float32),
            pos=torch.empty((0, 3), dtype=torch.float32),
        )
        return empty_graph, {
            "residue_indices": [],
            "residue_numbers": [],
            "residue_names": [],
            "chain_ids": [],
            "atom_indices": [],
            "atom_names": [],
            "atom_element_symbols": [],
            "atom_feature_names": ["atomic_number", "mass", "is_heavy_atom"],
        }

    atom_to_local = {int(atom_idx): i for i, atom_idx in enumerate(ligand_atom_indices)}
    atoms = [topology.atom(atom_idx) for atom_idx in ligand_atom_indices]

    atom_features = []
    atom_names = []
    atom_element_symbols = []
    atom_residue_indices = []
    atom_residue_names = []
    atom_residue_numbers = []
    atom_chain_ids = []
    for atom in atoms:
        atomic_number = int(_element_attr(atom, "atomic_number", 0))
        mass = float(_element_attr(atom, "mass", 0.0))
        atom_features.append([atomic_number, mass, float(atomic_number > 1)])
        atom_names.append(atom.name)
        atom_element_symbols.append(str(_element_attr(atom, "symbol", "")))
        atom_residue_indices.append(int(atom.residue.index))
        atom_residue_names.append(atom.residue.name)
        atom_residue_numbers.append(int(atom.residue.resSeq))
        atom_chain_ids.append(atom.residue.chain.chain_id)

    src = []
    dst = []
    edge_attr = []
    edge_source = "topology_bonds"
    for atom_a, atom_b in topology.bonds:
        a_idx = int(atom_a.index)
        b_idx = int(atom_b.index)
        if a_idx not in atom_to_local or b_idx not in atom_to_local:
            continue
        a = atom_to_local[a_idx]
        b = atom_to_local[b_idx]
        length = float(np.linalg.norm(reference_xyz[a_idx] - reference_xyz[b_idx]))
        src.extend([a, b])
        dst.extend([b, a])
        edge_attr.extend([[length], [length]])

    if not src and edge_cutoff_nm > 0:
        edge_source = "distance_cutoff"
        ligand_pos = reference_xyz[ligand_atom_indices]
        for i in range(len(ligand_atom_indices)):
            for j in range(i + 1, len(ligand_atom_indices)):
                length = float(np.linalg.norm(ligand_pos[i] - ligand_pos[j]))
                if 0.0 < length <= edge_cutoff_nm:
                    src.extend([i, j])
                    dst.extend([j, i])
                    edge_attr.extend([[length], [length]])

    pos = torch.tensor(reference_xyz[ligand_atom_indices], dtype=torch.float32)
    edge_index = (
        torch.tensor([src, dst], dtype=torch.long)
        if src else torch.empty((2, 0), dtype=torch.long)
    )
    edge_attr = (
        torch.tensor(edge_attr, dtype=torch.float32)
        if edge_attr else torch.empty((0, 1), dtype=torch.float32)
    )

    drug_graph = Data(
        x=torch.tensor(atom_features, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=pos,
    )
    if mean_xyz is not None:
        drug_graph.mean_pos = torch.tensor(mean_xyz, dtype=torch.float32)
    drug_graph.atom_indices = torch.tensor(ligand_atom_indices, dtype=torch.long)

    ligand_residues = [topology.residue(index) for index in ligand_residue_indices]
    metadata = {
        "residue_indices": [int(index) for index in ligand_residue_indices],
        "residue_numbers": [int(residue.resSeq) for residue in ligand_residues],
        "residue_names": [residue.name for residue in ligand_residues],
        "chain_ids": [residue.chain.chain_id for residue in ligand_residues],
        "atom_indices": [int(index) for index in ligand_atom_indices],
        "atom_names": atom_names,
        "atom_element_symbols": atom_element_symbols,
        "atom_residue_indices": atom_residue_indices,
        "atom_residue_names": atom_residue_names,
        "atom_residue_numbers": atom_residue_numbers,
        "atom_chain_ids": atom_chain_ids,
        "atom_feature_names": ["atomic_number", "mass", "is_heavy_atom"],
        "edge_feature_names": ["reference_bond_length_nm"],
        "edge_source": edge_source,
        "edge_cutoff_nm": float(edge_cutoff_nm),
    }
    return drug_graph, metadata


def process_local(local_id: str, local_dir: Path, args: argparse.Namespace) -> Dict[str, object]:
    md = require_mdtraj()
    file_info = collect_local_files(local_dir)
    pairs = file_info["replicates"]

    reference_traj = md.load(str(pairs[0][1]))
    topology = reference_traj.topology
    residue_groups = classify_residues(topology)
    protein_residue_indices = residue_groups["protein_residue_indices"]
    ligand_residue_indices = residue_groups["ligand_residue_indices"]
    ca_atom_indices = get_residue_ca_atom_indices(topology, protein_residue_indices)
    ligand_atom_indices = get_residue_atom_indices(topology, ligand_residue_indices)
    pair_indices = build_residue_pairs(len(protein_residue_indices))
    topology_residue_pairs = to_topology_residue_pairs(protein_residue_indices, pair_indices)
    pair_lookup = {(int(i), int(j)): idx for idx, (i, j) in enumerate(pair_indices)}

    if len(pair_indices) == 0:
        raise ValueError(f"{local_id} contains fewer than two protein residues")

    dist_sum = np.zeros(len(pair_indices), dtype=np.float64)
    dist_sq_sum = np.zeros(len(pair_indices), dtype=np.float64)
    occupancy_count = np.zeros(len(pair_indices), dtype=np.float64)
    sasa_sum = np.zeros(len(protein_residue_indices), dtype=np.float64)
    sasa_sq_sum = np.zeros(len(protein_residue_indices), dtype=np.float64)
    ligand_contact_count = np.zeros(len(protein_residue_indices), dtype=np.float64)
    ca_xyz_all = []
    ligand_xyz_all = []
    num_frames_total = 0
    static_contact = None
    temporal_slices: List[Dict[str, object]] = []

    protein_ca_pairs = [(ca_atom_indices[i], ca_atom_indices[j]) for i, j in pair_indices]
    ligand_contact_pairs = [
        (protein_residue_index, ligand_residue_index)
        for protein_residue_index in protein_residue_indices
        for ligand_residue_index in ligand_residue_indices
    ]

    for replicate_index, pdb_path, xtc_path in pairs:
        traj = md.load_xtc(str(xtc_path), top=str(pdb_path), stride=args.stride)
        traj = maybe_trim_frames(traj, args.max_frames_per_traj)
        traj.superpose(reference_traj, atom_indices=ca_atom_indices)

        pair_distances = md.compute_distances(traj, protein_ca_pairs)
        dist_sum += pair_distances.sum(axis=0)
        dist_sq_sum += np.square(pair_distances).sum(axis=0)
        occupancy_count += (pair_distances <= args.contact_cutoff).sum(axis=0)

        if static_contact is None:
            static_contact = (pair_distances[0] <= args.contact_cutoff).astype(np.float32)

        sasa = md.shrake_rupley(traj, mode="residue")
        protein_sasa = sasa[:, protein_residue_indices]
        sasa_sum += protein_sasa.sum(axis=0)
        sasa_sq_sum += np.square(protein_sasa).sum(axis=0)

        if ligand_contact_pairs:
            ligand_distances, _ = md.compute_contacts(
                traj,
                contacts=ligand_contact_pairs,
                scheme="closest-heavy",
                ignore_nonprotein=False,
                periodic=False,
            )
            ligand_distances = ligand_distances.reshape(traj.n_frames, len(protein_residue_indices), len(ligand_residue_indices))
            ligand_contact_count += (ligand_distances <= args.ligand_contact_cutoff).any(axis=2).sum(axis=0)
            ligand_contact_count_rep = (ligand_distances <= args.ligand_contact_cutoff).any(axis=2).sum(axis=0)
            ligand_contact_freq_rep = ligand_contact_count_rep / max(traj.n_frames, 1)
        else:
            ligand_contact_freq_rep = np.zeros(len(protein_residue_indices), dtype=np.float64)

        ca_xyz_rep = traj.xyz[:, ca_atom_indices, :]
        ligand_xyz_rep = traj.xyz[:, ligand_atom_indices, :] if ligand_atom_indices else None
        rmsf_rep = compute_rmsf(ca_xyz_rep)
        sasa_mean_rep = protein_sasa.mean(axis=0)
        sasa_std_rep = protein_sasa.std(axis=0)
        pocket_flag_rep = (ligand_contact_freq_rep >= args.pocket_threshold).astype(np.float32)
        dccm_rep = compute_dccm(ca_xyz_rep)
        static_contact_rep = (pair_distances[0] <= args.contact_cutoff).astype(np.float32)
        occupancy_rep = (pair_distances <= args.contact_cutoff).mean(axis=0)
        mean_distance_rep = pair_distances.mean(axis=0)
        std_distance_rep = pair_distances.std(axis=0)

        temporal_slice = {
            "replicate": int(replicate_index),
            "num_frames": int(traj.n_frames),
            "node_features": torch.tensor(
                np.stack(
                    [rmsf_rep, sasa_mean_rep, sasa_std_rep, pocket_flag_rep, ligand_contact_freq_rep],
                    axis=1,
                ),
                dtype=torch.float32,
            ),
            "edge_features_dense": {
                "pair_indices": torch.tensor(pair_indices, dtype=torch.long),
                "static_contact": torch.tensor(static_contact_rep, dtype=torch.float32),
                "dynamic_contact_occupancy": torch.tensor(occupancy_rep, dtype=torch.float32),
                "ca_dist_mean": torch.tensor(mean_distance_rep, dtype=torch.float32),
                "ca_dist_std": torch.tensor(std_distance_rep, dtype=torch.float32),
                "dccm": torch.tensor(dccm_rep, dtype=torch.float32),
            },
        }
        # Full CA trajectory [T, N, 3] — used for next-frame position prediction
        if args.store_ca_xyz:
            temporal_slice["ca_xyz"] = torch.tensor(ca_xyz_rep, dtype=torch.float32)
            temporal_slice["ca_xyz_mean"] = torch.tensor(ca_xyz_rep.mean(axis=0), dtype=torch.float32)
            if ligand_xyz_rep is not None:
                temporal_slice["drug_atom_xyz"] = torch.tensor(ligand_xyz_rep, dtype=torch.float32)
                temporal_slice["drug_atom_xyz_mean"] = torch.tensor(
                    ligand_xyz_rep.mean(axis=0), dtype=torch.float32
                )
        # Per-frame pairwise distances [T, E] — used for contact open/close prediction
        if args.store_frame_contacts:
            # Only store distances for edges in the sparse contact graph (edge_keep_mask applied later)
            # Store all pairs here; Teacher training will apply the mask
            temporal_slice["frame_pair_distances"] = torch.tensor(
                pair_distances.astype(np.float32), dtype=torch.float32
            )  # [T, num_pairs]
        temporal_slices.append(temporal_slice)

        ca_xyz_all.append(traj.xyz[:, ca_atom_indices, :])
        if ligand_xyz_rep is not None:
            ligand_xyz_all.append(ligand_xyz_rep)
        num_frames_total += traj.n_frames
        print(f"[{local_id}] replicate {replicate_index}: {traj.n_frames} frames")

    if num_frames_total == 0:
        raise ValueError(f"{local_id} contains zero usable frames")

    ca_xyz_all = np.concatenate(ca_xyz_all, axis=0)
    ligand_xyz_mean = (
        np.concatenate(ligand_xyz_all, axis=0).mean(axis=0)
        if ligand_xyz_all else None
    )
    mean_distance = dist_sum / num_frames_total
    std_distance = np.sqrt(np.maximum(dist_sq_sum / num_frames_total - np.square(mean_distance), 0.0))
    occupancy = occupancy_count / num_frames_total
    rmsf = compute_rmsf(ca_xyz_all)
    sasa_mean = sasa_sum / num_frames_total
    sasa_std = np.sqrt(np.maximum(sasa_sq_sum / num_frames_total - np.square(sasa_mean), 0.0))
    ligand_contact_freq = ligand_contact_count / num_frames_total if ligand_residue_indices else np.zeros_like(rmsf)
    pocket_flag = (ligand_contact_freq >= args.pocket_threshold).astype(np.float32)
    dccm = compute_dccm(ca_xyz_all)

    edge_keep_mask = np.logical_or(static_contact > 0, occupancy >= args.dynamic_edge_threshold)
    edge_pairs = pair_indices[edge_keep_mask]
    edge_index, edge_attr = stack_edge_features(
        edge_pairs=edge_pairs,
        pair_lookup=pair_lookup,
        static_contact=static_contact,
        occupancy=occupancy,
        dccm=dccm,
        mean_distance=mean_distance,
        std_distance=std_distance,
    )

    node_scalar_features = np.stack(
        [rmsf, sasa_mean, sasa_std, pocket_flag, ligand_contact_freq],
        axis=1,
    )

    data = Data(
        x=torch.tensor(node_scalar_features, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_attr,
    )
    data.num_nodes = len(protein_residue_indices)

    esm2_embedding = load_optional_embedding(args.esm2_dir, local_id, data.num_nodes)
    esmdance_embedding = load_optional_embedding(args.esmdance_dir, local_id, data.num_nodes)
    if esm2_embedding is not None:
        data.esm2_embedding = esm2_embedding
    if esmdance_embedding is not None:
        data.esmdance_embedding = esmdance_embedding

    drug_graph, drug_metadata = build_drug_graph_and_metadata(
        topology,
        ligand_residue_indices,
        reference_traj.xyz[0],
        ligand_xyz_mean,
        args.drug_edge_cutoff,
    )

    protein_residues = [topology.residue(index) for index in protein_residue_indices]
    ligand_residues = [topology.residue(index) for index in ligand_residue_indices]
    ion_residues = [topology.residue(index) for index in residue_groups["ion_residue_indices"]]

    payload = {
        "graph": data,
        "protein_graph": data,
        "drug_graph": drug_graph,
        "protein": {
            "graph_key": "protein_graph",
            "residue_indices": [int(index) for index in protein_residue_indices],
            "residue_numbers": [int(residue.resSeq) for residue in protein_residues],
            "residue_names": [residue.name for residue in protein_residues],
            "chain_ids": [residue.chain.chain_id for residue in protein_residues],
            "node_feature_names": [
                "rmsf",
                "sasa_mean",
                "sasa_std",
                "pocket_flag",
                "ligand_contact_freq",
            ],
        },
        "drug": drug_metadata,
        "ligand": drug_metadata,
        "node_feature_names": [
            "rmsf",
            "sasa_mean",
            "sasa_std",
            "pocket_flag",
            "ligand_contact_freq",
        ],
        "edge_feature_names": [
            "static_contact",
            "dynamic_contact_occupancy",
            "dccm",
            "ca_dist_mean",
            "ca_dist_std",
        ],
        "residue_numbers": [int(residue.resSeq) for residue in protein_residues],
        "residue_names": [residue.name for residue in protein_residues],
        "chain_ids": [residue.chain.chain_id for residue in protein_residues],
        "ligand_residue_names": [residue.name for residue in ligand_residues],
        "ion_residue_names": [residue.name for residue in ion_residues],
        "metadata": {
            "local_id": local_id,
            "num_frames": num_frames_total,
            "num_protein_residues": len(protein_residue_indices),
            "num_ligand_residues": len(ligand_residue_indices),
            "num_ion_residues": len(residue_groups["ion_residue_indices"]),
            "contact_cutoff_nm": args.contact_cutoff,
            "ligand_contact_cutoff_nm": args.ligand_contact_cutoff,
            "pocket_threshold": args.pocket_threshold,
            "dynamic_edge_threshold": args.dynamic_edge_threshold,
            "source_topology": str(file_info["topology"]),
            "source_replicates": [
                {"replicate": rep, "pdb": str(pdb), "xtc": str(xtc)}
                for rep, pdb, xtc in pairs
            ],
            "has_esm2_embedding": esm2_embedding is not None,
            "has_esmdance_embedding": esmdance_embedding is not None,
            "has_temporal_slices": len(temporal_slices) > 0,
        },
        "temporal_slices": temporal_slices,
        "dense_pair_features": {
            "pair_indices": torch.tensor(pair_indices, dtype=torch.long),
            "static_contact": torch.tensor(static_contact, dtype=torch.float32),
            "dynamic_contact_occupancy": torch.tensor(occupancy, dtype=torch.float32),
            "ca_dist_mean": torch.tensor(mean_distance, dtype=torch.float32),
            "ca_dist_std": torch.tensor(std_distance, dtype=torch.float32),
            "dccm": torch.tensor(dccm, dtype=torch.float32),
        },
    }
    return payload


def save_payload(payload: Dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    local_ids = discover_local_ids(args.input_dir, args.locals_file, args.only_local)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    failures = {}
    for local_id in local_ids:
        local_dir = args.input_dir / local_id
        if not local_dir.exists():
            failures[local_id] = f"missing directory: {local_dir}"
            continue
        try:
            payload = process_local(local_id, local_dir, args)
            save_payload(payload, args.output_dir / f"{local_id}.pt")
            print(
                f"[{local_id}] saved {args.output_dir / f'{local_id}.pt'} "
                f"with {payload['metadata']['num_protein_residues']} residues and "
                f"{payload['metadata']['num_frames']} frames"
            )
        except Exception as exc:
            failures[local_id] = str(exc)
            print(f"[{local_id}] FAILED: {exc}")

    summary = {
        "processed": len(local_ids) - len(failures),
        "failed": len(failures),
        "failures": failures,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
