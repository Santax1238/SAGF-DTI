# MDBind Data Exploration Report

## 1. Directory Structure Overview

### Raw Data: `<repo>/data/mdbind_50_locals/`

**Structure:** 50 protein-ligand complex folders, each containing:
- `{ID}_topology.prmtop` - AMBER topology file with full system (protein + ligand + water + ions)
- `{ID}.{REP}_structure.pdb` - Reference PDB structure for replicate REP (1-10 replicates per local)
- `{ID}.{REP}_trajectory.xtc` - MD trajectory for replicate REP (GROMACS XTC format, ~2.6-4.7 MB per file)

**Example:** A004B folder contains:
- A004B_topology.prmtop (5.8 MB)
- A004B.1_structure.pdb through A004B.10_structure.pdb (1.1 MB each)
- A004B.1_trajectory.xtc through A004B.10_trajectory.xtc (2.6 MB each)

### Processed Data: `<repo>/data/mdbind_processed/`

**Structure:** 50 PyTorch .pt files, one per local complex
- File sizes: 4-1800 MB (median ~20-60 MB depending on protein size & #frames)
- Format: torch.save() serialized Python dictionaries containing PyG Data objects and metadata

---

## 2. Protein Information Available

### Per-Local Protein Data (from `process_local()` in `preprocess_mdbind_dynamics.py`)

**Extracted from topology and trajectories:**

1. **Residue-level information:**
   - Residue numbers (PDB resSeq)
   - Residue names (3-letter codes: ALA, GLY, VAL, etc.)
   - Chain IDs (e.g., 'A', 'B')
   - Protein residue indices (from standard 24 amino acid set)

2. **Static node features (per residue, aggregated across all frames):**
   - `rmsf` - Root mean square fluctuation (flexibility measure)
   - `sasa_mean` - Mean solvent-accessible surface area
   - `sasa_std` - Std dev of SASA
   - `pocket_flag` - Boolean: is this residue part of ligand-binding pocket? (1 if ligand_contact_freq ≥ 0.1)
   - `ligand_contact_freq` - Fraction of frames where this residue contacts the ligand (≤0.45 nm cutoff, closest-heavy atom scheme)

3. **Dynamic residue-residue edge features (Cα-based pairs):**
   - `static_contact` - Whether residue pair is in contact (≤0.8 nm) in first frame
   - `dynamic_contact_occupancy` - Fraction of frames pair is in contact
   - `dccm` - Dynamical cross-correlation matrix (correlation of residue motions)
   - `ca_dist_mean` - Mean Cα-Cα distance
   - `ca_dist_std` - Std dev of Cα-Cα distance

4. **Optional embeddings (if provided during preprocessing):**
   - `esm2_embedding` - ESM-2 protein language model embeddings (1280-dim, optional)
   - `esmdance_embedding` - ESMDance embeddings (optional)

5. **Per-replicate temporal snapshots** (10 replicates each):
   - Frame count per replicate
   - Replicate-specific node features (RMSF, SASA, pocket flags, ligand contact freq)
   - Replicate-specific edge features (static contact, occupancy, DCCM, distances)
   - Optional: CA coordinate means for SE(3)-aware encoding

6. **Metadata:**
   - Local ID, total frame count, num protein residues
   - Num ligand residues, num ions
   - Contact cutoffs used (protein-protein: 0.8 nm, protein-ligand: 0.45 nm)
   - Source: which PDB/XTC files were processed
   - Has ESM2/ESMDance embeddings flags

### Data in Processed .pt Files

**Structure (torch_geometric.data.Data + auxiliary info):**

```python
payload = {
    'graph': Data(
        x=torch.Tensor([num_residues, 5]),       # Node features: [rmsf, sasa_mean, sasa_std, pocket_flag, ligand_contact_freq]
        edge_index=torch.Tensor([2, num_edges]), # Edge indices (sparse residue graph)
        edge_attr=torch.Tensor([num_edges, 5]),  # Edge features: [static_contact, occupancy, dccm, dist_mean, dist_std]
        num_nodes=int,                           # Number of residues
        esm2_embedding=torch.Tensor([num_residues, 1280]),      # Optional
        esmdance_embedding=torch.Tensor([num_residues, d]),     # Optional
    ),
    'node_feature_names': ['rmsf', 'sasa_mean', 'sasa_std', 'pocket_flag', 'ligand_contact_freq'],
    'edge_feature_names': ['static_contact', 'dynamic_contact_occupancy', 'dccm', 'ca_dist_mean', 'ca_dist_std'],
    'residue_numbers': List[int],                 # PDB residue numbers
    'residue_names': List[str],                   # 3-letter AA codes
    'chain_ids': List[str],                       # Chain IDs (usually 'A', 'B', etc.)
    'ligand_residue_names': List[str],           # Names of ligand molecules bound
    'ion_residue_names': List[str],              # Names of ions in structure
    'temporal_slices': List[Dict],               # Per-replicate snapshots
    'dense_pair_features': Dict[str, Tensor],    # Dense pairwise statistics
    'metadata': Dict[str, Any],                  # Processing parameters & counts
}
```

---

## 3. Ligand/Drug Information Available

### Ligand Information Stored

**In processed .pt files:**
- `ligand_residue_names` - **PDB residue names** of all ligand molecules bound (stored as strings)

**Example:**
- A004B: ['GDP'] - Guanosine diphosphate
- A03IE: ['A2P'] - Diadenosine phosphate
- A03O0: ['CMG', 'TPO'] - Multiple ligands
- A03OW: ['PGA'] - Prostaglandin A

### Ligand Inventory (50 MDBind Locals)

**Total unique ligands:** 47 distinct residue types (mostly PDB-named, small molecules/nucleotides)

**Distribution:**
- Single ligand per complex: 46 locals (92%)
- Multiple ligands per complex: 4 locals (8%)
  - A03IS: CSO + XK2
  - A03O0: CMG + TPO
  - A03O1: 4SP + TPO
  - A03O9: 1ZK + CSO

**Ligand categories observed:**
1. **Nucleotides/cofactors:** GDP, GAN, A2P, UNI, RUN, XMP, etc.
2. **Small organic molecules:** Various drug-like compounds with 3-letter PDB codes
3. **Modified amino acids:** ORN, CSO (cysteine sulfenic acid), etc.

### CRITICAL LIMITATION: No SMILES/Molecular Structures

**Important:** The MDBind processed .pt files contain:
- ✅ Ligand **residue names** (PDB 3-letter codes)
- ✅ Ligand-protein **contact information** (pocket flags, contact frequencies)
- ❌ **NO** SMILES strings
- ❌ **NO** molecular graph structures
- ❌ **NO** chemical properties/descriptors
- ❌ **NO** ligand embeddings

To access actual ligand structures, you would need to:
1. Extract ligand atoms from the original PDB files
2. Use external tools (RDKit, etc.) to parse ligand structure
3. Generate SMILES or molecular embeddings separately

### Protein-Ligand Interaction Features

**What IS stored about protein-ligand interactions:**

1. **Per-residue contact information:**
   - `pocket_flag` - Binary: residue in binding pocket (contact_freq ≥ 0.1)
   - `ligand_contact_freq` - Continuous: fraction of frames with protein-ligand contact

2. **Cumulative interaction counts:**
   - Total protein-ligand contact frames counted in preprocessing
   - Contact cutoff: 0.45 nm (closest-heavy atom scheme)

3. **No per-frame tracking:**
   - Temporal slices don't break down which specific atoms contacted the ligand
   - Just aggregate statistics

---

## 4. Frame/Temporal Information

### Trajectory Data

**Per replicate:**
- 10 replicates per local (indexed 1-10)
- Each replicate has 1 PDB reference structure + 1 XTC trajectory
- XTC files typically contain 100+ frames at 1ns intervals (variable based on MD protocol)

**Processing:**
- Script can stride frames (--stride N to keep every Nth frame)
- Can cap frames per trajectory (--max-frames-per-traj parameter)
- All frames across all replicates are concatenated for global statistics

**Per-replicate metadata stored:**
- `replicate` - Replicate number (1-10)
- `num_frames` - Frames in that replicate
- Node/edge features aggregated per replicate
- Optional: CA coordinate means

---

## 5. File Format Details

### Raw .pt File Structure (PyTorch Archive)

Each .pt is a ZIP with:
- `{local_id}/data.pkl` - Python pickle with the full payload dict
- `{local_id}/data/0-88` - Serialized tensor data (FloatStorage, LongStorage)
- Metadata: format version, byte order, storage alignment

### Loading in Code

```python
import torch
data = torch.load('data/mdbind_processed/A004B.pt', map_location='cpu', weights_only=False)

# Access protein graph
graph = data['graph']  # PyG Data object
x = graph.x  # [num_residues, 5] node features
edge_index = graph.edge_index  # [2, num_edges] sparse edges
edge_attr = graph.edge_attr  # [num_edges, 5] edge attributes

# Access protein metadata
residues = data['residue_names']  # ['ALA', 'GLY', ...]
ligands = data['ligand_residue_names']  # ['GDP']
chains = data['chain_ids']  # ['A', 'A', ...]

# Access temporal snapshots
temporal = data['temporal_slices']  # List of 10 dicts
for slice_dict in temporal:
    rep = slice_dict['replicate']  # 1-10
    frames = slice_dict['num_frames']
    node_feats = slice_dict['node_features']  # [num_residues, 5]
    edge_feats = slice_dict['edge_features_dense']  # Dict with pair_indices, etc.

# Metadata
meta = data['metadata']
print(f"Protein: {meta['local_id']}")
print(f"Residues: {meta['num_protein_residues']}")
print(f"Ligands: {meta['num_ligand_residues']}")
print(f"Total frames: {meta['num_frames']}")
```

---

## 6. Data Preprocessing Pipeline

**Source:** `preprocess_mdbind_dynamics.py`

1. **Load topology** → Classify residues (protein / ligand / ions / water)
2. **For each replicate (1-10):**
   - Load XTC trajectory with PDB reference
   - Superpose on first structure (CA-based alignment)
   - Compute pairwise distances, SASA, ligand contacts
   - Compute dynamic properties (RMSF, DCCM)
   - Store per-replicate snapshot
3. **Aggregate across replicates:**
   - Average distances, SASA, contact frequencies
   - Concatenate all CA coordinates for global RMSF/DCCM
4. **Build sparse graph:**
   - Keep edges where: static_contact > 0 OR occupancy ≥ 0.1
   - Create PyG Data object with edge_index and edge_attr
5. **Save payload** as .pt file with all metadata

---

## 7. Summary: What You Have

### ✅ Available in MDBind:

| Category | Available | Format |
|----------|-----------|--------|
| Protein sequences | Via residue names (implicit) | 3-letter codes |
| Protein structures | Via Cα coordinates | Stored as distances/correlations |
| Protein flexibility | RMSF per residue | Continuous [0, ∞) nm |
| Protein-protein contacts | Static + dynamic | Binary + occupancy fractions |
| Protein surface | SASA mean/std | Continuous values |
| Binding pocket residues | pocket_flag | Binary [0, 1] |
| Ligand types | Residue names (PDB codes) | Strings like 'GDP', 'A2P' |
| Ligand-protein contacts | Contact frequency per residue | Continuous [0, 1] |
| Molecular dynamics trajectories | Summarized statistics | Not raw frames |
| Optional: Sequence embeddings | ESM-2 (1280-dim) | If provided |
| Optional: Temporal snapshots | Per-replicate features | 10 snapshots per local |

### ❌ NOT Available in MDBind:

- SMILES strings or molecular graphs for ligands
- Individual ligand atom structures
- Atomic coordinates (only Cα distances aggregated)
- Raw trajectory frames (only statistics)
- Ligand embeddings or descriptors
- Per-frame ligand binding poses
- Binding affinities or activity labels

---

## 8. Size & Scale

- **50 protein-ligand complexes**
- **~7500 residues total** (average 150 per complex, range: 50-950)
- **~47 unique ligand types**
- **~500,000 MD frames** aggregated (10,000 per replicate on average)
- **5.5 GB total** for 50 processed .pt files

---

## 9. Example: Inspecting One Complex

```python
import torch

# Load A004B (GDP-bound protein, GTPase)
data = torch.load('data/mdbind_processed/A004B.pt', map_location='cpu', weights_only=False)

print(f"Local: {data['metadata']['local_id']}")
print(f"Protein residues: {data['metadata']['num_protein_residues']}")
print(f"Ligand(s): {data['ligand_residue_names']}")  # ['GDP']
print(f"Ions: {data['ion_residue_names']}")
print(f"Total MD frames: {data['metadata']['num_frames']}")
print(f"Graph edges: {data['graph'].edge_index.shape[1]}")
print(f"Node features: {data['graph'].x.shape}")  # [num_residues, 5]

# Identify binding pocket residues
pocket_mask = data['graph'].x[:, 3] > 0  # pocket_flag is feature 3
pocket_residues = [res for i, res in enumerate(data['residue_names']) if pocket_mask[i]]
print(f"Pocket residues: {len(pocket_residues)} of {len(data['residue_names'])}")
```

---

## References

- Preprocessing source: `<repo>/preprocess_mdbind_dynamics.py` (573 lines)
- Documentation: `<repo>/MDBIND_PREPROCESS.md`
- Data location: `<repo>/data/mdbind_50_locals/` (raw), `<repo>/data/mdbind_processed/` (processed)
