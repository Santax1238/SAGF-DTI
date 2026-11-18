================================================================================
MDBind Dataset Documentation Index
================================================================================

ABOUT MDBIND:
The MDBind dataset contains 50 protein-ligand complexes with:
  • Molecular dynamics trajectories (10 replicates each)
  • Residue-level graph representations
  • Dynamic protein features extracted from MD simulations
  • Limited ligand information (PDB codes only, no SMILES/structures)

DATASET LOCATION:
  Raw data:       <repo>/data/mdbind_50_locals/
  Processed .pt:  <repo>/data/mdbind_processed/
  Size:           5.5 GB processed + 37 MB embeddings

================================================================================
DOCUMENTATION FILES (In Order of Detail)
================================================================================

1. START HERE:
   📄 MDBIND_QUICK_REFERENCE.txt (8.6 KB)
      • One-page guide to data structure
      • Python code snippets
      • Field-by-field breakdown
      • What's available vs missing

2. FOR LIGAND DETAILS:
   📄 MDBIND_LIGAND_INVENTORY.txt (9.5 KB)
      • Complete list of 47 ligands
      • Per-complex breakdown
      • Ligand categories
      • Data access patterns
      • Important limitations

3. FOR COMPLETE DETAILS:
   📄 MDBIND_DATA_INVENTORY.md (13 KB)
      • Full directory structure
      • Detailed field descriptions
      • Processing pipeline explanation
      • Complete Python usage examples
      • File format specifications

4. FOR PREPROCESSING DETAILS:
   📄 MDBIND_PREPROCESS.md (5.4 KB)
      • Preprocessing overview
      • Node/edge feature definitions
      • Ligand handling strategy
      • Command-line examples

5. FOR SOURCE CODE:
   📄 preprocess_mdbind_dynamics.py (573 lines)
      • Full Python implementation
      • Key function: process_local() (lines 322-528)
      • Residue classification logic
      • Feature computation details

================================================================================
QUICK FACTS
================================================================================

PROTEINS:
  ✅ 50 complexes
  ✅ 50-950 residues each (mean 150)
  ✅ 10 MD replicates per protein
  ✅ ~500,000 frames total
  ✅ Residue-level graph features (10 dims total)
  ✅ Optional: ESM-2 embeddings (1280-dim)

LIGANDS:
  ✅ 47 unique PDB codes
  ✅ Pocket detection (pocket_flag)
  ✅ Contact frequency (0-1)
  ❌ NO SMILES/structures
  ❌ NO chemical properties
  ❌ NO embeddings

DYNAMICS:
  ✅ Per-replicate snapshots
  ✅ Flexibility metrics (RMSF)
  ✅ Surface area (SASA)
  ✅ Protein-protein correlations (DCCM)
  ✅ Distance statistics
  ❌ NO raw trajectory frames
  ❌ NO per-frame binding info

================================================================================
HOW TO USE
================================================================================

1. Load a processed .pt file:
   import torch
   data = torch.load('data/mdbind_processed/A004B.pt', map_location='cpu', weights_only=False)

2. Access the protein graph:
   graph = data['graph']  # torch_geometric.data.Data
   x = graph.x            # [num_residues, 5] node features
   edge_index = graph.edge_index  # [2, num_edges]
   edge_attr = graph.edge_attr    # [num_edges, 5] edge features

3. Get ligand information:
   ligands = data['ligand_residue_names']  # e.g., ['GDP']
   pocket_flag = data['graph'].x[:, 3]    # binding pocket indicator
   contact_freq = data['graph'].x[:, 4]   # per-residue contact frequency

4. Access temporal data:
   for rep in data['temporal_slices']:
       print(f"Replicate {rep['replicate']}: {rep['num_frames']} frames")

See MDBIND_QUICK_REFERENCE.txt for complete code examples.

================================================================================
LIGAND INVENTORY (47 types)
================================================================================

Nucleotides & Cofactors:
  GDP, GAN, A2P, UNI, RUN, XMP, NOG, UP6, UNA

Drug-like Molecules:
  A76, A77, A78, A79, INF, PSI, APR, BMP, BZQ, DHT, ESM, JE2, KNI, L04, LHY,
  MK1, PLU, RIT, ROC, RTL, TAD

Modified Residues & Other:
  ORN, CSO, PGA, CMG, 9PP, 0EM, 0ZQ, 137, 1ZK, 478, 4SP, FA1, HQO, IMH, TPO, XK2

Multi-ligand complexes (4):
  A03IS: CSO + XK2
  A03O0: CMG + TPO
  A03O1: 4SP + TPO
  A03O9: 1ZK + CSO

Full details in MDBIND_LIGAND_INVENTORY.txt

================================================================================
IMPORTANT LIMITATIONS
================================================================================

1. LIGAND DATA
   • Only PDB 3-letter codes (e.g., 'GDP')
   • No SMILES strings or molecular graphs
   • No chemical descriptors or properties
   • No ligand embeddings
   → Workaround: Extract from raw PDB files, use RDKit for SMILES

2. BINDING INFORMATION
   • No binding affinity labels (Kd, Ki, etc.)
   • No activity data
   • MDBind is for representation learning only
   → Use: Pre-training feature extractor, fine-tune on BindingDB/DAVIS

3. TRAJECTORY DATA
   • Only aggregated statistics stored (not raw frames)
   • No per-frame binding poses
   • No frame-by-frame interaction tracking
   → Could: Re-process raw XTC files for dynamics

4. STRUCTURAL DATA
   • No full atomic coordinates
   • Only Cα-based features
   • No atomic-level details
   → Use: Raw PDB files for atom-level information

================================================================================
COMMON TASKS
================================================================================

Find binding pocket residues:
  >>> pocket_mask = data['graph'].x[:, 3] > 0
  >>> pocket_res = [data['residue_names'][i] for i in range(len(data['residue_names'])) if pocket_mask[i]]

Get residue flexibility:
  >>> rmsf = data['graph'].x[:, 0]  # Feature 0

Count protein-ligand interactions:
  >>> contact_freq = data['graph'].x[:, 4]  # Feature 4
  >>> high_contact = (contact_freq > 0.5).sum().item()

Access per-replicate data:
  >>> for rep in data['temporal_slices']:
  ...     print(f"Rep {rep['replicate']}: {rep['num_frames']} frames")

Get all protein-protein contact pairs:
  >>> edge_index = data['graph'].edge_index
  >>> src, dst = edge_index[0], edge_index[1]

================================================================================
FOR DTI PREDICTION
================================================================================

MDBind is designed for PROTEIN representation learning.

For full DTI prediction, you'll need:

1. Protein features (MDBind provides):
   ✅ Residue-level graph representations
   ✅ Dynamic flexibility information
   ✅ Binding pocket detection
   ✅ Contact patterns
   ✅ Optional sequence embeddings (ESM-2)

2. Ligand features (not in MDBind):
   ❌ Need external source (e.g., KPGT, RDKit)
   ❌ Could extract from PDB files in mdbind_50_locals/
   ❌ Generate SMILES or fingerprints separately

3. Labels (not in MDBind):
   ❌ Need external binding data (BindingDB, DAVIS, etc.)
   ❌ Use MDBind for pre-training, BindingDB for fine-tuning

Example workflow:
  1. Pre-train protein encoder on MDBind (representation learning)
  2. Extract ligand features from BindingDB/external sources
  3. Fine-tune full DTI model on labeled data

================================================================================
FILE SIZES & COUNTS
================================================================================

Raw data (symbolic link):
  /data/mdbind_50_locals → /mnt/d/MD_project/mdbind_50_locals (external drive)
  50 directories with ~300 files total

Processed .pt files:
  /data/mdbind_processed/
  Size: 5.5 GB
  Count: 50 .pt files
  Range: 4 MB (A03IE.pt) to 1.8 GB (A03OK.pt)

Embeddings:
  /data/mdbind_temporal_embeddings/
  Size: 37 MB (if generated)

Documentation (new):
  MDBIND_QUICK_REFERENCE.txt      8.6 KB
  MDBIND_LIGAND_INVENTORY.txt     9.5 KB
  MDBIND_DATA_INVENTORY.md        13 KB
  (Total: ~31 KB)

================================================================================
NEXT STEPS
================================================================================

1. Read MDBIND_QUICK_REFERENCE.txt for immediate overview

2. Load a sample complex:
   python
   >>> import torch
   >>> data = torch.load('data/mdbind_processed/A004B.pt', ...)
   >>> print(data['graph'])

3. Explore temporal dynamics:
   >>> len(data['temporal_slices'])  # 10 replicates
   >>> data['metadata']['num_frames']  # total frames

4. Plan feature extraction:
   • Decide which node/edge features to use
   • Consider per-replicate vs aggregated statistics
   • Plan ligand feature integration

5. For publication/documentation:
   • Reference: preprocess_mdbind_dynamics.py
   • Cite preprocessing parameters in metadata
   • Document any custom feature engineering

================================================================================
CONTACT & REFERENCES
================================================================================

Source code:           preprocess_mdbind_dynamics.py
Preprocessing guide:   MDBIND_PREPROCESS.md
Project README:        README.md
Full documentation:    MDBIND_DATA_INVENTORY.md
Quick reference:       MDBIND_QUICK_REFERENCE.txt
Ligand inventory:      MDBIND_LIGAND_INVENTORY.txt

For preprocessing parameters, see:
  --contact-cutoff (default 0.8 nm for protein-protein)
  --ligand-contact-cutoff (default 0.45 nm for protein-ligand)
  --pocket-threshold (default 0.1 for pocket detection)
  --dynamic-edge-threshold (default 0.1 for edge filtering)

================================================================================
