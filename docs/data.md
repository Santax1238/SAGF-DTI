# Data contract

`--data-root` is the feature-root directory, not a single dataset directory. Paths in shipped configurations resolve under it. For example:

```text
DATA_ROOT/
  SNAP/biofusion1500_front/
    drugs.csv
    targets.csv
    kpgt/kpgt.npz
    contrastive/drug_knn_sim.npz
    contrastive/target_knn_sim.npz
    graphs/esmdance_node_cond35_dyn50/<Target_ID>.pt
    biofusion_official_splits/prot/seed_42/
      prot_train.csv
      prot_val.csv
      prot_test.csv
  DRH/biofusion1500_front/...
  Kinase/biofusion1500_front/...
```

Use `random`, `drug`, or `prot` and the configured seed in each split path. Unselected datasets need not be present for a single-config run. Every referenced pair must have its drug feature and target graph; new entry points reject missing inputs instead of silently filtering samples.

`validate-data` defaults to the training and validation splits. Add `--inspect-features` to load and inspect their distinct graphs; otherwise it checks graph presence, tables, KPGT and similarity data. To explicitly include test inputs in the full check, add `--split train --split val --split test`. The training wrapper performs the checks required by its run and selected-checkpoint evaluation.

## Tables and vectors

| Artifact | Required representation |
| --- | --- |
| `drugs.csv` | Unique `Drug_ID`; `Drug` contains SMILES for raw preparation. Preserve the historical row order. |
| `targets.csv` | Unique `Target_ID`; `Target` contains the corresponding amino-acid sequence. Preserve historical row order and sequence identity. |
| Split CSV | Canonical `pair_id`, `drug_id`, `sequence_hash`, `label`; `sequence_hash` equals the graph/target identifier. The preserved dataset can normalize `Drug_ID`, `Target_ID`, and `Y`/`Label` aliases. Labels are binary. |
| KPGT NPZ | `drug_ids[N]` and finite float `fps[N,2304]`; IDs correspond to the original molecules. For generated contrastive matrices, row order must match `drugs.csv`. |
| Drug similarity NPZ | Square matrix in `arr_0` (the original trainer reads the first array); rows correspond to `drugs.csv` order. |
| Target similarity NPZ | Square matrix in `arr_0`; rows correspond to `targets.csv` order. |

The historical similarity matrices contain no ID metadata. Shape validation and recorded CSV order cannot prove the originating identity alignment; keep their original CSV companions. The preserved generator constructs symmetric binary kNN similarities from KPGT vectors and mean ESM2 graph features, with diagonal positives, top fraction 0.05, minimum k 20, and maximum k 300.

## Protein caches

Each graph is a trusted `torch.save` PyG `Data` artifact. Load only artifacts obtained from a trusted source; these files are external inputs, and the legacy format uses Python object deserialization. The graph contains node-aligned tensors:

| Field | Shape / meaning |
| --- | --- |
| `x`, `x_esm` | `[L,1280]`, frozen ESM2-650M residue representation |
| `x_dyn` | `[L,50]`, predicted residue dynamic properties |
| `predicted_dynamic_residue_features`, `esmdance_residue_properties` | Aliases for the same dynamic50 representation |
| `esmdance_35m_embedding` | `[L,480]`, frozen ESM2-35M/ESMDance carrier |
| `esmdance35_embedding`, `esmdance_encoder_embedding` | Carrier aliases |
| `esmdance_embedding_mask` | `[L,1]`, binary prefix mask marking available auxiliary residues |
| `edge_index` | `[2,E]`, valid integer node indices; the preserved builder creates both directions of sequence-neighbor edges |
| Target metadata | `target_id`, `sequence_hash`, `graph_version`, `effective_len`, and available source metadata |

The raw ESM2 `.pt` payload stores `x_esm`/`embedding [L,1280]` plus length/chunk metadata. The raw carrier cache is a tensor `[L,480]`. The dynamic cache stores the two dynamic aliases, 50 feature names and source metadata. Dynamic50 concatenates six scalar properties (SASA mean/std, normalized RMSF, three NMA components), secondary structure probabilities (8), and chi/phi/psi bins (12 each).

## Historical length and split limitations

`biofusion1500_front` uses the first 1,500 residues. The preserved ESM650 generator embeds this view in chunks of at most 1,022 residues, then concatenates them. ESMDance auxiliary extraction uses the first 1,022 residues; zero padding and its mask align the auxiliary caches to the longer graph. The mask determines coverage. Some historical metadata fields such as `covered_len` disagree with the saved tensor; validation reports that discrepancy without rewriting the original cache.

Retain these tensors and alignment rules for paper replay. Regenerating full-length features, changing chunking, adding auxiliary coverage, or using later cleaned cold-drug splits changes the experiment. Historical official split CSVs and later `biofusion_protocol_splits` / revision split directories are distinct. Input preparation normalizes original official CSVs; it does not invent new random/cold splits.
