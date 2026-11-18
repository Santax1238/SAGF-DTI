# MDBind Dynamic Preprocessing

`preprocess_mdbind_dynamics.py` converts one MDBind local folder into a residue-level graph with dynamic features aggregated across its local trajectories.

## What The Script Reads

For each local such as `A03O6`, it expects:

- `A03O6_topology.prmtop`
- `A03O6.1_structure.pdb` ... `A03O6.10_structure.pdb`
- `A03O6.1_trajectory.xtc` ... `A03O6.10_trajectory.xtc`

## What It Writes

Each processed local is saved as `data/mdbind_processed/<local_id>.pt`.

The `.pt` payload contains:

- `graph`: a PyG `Data` object
- `graph.x`: node scalar features
- `graph.edge_index`: sparse residue graph edges
- `graph.edge_attr`: edge features
- `graph.esm2_embedding`: optional residue embeddings if provided
- `graph.esmdance_embedding`: optional residue embeddings if provided
- `temporal_slices`: per-replicate temporal snapshots for stage-A PINN pretraining
- `dense_pair_features`: dense pairwise statistics for later experiments
- residue names, numbers, chain ids, ligand metadata, and source-file metadata

## First-Pass Node Features

- `rmsf`
- `sasa_mean`
- `sasa_std`
- `pocket_flag`
- `ligand_contact_freq`

`pocket_flag` is defined from ligand-contact frequency and defaults to `ligand_contact_freq >= 0.1`.

## First-Pass Edge Features

- `static_contact`
- `dynamic_contact_occupancy`
- `dccm`
- `ca_dist_mean`
- `ca_dist_std`

Static contacts and occupancy both use a default C-alpha cutoff of `0.8 nm`.

## Ligand Handling

- Standard amino acids are treated as protein residues.
- Water residues are ignored.
- Metal ions such as `ZN` and `CA` are excluded from ligand-contact calculations in the first version.
- Non-protein, non-water, non-ion residues are treated as ligand-like residues.

## Required Dependency

This script currently requires `mdtraj` for XTC loading and trajectory statistics.

Suggested install:

```bash
pip install mdtraj
```

## Example Usage

Process one local:

```bash
python preprocess_mdbind_dynamics.py --only-local A03O6
```

Process the full `locals.txt` list with frame downsampling:

```bash
python preprocess_mdbind_dynamics.py --stride 10 --output-dir data/mdbind_processed_stride10
```

Attach precomputed embeddings when available:

```bash
python preprocess_mdbind_dynamics.py \
  --esm2-dir data/mdbind_embeddings/esm2 \
  --esmdance-dir data/mdbind_embeddings/esmdance
```

## Stage-A / Stage-B Commands

Stage-A temporal PINN pretraining:

```bash
python temporal_pretrain.py --processed-dir data/mdbind_processed --epochs 30
```

Export dynamic embeddings:

```bash
python temporal_export.py \
  --processed-dir data/mdbind_processed \
  --checkpoint checkpoints/temporal_pinn/best.pt \
  --output-dir data/mdbind_temporal_embeddings
```

Validate embedding alignment:

```bash
python validate_temporal_embeddings.py \
  --graph-dir data/mdbind_processed \
  --embedding-dir data/mdbind_temporal_embeddings
```

Stage-B GS-DTI fine-tuning with temporal embeddings:

```bash
python train_gsdti_temporal.py \
  --graph-folderpath data/bindingDB/targets/graph \
  --temporal-embedding-dir data/mdbind_temporal_embeddings
```

If `Target_ID` in BindingDB and local IDs in MDBind are different, build a mapping file first:

```bash
python build_temporal_target_mapping.py \
  --targets-csv data/BindingDB/targets/targets.csv \
  --processed-dir data/mdbind_processed \
  --output-json data/mdbind_temporal_embeddings/target_mapping.json \
  --min-ratio 0.95
```

Then pass it into training:

```bash
python train_gsdti_temporal.py \
  --graph-folderpath data/bindingDB/targets/graph \
  --temporal-embedding-dir data/mdbind_temporal_embeddings \
  --temporal-mapping-file data/mdbind_temporal_embeddings/target_mapping.json
```

## Notes

- The first version prioritizes a clean residue-level schema and direct compatibility with graph models.
- ESM-2 and ESMDance are optional inputs in this script because they are not derived from the XTC trajectories themselves.
- Once this pipeline is stable on `mdbind_50_locals`, it should scale to larger MDBind subsets with the same folder convention.

## 为什么把时间连续性与物理约束加进表示学习

### 1. 如何让动态特征映射到有用向量空间

目标不是重建全轨迹，而是学习对 DTI 有判别力的动力学摘要向量 `z_dyn`。

可行做法：
- 用时序编码器（TCN/LSTM/Transformer）输入轨迹片段，输出残基/边的动态 embedding。
- 将 PINN 约束作用在隐空间演化上，例如 `dz/dt = f_theta(z, context)`，并加上平滑、能量合理、残差约束等物理正则。
- 与 DTI 主任务联合优化：`L = L_DTI + λ1 L_phys + λ2 L_temporal`，让 `z_dyn` 同时满足物理与任务判别性。

判断“有用”的标准：
- 加入 `z_dyn` 后验证集 AUC/PRC 提升。
- 消融实验去掉动态分支性能下降。
- 相似 `z_dyn` 的蛋白在结合行为上更接近（可解释性增强）。

### 2. 为什么构象变化轨迹能提高 DTI 准确率

DTI 本质是“配体与蛋白可达构象的匹配”，而不是单一静态结构匹配。轨迹提供了额外信息：
- **构象可达性**：某些口袋仅在部分构象出现（瞬时口袋）。
- **诱导契合/构象选择**：配体更偏好轨迹中的某一段，而非单一端点。
- **柔性与熵效应代理**：轨迹统计能反映结合代价与稳定性。

因此动态特征能减少“静态结构误判”，提升 DTI 预测的准确性与鲁棒性。
