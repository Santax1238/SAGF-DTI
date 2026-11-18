# Real MD 621 public release

This directory is the lightweight, traceable release for the current real-MD trajectory work in GSDTI.

## Included

- The source-conditioned real-MD coverage state. The original 437 targets remain in place and 184 targets were added, giving 621 covered targets. The additions are mdCATH 15, MISATO 151, DynamicPDB 16, and MoDEL 2.
- The 184-target table and official-test coverage table. Local graph paths are replaced with `omitted_from_public_release`.
- Independent source and completion audits, including the 621-target trajectory-window patch manifest.
- The trajectory-window implementation and unit tests. The completed stage contains 1,550 window jobs across 775 trajectory-target mappings, with both halves available for all 621 eligible targets.
- Evaluation summaries and receipts for the original 621 comparison, the native-concat matched evaluation, and the full matched evaluation.
- MDBind preprocessing documentation and the residue-level preprocessing script.

## Reproduce locally

The code snapshots are provenance artifacts from the frozen experiment. Running them requires the local source data, frozen feature extractor, historical scaler, and graph inputs referenced by the receipts. The public release does not include those local inputs.

```bash
python -m unittest discover -s code -p 'test_*.py'
# Run the preprocessing script inside the project environment with mdtraj,
# PyTorch, and PyG installed.
python code/preprocess_mdbind_dynamics.py --help
```

## Data policy

Raw XTC/PDB/HDF trajectories, processed `.pt` features, checkpoints, runtime archives, prediction tables, and caches remain in the local experiment workspace. They are intentionally excluded from this GitHub release. The public files preserve identifiers, provenance hashes, counts, and result summaries without republishing those large artifacts.

## Scientific scope

The 621 view is source-conditioned. It does not establish an ordinary unconditioned DTI test set or a causal 437-versus-621 data increment. The matched Gaussian-versus-native-concat AUROC intervals cross zero in the current evaluation, and Gaussian NLL is higher in that comparison. The receipts and full local artifacts remain the authority for exact numerical provenance.
