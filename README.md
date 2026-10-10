# SAGF-DTI

Original SAGF-DTI paper implementation with portable training, evaluation, input validation, and result reporting. The Gaussian model and archived training recipe are preserved from the September 12 pre-revision source snapshot; this release does not include later CLS experiments. [中文说明](README.zh-CN.md)

SAGF-DTI combines cached KPGT molecular representations with ESM2-650M residue semantics and ESMDance predicted dynamic properties. The dynamic branch is conditioned on the ESM2-35M/ESMDance carrier. Residue Gaussian fusion uses a tempered product of experts and a semantic anchor; drug-conditioned query/key/value attention pools the fused residue features. The preserved code also contains separate branch prediction heads, reliability supervision, and drug/target contrastive losses. See [architecture and protocol](docs/protocol.md).

## Install

Use a new Python 3.11 environment. For Linux x86_64 with CUDA 12.8, including the local Blackwell GPU:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-blackwell.txt
python -m pip install --no-deps -e .
python scripts/check_environment.py --device cuda --output environment-check.json
```

For CPU installation, conda setup, verified wheel sources, and the exact local package versions, see [environment instructions](docs/environment.md). Cached-feature training does not import RDKit, SeqDance, KPGT, or Transformers. Raw feature extraction uses separately installed pretrained models and their dependencies.

## Prepare data

Feature caches, pretrained weights, and source datasets are external to this repository. Obtain inputs through their original providers and applicable terms; there is no SAGF-DTI dataset download URL in this release.

Set `DATA_ROOT` to a directory containing `SNAP/biofusion1500_front`, `DRH/biofusion1500_front`, and `Kinase/biofusion1500_front`. Each selected dataset needs CSV identifiers/sequences/SMILES, KPGT features, conditioned graphs, similarity matrices, and official split CSVs. [Data formats and layout](docs/data.md) lists every required field; [feature and split preparation](docs/preparation.md) provides commands from raw sequences/SMILES or existing caches.

```bash
export DATA_ROOT=/absolute/path/to/features
python -m sagf_dti validate-data \
  --config configs/paper/SNAP/prot/seed_42.yaml --data-root "$DATA_ROOT" --inspect-features
```

All 45 original configurations are under `configs/paper/{SNAP,DRH,Kinase}/{random,drug,prot}/seed_{42,1234,3407,5678,8888}.yaml`. Only input and run-root paths are normalized. `prot` means cold protein, `drug` means the historical official cold-drug setting, and `random` means the official random setting. Later cleaned cold-drug splits are a different protocol.

## Train, evaluate, and reproduce

```bash
python -m sagf_dti train \
  --config configs/paper/SNAP/prot/seed_42.yaml \
  --data-root "$DATA_ROOT" --output runs --require-paper-config

python -m sagf_dti evaluate \
  --config configs/paper/SNAP/prot/seed_42.yaml \
  --data-root "$DATA_ROOT" --checkpoint /absolute/path/to/best.pt \
  --output evaluations/snap_prot_42 --split test

python -m sagf_dti summarize --runs runs --output reports
```

The selected checkpoint maximizes validation AUROC. Test evaluation follows checkpoint selection. The wrapper validates complete inputs, checks checkpoint/config identity, and refuses to overwrite a completed run. After installation the package works from other working directories; pass an absolute config path when working outside the release directory.

For the original SNAP cold-protein five-seed row:

```bash
for seed in 42 1234 3407 5678 8888; do
  python -m sagf_dti train \
    --config "configs/paper/SNAP/prot/seed_${seed}.yaml" \
    --data-root "$DATA_ROOT" --output runs/snap_prot --require-paper-config
done
python -m sagf_dti summarize --runs runs/snap_prot --output reports/snap_prot
```

The original recipe uses one actual batch of 32, maximum 50 epochs, patience 8, learning rate and weight decay `1e-4`, AMP, unimodal/reliability supervision, two contrastive weights of `0.02`, and Cauchy gradient-alignment learning-rate updates. Gradient accumulation changes the contrastive negatives and is not an equivalent paper recipe. The retained August 28 correction converts each class logit to float32 before taking `logit1 - logit0`, ranks that binary margin, and uses its sigmoid for probabilities. The model forward still uses AMP. See [reproduction and evidence](docs/reproduction.md) for run artifacts, score-version differences, and the legacy entry point.

On the locally checked WSL host, add `--disable-host-pinning` to training/evaluation commands. This records `pin_memory: false` as an explicit runtime override; the canonical configurations retain `pin_memory: true`. Batch sizes, workers, model, losses, AMP and optimizer recipe remain unchanged.

## Reported results and verification

The paper reports SNAP official cold-protein five-seed AUROC **0.8885686 ± 0.0076321** and AUPR **0.8992116 ± 0.0146953** (mean ± sample SD). The independent [archived-prediction audit](provenance/archived_prediction_audit.json) exactly recovers all eight saved classification metrics in all 45 historical runs from `pred_prob`. The older `pred_logit` column stores only the positive-class logit, so it cannot reproduce the corrected binary-margin ranking without the missing negative-class logit. No exact main-paper checkpoints were retained. This audits existing artifacts; it is not fresh training.

CPU/CUDA checks passed with the recommended scientific dependency pins, reusing the existing compatible CUDA/PyG packages. The longest-input stress test passed actual train batch32 and validation batch64 at 1500 nodes per graph, with peak reserved memory **6.15625 GiB** and explicit host-pinning override. [Verification status](docs/verification.md) links the evidence. Fresh five-seed training results remain pending until independently completed runs and selected-checkpoint evaluations are recorded. Hardware, library and ranking-version changes can affect scores; this release does not promise bitwise replay.

## Provenance and external components

The [source manifest](provenance/source_manifest.json), [configuration manifest](provenance/config_manifest.json), and [preprocessing manifest](provenance/preprocessing_manifest.json) bind release files to preserved archive members. [Provenance](docs/provenance.md) explains the source boundary and historical protocol limitations. [Third-party notices](THIRD_PARTY_NOTICES.md) identifies external components and licensing evidence. A repository-wide license grant for the original SAGF-DTI materials has not been established, so this release does not invent one.
