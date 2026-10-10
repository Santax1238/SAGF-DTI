# Environment

Use an isolated Python 3.11 environment. The frozen cached-feature trainer needs PyTorch, PyG, NumPy, pandas, scikit-learn and PyYAML. RDKit/DGL and the pretrained protein models belong to feature preparation environments.

## CUDA 12.8 / Blackwell

From the release root:

```bash
conda env create -f environment.yml
conda activate sagf-dti
python -m pip install -r requirements-blackwell.txt
python -m pip install --no-deps -e .
python scripts/check_environment.py --device cuda --strict-versions --output environment-check.json
```

`environment.yml` creates the common Python environment; the second install selects the GPU runtime. The pinned CUDA wheels are `torch==2.7.1+cu128`, `torch-scatter==2.1.2+pt27cu128`, and `torch-sparse==0.6.18+pt27cu128`. PyTorch publishes the [2.7.1 CUDA 12.8 installation command](https://pytorch.org/get-started/previous-versions/), and the official [PyG torch 2.7 CUDA 12.8 wheel index](https://data.pyg.org/whl/torch-2.7.0+cu128.html) includes Python 3.11 Linux x86_64 scatter/sparse wheels. PyG extensions use the `torch-2.7.0` index for the 2.7 series.

## CPU

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install --only-binary=:all: torch-scatter==2.1.2 torch-sparse==0.6.18 \
  --find-links https://data.pyg.org/whl/torch-2.7.0+cpu.html
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python scripts/check_environment.py --device cpu --strict-versions
```

The CPU command follows [PyTorch's official previous-version instructions](https://pytorch.org/get-started/previous-versions/) and the matching [PyG CPU wheel index](https://data.pyg.org/whl/torch-2.7.0+cpu.html). The paper configurations specify CUDA. Use `--device cpu` for CPU training/evaluation diagnostics; those checks do not establish the original CUDA training reproduction.

## Versions and local evidence

| Package | Recommended runtime | Checked pinned venv | Earlier research environment |
| --- | --- | --- | --- |
| Python | 3.11 | 3.11.15 | 3.11.15 |
| PyTorch | 2.7.1 + matching runtime | 2.7.1+cu128 | 2.7.1+cu128 |
| PyG | 2.6.1 | 2.6.1 | 2.6.1 |
| torch-scatter | 2.1.2 + matching runtime | 2.1.2+pt27cu128 | 2.1.2+pt27cu128 |
| torch-sparse | 0.6.18 + matching runtime | 0.6.18+pt27cu128 | 0.6.18+pt27cu128 |
| NumPy | 1.26.4 | 1.26.4 | 2.4.6 |
| pandas | 2.2.3 | 2.2.3 | 2.3.3 |
| scikit-learn | 1.5.2 | 1.5.2 | 1.8.0 |
| PyYAML | 6.0.2 | 6.0.2 | 6.0.3 |

The checked venv passed `--strict-versions` CPU and CUDA forward/backward, scatter and sparse checks on an RTX 5070 (compute capability 12.0, approximately 12 GiB): [pinned CPU report](../verification/environment_isolated_cpu.json), [pinned CUDA report](../verification/environment_isolated_cuda.json). It installs the scientific dependency pins in a venv created with `--system-site-packages`, reusing existing compatible CUDA/PyG libraries. This verifies the runtime combination, not a complete fresh conda solve or a fresh CPU-wheel installation. Earlier research-environment checks remain recorded in the [original CPU report](../provenance/environment_cpu.json) and [original CUDA report](../provenance/environment_cuda.json).

The local WSL host cannot sustain the original pinned-host allocations. Pass `--disable-host-pinning` to train/evaluate to record the explicit `pin_memory: false` runtime override. The canonical configuration, batch32/64, worker count, AMP, model, losses and optimizer recipe remain preserved. The [full-batch CUDA diagnostic](../verification/full_batch_cuda.json) passed with this override; it is not a completed training reproduction.

The installed local RDKit 2023.9.6 has a NumPy 2 ABI compatibility problem; cached training avoids importing it. The recommended NumPy 1.26.4 pins reduce this native-extension compatibility risk for new environments. Install and run KPGT according to its own environment instructions; keep that environment separate. For ESMDance/ESM2 preparation, use the upstream SeqDance requirements (including Transformers) and locally available model/tokenizer files. Those optional dependencies are intentionally absent from cached training requirements.
