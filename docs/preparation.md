# Feature and official-split preparation

Use compatible existing caches for paper replay whenever possible. New raw extraction can differ with model/tokenizer revisions and library versions; record their versions and hashes. The utilities in `scripts/preprocessing/` are byte-preserved archive members, listed in [the preprocessing manifest](../provenance/preprocessing_manifest.json). `scripts/prepare_features.py` resolves these helpers independently of the current working directory, requires explicit input/model paths, disables implicit Hugging Face downloads, and rejects missing target artifacts.

## Existing compatible caches

Place the original CSVs, KPGT NPZ, conditioned graph cache, similarity matrices and official split CSVs under the [documented data root](data.md). Copy or link external inputs rather than adding them to git. Retain the companion table row order, tensor aliases, masks and original cache content. Then run:

```bash
python -m sagf_dti validate-data \
  --config configs/paper/SNAP/prot/seed_42.yaml --data-root "$DATA_ROOT" --inspect-features
```

The selected configuration determines the required dataset/setting/seed. Validation does not prepare or rewrite caches. Inspect the validation report before a full run, including auxiliary coverage and identity warnings.

## Raw inputs and external models

Start with `drugs.csv` containing unique `Drug_ID,Drug` (SMILES) and `targets.csv` containing unique `Target_ID,Target` (amino-acid sequence). Keep IDs and order aligned with the original split tables. For a new dataset, choosing identifiers, sampling labels and designing splits is a separate scientific protocol; these commands do not recreate unavailable historical source data.

Install [KPGT](https://github.com/lihan97/KPGT#setup-environment) in its own environment for SMILES processing and its pretrained base model. Install [SeqDance/ESMDance](https://github.com/ShenLab/SeqDance#seqdanceesmdance-usage) separately for the carrier and predicted dynamic50 features. Obtain ESMDance weights through the [upstream weight references](https://github.com/ShenLab/SeqDance#data-and-model-weights). Also obtain the [ESM650 model/tokenizer](https://huggingface.co/facebook/esm2_t33_650M_UR50D) and cache the [ESM35 tokenizer](https://huggingface.co/facebook/esm2_t12_35M_UR50D). These sources are external; no download automation or pretrained asset is shipped. Check the exact accompanying licenses as described in [third-party notices](../THIRD_PARTY_NOTICES.md).

Use new preparation directories. The preserved extraction scripts skip existing outputs; their presence alone cannot establish that a cache was generated from the current sequence/model. In the examples below, all paths are absolute:

```bash
export RELEASE_ROOT=/absolute/path/to/SAGF-DTI
export DATA_ROOT=/absolute/path/to/features
export BASE="$DATA_ROOT/SNAP/biofusion1500_front"
export PREP_ROOT=/absolute/path/to/new-preparation
export KPGT_ROOT=/absolute/path/to/KPGT
export KPGT_DATA=/absolute/path/to/kpgt-datasets
export SEQDANCE_ROOT=/absolute/path/to/SeqDance
export ESMDANCE_WEIGHTS=/absolute/path/to/esmdance_weights
export ESM650_MODEL=/absolute/path/to/esm2_t33_650M_UR50D
```

## SMILES to KPGT2304

First export a KPGT input CSV in exactly the `drugs.csv` row order. Its `label=0` column is an input placeholder for external KPGT extraction, not a DTI training label:

```bash
python "$RELEASE_ROOT/scripts/prepare_features.py" kpgt-input \
  --drugs-csv "$BASE/drugs.csv" \
  --output-csv "$KPGT_DATA/SNAP_biofusion1500_front/SNAP_biofusion1500_front.csv"
```

In the separately activated KPGT environment, use its original preprocessing and feature extraction commands from its `scripts` directory. These flags follow the [upstream KPGT feature instructions](https://github.com/lihan97/KPGT#generate-latent-features-for-your-datasets); paths must match your installed checkout:

```bash
cd "$KPGT_ROOT/scripts"
python preprocess_downstream_dataset.py --data_path "$KPGT_DATA" --dataset SNAP_biofusion1500_front
python extract_features.py --config base --model_path /absolute/path/to/base.pth \
  --data_path "$KPGT_DATA" --dataset SNAP_biofusion1500_front
```

Return to the SAGF-DTI cached-feature environment and standardize the resulting archive:

```bash
python "$RELEASE_ROOT/scripts/prepare_features.py" kpgt \
  --drugs-csv "$BASE/drugs.csv" \
  --kpgt-npz "$KPGT_DATA/SNAP_biofusion1500_front/kpgt_base.npz" \
  --output-npz "$BASE/kpgt/kpgt.npz"
```

The wrapper requires finite `[N,2304]` features, unique IDs and exact `drugs.csv` order. Some external KPGT preprocessing versions filter invalid SMILES. A changed row count must be corrected at the input; silently dropping molecules would change the experiment. Full raw KPGT extraction has not been rerun during this release check.

## Sequences to residue caches

Run these commands in an external protein-extraction environment containing SeqDance's dependencies and locally cached tokenizer files. The SAGF-DTI package need not be imported by these utilities.

```bash
python "$RELEASE_ROOT/scripts/prepare_features.py" esm2 \
  --targets-csv "$BASE/targets.csv" --output-dir "$PREP_ROOT/esm650" \
  --model-name-or-path "$ESM650_MODEL" --device cuda --max-len 1500 --chunk-len 1022

python "$RELEASE_ROOT/scripts/prepare_features.py" carrier \
  --targets "$BASE/targets.csv" --output-dir "$PREP_ROOT/carrier480_raw" \
  --seqdance-root "$SEQDANCE_ROOT" --weights-dir "$ESMDANCE_WEIGHTS" \
  --device cuda --max-len 1022

python "$RELEASE_ROOT/scripts/prepare_features.py" dynamic \
  --targets "$BASE/targets.csv" --output-dir "$PREP_ROOT/dynamic50_raw" \
  --seqdance-root "$SEQDANCE_ROOT" --weights-dir "$ESMDANCE_WEIGHTS" \
  --device cuda --max-len 1022 --precision fp32
```

The dynamic utility concatenates the pretrained residue prediction heads into 50 properties. The carrier utility extracts 480-dimensional encoder residue embeddings. Special tokens are removed. ESM650 concatenates independently embedded chunks; changing that chunking or the auxiliary maximum length changes the cached representation. `--help` after each stage shows the preserved script's full options. Full pretrained extraction has not been rerun during this release check.

## Pad, build and merge graphs

Run lightweight cache assembly in the SAGF-DTI runtime. Pad both carrier480 and dynamic50 to the same effective `min(sequence length,1500)` view, with zero-valued uncovered residues and a binary coverage mask:

```bash
python "$RELEASE_ROOT/scripts/prepare_features.py" pad \
  --targets-csv "$BASE/targets.csv" --input-dir "$PREP_ROOT/carrier480_raw" \
  --output-dir "$PREP_ROOT/carrier480_padded" --max-len 1500

python "$RELEASE_ROOT/scripts/prepare_features.py" pad \
  --targets-csv "$BASE/targets.csv" --input-dir "$PREP_ROOT/dynamic50_raw" \
  --output-dir "$PREP_ROOT/dynamic50_padded" --max-len 1500

python "$RELEASE_ROOT/scripts/prepare_features.py" graphs \
  --targets-csv "$BASE/targets.csv" --esm2-dir "$PREP_ROOT/esm650" \
  --esmdance-dir "$PREP_ROOT/carrier480_padded" --output-root "$BASE/graphs" \
  --graph-version esmdance_node --max-len 1500

python "$RELEASE_ROOT/scripts/prepare_features.py" graphs \
  --targets-csv "$BASE/targets.csv" --esm2-dir "$PREP_ROOT/esm650" \
  --esmdance-dir "$PREP_ROOT/dynamic50_padded" --output-root "$BASE/graphs" \
  --graph-version esmdance_node_dyn50 --max-len 1500

python "$RELEASE_ROOT/scripts/prepare_features.py" merge \
  --feature-root "$DATA_ROOT" --datasets SNAP --view biofusion1500_front \
  --dynamic-version esmdance_node_dyn50 --esmdance35-version esmdance_node \
  --output-version esmdance_node_cond35_dyn50

python "$RELEASE_ROOT/scripts/prepare_features.py" similarity \
  --feature-root "$DATA_ROOT" --datasets SNAP --view biofusion1500_front \
  --graph-version esmdance_node_cond35_dyn50 --top-frac 0.05 --min-k 20 --max-k 300
```

The preserved helpers sanitize nonfinite values and use minimum aligned lengths when assembling graphs. Those are historical preparation behaviors, not proof that arbitrary mismatched input caches are compatible. Do not use new outputs to overwrite original paper caches. The final full data validation checks dimensions, aliases, masks, graph indices and every referenced pair.

## Original official splits

`--official-root` contains the original dataset folders `{SNAP,DRH,Kinase}` and official train/validation/test CSVs for each setting and seed. Inspect `scripts/prepare_splits.py --help` for the source filename contract, then normalize existing original split tables:

```bash
python "$RELEASE_ROOT/scripts/prepare_splits.py" \
  --official-root /absolute/path/to/original-official-datasets \
  --data-root "$DATA_ROOT" --datasets SNAP --settings prot \
  --seeds 42 1234 3407 5678 8888 --report "$PREP_ROOT/split-import.json"

python -m sagf_dti validate-data \
  --config "$RELEASE_ROOT/configs/paper/SNAP/prot/seed_42.yaml" --data-root "$DATA_ROOT" --inspect-features
```

The split importer preflights all requested source files and refuses existing outputs unless `--force` is explicitly supplied. It calls the preserved normalization logic; it does not redraw splits. Repeat dataset/setting selections only when the corresponding original tables and all compatible feature caches are available. DRH/Kinase conditioned graphs and similarity caches were incomplete locally at release preparation time.
