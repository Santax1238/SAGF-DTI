# Reproduction and evidence

The paper's SNAP official cold-protein row uses seeds 42, 1234, 3407, 5678 and 8888. Its reported AUROC is `0.8885686 ± 0.0076321`; AUPR is `0.8992116 ± 0.0146953`, with mean ± sample SD. This is one row of the full 45-trial suite (three datasets × three settings × five seeds). All 45 original configurations are shipped.

There are no retained exact main-paper checkpoints. Recomputing a metric from archived predictions audits a reported artifact; it does not demonstrate fresh training reproduction. A smoke run likewise establishes only that the tested path executes. [Verification status](verification.md) separates these checks.

## Completed local reproduction

All five SNAP official cold-protein seeds completed from scratch on the RTX 5070, using the preserved recipe and explicit `pin_memory: false` WSL override. Validation AUROC selected each checkpoint before test access; all five checkpoints passed independent reload evaluation. The coordinator completed October 11, 2026 at about 05:17 Asia/Shanghai (October 10, 21:17 UTC).

| Seed | Epochs run | Best epoch | Test AUROC | Test AUPR |
| --- | --- | --- | --- | --- |
| 42 | 13 | 5 | 0.876872 | 0.884419 |
| 1234 | 18 | 10 | 0.901375 | 0.921819 |
| 3407 | 17 | 9 | 0.879244 | 0.893810 |
| 5678 | 19 | 11 | 0.896374 | 0.921622 |
| 8888 | 33 | 25 | 0.891163 | 0.906710 |

Mean ± sample SD is **AUROC 0.88900547 ± 0.01065910**, **AUPR 0.90567605 ± 0.01664792**. These aggregates use the original selected-checkpoint test metrics stored at completion. Reload ranking metrics differ by at most `6.76452e-7`, below the documented `5e-5` tolerance; prediction values are not bitwise identical. No run was rejected, failed or recorded OOM. The formal runs reached 8.23242 GiB peak reserved CUDA memory.

[Sanitized fresh evidence](../verification/fresh_reproduction.json) records full per-seed metrics, checkpoint/artifact hashes, source-freeze bindings and reload comparisons. These results are similar to the historical paper-level reference, with a different ranking implementation and documented hardware/transfer settings. They establish completion of this five-seed reproduction, not exact historical score equality or a model improvement. The other eight dataset/setting groups have historical audits and shipped configurations; they were not freshly retrained here.

## Archived prediction audit

The independent [audit artifact](../provenance/archived_prediction_audit.json) recomputes all eight saved classification metrics from each historical `pred_prob` column. All 45 runs match exactly, with maximum difference 0. The SNAP cold-protein means/sample SD recover the reported row. The audit also records population SD: AUROC `0.0068263957`, AUPR `0.0131438953`; these smaller values use `ddof=0` and are not the paper's displayed spread.

The older CSV's `pred_logit` is the positive-class logit alone. The retained August 28 trainer instead ranks the float32 binary margin `logit1 - logit0`. The old single-logit column lacks `logit0`, while rounded/saturated probabilities lose margin information. Its diagnostic re-ranking is a different score, not a corrected historical result; no corrected binary-margin paper metric can be recovered exactly from the saved CSVs. This ranking/version boundary can affect validation checkpoint selection as well as test metrics in new runs.

## Original training

Validate all referenced data before running. The original recipe requires one actual forward batch32, AMP, max50, patience8, learning rate/weight decay `1e-4`, original auxiliary/reliability and contrastive losses, and Cauchy gradient-alignment LR updates. Preserve the original loader RNG and validate on AUROC. Evaluate the test set only after selecting the checkpoint on validation data. The default original trainer evaluates the selected checkpoint after training; a separate `evaluate --split test` also checks checkpoint/config identity.

```bash
export RELEASE_ROOT=/absolute/path/to/SAGF-DTI
export DATA_ROOT=/absolute/path/to/features
export RUN_ROOT=/absolute/path/to/new-runs/snap_prot

for seed in 42 1234 3407 5678 8888; do
  python -m sagf_dti train \
    --config "$RELEASE_ROOT/configs/paper/SNAP/prot/seed_${seed}.yaml" \
    --data-root "$DATA_ROOT" --output "$RUN_ROOT" --require-paper-config
done
python -m sagf_dti summarize --runs "$RUN_ROOT" --output /absolute/path/to/reports/snap_prot
```

Use a new run root. On the locally checked WSL host, add `--disable-host-pinning` to each train/evaluate command: the original pinned-host allocation failed before model forward, and the explicit override records `pin_memory: false`. It retains the configured actual batch32/64, workers, losses, AMP and optimizer behavior. Canonical configurations keep their original pinning setting.

Completed runs are protected from overwrite. A five-seed summary is valid only when five independently completed runs have selected-checkpoint evaluations. Partial runs remain partial and cannot establish the reported five-seed mean/standard deviation. The reporter uses sample SD, matching the displayed SNAP paper row; the archived audit provides both conventions.

The [longest-input GPU stress test](../verification/full_batch_cuda.json) passed on the approximately 12 GiB RTX 5070: actual train batch32 and validation batch64, every selected graph at 1500 nodes, three optimizer updates with no skipped scaler steps, and nonzero finite dynamic/alpha/variance gradients. Auxiliary, contrastive and Cauchy LR behavior remained enabled. Peak reserved memory was 6.15625 GiB and peak allocated memory 3.85148 GiB, with the recorded pinning override. Repeated stress-test batches are not completed paper training.

For an automated SNAP cold-protein reproduction, first generate your own passing profile, then use the sequential five-seed coordinator. This example includes the explicit local WSL transfer override:

```bash
export PROFILE_ROOT=/absolute/path/to/new-profile
python -m sagf_dti.benchmark \
  --config "$RELEASE_ROOT/configs/paper/SNAP/prot/seed_42.yaml" \
  --data-root "$DATA_ROOT" --output "$PROFILE_ROOT" --device cuda \
  --max-reserved-gib 8.5 --disable-host-pinning
python "$RELEASE_ROOT/scripts/reproduce_paper.py" \
  --data-root "$DATA_ROOT" --output "$RUN_ROOT" \
  --profile "$PROFILE_ROOT/benchmark.json" --device cuda \
  --disable-host-pinning --background
```

The coordinator writes `reproduction_status.json` and seed logs in `RUN_ROOT`, reloads each selected checkpoint for an independent test evaluation, and summarizes only verified completed runs. It refuses to reuse partial runs or changed source/input identities. Keep code and source manifests unchanged while training is active. Omit `--disable-host-pinning` from both commands when using the original transfer setting on a compatible host.

Smaller microbatches with accumulation change contrastive negatives; BF16, changed truncation or a revised model are different experiments. The retained FP32 correction casts each AMP-produced class logit to float32 before taking their difference, then ranks the difference and uses its sigmoid for probabilities. The full model forward retains AMP.

## Run artifacts

Retain the resolved/base configuration, source/config/input hashes, seed and full train sample count, environment, per-epoch metrics, time/memory evidence, selected checkpoint, and evaluation files/hashes. The frozen trainer writes `metrics.csv`, `config.yaml`, validation/test predictions and `checkpoints/{best,last}.pt` under the configured run name. The portable wrapper adds manifest/completion evidence. A checkpoint on its own does not prove completion or establish the input identity.

Use `python -m sagf_dti summarize --runs RUN_ROOT --output REPORT_DIR` to inspect completion and group results. Review fresh results against the historical reference only after all required checks and selected-checkpoint test evaluations complete. Hardware, library and source-version changes prevent a promise of bitwise or exact-score reproduction.

## Legacy entry point

The numeric original trainer remains at `my_experiments/scripts/train/09_train_experiment.py`. Installed callers can load it without flat-name import conflicts:

```bash
python -c 'from sagf_dti.legacy import load_trainer; load_trainer().main()' \
  --config /absolute/path/to/fully-resolved-config.yaml
```

This advanced legacy entry expects fully resolved data/run paths and retains the historical script behavior, including its input filtering. Use the portable commands for strict completeness and run protection. To create a resolved configuration programmatically, use `sagf_dti.config.resolve_config(config_path, data_root, run_root)` and serialize the returned dictionary as YAML. This only resolves paths; it does not create features or splits.
