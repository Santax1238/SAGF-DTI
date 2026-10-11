# Local verification status

Status recorded during release preparation on October 11, 2026. A successful environment smoke check, preserved-model parity test, archived-prediction audit, and fresh training run answer different questions; each is reported separately.

| Check | Status / evidence |
| --- | --- |
| CPU runtime | Passed with strict release pins: tensor forward/backward, scatter and sparse operations; [JSON](../verification/environment_isolated_cpu.json). |
| CUDA runtime | Passed with strict release pins on RTX 5070 / CUDA 12.8 / compute capability 12.0; [JSON](../verification/environment_isolated_cuda.json). |
| Environment isolation | Scientific pins installed in a venv using `--system-site-packages` to reuse compatible CUDA/PyG libraries. A complete fresh conda solve and CPU-wheel install were not verified. |
| Frozen source / recipe | All 17 frozen source members retain archive hashes. 109 tests passed, including model parity, installed-wheel imports, malformed inputs, run identity and checkpoint reload; [release checks](../verification/release_checks.json). Independent final review also passed 89 targeted tests and a fresh train/reload/summary exercise. |
| Cached input readiness | All five SNAP official cold-protein split caches available locally; DRH/Kinase conditioned graph/similarity preparation still required locally. |
| Lightweight preparation | Passed outside the release working directory: KPGT export/normalization, both padded branches, graph construction/merge, similarity generation, missing-target/overwrite guards and full three-split validation. The synthetic long graph preserves 1500 nodes / 1022 auxiliary coverage; [JSON](../provenance/preparation_verification.json). |
| Full pretrained feature extraction | Not rerun. Existing original utilities are preserved and lightweight assembly is checked separately. |
| Archived saved-prediction audit | Passed: all 45 runs exactly match all eight saved classification metrics from probabilities; maximum difference 0. [Sanitized audit with 135 input hashes](../provenance/archived_prediction_audit.json). |
| Longest-input full-batch CUDA check | Passed actual train batch32 / validation batch64 with every graph at 1500 nodes, three optimizer updates / zero skipped scaler steps, finite losses and nonzero finite dynamic/alpha/variance gradients. Peak reserved 6.15625 GiB / allocated 3.85148 GiB; [JSON](../verification/full_batch_cuda.json). |
| Host-pinning override | Original WSL pinned-host allocation failed before forward. `--disable-host-pinning` records `pin_memory: false`; canonical recipe, actual batches, workers, losses and AMP remain preserved. |
| Fresh SNAP cold-protein reproduction | **5/5 completed from scratch**, October 11 about 05:17 Asia/Shanghai. AUROC **0.88900547 ± 0.01065910**, AUPR **0.90567605 ± 0.01664792** (mean ± sample SD), epochs 13/18/17/19/33. No rejected run, failure or OOM evidence; [sanitized evidence](../verification/fresh_reproduction.json). |
| Selected-checkpoint reload | All five pass `5e-5` AUROC/AUPR tolerance; maximum difference `6.76452e-7`. Pair membership/order/labels agree; prediction values have small GPU roundoff differences and are not bitwise equal. Checkpoint and evaluation hashes are recorded in the fresh evidence. |
| Original main-paper checkpoints | Not retained. |

The checked pinned runtime is Python 3.11.15, torch 2.7.1+cu128, PyG 2.6.1, scatter 2.1.2+pt27cu128, sparse 0.6.18+pt27cu128, NumPy 1.26.4, pandas 2.2.3, scikit-learn 1.5.2 and PyYAML 6.0.2. RDKit is not imported by the runtime check. The earlier research environment's different scientific package versions remain recorded separately. See [environment instructions](environment.md) for both version tables and CPU/CUDA install commands.

The audit recovers SNAP cold-protein AUROC `0.8885686063 ± 0.0076321424` and AUPR `0.8992115872 ± 0.0146953217` (mean ± sample SD), agreeing with the rounded paper values. Population SD is separately recorded. The older saved `pred_logit` is the positive-class logit alone, whereas this release's ranking score is the float32 difference between both class logits. Old-column re-ranking changes the score and cannot establish corrected historical metrics because the negative-class logit and exact unsaturated margin are unavailable.

The completed fresh results use the release's FP32 binary-margin ranking. They reach similar paper-level results under the documented ranking-version, hardware and host-pinning differences; they do not establish exact historical equality or a model improvement. The formal runs peaked at 8.23242 GiB reserved memory, higher than the short diagnostic's 6.15625 GiB. All training manifests match the unchanged [source freeze](../verification/source_freeze.json). Only SNAP cold-protein was freshly retrained; the remaining eight groups retain historical audit evidence.

Update this record only with verified artifacts. Report incomplete seeds and any failed memory/recipe check directly; archived results, short smoke runs, and revised experiments are not substitutes for from-scratch original-paper reproduction.
