# Original architecture and training protocol

The release uses the archived `E4_GAUSS_NODE_FUSION` model. A 2304-dimensional KPGT vector is encoded as the drug representation. Each protein residue has a 1280-dimensional ESM650 semantic feature, a 480-dimensional ESM35/ESMDance carrier, and 50 predicted dynamic properties. The conditioned dynamic branch and semantic branch parameterize residue Gaussian representations; tempered product-of-experts fusion and a semantic anchor combine them with the auxiliary coverage mask. The paper configuration uses the fused mean (`sample_fused: false`).

Drug-conditioned pooling has query/key/value attention and then an interaction MLP predicts binary labels. Separate unimodal branch heads provide auxiliary prediction, with reliability supervision. This describes the executable frozen code; simplified manuscript prose may omit these training components. Module construction order, state-dict keys, and unused/frozen parameters are retained because they can affect seeded initialization and checkpoint compatibility.

| Setting | Original value |
| --- | --- |
| Training / evaluation batch | 32 / 64 |
| Maximum epochs / early-stopping patience | 50 / 8 |
| Initial learning rate / weight decay | 1e-4 / 1e-4 |
| Selection metric | Validation AUROC |
| AMP / gradient clipping | Enabled / 1.0 |
| Hidden width / dropout | 256 / 0.2 |
| Gaussian variance bounds | 1e-4 to 50 |
| Semantic-anchor alpha initialization | 0.1 |
| Unimodal / reliability weights | 0.2 / 0.03 |
| Drug / target contrastive weights | 0.02 / 0.02 |
| Contrastive temperature | 0.07 |
| Cauchy LR beta / bounds | 0.01 / 1e-6 to 2e-4 |
| Data-loader workers | 4; persistent workers and prefetch factor 2 |

Contrastive negatives come from the actual forward batch. Preserve batch32 rather than replacing it with accumulated smaller batches. The Cauchy learning-rate update uses gradient alignment; the original trainer does not perform higher-order differentiation or an extra backward pass.

The retained trainer includes the August 28 FP32 ranking correction: it converts each class logit to float32, computes `logit1 - logit0`, ranks that binary margin, and applies sigmoid to it for probabilities. The model forward retains AMP. The older trainer ranked produced probabilities and saved only `logit1` in its `pred_logit` column; the current column stores the binary margin. This score/export change must be considered when comparing historical artifacts and new training. The September source snapshot corroborates the original Gaussian model, but an exact contemporaneous source commit for the 45 historical trials is unavailable. See [provenance](provenance.md).
