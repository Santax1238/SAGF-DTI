# Third-party and licensing notices

No repository-wide license grant for the original SAGF-DTI code, manuscript materials, or datasets has been established. This release adds no inferred license grant and no fabricated paper citation. Obtain permission from the rights holders for uses that require it.

## External pretrained feature tools

- **SeqDance / ESMDance**, Chao Hou and collaborators: [upstream repository](https://github.com/ShenLab/SeqDance), locally inspected revision `73ccea0fa4d54947441c5df9b5ae2f02238153b3`. The repository code includes a GNU GPL v3 license. Upstream provides [model and data links in its README](https://github.com/ShenLab/SeqDance#data-and-model-weights). The locally inspected ESMDance weight directory includes a CC BY 4.0 `LICENSE`, while its README metadata says MIT; these inconsistent labels do not establish a license for this release. Check the terms accompanying the exact weight artifact before using or redistributing it. Neither code nor weights are bundled.
- **KPGT**, Han Li and collaborators: [upstream repository](https://github.com/lihan97/KPGT), locally inspected revision `47dc1646c70b2138a157de481d24a1ac35d174cd`. Its code is licensed under Apache License 2.0. Upstream supplies environment, feature extraction and pretrained-model instructions in its README. Neither its source nor weights are bundled.
- **ESM2**, Meta/Facebook AI: the preparation utilities use [ESM2-650M](https://huggingface.co/facebook/esm2_t33_650M_UR50D) and [ESM2-35M](https://huggingface.co/facebook/esm2_t12_35M_UR50D). Their official model cards label these models MIT; check the terms attached to the exact assets obtained. Models and tokenizers are not bundled.

Feature extraction utilities copied from the SAGF-DTI snapshot import external libraries; copying those wrappers does not redistribute the SeqDance/KPGT implementations. Third-party package licenses remain with their respective packages. PyTorch, PyG, NumPy, pandas, scikit-learn and PyYAML are installed as dependencies rather than copied into this repository.

## Attribution references

KPGT: Han Li et al., *A knowledge-guided pre-training framework for improving molecular representation learning*, Nature Communications 14, 7568 (2023). See the [upstream citation](https://github.com/lihan97/KPGT#citation).

SeqDance/ESMDance attribution and current publication details are provided by the [upstream README](https://github.com/ShenLab/SeqDance#citation). Use the citation associated with the exact external model version used in your experiment.
