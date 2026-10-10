# SAGF-DTI 原论文代码

本项目整理原始 SAGF-DTI 高斯融合模型及训练流程，提供可移植的训练、评估、数据检查与结果汇总入口。模型源码保留自 9 月 12 日修订前快照，45 份原始配置仅调整数据及输出路径。后续 CLS 实验不属于本次发布。[English README](README.md)

模型使用缓存的 KPGT 2304 维药物表示、ESM2-650M 1280 维逐残基语义表示、ESM2-35M/ESMDance 480 维载体及 50 维预测动态属性，执行条件化动态编码、高斯融合与药物条件注意力池化。原代码同时保留分支辅助预测、可靠性监督及对比损失。

## 环境与数据

建议新建 Python 3.11 环境；Linux x86_64、CUDA 12.8 / Blackwell 安装示例：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-blackwell.txt
python -m pip install --no-deps -e .
python scripts/check_environment.py --device cuda
```

CPU / conda 安装及实际验证版本见 [环境说明](docs/environment.md)。缓存特征训练不需要 RDKit；原始 SMILES 特征准备在独立 KPGT 环境中进行。数据、预训练权重及第三方仓库不随代码分发。

`DATA_ROOT` 下应包含 `{SNAP,DRH,Kinase}/biofusion1500_front/`，每个所选数据集需要药物/蛋白 CSV、KPGT NPZ、条件化图、对比相似矩阵和原始官方划分。具体格式见 [数据说明](docs/data.md)，从原始序列/SMILES 或已有缓存生成输入的命令见 [预处理说明](docs/preparation.md)。保留原始前 1500 残基视图及辅助特征前 1022 残基覆盖限制；不要混用后续清洗的 cold-drug 划分。

## 使用

```bash
export DATA_ROOT=/absolute/path/to/features
python -m sagf_dti validate-data \
  --config configs/paper/SNAP/prot/seed_42.yaml --data-root "$DATA_ROOT" --inspect-features
python -m sagf_dti train \
  --config configs/paper/SNAP/prot/seed_42.yaml --data-root "$DATA_ROOT" --output runs
python -m sagf_dti evaluate \
  --config configs/paper/SNAP/prot/seed_42.yaml --data-root "$DATA_ROOT" \
  --checkpoint /absolute/path/to/best.pt --output evaluations/snap_prot_42 --split test
python -m sagf_dti summarize --runs runs --output reports
```

配置目录为 `configs/paper/数据集/划分/seed_种子.yaml`：数据集为 SNAP、DRH、Kinase，划分为 random、drug、prot，种子为 42、1234、3407、5678、8888。`prot` 表示 cold protein。原始训练使用实际 batch32、最多 50 epoch、patience8、LR/WD=1e-4、AMP、辅助/可靠性及两项权重 0.02 的对比损失，并使用 Cauchy 梯度对齐学习率更新。梯度累积不能等价替代原始对比学习 batch32。

如通过 Python 直接实例化模型，请使用 `SAGFDTI(完整配置字典)`；无参数构造会使用旧源码的通用默认值，模型尺寸不等同于论文配置。上述命令会自动传入完整配置。

保留的 8 月 28 日排名修正先将两个类别的 logit 分别转为 float32，再计算 `logit1 - logit0`，使用该差值计算排名指标，并用 sigmoid 得到概率。模型前向仍使用 AMP，并非整网 FP32 推理。按验证 AUROC 选择模型后才进行测试集评估；新入口会检查完整输入并保护已完成运行。

本地 WSL 的锁页主机内存限制导致原始 pinning 在前向前失败；本机训练及评估命令需添加 `--disable-host-pinning`。该选项明确记录 `pin_memory: false` 运行时覆盖，原配置仍保留 true，batch32/64、worker 数、模型、损失及 AMP/优化器设置不变。

## 结果与来源

本次整理、九组历史结果及正在进行的新训练，用更直观的中文记录在 [本地验证说明](docs/local_report.zh-CN.md)。

论文 SNAP 官方 cold-protein 五种子结果为 AUROC **0.8885686 ± 0.0076321**，AUPR **0.8992116 ± 0.0146953**（均值 ± 样本标准差）。[独立历史预测审计](provenance/archived_prediction_audit.json) 使用 `pred_prob` 精确恢复全部 45 个运行的八项分类指标，最大差值为 0。旧 CSV 的 `pred_logit` 仅保存正类 logit，缺失负类 logit，不能据此计算修正后的二分类差值排名。原始主论文 checkpoint 未留存；历史预测审计不代表从头训练复现。

推荐科学依赖版本的 CPU/CUDA 检查已通过，CUDA/PyG 库沿用现有兼容安装。真实最长图压力测试通过实际 batch32 / 验证 batch64（每图 1500 节点），峰值保留显存 6.15625 GiB，并记录上述 pinning 覆盖。新五种子复现结果仍待完成，见 [验证状态](docs/verification.md)。硬件、依赖及排名实现版本变化可能影响得分。

源码、配置及预处理文件 SHA256 记录在 `provenance/`。外部 SeqDance、KPGT 与权重须自行按其来源和许可获取；见 [第三方声明](THIRD_PARTY_NOTICES.md)。本项目未擅自授予尚未确认的原始材料许可。
