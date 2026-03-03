# InjuryPredict — 乘员损伤预测（使用说明）

## 1. 简要说明
基于碰撞波形与标量工况，预测乘员的 HIC（头部）、Dmax（胸部）与 Nij（颈部）指标，以及对应的 AIS/MAIS 等级。此 README 提供主要文件的使用步骤。

## 2. 目录概览
```
InjuryPredict/
├─ utils/                # 模型/损失/优化器实现
├─ Injurydata_prepare.py # 生成 data/processed/injury/*.pt（依赖根目录 prepare_data.py 的运行产物）
├─ train.py              # 单次训练入口
├─ train_KFold.py        # K-Fold 训练入口
├─ eval_model.py         # 评估与可视化（图表 + 报告）
└─ test_all_data.py      # 在完整数据集上导出预测 CSV
```

## 3. 环境准备
在项目根目录运行：

```bash
pip install -r requirements.txt
```

## 4. 数据准备（必须先做）
注意：**严格依赖**先运行根目录的 `prepare_data.py` 来生成原始打包与索引文件。

1) 生成原始打包与索引：
```bash
python prepare_data.py
```
2) 生成 processed `.pt`（且依赖上一步产生的文件）：
```bash
python -m InjuryPredict.Injurydata_prepare --out-dir data/processed/injury
```
产出：`data/processed/injury/{train_dataset.pt,val_dataset.pt,test_dataset.pt}` 与 `data/normalization_config.json`。

> 若缺少任何前置产物，脚本会明确报错并退出。

## 5. 训练
- 单次训练：
```bash
  python -m InjuryPredict.train
```
  输出保存在 `runs/`（包含 checkpoints、TrainingRecord.json、tensorboard logs）。

- K-Fold 训练：
```bash
  python -m InjuryPredict.train_KFold
```
  注: K-Fold 训练的输出文件已经包含了每折的训练记录、模型权重和详细的评估结果，不需要单独再运行评估脚本。

模型/训练超参统一配置于 `InjuryPredict/config.py`，无需改源码即可调整。

## 6. 评估与导出
- 生成评估报告与图表：
```bash
  python -m InjuryPredict.eval_model
```
  （建议仅用于评估 InjuryPredict.train 训练的模型. 在脚本 `__main__` 中设置需要评估的 `run_dir` 与 `weight_file`）

- 在完整数据集上导出预测 CSV：
```bash
  python -m InjuryPredict.test_all_data
```

## 7. 可视化（TensorBoard）
```bash
tensorboard --logdir=./runs
```
然后在浏览器访问 `http://localhost:6006`。
