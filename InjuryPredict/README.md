# InjuryPredict 使用说明

## 1. 简要说明

`InjuryPredict` 用于基于碰撞波形与标量工况，预测乘员的 `HIC`、`Dmax`、`Nij`，以及对应的 `AIS/MAIS` 等级。此 README 提供主要文件的使用步骤

## 2. 目录概览

```text
InjuryPredict/
├─ runs/                 # 训练输出目录
├─ utils/                # 模型、损失、工具函数
├─ config.py             # 模型/训练/评估配置
├─ Injurydata_prepare.py # 生成 processed .pt 数据集
├─ train.py              # 单次训练入口
├─ train_KFold.py        # K-Fold 训练入口
├─ eval_model.py         # 模型评估与图表导出
└─ test_all_data.py      # 基于完整 train/val/test 数据集的批量测试脚本
```

## 3. 路径约定

本子项目涉及的共享数据路径统一来自 [common/settings.py](../common/settings.py)，尤其是：

- `NORMALIZATION_CONFIG_PATH`
- `INJURY_SPLIT_DIR`
- `INJURY_PROCESSED_DIR`
- `get_injury_processed_dataset_path(...)`

## 4. 数据准备

`InjuryPredict` 严格依赖根目录先完成统一打包与 split 生成。

### 4.1 先生成共享打包数据与索引

在项目根目录执行：

```bash
python -m prepare_data
```

### 4.2 再生成 `InjuryPredict` 使用的 processed `.pt`

```bash
python -m InjuryPredict.Injurydata_prepare
```

默认会读取 `common.settings.INJURY_SPLIT_DIR`，并把 `.pt` 文件写入 `common.settings.INJURY_PROCESSED_DIR`，两者都对应 `combined` 视角。

如需为主驾或副驾单独生成 processed `.pt`，可显式指定：

```bash
python -m InjuryPredict.Injurydata_prepare --split-variant driver
python -m InjuryPredict.Injurydata_prepare --split-variant passenger
```

如需显式指定输出目录，也可以：

```bash
python -m InjuryPredict.Injurydata_prepare --out-dir <your_processed_dir>
```

默认输出文件名固定为：

- `train_dataset.pt`
- `val_dataset.pt`
- `test_dataset.pt`

默认输出目录由 `common.settings.INJURY_PROCESSED_DIR` 决定。

## 5. 训练

### 5.1 单次训练

```bash
python -m InjuryPredict.train
```

输出保存在 `InjuryPredict/runs/`，通常包括：

- 模型权重
- `TrainingRecord.json`
- tensorboard 日志

训练超参数统一配置于 [InjuryPredict/config.py](./config.py)。

### 5.2 K-Fold 训练

```bash
python -m InjuryPredict.train_KFold
```

`train_KFold.py` 的输出已包含各折的训练记录、模型权重和评估结果，一般不需要再额外运行 `eval_model.py`。

## 6. 评估与导出

```bash
python -m InjuryPredict.eval_model
```

说明：

- 该脚本当前默认仍是在 `__main__` 中指定 `run_dir` 与 `weight_file`
- 评估数据来自 `get_injury_processed_dataset_path("val")` 和 `get_injury_processed_dataset_path("test")`

## 7. TensorBoard

```bash
tensorboard --logdir=./InjuryPredict/runs
```

然后在浏览器访问 `http://localhost:6006`。

## 8. 推荐阅读顺序

建议按下面顺序理解本子项目：

1. 先看根目录 [README.md](../README.md)
2. 再看 [common/settings.py](../common/settings.py) 中的路径约定
3. 再运行 `python -m prepare_data`
4. 再运行 `python -m InjuryPredict.Injurydata_prepare`
5. 最后按需运行训练或评估脚本
