# 使用总览

## 1. 项目说明

项目Pipeline：

1. 数据准备
   - 将原始样本打包为统一的 `npz`
   - 生成训练、验证、测试索引

2. 碰撞波形预测
   - `PulsePredict` 子项目

3. 乘员损伤预测
   - `InjuryPredict` 子项目

4. 约束系统参数寻优
   - `ARS_optim` 子项目

根目录还提供了一个独立推理工具：

- `run_pulse_injury_inference.py`
  - 对外部 CSV 批量执行碰撞波形预测与损伤预测

## 2. 目录导航

```text
LX_project/
├─ prepare_data.py                 # 数据打包与索引生成
├─ run_pulse_injury_inference.py   # 根目录独立批处理推理工具
├─ requirements.txt                # 依赖
├─ PulsePredict/                   # 碰撞波形预测子项目
├─ InjuryPredict/                  # 乘员损伤预测子项目
├─ ARS_optim/                      # 自适应约束系统参数寻优子项目
├─ common/                         # 公共接口、归一化、AIS 计算、路径常量
└─ <DATA_DIR>/                     # 数据、索引、归一化配置、推理输出
```

## 3. 路径约定

项目内共享数据路径统一以 [common/settings.py](./common/settings.py) 为准：

- `DATA_DIR`
- `RAW_DATA` / `RAW_DATA_DIR`
- `SPLIT_INDICES_DIR`
- `NORMALIZATION_CONFIG_PATH`
- `PROCESSED_DATA_DIR`
- `INJURY_PROCESSED_DIR`

这意味着如果你后续将数据目录从 `data/` 改成 `data_DS/`、`data_PS/` 或其他名称，主流程脚本应优先通过 `common.settings` 适配，而不是到各个子项目里逐处改硬编码路径。

## 4. 建议阅读顺序

如果你是第一次使用本项目，建议按下面顺序阅读和使用：

1. 先阅读本 README
2. 再进入具体子项目 README
   - [PulsePredict/README.md](./PulsePredict/README.md)
   - [InjuryPredict/README.md](./InjuryPredict/README.md)
   - [ARS_optim/README.md](./ARS_optim/README.md)

## 5. 环境准备

建议先激活当前项目所使用的 conda 环境，例如：

```bash
conda activate pytorch
```

然后在 `LX_project` 根目录安装依赖：

```bash
pip install -r requirements.txt
```

## 6. 推荐使用流程

### 步骤 1：数据准备

根目录的 `prepare_data.py` 负责生成统一数据打包与切分索引，是后续子项目的共同前提。

```bash
python -m prepare_data
```

运行完成后，共享数据目录下通常会得到：

- `raw_packed/raw_data_packed.npz`
- `split_indices/` 下的训练、验证、测试索引
- `normalization_config.json`

以上目录的实际根路径以 `common/settings.py` 当前配置为准。

### 步骤 2：训练或评估碰撞波形预测模型

详细说明请阅读：

- [PulsePredict/README.md](./PulsePredict/README.md)

常用入口：

```bash
python -m PulsePredict.train -c PulsePredict/config.json
python -m PulsePredict.test -r PulsePredict/saved/models/.../model_best.pth
```

### 步骤 3：训练或评估乘员损伤预测模型

详细说明请阅读：

- [InjuryPredict/README.md](./InjuryPredict/README.md)

常用入口：

```bash
python -m InjuryPredict.Injurydata_prepare
python -m InjuryPredict.train
python -m InjuryPredict.eval_model
```

### 步骤 4：训练或评估 ARS 参数寻优模块

详细说明请阅读：

- [ARS_optim/README.md](./ARS_optim/README.md)

常用入口：

```bash
python -m ARS_optim.run_train
python -m ARS_optim.run_eval
```

## 7. 根目录独立功能

### 7.1 `prepare_data.py`

用途：

- 整理和打包项目后续所需的统一数据文件
- 生成训练、验证、测试索引
- 为统一归一化配置提供基础输入

推荐场景：

- 初次搭建项目数据
- 原始数据更新后重新生成打包结果

运行方式：

```bash
python -m prepare_data
```

### 7.2 `run_pulse_injury_inference.py`

用途：

- 输入一个外部 CSV
- 先调用 `PulsePredict` 预测碰撞波形
- 再调用 `InjuryPredict` 预测 `HIC15`、`Dmax`、`Nij` 和 AIS 结果

这是根目录下的独立批处理推理功能，不属于 `ARS_optim`。

#### 输入要求

输入 CSV 应满足：

- 扩展名为 `csv`
- 第一列为 `case_id`
- 后续特征列按 `FEATURE_ORDER` 顺序给出
- 包含 `impact_velocity`、`impact_angle`、`overlap`、`LL1`、`LL2`、`BTF`、`LLATTF`、`AFT`、`SP`、`SH`、`RA`、`is_driver_side`、`OT`

#### 运行方式

```bash
python -m run_pulse_injury_inference
```

或显式指定输入与权重：

```bash
python -m run_pulse_injury_inference --input-csv path/to/input.csv --pulse-checkpoint PulsePredict/saved/models/.../model_best.pth --injury-checkpoint InjuryPredict/runs/.../best_val_loss.pth
```

#### 输出结果

默认输出到：

```text
<DATA_DIR>/inference_outputs/<输入文件名>_时间戳/
```

其中 `<DATA_DIR>` 由 `common/settings.py` 决定。

典型输出包括：

- `waveforms/predicted_waveforms.csv`
- `waveforms/plots/` 下的单 case 波形图
- `injuries/predicted_injuries.csv`
- `manifest.json`

## 8. 各子项目职责划分

### PulsePredict

职责：

- 根据碰撞工况标量预测碰撞加速度波形

使用说明：

- 请阅读 [PulsePredict/README.md](./PulsePredict/README.md)

### InjuryPredict

职责：

- 根据碰撞波形与标量工况预测乘员损伤指标

使用说明：

- 请阅读 [InjuryPredict/README.md](./InjuryPredict/README.md)

### ARS_optim

职责：

- 基于已训练的 `PulsePredict` 与 `InjuryPredict` 模型，训练策略网络并执行参数寻优

使用说明：

- 请阅读 [ARS_optim/README.md](./ARS_optim/README.md)

## 9. 运行约定

本项目当前默认约定：

1. 所有命令都在 `LX_project` 根目录下运行
2. 推荐统一使用 `python -m xxx` 方式启动入口脚本
3. 子项目之间通过 `common` 和共享数据目录协同
4. `normalization_config.json` 为公共归一化基准
5. 各子项目 README 只解释本子项目范围内的功能，不重复展开整个 pipeline

## 10. 快速上手建议

如果你只想快速体验已有模型推理：

1. 准备一个满足要求的输入 CSV
2. 运行 `python -m run_pulse_injury_inference`

如果你要复现实验训练流程：

1. 运行 `python -m prepare_data`
2. 训练或测试 `PulsePredict`
3. 生成 `InjuryPredict` 所需 processed 数据并训练 `InjuryPredict`
4. 训练策略网络
5. 运行 `ARS_optim` 评估

## 11. 文档入口

请根据目标任务进入对应 README：

- 项目总体说明：本文件
- 波形预测：[PulsePredict/README.md](./PulsePredict/README.md)
- 损伤预测：[InjuryPredict/README.md](./InjuryPredict/README.md)
- 参数寻优：[ARS_optim/README.md](./ARS_optim/README.md)
