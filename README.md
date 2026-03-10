# LX_project 使用总览

## 1. 项目说明

LX_project 是一个围绕汽车碰撞场景构建的数据驱动预测与参数寻优项目，当前主要包含四层能力：

1. 数据准备
   - 将原始样本打包为统一 npz
   - 生成训练、验证、测试索引

2. 碰撞波形预测
   - PulsePredict 子项目

3. 乘员损伤预测
   - InjuryPredict 子项目

4. 约束系统参数寻优
   - ARS_optim 子项目

此外，根目录还提供了一个独立推理工具：

- run_pulse_injury_inference.py
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
└─ data/                           # 数据、索引、归一化配置、推理输出
```


## 3. 建议阅读顺序

如果你是第一次使用本项目，建议按下面顺序阅读和使用：

1. 阅读本 README
   - 了解每个子项目的职责与使用顺序

2. 然后进入具体子项目 README
   - PulsePredict/README.md
   - InjuryPredict/README.md
   - ARS_optim/README.md


## 4. 环境准备

建议先激活当前项目所使用的 conda 环境，例如：

```bash
conda activate pytorch
```

然后在 LX_project 根目录安装依赖：

```bash
pip install -r requirements.txt
```


## 5. 项目推荐使用流程

### 步骤 1：数据准备

根目录的 prepare_data.py 负责生成统一数据打包与切分索引，是后续子项目的共同前提。

```bash
python prepare_data.py
```

运行完成后，通常会在 data 目录下得到：

- raw_packed/raw_data_packed.npz
- split_indices 下的训练、验证、测试索引
- normalization_config.json

### 步骤 2：训练或评估碰撞波形预测模型

详细说明请阅读：

- PulsePredict/README.md

常用入口：

```bash
python -m PulsePredict.train -c PulsePredict/config.json
python -m PulsePredict.test -r PulsePredict/saved/models/.../model_best.pth
```

### 步骤 3：训练或评估乘员损伤预测模型

详细说明请阅读：

- InjuryPredict/README.md

常用入口：

```bash
python -m InjuryPredict.Injurydata_prepare --out-dir data/processed/injury
python -m InjuryPredict.train
python -m InjuryPredict.eval_model
```

### 步骤 4：训练或评估 ARS 参数寻优模块

详细说明请阅读：

- ARS_optim/README.md

常用入口：

```bash
python -m ARS_optim.run_train
python -m ARS_optim.run_eval
```


## 6. 根目录独立功能

### 6.1 prepare_data.py

用途：

- 整理和打包项目后续所需的统一数据文件
- 生成训练、验证、测试索引
- 为统一归一化配置提供基础输入

推荐场景：

- 初次搭建项目数据
- 原始数据更新后重新生成打包结果

运行方式：

```bash
python prepare_data.py
```

### 6.2 run_pulse_injury_inference.py

用途：

- 输入一个外部 CSV
- 先调用 PulsePredict 预测碰撞波形
- 再调用 InjuryPredict 预测 HIC15、Dmax、Nij 和 AIS 结果

这是根目录下的独立批处理推理功能，不属于 ARS_optim。

#### 输入要求

输入 CSV 应满足：

- 扩展名为 csv
- 第一列为 case_id
- 后续特征列按 FEATURE_ORDER 顺序给出
- 包含 impact_velocity、impact_angle、overlap、LL1、LL2、BTF、LLATTF、AFT、SP、SH、RA、is_driver_side、OT

#### 运行方式

```bash
python run_pulse_injury_inference.py
```

或显式指定输入与权重：

```bash
python run_pulse_injury_inference.py --input-csv path/to/input.csv --pulse-checkpoint PulsePredict/saved/models/.../model_best.pth --injury-checkpoint InjuryPredict/runs/.../best_val_loss.pth
```

#### 输出结果

默认输出到：

```text
data/inference_outputs/<输入文件名>_时间戳/
```

典型输出包括：

- waveforms/predicted_waveforms.csv
- waveforms/plots 下的单 case 波形图
- injuries/predicted_injuries.csv
- manifest.json


## 7. 各子项目职责划分

### PulsePredict

职责：

- 根据碰撞工况标量预测碰撞加速度波形

使用说明：

- 请阅读 PulsePredict/README.md

### InjuryPredict

职责：

- 根据碰撞波形与标量工况预测乘员损伤指标

使用说明：

- 请阅读 InjuryPredict/README.md

### ARS_optim

职责：

- 基于已训练的 PulsePredict 与 InjuryPredict 模型，训练策略网络并执行参数寻优

使用说明：

- 请阅读 ARS_optim/README.md


## 8. 运行约定

本项目当前默认约定：

1. 所有命令都在 LX_project 根目录下运行

2. 子项目之间通过 common 和 data 中的统一接口协同

3. normalization_config.json 为公共归一化基准

4. 各子项目的 README 只负责说明本子项目范围内的功能，不重复解释整个 pipeline


## 9. 快速上手建议

如果你只想快速体验已有模型推理：

1. 准备一个满足要求的输入 CSV
2. 运行 run_pulse_injury_inference.py

如果你要复现实验训练流程：

1. 运行 prepare_data.py
2. 训练或测试 PulsePredict
3. 生成 InjuryPredict 所需 processed 数据并训练 InjuryPredict
4. 训练策略网络
5. 运行 ARS_optim 评估


## 10. 文档入口

请根据目标任务进入对应 README：

- 项目总体说明：本文件
- 波形预测：PulsePredict/README.md
- 损伤预测：InjuryPredict/README.md
- 参数寻优：ARS_optim/README.md
