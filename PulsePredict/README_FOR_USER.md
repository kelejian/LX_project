# LX-Model 项目使用指南

## 1. 项目简介

本项目是一个基于 PyTorch 实现的深度学习模型，旨在根据车辆碰撞刚性墙的初始工况参数，快速预测碰撞加速度时序波形。其核心任务是创建一个数据驱动的代理模型，以替代传统、计算成本高昂的有限元仿真软件。

* **输入**: 3个碰撞工况标量 (碰撞速度、碰撞角度、重叠率)。
* **输出**: 3条加速度时序波形 (X/Y向线加速度, Z向角加速度)，每条波形包含150个时间点。

## 2. 环境配置

请确保已安装 Python 和 PyTorch。然后，通过 pip 安装项目所需的依赖包：

```bash
# 建议在一个虚拟环境中执行
pip install -r requirements.txt
```

`requirements.txt` 文件中主要包含以下依赖：
* `torch`
* `torchvision`
* `numpy`
* `tqdm`
* `tensorboard`

此外，项目中还使用了一些 `sklearn` 和 `joblib` 等库，请根据需要进行安装。

## 3. 数据准备

模型的训练和测试需要使用特定格式的 `.npz` 数据文件。您可以通过运行 `utils/data_prepare.py` 脚本来生成这些文件。

1.  **准备原始数据**:
    * 一个包含原始波形 `.csv` 文件的目录 (例如, `x1.csv`, `y1.csv`, `z1.csv` ...)。
    * 一个包含所有工况参数的 `.npz` 文件，其中必须包含 `case_id`, `impact_velocity`, `impact_angle`, `overlap` 这几个字段。

2.  **修改 `data_prepare.py`**:
    * 打开 `utils/data_prepare.py` 文件。
    * 在文件底部的 `if __name__ == '__main__':` 部分，修改 `pulse_dir` (波形目录) 和 `params_path` (参数文件路径) 为您的实际路径。

3.  **运行脚本**:
    ```bash
    python utils/data_prepare.py
    ```
    该脚本会自动分割训练集和测试集，并生成 `packaged_data_train.npz` 和 `packaged_data_test.npz` 文件。

4.  **更新配置文件**:
    * 打开您的 `config.json` 文件。
    * 在 `data_loader_train` 和 `data_loader_test` 部分，将 `packaged_data_path` 的值更新为您刚刚生成的 `.npz` 文件的路径。

## 4. 模型训练

模型训练通过 `train.py` 脚本启动。您可以进行从零训练、恢复训练或微调。

### 4.1. 从零开始训练

这是最常见的场景。确保您的配置文件 (`config.json`) 已正确设置，然后运行：

```bash
python train.py -c config.json
```

* `-c`: 指定您要使用的配置文件。

训练过程中，项目会在 `saved/` 目录下创建一个新的实验文件夹，其结构如下：
* `saved/models/<实验名>/<时间戳>/`: 存放模型检查点 (`.pth`) 和该次运行的 `config.json` 副本。
* `saved/log/<实验名>/<时间戳>/`: 存放 `info.log` 日志文件和 TensorBoard 的 `events` 文件。

### 4.2. 恢复中断的训练

如果训练意外中断，您可以从最后一个保存的检查点恢复。

```bash
python train.py -r saved/models/<实验名>/<时间戳>/checkpoint-epoch10.pth
```

* `-r`: 指定要恢复的检查点文件路径。

程序会自动加载模型权重、优化器状态以及中断时的 epoch 数，并从下一轮 (`epoch 11`) 继续训练，日志和模型也会保存在**原有的目录**下的一个新建的 `resume_<时间戳>` 子文件夹中。

### 4.3. 微调 (Fine-tuning)

微调是指在一个已经训练好的模型基础上，使用一组新的超参数（如不同的学习率或数据集）继续训练。

```bash
python train.py -c new_config.json -r saved/models/<实验名>/<时间戳>/model_best.pth
```

* 同时提供 `-c` (新配置文件) 和 `-r` (预训练模型) 就会触发微调模式。
* 程序会加载 `-r` 指定的模型权重，但使用 `-c` 指定的新配置进行训练。
* 同时，程序会创建一个**全新的实验文件夹** (`<实验名>_finetuned/<时间戳>`) 来存放微调的结果，以和原始训练区分。

## 5. 模型测试与评估

使用 `test.py` 脚本来评估已训练好的模型在测试集上的性能。

### 5.1. 运行测试

```bash
python test.py -r saved/models/<实验名>/<时间戳>/model_best.pth
```

* `-r`: 指定您要评估的最佳模型检查点。

测试脚本会执行以下操作：
1.  加载模型并在测试集上进行预测。
2.  在原模型文件夹内创建一个名为 `test_<时间戳>` 的子文件夹。
3.  将测试过程的日志和第一个批次的样本对比图保存在这个新的子文件夹中。
4.  在控制台打印出模型在**整个测试集**上的总体性能指标。

### 5.2. 分组评估

`test.py` 支持按工况参数对指标进行分组统计，以分析模型在不同场景下的性能。

* **如何配置**:
    1.  打开 `test.py` 文件。
    2.  找到 `main` 函数开头的 `grouping_config` 字典。
    3.  根据需要修改 `param_name` (分组依据的参数名), `param_index` (参数索引) 和 `ranges` (区间定义)。

例如，按碰撞角度分组：

```python
grouping_config = {
    'param_name': 'angle',
    'param_index': 1,
    'ranges': {
        'negative_angle': [-60, -20],
        'central_impact': [-20, 20],
        'positive_angle': [20, 60]
    }
}
```

运行测试后，脚本会在打印完全量指标后，接着打印出每个定义好的区间的独立指标。

### 5.3. 绘制预测精度的散点图
`interfere_data.py` 脚本用于绘制（测试集上）各个case的预测精度的散点图，以便更直观地分析模型性能在不同工况下的表现。在该脚本中设置好相关参数（包括模型路径、数据路径等）后，可以直接运行该脚本：

```bash
python interfere_data.py
```

## 6. 配置文件详解 (`config.json`)

配置文件是控制整个项目的核心。以下是一些关键字段的说明：

* `name`: 实验名称，用于创建文件夹。
* `arch`: 定义模型架构。
    * `type`: 模型类名，如 `HybridPulseCNN`
    * `args`: 模型的初始化参数，如 `GauNll_use` (是否使用高斯NLLLoss)。
* `data_loader_train` / `data_loader_test`: 数据加载器的配置。
    * `packaged_data_path`: 数据文件路径。
    * `scaler_path`: 归一化 scaler 文件的保存/加载路径。
* `loss`: 损失函数的配置。
* `metrics`: 在日志中要追踪的评估指标列表，对应 `model/metric.py` 中的函数名。
* `trainer`: 训练器的配置。
    * `epochs`: 总训练轮数。
    * `monitor`: 模型性能监控指标，格式为 `'min/max <metric_name>'`。例如 `'max val_iso_rating_x'` 表示保存验证集上 `iso_rating_x` 指标最大时的模型。
    * `early_stop`: 如果监控指标连续 `n` 轮没有改善，则提前停止训练。

## 7. 项目结构

```
LX_model_PulsePredict/
├── base/             # 抽象基类 (模型, 数据加载器, 训练器)
├── data_loader/      # 数据加载相关模块
├── logger/           # 日志和可视化模块
├── model/            # 模型架构、损失函数、评估指标
├── saved/            # 存放所有日志、实验结果的默认目录; 包括超参配置(config.json)、最佳模型权重.pth文件(model_best.pth)！
├── trainer/          # 训练器实现
├── utils/            # 辅助工具函数
├── config.json       # 主配置文件
├── train.py          # 训练脚本
├── test.py           # 测试脚本
└── interfere_data.py # 使用训练好的模型绘制预测精度散点图
---

