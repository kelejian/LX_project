# PulsePredict 使用说明

## 1. 项目简介

`PulsePredict` 用于根据碰撞工况参数预测三通道碰撞脉冲波形。

- 输入：碰撞速度、碰撞角度、重叠率等工况标量
- 输出：三通道波形序列

## 2. 项目结构

```text
PulsePredict/
├─ base/              # 抽象基类（模型、数据、训练器）
├─ data_loader/       # 数据加载实现
├─ logger/            # 日志与可视化工具
├─ model/             # 模型、损失、评估指标
├─ trainer/           # 训练器实现
├─ utils/             # 辅助脚本
├─ saved/             # 训练/测试输出目录
├─ config.json        # 默认配置
├─ train.py           # 训练入口
├─ test.py            # 测试入口
└─ interfere_data.py  # 推理与绘图脚本
```

## 3. 数据路径约定

当 `PulsePredict/config.json` 中的数据路径字段为 `null` 时，程序会回退到 [common/settings.py](../common/settings.py) 中的统一路径配置，例如：

- `RAW_DATA`
- `NORMALIZATION_CONFIG_PATH`
- `PULSE_SPLIT_DIR`

因此一般不需在 `PulsePredict/config.json` 中写死共享数据路径，保留 `config.json` 中的 `null` 设置即可。

## 4. 环境准备

确保已经在项目根目录下执行：

```bash
pip install -r requirements.txt
```

## 5. 数据准备

本项目依赖根目录已经生成好的打包数据与 split 索引。

先在项目根目录执行：

```bash
python -m prepare_data
```

`PulsePredict` 读取的核心输入包括：

- 打包后的 `raw_data_packed.npz`
- `normalization_config.json`
- `split_indices/pulse/pulse_train_indices.csv`
- `split_indices/pulse/pulse_val_indices.csv`
- `split_indices/pulse/pulse_test_indices.csv`

其中默认 split 目录由 `common.settings.PULSE_SPLIT_DIR` 决定。

## 6. 训练

从零开始训练：

```bash
python -m PulsePredict.train -c PulsePredict/config.json
```

修改常用超参数：

```bash
python -m PulsePredict.train -c PulsePredict/config.json --bs 64 --lr 0.001
```

恢复训练：

```bash
python -m PulsePredict.train -r PulsePredict/saved/models/<实验名>/<时间戳>/checkpoint-epochX.pth
```

训练结果与日志保存在 `PulsePredict/saved/` 下。

## 7. 测试

```bash
python -m PulsePredict.test -r PulsePredict/saved/models/<实验名>/<时间戳>/model_best.pth
```

`test.py` 会加载模型并在测试集上计算指标，将图像和日志写入对应的 `saved` 目录。若 `pulse_test_indices.csv` 为空，会直接报错。

## 8. `interfere_data.py`

使用方式：

```bash
python -m PulsePredict.interfere_data
```

该脚本会结合 checkpoint 和配置文件绘制精度分布图、散点图或导出 case 数据。
注意：

- 波形数据、归一化配置默认依然按 `common.settings`
- checkpoint 路径需在脚本中显式设置