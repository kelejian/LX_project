# PulsePredict 使用说明

## 1. 简要说明

`PulsePredict` 是基于 PyTorch 的碰撞加速度时序波形预测子项目。输入为三维工况标量（速度、角度、重叠率），输出为三通道加速度波形。该 README 仅包含运行所需的必要说明。


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
└─ interfere_data.py  # 绘图与导出工具
```

## 3. 路径约定

当前主流程不在 `config.json` 里硬编码共享数据路径。

当`PulsePredict/config.json` 中与数据相关的字段为 `null`，运行时会自动回落到 [common/settings.py](../common/settings.py) 中的统一路径约定，包括：

- `RAW_DATA`
- `NORMALIZATION_CONFIG_PATH`
- `PULSE_SPLIT_DIR`

建议保留 `config.json` 中的 `null` 设置。

## 4. 环境与依赖

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

从零训练：

```bash
python -m PulsePredict.train -c PulsePredict/config.json
```

调整常用参数示例：

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

需要注意：

- 波形数据、归一化配置默认仍走 `common.settings`
- checkpoint 默认值仍是脚本内的便捷实验路径，如需复用到其他运行目录，请显式修改或传入相应路径

## 9. 推荐阅读顺序

1. 先看根目录 [README.md](../README.md)
2. 再看 [common/settings.py](../common/settings.py) 中的共享路径约定
3. 运行 `python -m prepare_data`
4. 再运行 `PulsePredict.train` 或 `PulsePredict.test`
