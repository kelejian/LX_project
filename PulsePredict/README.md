# PulsePredict 使用说明

## 1. 简要说明
PulsePredict 是基于 PyTorch 的碰撞加速度时序波形预测子项目。输入为三维工况标量（速度、角度、重叠率），输出为三通道加速度波形。该 README 仅包含运行所需的必要说明。


## 2. 碰撞波形预测子项目结构（概要）

```
PulsePredict/
├─ base/              # 抽象基类（模型、数据、训练器）
├─ data_loader/       # 数据加载实现
├─ logger/            # 日志与可视化工具
├─ model/             # 模型、损失、评估指标
├─ trainer/           # 训练器实现
├─ utils/             # 辅助脚本（含绘图、配置解析）
├─ saved/             # 训练/测试输出目录
├─ config.json        # 默认配置
├─ train.py           # 训练入口（包方式运行）
├─ test.py            # 测试入口（包方式运行）
└─ interfere_data.py  # 绘图与导出工具
```

## 3. 环境与依赖
确保已经在项目根目录下执行：

```bash
pip install -r requirements.txt
```

## 4. 数据准备
本项目从已打包的 `.npz` 数据和 `normalization_config.json` 读取数据；默认路径在 `data/` 下，配置见 `config.json` 中 `data_loader_train` / `data_loader_test`。如果需要重新打包数据，请使用仓库根的 `prepare_data.py`。

## 5. 训练
使用配置文件启动训练（默认使用 `PulsePredict/config.json`）：

```bash
# 从零训练
python -m PulsePredict.train -c PulsePredict/config.json

# 调整常用参数示例：
python -m PulsePredict.train -c PulsePredict/config.json --bs 64 --lr 0.001
```

恢复训练：

```bash
python -m PulsePredict.train -r PulsePredict/saved/models/<实验名>/<时间戳>/checkpoint-epochX.pth
```

训练结果与日志保存在 `PulsePredict/saved/` 下（由 `config.json` 中 `trainer.save_dir` 指定）。

## 6. 测试
评估已训练模型：

```bash
python -m PulsePredict.test -r PulsePredict/saved/models/<实验名>/<时间戳>/model_best.pth
```

`test.py` 会加载模型并在测试集上计算指标，按配置会将图像和日志写入对应 `saved` 目录。

## 7. 绘制精度分布/散点图（interfere_data）
使用 `interfere_data.py` 在测试结果上绘制 ISO/精度散点图或导出 case 数据：

```bash
python -m PulsePredict.interfere_data
```

在脚本内按需修改 `CHECKPOINT_PATH` 或从 checkpoint 推断的 `config.json` 路径以定位数据与模型。图像和导出文件会写到由脚本或 checkpoint 指定的输出目录。


