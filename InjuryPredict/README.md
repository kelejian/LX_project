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
- `DEFAULT_INJURY_VARIANT`
- `INJURY_SPLIT_ROOT_DIR`
- `INJURY_SPLIT_DIR`
- `INJURY_PROCESSED_ROOT_DIR`
- `INJURY_PROCESSED_DIR`
- `get_injury_processed_dataset_path(...)`

其中 `*_ROOT_DIR` 是容器目录，例如 `data/split_indices/injury/` 或 `data/processed/injury/`，其下再分 `combined / driver / passenger`。`INJURY_SPLIT_DIR` 和 `INJURY_PROCESSED_DIR` 是当前默认入口，默认由 `DEFAULT_INJURY_VARIANT` 决定。

做主副驾数据源消融时，通常先生成三套 processed `.pt`，再只切换 `DEFAULT_INJURY_VARIANT`：

```python
DEFAULT_INJURY_VARIANT = "combined"   # 主副驾合训
DEFAULT_INJURY_VARIANT = "driver"     # 仅主驾训练/评估
DEFAULT_INJURY_VARIANT = "passenger"  # 仅副驾训练/评估
```

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

默认会读取 `common.settings.INJURY_SPLIT_DIR`，并把 `.pt` 文件写入 `common.settings.INJURY_PROCESSED_DIR`。这两个目录都由 `common.settings.DEFAULT_INJURY_VARIANT` 决定，默认是 `combined`。

 processed `.pt` 会同时保存两种波形源：

- `x_acc_gt`：真值碰撞波形。
- `x_acc_pred`：冻结 `PulsePredict` 生成的预测波形。

训练、验证和评估默认以 `x_acc_pred` 作为部署域波形源。可在config.py中指定或者命令行参数中传入训练好的 PulsePredict 模型权重。

如需为主驾或副驾单独生成 processed `.pt`，可显式指定：

```bash
python -m InjuryPredict.Injurydata_prepare --split-variant driver --overwrite
python -m InjuryPredict.Injurydata_prepare --split-variant passenger --overwrite
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

当前默认训练流程为三阶段波形源课程学习：先使用真值波形 warm-up，再从真值波形平滑过渡到预测波形，最后只使用预测波形微调。主任务损失对 `HIC`、`Dmax`、`Nij` 使用 Kendall 同质不确定性加权，并保留配置中的人工先验权重。

### 5.2 K-Fold 训练

```bash
python -m InjuryPredict.train_KFold
```

`train_KFold.py` 的输出已包含各折的训练记录、模型权重和评估结果，一般不需要再额外运行 `eval_model.py`。

## 6. 评估与导出

```bash
python -m InjuryPredict.eval_model --run_dir <your_run_dir> --weight_file <your_weight_file>
```

说明：
- 评估数据来自 `get_injury_processed_dataset_path("val")` 和 `get_injury_processed_dataset_path("test")`
- 评估默认使用 `x_acc_pred` 传入损伤预测模型，即冻结 PulsePredict 产生的预测波形源。

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

## 9. TensorBoard 记录项说明

`train.py` 与 `train_KFold.py` 使用同一套 TensorBoard 命名结构。一级分组只区分 `Train` 与 `Val`，二级分组只区分 `Loss` 与 `InjuryMetrics`。

### 9.1 代码溯源

TensorBoard 标量的统一写入入口是 [utils/training.py](./utils/training.py) 中的 `log_injury_tensorboard_metrics(...)`。训练和验证的指标字典来自同一文件中的 `run_one_epoch(...)`，Kendall 主任务损失定义来自 [utils/loss.py](./utils/loss.py) 中的 `InjuryKendallMultiTaskLoss`，输出一致性和特征一致性分别来自 `OutputConsistencyLoss` 与 `FeatureConsistencyLoss`。

### 9.2 Loss 组

`Train/Loss/*` 与 `Val/Loss/*` 记录损失函数、课程学习系数和优化相关系数。

- `*/Loss/total_loss`：当前 epoch 实际用于统计的总损失。训练阶段按课程阶段变化：

```text
warmup:          L_train = L_main_gt
transition:      L_train = (1 - alpha) L_main_gt + alpha L_main_pred + lambda_out L_out_cons + lambda_feat L_feat_cons
target_finetune: L_train = L_main_pred
validation:      L_val   = L_main_pred
```

验证阶段固定使用预测波形源，且不包含 `out_cons_loss` 与 `feat_cons_loss`，因此 `Val/Loss/total_loss` 可跨训练阶段直接比较。

- `Train/Loss/main_gt_loss`：真值波形分支上的 Kendall 主任务损失，即上式中的 \(L_{\mathrm{main\_gt}}\)。代码来源是 `run_one_epoch(...)` 中对 `model(x_acc_gt, ...)` 的前向结果调用 `InjuryKendallMultiTaskLoss`。
- `*/Loss/main_pred_loss`：预测波形分支上的 Kendall 主任务损失，即 \(L_{\mathrm{main\_pred}}\)。Phase III、验证和部署评估均使用该波形源。
- `Train/Loss/gt_hic_loss`、`Train/Loss/gt_dmax_loss`、`Train/Loss/gt_nij_loss`：真值波形分支上 HIC、Dmax、Nij 三个主任务的原始加权子损失。
- `*/Loss/pred_hic_loss`、`*/Loss/pred_dmax_loss`、`*/Loss/pred_nij_loss`：预测波形分支上 HIC、Dmax、Nij 三个主任务的原始加权子损失。

三个主任务的 Kendall 主损失为：

```text
L_main = sum_i p_i * (0.5 * exp(-s_i) * L_i + 0.5 * s_i),  i in {HIC, Dmax, Nij}
```

其中 \(L_i\) 是对应任务的原始加权子损失，\(p_i\) 对应 `task_prior_weights`，\(s_i\) 是可学习的 `log_var`。`Train/Loss/Kendall/<task>/precision` 对应 \(\exp(-s_i)\)，`effective_loss_weight` 对应 \(0.5 p_i \exp(-s_i)\)。

- `Train/Loss/out_cons_loss`：真值波形分支输出与预测波形分支输出之间的一致性损失，记录的是尚未乘 `lambda_out` 的原始值。其形式为：

```text
L_out_cons = mean(abs((y_hat_gt - y_hat_pred) * output_weights))
```

其中 `output_weights` 由训练集 HIC、Dmax、Nij 标签标准差的倒数构造，用于降低不同量纲输出对一致性损失的影响。

- `Train/Loss/feat_cons_loss`：真值波形分支与预测波形分支的波形编码特征一致性损失，记录的是尚未乘 `lambda_feat` 的原始值。其形式为：

```text
L_feat_cons = mean(abs(h_gt_wave - h_pred_wave))
```

这里的 `h_*_wave` 是模型编码特征中与波形编码器对应的切片，由 `run_one_epoch(...)` 根据 `model.wave_feature_dim` 选取。

- `Train/Loss/alpha`：transition 阶段从真值波形分支切换到预测波形分支的主任务损失权重。transition 内部使用 smoothstep：

```text
alpha(u) = 3u^2 - 2u^3,  u in [0, 1]
```

warmup 阶段 `alpha=0`，Phase III 阶段 `alpha=1`。

- `Train/Loss/lambda_out` 与 `Train/Loss/lambda_feat`：当前 epoch 的一致性正则实际权重。二者只在 transition 阶段非零，并使用钟形调度：

```text
lambda_out  = lambda_out_max  * 4 * alpha * (1 - alpha)
lambda_feat = lambda_feat_max * 4 * alpha * (1 - alpha)
```

- `Train/Loss/learning_rate`：当前 epoch 实际使用的 optimizer 第一个参数组学习率。当前实现中学习率调度与课程阶段解耦，整个训练流程共用一条 `CosineAnnealingLR(T_max=Epochs)`，不会在 Phase III 重启。
- `Train/Loss/Kendall/<task>/*`：Kendall 同质不确定性加权的任务权重状态，`<task>` 为 `HIC`、`Dmax` 或 `Nij`。这些量用于观察多任务损失的相对权重变化，不是额外的验证指标。

### 9.3 InjuryMetrics 组

`Train/InjuryMetrics/*` 与 `Val/InjuryMetrics/*` 记录与实际损伤预测物理量直接相关的监控指标。代码来源是 `utils/training.py::_collect_common_outputs(...)`。

- `*/InjuryMetrics/accu_head`、`accu_chest`、`accu_neck`：先分别用 `AIS_cal_head`、`AIS_cal_chest`、`AIS_cal_neck` 将 HIC、Dmax、Nij 换算为 AIS 等级，再计算分类准确率。
- `*/InjuryMetrics/accu_mais`：六分类 MAIS 准确率，MAIS 由头、胸、颈三处 AIS 的最大值得到。
- `*/InjuryMetrics/accu_mais_3c`：三分类 MAIS 准确率，代码来源是 `convert_mais_to_3c(...)` 与 `accuracy_score(...)`。
- `*/InjuryMetrics/mae_hic`、`mae_dmax`、`mae_nij`：HIC、Dmax、Nij 的平均绝对误差。
- `*/InjuryMetrics/rmse_hic`、`rmse_dmax`、`rmse_nij`：HIC、Dmax、Nij 的均方根误差。
- `*/InjuryMetrics/r2_hic`、`r2_dmax`、`r2_nij`：HIC、Dmax、Nij 的决定系数。
