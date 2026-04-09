# ARS_optim 说明

## 1. 子项目定位

`ARS_optim` 负责在物理约束和优化边界约束下，对控制参数执行两阶段寻优：

1. `Opt1`
   由策略网络直接给出摊销优化解。
2. `Opt2`
   在单个 `case` 上，以 `Opt1` 或 `default` 为起点做局部梯度精调。

`ARS_optim` 不重新训练 `PulsePredict` 或 `InjuryPredict`，而是复用这两个子项目的已训练权重作为代理链路。

## 2. 目录结构

```text
ARS_optim/
├─ configs/
│  ├─ default_config.yaml
│  └─ param_space.yaml
├─ src/
│  ├─ constraints.py
│  ├─ data_sampler.py
│  ├─ distribution_penalty.py
│  ├─ optimizer.py
│  ├─ param_manager.py
│  ├─ strategy_net.py
│  └─ surrogate.py
├─ run_train.py
├─ run_eval.py
├─ plot_eval_cases.py
├─ saved_models/
└─ saved_eval/
```

## 3. 配置文件职责

### 3.1 `configs/default_config.yaml`

负责配置：

- 代理模型权重路径
- 策略网络结构
- 自监督训练超参数
- 局部精调超参数
- 分布偏离惩罚与评估设置

### 3.2 `configs/param_space.yaml`

负责配置：

- 参数角色划分：`state / control`
- `trainable` 控制参数与固定控制参数
- `base_min/base_max`
- `opt_min/opt_max`
- 耦合约束、额外输出约束
- 座椅可行域与 RA 条件区间

运行时只会读取 `param_space.yaml`，不会去解析说明文档。

## 4. 共享数据依赖

`ARS_optim` 通过 [`common/settings.py`](/e:/WPS Office/1628575652/WPS企业云盘/清华大学/我的企业文档/课题组相关/理想项目/LX_project/common/settings.py) 读取共享数据目录。当前默认依赖：

- `data/raw_packed/raw_data_packed.npz`
- `data/normalization_config.json`
- `data/split_indices/injury/combined/`

说明：

- `injury` 任务保留 `combined / driver / passenger` 三套 split
- `pulse` 任务只保留一套完整 split
- `ARS_optim` 默认按 `combined` 视角训练与评估
- 输入归一化始终使用根目录共享的 `normalization_config.json`

## 5. 训练流程

训练入口：

```bash
conda activate pytorch
python -m ARS_optim.run_train
```

训练阶段主要做四件事：

1. 从 `injury_train` 经验池采样 `context`
2. 在输入端按规则做轻微扰动与拒绝采样
3. 通过 `PulsePredict + InjuryPredict` 组成代理链路，计算损伤目标与约束惩罚
4. 保存训练产物与配置快照

训练输出目录位于：

```text
ARS_optim/saved_models/<run_dir>/
├─ checkpoints/
├─ configs_used/
├─ records/
└─ tensorboard/
```

其中会保存：

- `train_best_model.pth`
- `val_best_model.pth`
- `final_model.pth`

## 6. 评估流程

评估入口：

```bash
python -m ARS_optim.run_eval
```

也可以读取用户自定义 CSV：

```bash
python -m ARS_optim.run_eval --input_csv your_cases.csv
```

评估逻辑说明：

- 如果提供 `--input_csv`，则按输入文件中的 `context` 与可选 baseline control 做评估
- 如果不提供 `--input_csv`，则优先使用 `injury test split`
- 若 test split 不可用，则自动回退到 `injury val split`
- `input_csv` 模式下，baseline trainable control 只有整组合法才采用；否则整组回退为 `param_space.yaml` 中的 `default`
- 若 baseline 回退为 `default` 后，和当前 `context` 联合起来仍不合法，则该 `case` 会直接跳过，并在结果记录中写明原因
- 若启用 `direct_inference`，先生成 `Opt1`
- 若 `refine_steps > 0`，再继续生成 `Opt2`

评估输出目录位于：

```text
ARS_optim/saved_eval/<run_dir>/
├─ configs_used/
└─ results/
   ├─ evaluation_results.csv
   ├─ evaluation_record.yaml
   └─ optimization_traces/   # 仅在指定 trace_case_ids 时生成
```

## 7. 结果表字段

`evaluation_results.csv` 会保留以下主要列：

- `Base_*`
- `Opt1_*`
- `Opt2_*`
- `Reduction_*`
- `True_*`
- `True_vs_*`
- `Opt2_ConvergenceStep`

这些列分别用于表示：

- `Base_*`
  baseline 输入下的预测结果
- `Opt1_*`
  策略网络直推结果
- `Opt2_*`
  局部精调结果
- `Reduction_*`
  相对 `Base` 的下降幅度
- `True_*`
  测试集真值或输入侧提供的对照值
- `True_vs_*`
  真值与各阶段预测结果的差值
- `Opt2_ConvergenceStep`
  局部精调的收敛步数

## 8. 画图脚本

画图入口：

```bash
python -m ARS_optim.plot_eval_cases --eval_csv ARS_optim/saved_eval/.../results/evaluation_results.csv --case_ids 1 2 3
```

也可以直接选择优化效果最好的若干 `case`：

```bash
python -m ARS_optim.plot_eval_cases --eval_csv ARS_optim/saved_eval/.../results/evaluation_results.csv --topn_joint_risk 10
```

输出目录：

- `plots_case_ids/`
- `plots_top{N}_joint_risk/`

同一批 `case` 的图片平铺在同一目录中，通过文件名前缀区分 `case_id`，不会为每个 case 再单独建子目录。

## 9. 运行约定

- 统一在 `LX_project` 根目录下运行
- 统一使用 `python -m xxx`
- 评估阶段若加载策略权重，会严格校验当前 `param_space.yaml` 和 `normalization_config.json` 是否与权重目录中的快照一致
- 不保留旧数据划分与旧路径布局的兼容逻辑
