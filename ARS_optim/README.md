# ARS_optim 项目说明

## 1. 项目定位

`ARS_optim` 是 `LX_project` 中用于约束条件下参数寻优的子项目。

它不重新训练碰撞仿真模型，而是基于已经训练好的：

- `PulsePredict`
- `InjuryPredict`

在给定工况下，为可调控制参数生成更优建议，并输出优化前后的损伤结果对比。

简单理解，`ARS_optim` 主要做四件事：

1. 读取给定工况
2. 通过策略网络直接给出初始解
3. 对单个 case 做局部迭代精调
4. 汇总优化前后参数与损伤指标变化

## 2. 子项目目录

当前 `ARS_optim` 目录可以按“入口脚本 / 配置 / 核心模块 / 输出结果”来理解：

```text
ARS_optim/
├─ configs/
│  ├─ default_config.yaml     # 训练与评估的主配置
│  └─ param_space.yaml        # 参数角色、默认值、边界和约束编码
├─ src/
│  ├─ constraints.py          # 约束校验、DAG 投影、软惩罚
│  ├─ data_sampler.py         # 经验池采样与带掩码扰动拒绝采样
│  ├─ distribution_penalty.py # 分布偏离惩罚
│  ├─ optimizer.py            # 局部精调优化器
│  ├─ param_manager.py        # 参数定义解析与 trainable/context 管理
│  ├─ strategy_net.py         # 策略网络结构、构建与权重加载
│  └─ surrogate.py            # PulsePredict + InjuryPredict 代理接口封装
├─ run_train.py               # 策略网络训练入口
├─ run_eval.py                # 两阶段寻优评估入口
├─ plot_eval_cases.py         # 评估结果绘图入口
├─ params_constraint.md       # 参数规则参考文档，不作为运行时读取文件
├─ ARS_codetask_prompt.md     # 项目开发说明材料
├─ saved_models/              # 训练产物目录
└─ saved_eval/                # 评估产物目录
```

如果你是第一次进入这个子项目，推荐阅读顺序是：

1. 先看 `configs/param_space.yaml`
2. 再看 `run_train.py` 和 `run_eval.py`
3. 最后按需展开 `src/` 中的实现模块

## 3. 整体工作流程

项目整体流程与 `ARS_Pipeline.md` 保持一致，可以概括为以下三个阶段。

### 3.1 策略网络直推

系统根据输入工况直接给出一组可调参数建议，对应结果中的 `Opt1`。

### 3.2 局部迭代精调

在直推结果基础上，再针对当前 case 做逐点优化，对应结果中的 `Opt2`。

### 3.3 结果评估与可视化

系统会把以下方案放在同一份结果中对比：

- `Base`：优化前基线方案
- `Opt1`：策略网络直推方案
- `Opt2`：局部精调后的方案
- `True`：如果评估输入自带真值，则额外包含仿真真值；未指定 `--input_csv` 的默认 split 评估一定会包含

## 4. 关键文件职责

### 4.1 配置层

- `configs/default_config.yaml`
  - 训练与评估的主配置入口
  - 包括 surrogate、sampling、training、evaluation、local_refine 等超参数

- `configs/param_space.yaml`
  - 把 `params_constraint.md` 中真正影响运行的规则编码为可执行配置
  - 同时描述每个参数的 `state/control`、`trainable`、`default`、`opt_min/opt_max`、约束规则等

### 4.2 训练入口

- `run_train.py`
  - 加载代理模型
  - 构建策略网络
  - 从 injury train 经验池流式采样 context
  - 进行自监督训练
  - 保存 `train_best / val_best / final` 权重与训练记录

### 4.3 评估入口

- `run_eval.py`
  - 读取 `input_csv` 或默认内部 split（test 优先，不可用时回退 val）
  - 计算 baseline 结果
  - 依次执行 `Opt1` 和 `Opt2`
  - 输出完整结果表和汇总记录

- `plot_eval_cases.py`
  - 对 `evaluation_results.csv` 做 case 级柱状图可视化
  - 支持指定 `case_id`
  - 也支持自动选取 `JointRisk` 下降最多的 TopN case

### 4.4 核心实现层

- `src/param_manager.py`
  - 对 `param_space.yaml` 做解析和一致性检查
  - 统一提供 context/trainable 参数索引、名称和默认值

- `src/constraints.py`
  - 负责输入端合法性判断
  - 负责输出端的前向拓扑投影和软惩罚

- `src/data_sampler.py`
  - 负责经验池采样
  - 负责按规则冻结部分参数并进行扰动
  - 负责拒绝采样统计

- `src/surrogate.py`
  - 负责衔接 `PulsePredict` 与 `InjuryPredict`
  - 向训练和评估阶段提供统一损伤目标接口

- `src/strategy_net.py`
  - 定义策略网络结构
  - 负责默认值初始化逻辑
  - 负责从 `saved_models/.../configs_used/` 恢复结构配置

- `src/optimizer.py`
  - 实现局部精调阶段的逐点优化

- `src/distribution_penalty.py`
  - 实现对经验池分布偏离的惩罚项

## 5. 前置条件

本项目依赖以下前置产物：

- 已完成统一打包与训练/验证/测试划分
- 已具备可用的 `PulsePredict` 模型权重
- 已具备可用的 `InjuryPredict` 模型权重
- 在 `LX_project` 根目录下通过 `python -m xxx` 方式运行

推荐先阅读根目录 [README.md](../README.md)。

## 6. 路径约定

`ARS_optim` 读取共享数据时，统一依赖 [common/settings.py](../common/settings.py)，尤其是：

- `RAW_DATA`
- `NORMALIZATION_CONFIG_PATH`
- `SPLIT_INDICES_DIR`
- `get_split_indices_path(...)`

因此这里不再把测试集或 split 路径写死成某个 `data/...`。如果后续切换共享数据目录名，应优先修改 `common/settings.py`。

## 7. 训练输入与评估输入

### 7.1 训练策略网络

`run_train.py` 的训练数据流来自损伤预测任务的训练集经验池，并结合扰动采样构造。固定验证评估来自损伤预测任务的 `val` split。

如果 `injury_val` 为空：

- 训练仍可继续
- 固定验证会被关闭
- `val_best_model.pth` 不会生成

### 7.2 评估

`run_eval.py` 支持两类输入：

#### 自定义 CSV

用户提供一份包含 context 参数的本地 CSV。系统会读取并校验输入，在合法前提下完成寻优与评估。

#### 默认内部 split

若未指定 `--input_csv`，系统会优先使用损伤预测任务对应的 test split 进行评估；如果 test split 文件不存在或该 split 为空，则自动回退到 val split，评估逻辑保持完全一致。

## 8. 项目输出

### 8.1 训练输出

策略网络训练完成后，会在 `ARS_optim/saved_models/` 下生成对应运行目录，主要包含：

- `configs_used/`
- `checkpoints/`
- `records/`
- `tensorboard/`

常见权重包括：

- `train_best_model.pth`
- `val_best_model.pth`
- `final_model.pth`

其中 `val_best_model.pth` 仅在验证集非空时生成。

### 8.2 评估输出

评估完成后，会在 `ARS_optim/saved_eval/` 下生成独立结果目录，主要包含：

- `configs_used/`
- `results/`

其中常见结果包括：

- `results/evaluation_results.csv`
- `results/evaluation_record.yaml`

结果目录名会按输入来源区分，例如：

- `eval_injury_test_split_*`
- `eval_injury_val_split_*`
- `eval_<input_csv文件名>_*`

### 8.3 绘图输出

`plot_eval_cases.py` 会在评估结果目录旁边继续生成绘图子目录，用于保存：

- 参数优化前后对比图
- 损伤指标对比图
- AIS/MAIS 对比图
- 风险概率与 `JointRisk` 对比图

## 9. 结果如何理解

### 9.1 Base / Opt1 / Opt2 / True

- `Base`：基线方案，即未优化前的输入方案
- `Opt1`：策略网络直接输出的第一阶段优化结果
- `Opt2`：在 `Opt1` 基础上进一步精调后的第二阶段优化结果
- `True`：当评估输入侧带有真值时存在；默认 split 评估下表示对应 split 的仿真真值

### 9.2 常见指标

结果表和图中通常会包含以下信息：

- `HIC / Dmax / Nij`
- `AIS_head / AIS_chest / AIS_neck`
- `MAIS`
- `Phead / Pchest / Pneck`
- `JointRisk`

### 9.3 Reduction 的含义

`Reduction` 表示与基线方案相比的降低幅度，例如：

- `Opt1` 相比 `Base` 的下降值
- `Opt2` 相比 `Base` 的下降值

## 10. 典型使用方式

### 10.1 训练策略网络

```bash
python -m ARS_optim.run_train
```

### 10.2 进行评估

```bash
python -m ARS_optim.run_eval
```

或指定自定义输入文件：

```bash
python -m ARS_optim.run_eval --input_csv your_cases.csv
```

### 10.3 绘制结果图

```bash
python -m ARS_optim.plot_eval_cases --eval_csv path_to_evaluation_results.csv --case_ids 1 2 3
```

或直接绘制优化效果最好的若干 case：

```bash
python -m ARS_optim.plot_eval_cases --eval_csv path_to_evaluation_results.csv --topn_joint_risk 10
```

## 11. 注意事项

- 本项目默认在 `LX_project` 根目录下通过 `python -m xxx` 运行
- 本项目依赖已有的波形预测模型和损伤预测模型权重，不负责重新训练这两个模型
- 评估阶段若使用策略权重，会严格比对当前 `param_space.yaml` 和 `normalization_config.json` 与权重目录中的快照，避免结构性错配
