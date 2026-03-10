# ARS_optim 使用说明

## 1. 简要说明

ARS_optim 是 LX_project 中面向自适应乘员约束系统参数寻优的子项目。

它不直接重新训练碰撞波形预测模型和损伤预测模型，而是在二者已经训练完成并具备可用权重的前提下，完成以下两类任务：

1. 训练策略网络
   - 输入为 context 参数，即 state 参数与当前不可调 control 参数
   - 输出为当前可调 control 参数
   - 训练方式为自监督的无限数据流训练，不使用 epoch 概念

2. 执行完整寻优评估
   - baseline 评估
   - 策略网络直推 Opt1
   - 局部梯度精调 Opt2

整个子项目严格依赖根目录下统一的 normalization_config.json，并调用以下现有模块：

- PulsePredict：生成归一化碰撞波形
- InjuryPredict：根据波形和标量参数预测 HIC、Dmax、Nij
- common：提供统一归一化、AIS 计算、路径常量等公共接口


## 2. 目录概览

```text
ARS_optim/
├─ configs/
│  ├─ default_config.yaml   # 训练与评估主配置
│  └─ param_space.yaml      # 参数角色、默认值、范围、耦合规则
├─ src/
│  ├─ constraints.py        # 统一硬约束与可微投影
│  ├─ data_sampler.py       # 基于 injury_train 经验池的训练数据流
│  ├─ distribution_penalty.py
│  ├─ optimizer.py          # 局部精调优化器
│  ├─ param_manager.py      # 参数空间定义解析与索引管理
│  ├─ strategy_net.py       # 策略网络
│  └─ surrogate.py          # PulsePredict + InjuryPredict 代理接口
├─ saved_models/            # 策略网络训练结果
├─ saved_eval/              # 寻优评估结果
├─ run_train.py             # 策略网络训练入口
└─ run_eval.py              # 寻优评估入口
```


## 3. 运行前提

在使用 ARS_optim 之前，需满足以下前提：

1. 已在项目根目录完成数据打包与索引划分
   - 即 prepare_data.py 已成功运行

2. 已训练好 PulsePredict 模型
   - 当前默认权重路径由 ARS_optim/configs/default_config.yaml 中 surrogate.pulse_checkpoint 指定

3. 已训练好 InjuryPredict 模型
   - 当前默认权重路径由 ARS_optim/configs/default_config.yaml 中 surrogate.checkpoint_rel_path 指定

4. 根目录下存在统一归一化文件
   - data/normalization_config.json

5. 运行命令必须在 LX_project 根目录下执行
   - 本子项目当前只考虑这种运行方式


## 4. 参数与数据流约定

### 4.1 参数角色

本项目中的参数分为以下两类：

- state 参数
  - 必然不可调
- control 参数
  - 当前可调或当前不可调都属于 control
  - trainable=false 的 control 在当前配置下会并入 context
  - trainable=true 的 control 由策略网络输出或局部精调优化

因此：

- context = state + 当前 trainable=false 的 control
- trainable_control = 当前 trainable=true 的 control

这种划分由 ARS_optim/configs/param_space.yaml 统一定义。后续若修改 trainable 属性，代码会按配置自动切换角色，不需要额外改训练主链路。

### 4.2 训练数据流

策略网络训练不是基于固定训练集 epoch 循环，而是基于 injury_train 经验池构造无限数据流：

1. 从损伤预测任务的训练集索引中取样
2. 对连续 context 特征加入轻微扰动
3. 扰动后统一通过约束引擎回收至合法可行域

验证阶段则使用 injury_val 全量样本，不加扰动，并按 val_interval 定期评估完整验证集。

### 4.3 约束规则来源

参数范围、耦合关系、座椅几何约束、离散 RA 档位等规则统一集中在：

- ARS_optim/configs/param_space.yaml
- ARS_optim/src/constraints.py

不要在其他脚本中重复硬编码同类逻辑。


## 5. 配置文件

### 5.1 default_config.yaml

该文件统一管理以下内容：

- 设备与随机种子
- 外部代理模型权重路径
- 策略网络结构超参数
- 策略网络训练超参数
- 局部精调超参数
- 分布偏离惩罚超参数
- 评估批大小与可选策略权重路径

常用配置项包括：

- strategy_net.train.batch_size
- strategy_net.train.max_iterations
- strategy_net.train.val_interval
- optimization.direct_inference
- optimization.refine_steps
- optimization.lr
- optimization.distribution_penalty.weight
- evaluation.strategy_checkpoint

### 5.2 param_space.yaml

该文件定义：

- 参数顺序
- 参数角色与 trainable 属性
- 默认值
- 连续参数范围
- 离散参数允许值
- AFT/BTF、LL2/LL1、LLATTF/BTF 等耦合规则
- overlap 与 angle 的规则
- 座椅 SP/SH 多边形约束
- RA 离散档位

如果后续要切换某个 control 参数是否可调，应优先修改此文件，而不是修改训练或评估脚本。


## 6. 训练策略网络

### 6.1 基本命令

请在 LX_project 根目录执行：

```bash
python -m ARS_optim.run_train
```

### 6.2 常用参数覆盖

```bash
python -m ARS_optim.run_train --config ARS_optim/configs/default_config.yaml

python -m ARS_optim.run_train --batch_size 512 --lr 0.0005 --max_iterations 30000

python -m ARS_optim.run_train --device cpu
```

### 6.3 训练过程说明

训练入口会自动完成以下工作：

1. 读取参数空间定义与主配置
2. 加载 PulsePredict 与 InjuryPredict 权重
3. 构建策略网络
4. 从 injury_train 构造训练数据流
5. 如启用分布惩罚，则先拟合训练参考分布
6. 在正式训练前执行一次梯度流自检
7. 按最大 iteration 数进行训练
8. 周期性评估 injury_val 全量样本
9. 保存 train_best、val_best、final 三套权重

### 6.4 训练输出目录

训练结果保存在：

```text
ARS_optim/saved_models/strategy_net_MMDD_HHMMSS/
```

目录中通常包含：

- train_best_model.pth
- val_best_model.pth
- final_model.pth
- training_history.csv
- training_summary.yaml
- config_used.yaml
- param_space.yaml
- normalization_config.json
- TensorBoard 事件文件


## 7. 执行寻优评估

### 7.1 基本命令

默认使用 injury test split 进行评估：

```bash
python -m ARS_optim.run_eval
```

指定输入 CSV：

```bash
python -m ARS_optim.run_eval --input_csv path/to/input.csv
```

启用策略网络直推并显式指定权重：

```bash
python -m ARS_optim.run_eval --input_csv path/to/input.csv --direct_inference --strategy_ckpt ARS_optim/saved_models/strategy_net_xxxx/val_best_model.pth
```

自定义输出 CSV 文件名：

```bash
python -m ARS_optim.run_eval --input_csv path/to/input.csv --output_csv my_eval.csv
```

### 7.2 评估输入规则

若使用 input_csv：

- 应包含 context 参数列
- 若缺失某个 context 参数，会自动回填 param_space.yaml 中的 default，并发出提醒
- 若额外提供了 trainable control 列，则这些值作为 baseline 输入
- 若未提供 trainable control 列，则 baseline 使用 default 值
- 对于用户已显式提供的列，如果不合法或违反硬约束，会直接报错

若未提供 input_csv：

- 自动使用损伤预测任务的测试集工况点
- baseline 直接使用测试集已有参数
- 同时读取真值标签并写入输出结果，便于和 baseline、优化结果做对比

### 7.3 评估阶段定义

- Base
  - baseline 结果
- Opt1
  - 策略网络直推结果
  - 仅在 direct_inference=true 且提供兼容权重时存在
- Opt2
  - 局部精调结果
  - 仅在 refine_steps 大于 0 时存在

局部精调是按 case 独立并行的逐点优化，不是分布级优化。

### 7.4 评估输出目录

每次评估都会在以下目录下创建新的时间戳子目录：

```text
ARS_optim/saved_eval/eval_xxx_MMDD_HHMMSS/
```

目录中通常包含：

- evaluation_results.csv 或用户指定名称的输出 CSV
- eval_info.yaml
- config_used.yaml
- param_space.yaml
- normalization_config.json

### 7.5 输出 CSV 内容

输出结果包含：

- metadata 列
  - 如 case_id
- 完整 context 参数
- Base_ 前缀的 baseline trainable control
- Opt_ 前缀的最终优化后 trainable control
- Base 与 Opt 两组损伤预测结果
  - HIC
  - Dmax
  - Nij
  - 三部位风险
  - 联合损伤风险
  - AIS_head、AIS_chest、AIS_neck、AIS_max
- Reduction_ 前缀的绝对降低量
- 若为测试集模式，还会包含 True_ 前缀的真值列

### 7.6 eval_info.yaml 内容

eval_info.yaml 中会记录：

- 输入来源路径
- 使用的策略权重路径
- 当前配置快照
- 参数角色划分
- 输入校验策略
- 各阶段状态
- 宏观降损指标汇总
- 总耗时与平均耗时


## 8. 推荐使用顺序

建议按照以下顺序使用 ARS_optim：

1. 先确认 PulsePredict 和 InjuryPredict 权重路径正确
2. 检查 ARS_optim/configs/default_config.yaml
3. 训练策略网络
4. 用测试集先做一次评估
5. 再用自定义 input_csv 做实际工况评估


## 9. 常见注意事项

1. 命令需在 LX_project 根目录下执行

2. normalization_config.json 必须与当前数据和模型一致

3. 若启用 direct_inference，但未提供兼容当前参数空间的策略权重，会直接报错

4. 若用户输入 CSV 中显式提供了非法参数，评估不会静默修正，而会报错退出

5. param_space.yaml 中的 default 值被视为用户已确认的合法值，不会额外做兼容性兜底


## 10. 与其他子项目的关系

ARS_optim 不负责以下工作：

- 原始数据打包与索引生成
- PulsePredict 模型训练
- InjuryPredict 模型训练

这些工作请分别参考：

- 根目录 README
- PulsePredict/README.md
- InjuryPredict/README.md
