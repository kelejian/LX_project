# LX_project 说明

## 1. 项目结构

`LX_project` 由四部分组成：

1. `common/`
   共享特征顺序、数据路径、归一化接口、划分读写工具和通用指标。
2. `PulsePredict/`
   碰撞波形预测模型与训练、测试代码。
3. `InjuryPredict/`
   损伤预测模型、`.pt` 数据准备脚本和训练评估代码。
4. `ARS_optim/`
   在约束条件下执行“策略直推 + 局部精调”的两阶段参数寻优。

根目录还提供统一数据准备脚本 [`prepare_data.py`](/e:/WPS Office/1628575652/WPS企业云盘/清华大学/我的企业文档/课题组相关/理想项目/LX_project/prepare_data.py) 和联合推理脚本 `run_pulse_injury_inference.py`。

## 2. 共享数据接口

跨子项目共享的数据接口由 [`common/settings.py`](/e:/WPS Office/1628575652/WPS企业云盘/清华大学/我的企业文档/课题组相关/理想项目/LX_project/common/settings.py) 统一管理。当前默认目录结构如下：

```text
data/
├─ raw_packed/
│  └─ raw_data_packed.npz
├─ split_indices/
│  ├─ injury/
│  │  ├─ combined/
│  │  ├─ driver/
│  │  └─ passenger/
│  └─ pulse/
├─ processed/
│  └─ injury/
│     ├─ combined/
│     ├─ driver/
│     └─ passenger/
└─ normalization_config.json
```

当前约定：

- `injury` 任务保留三套划分：`combined / driver / passenger`
- `pulse` 任务只保留一套完整划分
- 全项目共享一份 `normalization_config.json`
- 默认开发与训练口径为 `combined`

## 3. 数据准备

根目录脚本 [`prepare_data.py`](/e:/WPS Office/1628575652/WPS企业云盘/清华大学/我的企业文档/课题组相关/理想项目/LX_project/prepare_data.py) 负责：

- 打包 `raw_packed/raw_data_packed.npz`
- 生成 `pulse` 任务的一套完整 split
- 生成 `injury` 任务的 `combined / driver / passenger` 三套 split
- 基于合并主副驾分布生成共享 `normalization_config.json`

运行方式：

```bash
conda activate pytorch
python -m prepare_data
```

运行后通常会得到：

- `data/raw_packed/raw_data_packed.npz`
- `data/split_indices/injury/combined/`
- `data/split_indices/injury/driver/`
- `data/split_indices/injury/passenger/`
- `data/split_indices/pulse/`
- `data/normalization_config.json`

## 4. 子项目入口

### 4.1 PulsePredict

```bash
python -m PulsePredict.train -c PulsePredict/config.json
python -m PulsePredict.test -r PulsePredict/saved/models/.../model_best.pth
```

### 4.2 InjuryPredict

先把 `raw_packed + split_indices + normalization_config` 转成 `.pt` 子集：

```bash
python -m InjuryPredict.Injurydata_prepare
```

再训练或评估模型：

```bash
python -m InjuryPredict.train
python -m InjuryPredict.eval_model
```

### 4.3 ARS_optim

```bash
python -m ARS_optim.run_train
python -m ARS_optim.run_eval
python -m ARS_optim.plot_eval_cases --eval_csv ARS_optim/saved_eval/.../results/evaluation_results.csv --case_ids 1 2 3
```

## 5. 当前数据口径说明

### 5.1 PulsePredict

- 使用 `data/raw_packed/raw_data_packed.npz`
- 使用 `data/split_indices/pulse/`
- 不区分主驾 / 副驾划分

### 5.2 InjuryPredict

- 使用同一份 `raw_packed`
- 使用 `data/split_indices/injury/<variant>/`
- 默认 `.pt` 生成目录为 `data/processed/injury/combined/`

### 5.3 ARS_optim

- 默认从 `injury/combined` 视角读取经验池与验证 / 测试划分
- 与 `PulsePredict`、`InjuryPredict` 共用同一份 `normalization_config.json`
- 通过配置文件和权重快照重建代理链路
- `run_eval` 在 `input_csv` 模式下只接受整组合法的 baseline trainable control；否则整组回退为 `param_space.yaml` 的 `default`
- 若回退后的 `default` 与当前 `context` 联合后仍不合法，则该 `case` 会直接跳过，不再输出伪造的 baseline 评估结果

## 6. 运行约定

- 统一在 `LX_project` 根目录下运行
- 统一使用 `python -m xxx`
- 建议先激活项目环境：

```bash
conda activate pytorch
```

- 如需安装依赖：

```bash
pip install -r requirements.txt
```

## 7. 进一步说明

- [PulsePredict/README.md](./PulsePredict/README.md)
- [InjuryPredict/README.md](./InjuryPredict/README.md)
- [ARS_optim/README.md](./ARS_optim/README.md)
