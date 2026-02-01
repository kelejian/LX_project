# LX_project - Unified Architecture

## 项目概述 (Project Overview)

LX_project 是一个统一的汽车碰撞安全分析与优化平台，整合了三个核心子项目：

- **PulsePredict (b)**: 碰撞加速度波形预测 - 基于初始工况参数快速预测碰撞加速度时序波形
- **InjuryPredict (c)**: 乘员损伤预测 - 预测车辆乘员在碰撞事故中的损伤风险
- **ARS_optim (d)**: 约束系统优化 - 自适应约束系统参数优化

## 项目架构 (Architecture)

```
LX_project/
├── shared/                  # 共享模块 (Shared modules)
│   ├── data/               # 数据处理工具 (Data handling utilities)
│   │   ├── data_loader.py  # 数据加载 (Data loading)
│   │   └── preprocessing.py # 预处理 (Preprocessing)
│   ├── utils/              # 通用工具 (Common utilities)
│   │   ├── injury_metrics.py # AIS损伤指标计算 (AIS calculations)
│   │   └── random_seed.py    # 随机种子管理 (Random seed)
│   └── config/             # 配置管理 (Configuration)
│
├── PulsePredict/           # 子项目 b: 波形预测
│   ├── base/               # 抽象基类
│   ├── data_loader/        # 数据加载器
│   ├── model/              # 模型定义
│   ├── trainer/            # 训练器
│   ├── utils/              # 项目特定工具
│   ├── config.json         # 配置文件
│   └── train.py            # 训练脚本
│
├── InjuryPredict/          # 子项目 c: 损伤预测
│   ├── utils/              # 工具函数
│   ├── config.py           # 配置文件
│   ├── train.py            # 训练脚本
│   └── eval_model.py       # 评估脚本
│
├── ARS_optim/              # 子项目 d: 约束系统优化
│   ├── src/                # 源代码
│   │   ├── core/           # 核心优化逻辑
│   │   ├── interface/      # 接口适配器
│   │   ├── models/         # 策略网络
│   │   └── utils/          # 工具函数
│   ├── configs/            # 配置文件
│   └── run_train_strategy.py
│
├── requirements.txt        # 统一依赖 (Unified dependencies)
├── README.md              # 本文件 (This file)
├── MIGRATION_GUIDE.md     # 迁移指南 (Migration guide)
├── API_REFERENCE.md       # API文档 (API reference)
├── examples_data_module.py # 数据模块示例 (Data module examples)
└── examples_utils_module.py # 工具模块示例 (Utils module examples)
```

## 核心特性 (Key Features)

### 1. 统一的数据格式 (Unified Data Format)
所有子项目使用 `.npz` 格式存储原始数据，确保数据处理的一致性。

### 2. 共享模块 (Shared Modules)
- **数据处理**: 统一的数据加载、预处理和归一化函数
- **损伤指标**: AIS (Abbreviated Injury Scale) 计算函数
- **工具函数**: 随机种子管理、配置管理等

### 3. 模块化设计 (Modular Design)
每个子项目保持独立性，同时可以复用共享模块中的功能。

## 快速开始 (Quick Start)

### 环境配置 (Environment Setup)

```bash
# 克隆项目
git clone <repository-url>
cd LX_project

# 创建虚拟环境 (推荐)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 使用子项目 (Using Sub-projects)

#### PulsePredict (波形预测)
```bash
cd PulsePredict
python train.py -c config.json
python test.py -r saved/models/.../model_best.pth
```
详细说明请参考: [PulsePredict/README_FOR_USER.md](PulsePredict/README_FOR_USER.md)

#### InjuryPredict (损伤预测)
```bash
cd InjuryPredict
python train.py
python eval_model.py
```
详细说明请参考: [InjuryPredict/README_FOR_USER.md](InjuryPredict/README_FOR_USER.md)

#### ARS_optim (约束系统优化)
```bash
cd ARS_optim
python run_train_strategy.py
python run_evaluation.py
```

## 共享模块使用 (Using Shared Modules)

### 数据加载示例 (Data Loading Example)
```python
from shared.data import load_npz_data, split_train_test

# 加载数据
data = load_npz_data('path/to/data.npz')

# 分割训练/测试集
train_ids, test_ids = split_train_test(data['case_ids'], train_ratio=0.86)
```

### 预处理示例 (Preprocessing Example)
```python
from shared.data import normalize_waveform_data, save_preprocessors

# 归一化波形数据
normalized_waveforms, scaler = normalize_waveform_data(
    waveforms, 
    method='minmax', 
    fit=True
)

# 保存预处理器
save_preprocessors('preprocessors.joblib', waveform_scaler=scaler)
```

### 损伤指标计算示例 (Injury Metrics Example)
```python
from shared.utils import AIS_cal_head, AIS_cal_chest, AIS_cal_neck

# 计算AIS等级
head_ais = AIS_cal_head(HIC15=800.0)
chest_ais = AIS_cal_chest(Dmax=45.0, OT=2)
neck_ais = AIS_cal_neck(Nij=0.5)
```

## 数据处理流程 (Data Processing Pipeline)

### 统一的数据处理流程:
1. **原始数据准备**: CSV波形文件 + 工况参数文件
2. **数据打包**: 使用 `.npz` 格式打包数据
3. **数据预处理**: 归一化、特征工程
4. **数据集划分**: 训练集、验证集、测试集

### 共享的数据处理函数:
- `load_npz_data()`: 加载.npz文件
- `validate_case_ids()`: 验证case ID一致性
- `normalize_waveform_data()`: 波形归一化
- `normalize_features()`: 特征归一化
- `split_train_test()`: 数据集划分

## 项目状态 (Project Status)

- **PulsePredict**: ✅ 运行稳定 (Running stably)
- **InjuryPredict**: ✅ 运行稳定 (Running stably)
- **ARS_optim**: 🚧 开发中 (In development)

## 依赖关系 (Dependencies)

核心依赖:
- PyTorch >= 1.10.0
- NumPy >= 1.20.0
- Pandas >= 1.3.0
- Scikit-learn >= 0.24.0
- SciPy >= 1.7.0

详细依赖列表请参考 [requirements.txt](requirements.txt)

## 文档 (Documentation)

- **[README.md](README.md)** - 项目概览和快速开始 (Project overview and quick start)
- **[MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)** - 迁移到新架构的指南 (Guide for migrating to new architecture)
- **[API_REFERENCE.md](API_REFERENCE.md)** - 共享模块API文档 (Shared modules API reference)
- **[examples_data_module.py](examples_data_module.py)** - 数据模块使用示例 (Data module usage examples)
- **[examples_utils_module.py](examples_utils_module.py)** - 工具模块使用示例 (Utils module usage examples)

## 开发指南 (Development Guide)

### 添加共享功能 (Adding Shared Features)
1. 在 `shared/` 目录下添加新模块
2. 更新相应的 `__init__.py` 文件
3. 在子项目中导入使用

### 添加新的子项目 (Adding New Sub-projects)
1. 在根目录创建新的子项目文件夹
2. 复用 `shared/` 模块中的功能
3. 更新本 README 文件

## 贡献指南 (Contributing)

欢迎贡献! 请确保:
1. 遵循现有的代码结构和风格
2. 新增功能优先考虑添加到 `shared/` 模块
3. 更新相关文档

## 许可证 (License)

[根据实际情况填写]

## 联系方式 (Contact)

[根据实际情况填写]

---

**注意**: 本项目架构旨在提供统一、可扩展的碰撞安全分析平台，便于各子项目间的协作和功能复用。
