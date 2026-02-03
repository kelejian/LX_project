# 乘员损伤预测模型 (Injury Prediction Model)

本项目包含用于预测车辆乘员（特别是副驾驶侧）在碰撞事故中损伤风险的深度学习模型代码。主要能够预测 HIC (头部)、Dmax (胸部压缩量)、Nij (颈部) 的数值及其对应的 AIS 损伤等级和整体 MAIS 等级。

---

## 1. 环境准备

主要的依赖库如下：

```bash
pip install torch numpy pandas scikit-learn matplotlib imbalanced-learn tensorboard tqdm joblib
```

---

## 2. 项目结构说明
*   `data/`: 存放处理后的数据集文件 (.npz, .pt)。
*   `runs/`: 训练输出目录，保存训练配置、评估日志和模型权重文件等。
*   `utils/`: 工具脚本包。
    *   `data_package.py`: 用于提取原始波形和参数，打包成 .npz 文件。
    *   `dataset_prepare.py`: 数据预处理、归一化、划分训练/验证/测试集，生成 .pt 文件。
    *   `models.py`: 定义损伤预测神经网络模型结构
    *   `weighted_loss.py`: 自定义损失函数。
    *   `AIS_cal.py`: 损伤指标到 AIS 等级的计算逻辑。
*   `config.py`: **核心配置文件**，包含所有训练超参数、模型结构参数和损失权重。
*   `train.py`: 单次训练脚本（训练集+验证集）。
*   `train_KFold.py`: K折交叉验证训练脚本。
*   `eval_model.py`: 模型评估脚本，生成图表和报告。

---

## 3. 使用流程

### 注: 默认已经运行完成第一步的数据准备步骤，生成了 `data/train_dataset.pt`, `data/val_dataset.pt`, `data/test_dataset.pt` 和 `data/preprocessors.joblib`。

### 第一步：数据准备 (Data Preparation)

在使用模型训练之前，需要将原始的 CSV 波形数据和工况参数表转换为模型可读取的格式。

1.  **打包原始数据**:
    *   打开 `utils/data_package.py`。
    *   修改 `if __name__ == '__main__':` 部分的路径配置：
        *   `pulse_dir`: 原始波形 CSV 文件所在的文件夹路径。
        *   `params_path`: 包含工况参数的 CSV 或 NPZ 文件路径。
        *   `output_dir`: 输出目录，通常设置为 `./data`。
    *   运行脚本：
        ```bash
        python -m utils.data_package
        ```
    *   **产出**: `data/data_input.npz` 和 `data/data_labels.npz`。

2.  **生成数据集 (预处理)**:
    *   打开 `utils/dataset_prepare.py`。
    *   (可选) 可以在 `if __name__ == '__main__':` 中修改 `split_data` 的 `train_ratio`, `val_ratio`, `test_ratio` 比例，或设置 `special_assignments` 强制指定某些案例的划分。
    *   运行脚本：
        ```bash
        python -m utils.dataset_prepare
        ```
    *   **产出**:
        *   `data/train_dataset.pt`: 训练集 Tensor 数据。
        *   `data/val_dataset.pt`: 验证集 Tensor 数据。
        *   `data/test_dataset.pt`: 测试集 Tensor 数据。
        *   `data/preprocessors.joblib`: 保存的归一化参数，用于后续推理。

### 第二步：配置模型 (Configuration)

所有可调参数都集中在 `config.py` 中，无需深入修改训练代码。

*   **Training Params**: 轮数 (`Epochs`)、批次大小 (`Batch_size`)、学习率 (`Learning_rate`) 等。
*   **Loss Params**: 各个损伤部位的权重 (`loss_weights`)。
*   **Model Params**: TCN 通道数、MLP 层数及维度等。
*   **K-Fold Params**: K折交叉验证的折数 (`K`)。

### 第三步：模型训练 (Training)

提供两种训练模式：

#### 模式 A：单次训练
适用于快速验证模型效果或最终全量训练。

*   运行：
    ```bash
    python train.py
    ```
*   **输出**:
    *   日志和模型权重保存在 `runs/InjuryPredictModel_MMDDHHMM` 目录下。
    *   包含最佳 val Loss、最佳 MAIS 准确率等多个 checkpoints (.pth)。
    *   `TrainingRecord.json` 记录了本次训练的配置和最终结果。

#### 模式 B：K折交叉验证
适用于评估模型的泛化能力和稳定性。

*   运行：
    ```bash
    python train_KFold.py
    ```
*   **输出**:
    *   结果保存在 `runs/InjuryPredictModel_KFold_MMDDHHMM` 目录下。
    *   包含每一折 (Fold_1, Fold_2...) 的独立子文件夹和权重。
    *   `KFold_TrainingRecord.json` 汇总了所有折的平均性能指标。

### 第四步：模型评估 (Evaluation)

对训练好的模型在测试集上进行详细评估。
注: 建议仅用于测试 train.py 输出的模型权重文件

1.  打开 `eval_model.py`。
2.  修改 `if __name__ == "__main__":` 中的 `args` 类参数：
    *   `run_dir`: 指定你想要评估的特定训练记录文件夹路径 (例如 `runs/InjuryPredictModel_xxxx`).
    *   `weight_file`: 指定要加载的权重文件 (例如 `'best_loss.pth'`).
3.  运行脚本：
    ```bash
    python eval_model.py
    ```
4.  **产出**:
    *   会在 `run_dir` 下生成 Markdown 格式的评估报告 (`TestResults_....md`)。
    *   生成散点图 (HIC, Dmax, Nij 预测值 vs 真实值)。
    *   生成混淆矩阵图 (各部位及 MAIS 的分类表现)。

---

## 4. 可视化 (TensorBoard)

在训练过程中或训练后，可以使用 TensorBoard 查看损失曲线和指标变化：

```bash
tensorboard --logdir=./runs
```
```
然后在浏览器访问 `http://localhost:6006`。
