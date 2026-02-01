"""
对损伤预测模型在测试集上的性能进行全面评估。
功能包括：
1. 计算三个损伤部位（头、胸、颈）的回归指标 (MAE, RMSE, R^2)。
2. 计算对应AIS等级以及MAIS的分类指标 (Accuracy, G-mean, Confusion Matrix, Report)。
3. 为HIC额外计算AIS-3C的分类指标。
4. 生成并保存在指定运行目录下的详细评估报告 (Markdown格式)。
5. 生成并保存所有损伤指标的散点图和所有AIS分类的混淆矩阵图。
"""
# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')
import os, json
import pandas as pd
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, ConcatDataset
from imblearn.metrics import geometric_mean_score, classification_report_imbalanced
from sklearn.metrics import confusion_matrix, r2_score, accuracy_score
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from utils import models
from utils.dataset_prepare import CrashDataset
from utils.AIS_cal import AIS_cal_head, AIS_cal_chest, AIS_cal_neck 

from utils.set_random_seed import set_random_seed
set_random_seed()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test(model, loader):
    """
    在测试集上运行模型并收集所有预测和真实标签。

    返回:
        preds (np.ndarray): 模型对 [HIC, Dmax, Nij] 的预测值, 形状 (N, 3)。
        trues (dict): 包含所有真实标签的字典。
    """
    model.eval()
    all_preds = []
    all_trues_regression = []
    all_true_ais_head, all_true_ais_chest, all_true_ais_neck, all_true_mais = [], [], [], []
    all_ot = []
    with torch.no_grad():
        for batch in loader:
            (batch_x_acc, batch_x_att_continuous, batch_x_att_discrete,
             batch_y_HIC, batch_y_Dmax, batch_y_Nij,
             batch_ais_head, batch_ais_chest, batch_ais_neck, batch_y_MAIS, 
             batch_OT) = [d.to(device) for d in batch]
            
            batch_pred, _, _ = model(batch_x_acc, batch_x_att_continuous, batch_x_att_discrete)

            # 收集回归和分类的标签
            batch_y_true = torch.stack([batch_y_HIC, batch_y_Dmax, batch_y_Nij], dim=1)
            all_preds.append(batch_pred.cpu().numpy())
            all_trues_regression.append(batch_y_true.cpu().numpy())
            all_true_ais_head.append(batch_ais_head.cpu().numpy())
            all_true_ais_chest.append(batch_ais_chest.cpu().numpy())
            all_true_ais_neck.append(batch_ais_neck.cpu().numpy())
            all_true_mais.append(batch_y_MAIS.cpu().numpy())
            all_ot.append(batch_OT.cpu().numpy()) # 保存OT
    
    preds = np.concatenate(all_preds)
    trues = {
        'regression': np.concatenate(all_trues_regression),
        'ais_head': np.concatenate(all_true_ais_head),
        'ais_chest': np.concatenate(all_true_ais_chest),
        'ais_neck': np.concatenate(all_true_ais_neck),
        'mais': np.concatenate(all_true_mais),
        'ot': np.concatenate(all_ot)
    }
    
    return preds, trues

def get_regression_metrics(y_true, y_pred):
    """计算并返回一组回归指标"""
    return {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': root_mean_squared_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }

def get_classification_metrics(y_true, y_pred, labels):
    """计算并返回一组分类指标 - 改进版"""
    # 检查缺失的类别
    present_labels = set(np.unique(np.concatenate([y_true, y_pred])))
    missing_labels = set(labels) - present_labels
    
    if missing_labels:
        print(f"\n*Warning: Labels {missing_labels} are not present in the data\n")
    
    return {
        'accuracy': accuracy_score(y_true, y_pred) * 100,
        'g_mean': geometric_mean_score(y_true, y_pred, labels=labels, average='multiclass'),
        'conf_matrix': confusion_matrix(y_true, y_pred, labels=labels),
        'report': classification_report_imbalanced(
            y_true, y_pred, labels=labels, digits=3, 
            zero_division=0  # 处理除零情况
        )
    }

def plot_scatter(y_true, y_pred, ais_true, title, xlabel, save_path):
    """改进的散点图函数"""
    plt.figure(figsize=(8, 7))
    colors = ['blue', 'green', 'yellow', 'orange', 'red', 'darkred']
    ais_colors = [colors[min(ais, 5)] for ais in ais_true]
    plt.scatter(y_true, y_pred, c=ais_colors, alpha=0.5, s=40)

    # 显示所有可能的类别，即使数据中不存在
    # all_possible_ais = range(6)
    # legend_elements = [
    #     Patch(facecolor=colors[i], 
    #           label=f'AIS {i}' + (' (absent)' if i not in np.unique(ais_true) else ''))
    #     for i in all_possible_ais
    # ]

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[i], label=f'AIS {i}') for i in range(6) if i in np.unique(ais_true)]
    
    max_val = max(np.max(y_true), np.max(y_pred)) * 1.05
    plt.plot([0, max_val], [0, max_val], 'r--', label="Ideal Line")
    plt.xlabel(f"Ground Truth ({xlabel})", fontsize=16)
    plt.ylabel(f"Predictions ({xlabel})", fontsize=16)
    plt.title(f"Scatter Plot of Predictions vs Ground Truth\n({title})", fontsize=18)
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    
    first_legend = plt.legend(handles=legend_elements, title='AIS Level', loc='upper left')
    plt.gca().add_artist(first_legend)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(cm, labels, title, save_path):
    """绘制并保存混淆矩阵图"""
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, fontsize=12)
    plt.yticks(tick_marks, labels, fontsize=12)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    thresh = cm.max() / 2. if cm.max() > 0 else 0.5 
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=12)
    plt.tight_layout(pad=0.5)  # 减少边距，从默认的 1.08 降低到 0.5
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)  # 添加紧凑保存选项
    plt.close()

def generate_report_section(title, reg_metrics, cls_metrics_6c):
    """生成Markdown报告的一个区域"""
    section = f"## {title} Metrics\n\n"
    section += f"- **MAE**: {reg_metrics['mae']:.4f}\n"
    section += f"- **RMSE**: {reg_metrics['rmse']:.4f}\n"
    section += f"- **R² Score**: {reg_metrics['r2']:.4f}\n\n"
    
    section += f"### AIS-6C Classification\n\n"
    section += f"- **Accuracy**: {cls_metrics_6c['accuracy']:.2f}%\n"
    section += f"- **G-Mean**: {cls_metrics_6c['g_mean']:.4f}\n"
    section += f"- **Confusion Matrix**:\n```\n{cls_metrics_6c['conf_matrix']}\n```\n"
    section += f"- **Classification Report**:\n```\n{cls_metrics_6c['report']}\n```\n"
    
    return section

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Evaluate a trained injury prediction model")
    # parser.add_argument("--run_dir", '-r', type=str, required=True, help="Directory of the training run to evaluate.")
    # parser.add_argument("--weight_file", '-w', type=str, default="best_mais_accu.pth", help="Name of the model weight file.")
    # args = parser.parse_args()

    from dataclasses import dataclass
    @dataclass
    class args:
        run_dir: str = r'E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\LX_model_injurypredict\runs\InjuryPredictModel_01281059'
        weight_file: str = 'best_val_loss.pth'

    # --- 1. 加载模型和数据 ---
    # 尝试加载普通训练记录
    record_path = os.path.join(args.run_dir, "TrainingRecord.json")
    
    # 如果找不到，尝试查找是否是 K-Fold 训练的子目录结构
    if not os.path.exists(record_path):
        # 尝试在当前目录找 KFold 记录
        kfold_record_path = os.path.join(args.run_dir, "KFold_TrainingRecord.json")
        if os.path.exists(kfold_record_path):
             record_path = kfold_record_path
        else:
            # 尝试在父目录找 KFold 记录 (标准 K-Fold 结构)
            parent_dir = os.path.dirname(args.run_dir)
            parent_kfold_record = os.path.join(parent_dir, "KFold_TrainingRecord.json")
            if os.path.exists(parent_kfold_record):
                record_path = parent_kfold_record

    print(f"Reading training record from: {record_path}")
    with open(record_path, "r") as f:
        training_record = json.load(f)
    
    model_params = training_record["hyperparameters"]["model"]
    
    train_dataset = torch.load("./data/train_dataset.pt", weights_only=False) # 仅用于获取 num_classes_of_discrete
    test_dataset1 = torch.load("./data/val_dataset.pt", weights_only=False)
    test_dataset2 = torch.load("./data/test_dataset.pt", weights_only=False)
    test_dataset = ConcatDataset([test_dataset1, test_dataset2])
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)

    print(f"加载 InjuryPredictModel 架构 (来自 {args.run_dir})")
    model = models.InjuryPredictModel(**model_params).to(device)
    
    model.load_state_dict(torch.load(os.path.join(args.run_dir, args.weight_file)))

    # --- 2. 执行预测 ---
    predictions, ground_truths = test(model, test_loader)
    
    ot = ground_truths['ot']  # 乘员体征类别
    pred_hic, pred_dmax, pred_nij = predictions[:, 0], predictions[:, 1], predictions[:, 2]
    true_hic, true_dmax, true_nij = ground_truths['regression'][:, 0], ground_truths['regression'][:, 1], ground_truths['regression'][:, 2]

    # --- 3. 计算所有指标 ---
    # 回归指标
    reg_metrics_hic = get_regression_metrics(true_hic, pred_hic)
    reg_metrics_dmax = get_regression_metrics(true_dmax, pred_dmax)
    reg_metrics_nij = get_regression_metrics(true_nij, pred_nij)

    # 分类指标 
    AIS_head = AIS_cal_head(pred_hic)
    AIS_chest = AIS_cal_chest(pred_dmax, ot)
    AIS_neck = AIS_cal_neck(pred_nij)
    cls_metrics_head = get_classification_metrics(ground_truths['ais_head'], AIS_head,  list(range(6)))
    cls_metrics_chest = get_classification_metrics(ground_truths['ais_chest'], AIS_chest, list(range(6)))
    cls_metrics_neck = get_classification_metrics(ground_truths['ais_neck'], AIS_neck, list(range(6)))

    # MAIS 指标
    mais_pred = np.maximum.reduce([AIS_head, AIS_chest, AIS_neck])
    cls_metrics_mais = get_classification_metrics(ground_truths['mais'], mais_pred, list(range(6)))
    

    # --- 4. 生成并保存所有可视化图表 ---
    plot_scatter(true_hic, pred_hic, ground_truths['ais_head'], 'Head Injury Criterion (HIC)', 'HIC', os.path.join(args.run_dir, "scatter_plot_HIC.png"))
    plot_scatter(true_dmax, pred_dmax, ground_truths['ais_chest'], 'Chest Displacement (Dmax)', 'Dmax (mm)', os.path.join(args.run_dir, "scatter_plot_Dmax.png"))
    plot_scatter(true_nij, pred_nij, ground_truths['ais_neck'], 'Neck Injury Criterion (Nij)', 'Nij', os.path.join(args.run_dir, "scatter_plot_Nij.png"))

    plot_confusion_matrix(cls_metrics_head['conf_matrix'], list(range(6)), 'Confusion Matrix - AIS Head (6C)', os.path.join(args.run_dir, "cm_head_6c.png"))
    plot_confusion_matrix(cls_metrics_chest['conf_matrix'], list(range(6)), 'Confusion Matrix - AIS Chest (6C)', os.path.join(args.run_dir, "cm_chest_6c.png"))
    plot_confusion_matrix(cls_metrics_neck['conf_matrix'], list(range(6)), 'Confusion Matrix - AIS Neck (6C)', os.path.join(args.run_dir, "cm_neck_6c.png"))
    plot_confusion_matrix(cls_metrics_mais['conf_matrix'], list(range(6)), 'Confusion Matrix - MAIS (6C)', os.path.join(args.run_dir, "cm_mais_6c.png"))
    print(f"All plots have been saved to {args.run_dir}")

    # 模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {total_params} parameters.")

    # 打印回归指标
    print("\n--- Regression Metrics ---")
    print(f"HIC - MAE: {reg_metrics_hic['mae']:.4f}, RMSE: {reg_metrics_hic['rmse']:.4f}, R²: {reg_metrics_hic['r2']:.4f}")
    print(f"Dmax - MAE: {reg_metrics_dmax['mae']:.4f}, RMSE: {reg_metrics_dmax['rmse']:.4f}, R²: {reg_metrics_dmax['r2']:.4f}")
    print(f"Nij - MAE: {reg_metrics_nij['mae']:.4f}, RMSE: {reg_metrics_nij['rmse']:.4f}, R²: {reg_metrics_nij['r2']:.4f}")
    # 打印MAIS准确率, 和三个部位多分类准确率
    print(f"MAIS Accuracy: {cls_metrics_mais['accuracy']:.2f}%")
    print(f"Head AIS-6C Accuracy: {cls_metrics_head['accuracy']:.2f}%")
    print(f"Chest AIS-6C Accuracy: {cls_metrics_chest['accuracy']:.2f}%")
    print(f"Neck AIS-6C Accuracy: {cls_metrics_neck['accuracy']:.2f}%")

    # --- 5. 生成并保存 Markdown 报告 ---
    markdown_content = f"""# Model Evaluation Report

## Model Identification
- **Run Directory**: `{args.run_dir}`
- **Weight File**: `{args.weight_file}`
- **Total Parameters**: {total_params}
- **Trainset size**: {len(train_dataset)}
- **Testset size**: {len(test_dataset)}
```

## Overall Injury Assessment (MAIS)

- **Accuracy**: {cls_metrics_mais['accuracy']:.2f}%
- **G-Mean**: {cls_metrics_mais['g_mean']:.4f}
- **Confusion Matrix**:
{cls_metrics_mais['conf_matrix']}
- **Classification Report**:
{cls_metrics_mais['report']}

---
"""
    markdown_content += generate_report_section("Head (HIC)", reg_metrics_hic, cls_metrics_head)
    markdown_content += "---\n"
    markdown_content += generate_report_section("Chest (Dmax)", reg_metrics_dmax, cls_metrics_chest)
    markdown_content += "---\n"
    markdown_content += generate_report_section("Neck (Nij)", reg_metrics_nij, cls_metrics_neck)

    report_path = os.path.join(args.run_dir, f"TestResults_{args.weight_file.replace('.pth', '')}.md")
    with open(report_path, "w", encoding="utf-8") as md_file:
        md_file.write(markdown_content)
    
    print(f"Comprehensive evaluation report saved to {report_path}")