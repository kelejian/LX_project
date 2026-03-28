"""
对损伤预测模型在测试集上的性能进行全面评估。
功能包括：
1. 计算三个损伤部位（头、胸、颈）的回归指标 (MAE, RMSE, R^2)。
2. 计算对应AIS等级以及MAIS的分类指标 (六分类与三分类的 Accuracy, G-mean, Confusion Matrix, Report)。
3. 为HIC额外计算AIS-3C的分类指标。
4. 生成并保存在指定运行目录下的详细评估报告 (Markdown格式)。
5. 生成并保存所有损伤指标的散点图和所有AIS分类的混淆矩阵图。
"""
# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')
import os, json
import torch
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset

from InjuryPredict.Injurydata_prepare import InjuryPackedDataset, load_processed_subset
from InjuryPredict.utils import models
from InjuryPredict.utils.tools import (
    get_regression_metrics,
    get_classification_metrics,
    get_mais_3c_metrics,
    MAIS_3C_DISPLAY_LABELS,
    plot_scatter,
    plot_confusion_matrix,
)

from common.metrics.injury_risk import AIS_cal_head, AIS_cal_chest, AIS_cal_neck
from common.tools.seeding import set_random_seed
from common.settings import get_injury_processed_dataset_path

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
        'ot': np.concatenate(all_ot),
    }

    return preds, trues

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

    set_random_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    # parser = argparse.ArgumentParser(description="Evaluate a trained injury prediction model")
    # parser.add_argument("--run_dir", '-r', type=str, default=r".\InjuryPredict\runs\InjuryPredictModel_03032051", help="Directory of the training run to evaluate.")
    # parser.add_argument("--weight_file", '-w', type=str, default="best_val_loss.pth", help="Name of the model weight file.")
    # args = parser.parse_args()

    from dataclasses import dataclass
    @dataclass
    class args:
        run_dir: str = r".\InjuryPredict\runs\InjuryPredictModel_03280055"
        weight_file: str = 'best_val_loss.pth'

    record_path = os.path.join(args.run_dir, "TrainingRecord.json")

    if not os.path.exists(record_path):
        kfold_record_path = os.path.join(args.run_dir, "TrainingRecord.json")
        if os.path.exists(kfold_record_path):
            record_path = kfold_record_path
        else:
            parent_dir = os.path.dirname(args.run_dir)
            parent_kfold_record = os.path.join(parent_dir, "TrainingRecord.json")
            if os.path.exists(parent_kfold_record):
                record_path = parent_kfold_record

    print(f"Reading training record from: {record_path}")
    with open(record_path, "r") as f:
        training_record = json.load(f)

    model_params = training_record["hyperparameters"]["model"]

    train_pt = get_injury_processed_dataset_path("train")
    val_pt = get_injury_processed_dataset_path("val")
    test_pt = get_injury_processed_dataset_path("test")
    train_dataset = load_processed_subset(train_pt)
    eval_subsets = []
    for subset_path in (val_pt, test_pt):
        subset = load_processed_subset(subset_path)
        if len(subset) > 0:
            eval_subsets.append(subset) # 只添加非空的验证集和测试集到评估列表中, 如果其中一个为空则只评估另一个，两个都不空则合并评估。
    if not eval_subsets:
        raise ValueError("val/test 数据集同时为空，eval_model 无可评估样本。")
    test_dataset = eval_subsets[0] if len(eval_subsets) == 1 else ConcatDataset(eval_subsets)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)

    print(f"加载 InjuryPredictModel 架构 (来自 {args.run_dir})")
    model = models.InjuryPredictModel(**model_params).to(device)

    model.load_state_dict(torch.load(os.path.join(args.run_dir, args.weight_file)))

    predictions, ground_truths = test(model, test_loader)

    ot = ground_truths['ot']
    pred_hic, pred_dmax, pred_nij = predictions[:, 0], predictions[:, 1], predictions[:, 2]
    true_hic, true_dmax, true_nij = ground_truths['regression'][:, 0], ground_truths['regression'][:, 1], ground_truths['regression'][:, 2]

    reg_metrics_hic = get_regression_metrics(true_hic, pred_hic)
    reg_metrics_dmax = get_regression_metrics(true_dmax, pred_dmax)
    reg_metrics_nij = get_regression_metrics(true_nij, pred_nij)

    AIS_head = AIS_cal_head(pred_hic)
    AIS_chest = AIS_cal_chest(pred_dmax, ot)
    AIS_neck = AIS_cal_neck(pred_nij)
    print(f"processed {len(pred_hic)} samples for classification metrics.")
    cls_metrics_head = get_classification_metrics(ground_truths['ais_head'], AIS_head, list(range(6)))
    print(f"Head metrics: {cls_metrics_head['accuracy']:.2f}%")
    cls_metrics_chest = get_classification_metrics(ground_truths['ais_chest'], AIS_chest, list(range(6)))
    print(f"Chest metrics: {cls_metrics_chest['accuracy']:.2f}%")
    cls_metrics_neck = get_classification_metrics(ground_truths['ais_neck'], AIS_neck, list(range(6)))
    print(f"Neck metrics: {cls_metrics_neck['accuracy']:.2f}%")

    mais_pred = np.maximum.reduce([AIS_head, AIS_chest, AIS_neck])
    cls_metrics_mais = get_classification_metrics(ground_truths['mais'], mais_pred, list(range(6)))
    cls_metrics_mais_3c = get_mais_3c_metrics(ground_truths['mais'], mais_pred)

    plot_scatter(true_hic, pred_hic, ground_truths['ais_head'], 'Head Injury Criterion (HIC)', 'HIC', os.path.join(args.run_dir, "scatter_plot_HIC.png"))
    plot_scatter(true_dmax, pred_dmax, ground_truths['ais_chest'], 'Chest Displacement (Dmax)', 'Dmax (mm)', os.path.join(args.run_dir, "scatter_plot_Dmax.png"))
    plot_scatter(true_nij, pred_nij, ground_truths['ais_neck'], 'Neck Injury Criterion (Nij)', 'Nij', os.path.join(args.run_dir, "scatter_plot_Nij.png"))

    plot_confusion_matrix(cls_metrics_head['conf_matrix'], list(range(6)), 'Confusion Matrix - AIS Head (6C)', os.path.join(args.run_dir, "cm_head_6c.png"))
    plot_confusion_matrix(cls_metrics_chest['conf_matrix'], list(range(6)), 'Confusion Matrix - AIS Chest (6C)', os.path.join(args.run_dir, "cm_chest_6c.png"))
    plot_confusion_matrix(cls_metrics_neck['conf_matrix'], list(range(6)), 'Confusion Matrix - AIS Neck (6C)', os.path.join(args.run_dir, "cm_neck_6c.png"))
    plot_confusion_matrix(cls_metrics_mais['conf_matrix'], list(range(6)), 'Confusion Matrix - MAIS (6C)', os.path.join(args.run_dir, "cm_mais_6c.png"))
    plot_confusion_matrix(cls_metrics_mais_3c['conf_matrix'], MAIS_3C_DISPLAY_LABELS, 'Confusion Matrix - MAIS (3C)', os.path.join(args.run_dir, "cm_mais_3c.png"))
    print(f"All plots have been saved to {args.run_dir}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {total_params} parameters.")

    print("\n--- Regression Metrics ---")
    print(f"HIC - MAE: {reg_metrics_hic['mae']:.4f}, RMSE: {reg_metrics_hic['rmse']:.4f}, R²: {reg_metrics_hic['r2']:.4f}")
    print(f"Dmax - MAE: {reg_metrics_dmax['mae']:.4f}, RMSE: {reg_metrics_dmax['rmse']:.4f}, R²: {reg_metrics_dmax['r2']:.4f}")
    print(f"Nij - MAE: {reg_metrics_nij['mae']:.4f}, RMSE: {reg_metrics_nij['rmse']:.4f}, R²: {reg_metrics_nij['r2']:.4f}")
    print(f"MAIS Accuracy (6C): {cls_metrics_mais['accuracy']:.2f}%")
    print(f"MAIS Accuracy (3C): {cls_metrics_mais_3c['accuracy']:.2f}%")
    print(f"Head AIS-6C Accuracy: {cls_metrics_head['accuracy']:.2f}%")
    print(f"Chest AIS-6C Accuracy: {cls_metrics_chest['accuracy']:.2f}%")
    print(f"Neck AIS-6C Accuracy: {cls_metrics_neck['accuracy']:.2f}%")

    markdown_content = f"""# Model Evaluation Report

## Model Identification
- **Run Directory**: `{args.run_dir}`
- **Weight File**: `{args.weight_file}`
- **Total Parameters**: {total_params}
- **Trainset size**: {len(train_dataset)}
- **Testset size**: {len(test_dataset)}
```

## Overall Injury Assessment (MAIS)

- **AIS-6C Accuracy**: {cls_metrics_mais['accuracy']:.2f}%
- **AIS-6C G-Mean**: {cls_metrics_mais['g_mean']:.4f}
- **AIS-6C Confusion Matrix**:
{cls_metrics_mais['conf_matrix']}
- **AIS-6C Classification Report**:
{cls_metrics_mais['report']}

- **AIS-3C Accuracy**: {cls_metrics_mais_3c['accuracy']:.2f}%
- **AIS-3C G-Mean**: {cls_metrics_mais_3c['g_mean']:.4f}
- **AIS-3C Confusion Matrix**:
{cls_metrics_mais_3c['conf_matrix']}
- **AIS-3C Classification Report**:
{cls_metrics_mais_3c['report']}

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
