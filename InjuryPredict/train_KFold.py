# -*- coding: utf-8 -*-
"""
使用 K-Fold 交叉验证训练。
加载由 dataset_prepare.py 生成的 train_dataset.pt 和 val_dataset.pt，
将它们合并后进行 K-Fold 划分，并在每个 fold 上独立训练和验证模型。
最终报告 K-Fold 的平均性能。
"""

import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = 'T' # 似乎是特定环境的设置
import warnings
warnings.filterwarnings('ignore')
import json
import time
from datetime import datetime
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset, ConcatDataset # 引入 Subset 和 ConcatDataset
import torch.optim as optim
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, accuracy_score, confusion_matrix, r2_score
from sklearn.model_selection import StratifiedKFold # 引入 StratifiedKFold
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from imblearn.metrics import geometric_mean_score, classification_report_imbalanced
# --- 从 utils 导入必要的模块 ---
from utils import models
from utils.weighted_loss import weighted_loss
from utils.dataset_prepare import CrashDataset # 需要导入以加载 .pt 文件
from utils.AIS_cal import AIS_cal_head, AIS_cal_chest, AIS_cal_neck 
from utils.set_random_seed import GLOBAL_SEED, set_random_seed # 导入 GLOBAL_SEED
from utils.optimizer_utils import get_parameter_groups

from config import training_params, loss_params, model_params, kfold_params
set_random_seed() # 设置全局随机种子

def get_compare_func(func_indicator):
    """根据配置中的指示器返回比较函数和初始值"""
    if func_indicator == max or (isinstance(func_indicator, str) and func_indicator.lower() == 'max'):
        return max, float('-inf'), lambda curr, best: curr > best
    else:  # min
        return min, float('inf'), lambda curr, best: curr < best

def build_metric_trackers(val_metrics_to_track):
    """
    根据配置构建指标跟踪器字典。
    
    返回:
        trackers (dict): 键为指标名，值为包含比较函数、初始值等的字典。
    """
    trackers = {}
    for metric_name, compare_indicator in val_metrics_to_track:
        _, initial_value, is_better = get_compare_func(compare_indicator)
        trackers[metric_name] = {
            'compare_indicator': compare_indicator,
            'initial_value': initial_value,
            'is_better': is_better,
            'model_filename': f"best_{metric_name}_model.pth"
        }
    return trackers

def run_one_epoch(model, loader, criterion, device, optimizer=None):
    """
    执行一个完整的训练或验证周期。
    """
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    loss_batch = []
    all_preds, all_trues = [], []
    all_true_ais_head, all_true_ais_chest, all_true_ais_neck = [], [], []
    all_true_mais = []
    all_ot = []
    
    with torch.set_grad_enabled(is_train):
        for batch in loader:
            (batch_x_acc, batch_x_att_continuous, batch_x_att_discrete,
             batch_y_HIC, batch_y_Dmax, batch_y_Nij,
             batch_ais_head, batch_ais_chest, batch_ais_neck, batch_y_MAIS, 
             batch_OT) = [d.to(device) for d in batch]

            if is_train:
                optimizer.zero_grad()

            batch_y_true = torch.stack([batch_y_HIC, batch_y_Dmax, batch_y_Nij], dim=1)
            batch_pred, _, _ = model(batch_x_acc, batch_x_att_continuous, batch_x_att_discrete)
            loss = criterion(batch_pred, batch_y_true, batch_OT)

            if is_train:
                loss.backward()
                optimizer.step()

            loss_batch.append(loss.item())
            all_preds.append(batch_pred.detach().cpu().numpy())
            all_trues.append(batch_y_true.detach().cpu().numpy())
            all_true_ais_head.append(batch_ais_head.cpu().numpy())
            all_true_ais_chest.append(batch_ais_chest.cpu().numpy())
            all_true_ais_neck.append(batch_ais_neck.cpu().numpy())
            all_true_mais.append(batch_y_MAIS.cpu().numpy())
            all_ot.append(batch_OT.cpu().numpy())

    avg_loss = np.mean(loss_batch)
    preds, trues = np.concatenate(all_preds), np.concatenate(all_trues)
    ot = np.concatenate(all_ot)
    pred_hic, pred_dmax, pred_nij = preds[:, 0], preds[:, 1], preds[:, 2]
    true_hic, true_dmax, true_nij = trues[:, 0], trues[:, 1], trues[:, 2]
    
    ais_head_pred, ais_chest_pred, ais_neck_pred = AIS_cal_head(pred_hic), AIS_cal_chest(pred_dmax, ot), AIS_cal_neck(pred_nij)
    true_ais_head, true_ais_chest, true_ais_neck = np.concatenate(all_true_ais_head), np.concatenate(all_true_ais_chest), np.concatenate(all_true_ais_neck)
    true_mais = np.concatenate(all_true_mais)
    mais_pred = np.maximum.reduce([ais_head_pred, ais_chest_pred, ais_neck_pred])
    
    metrics = {
        'loss': avg_loss,
        'accu_head': accuracy_score(true_ais_head, ais_head_pred) * 100,
        'accu_chest': accuracy_score(true_ais_chest, ais_chest_pred) * 100,
        'accu_neck': accuracy_score(true_ais_neck, ais_neck_pred) * 100,
        'accu_mais': accuracy_score(true_mais, mais_pred) * 100,
        'mae_hic': mean_absolute_error(true_hic, pred_hic), 'rmse_hic': root_mean_squared_error(true_hic, pred_hic),
        'mae_dmax': mean_absolute_error(true_dmax, pred_dmax), 'rmse_dmax': root_mean_squared_error(true_dmax, pred_dmax),
        'mae_nij': mean_absolute_error(true_nij, pred_nij), 'rmse_nij': root_mean_squared_error(true_nij, pred_nij),
        'r2_hic': r2_score(true_hic, pred_hic),
        'r2_dmax': r2_score(true_dmax, pred_dmax),
        'r2_nij': r2_score(true_nij, pred_nij),
    }
    return metrics

def evaluate_fold(model, loader, device):
    """
    在验证集上运行模型并收集所有预测和真实标签。

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
            
            # 前向传播
            batch_pred, _, _ = model(batch_x_acc, batch_x_att_continuous, batch_x_att_discrete)

            # 收集回归和分类的标签
            batch_y_true = torch.stack([batch_y_HIC, batch_y_Dmax, batch_y_Nij], dim=1)
            all_preds.append(batch_pred.cpu().numpy())
            all_trues_regression.append(batch_y_true.cpu().numpy())
            all_true_ais_head.append(batch_ais_head.cpu().numpy())
            all_true_ais_chest.append(batch_ais_chest.cpu().numpy())
            all_true_ais_neck.append(batch_ais_neck.cpu().numpy())
            all_true_mais.append(batch_y_MAIS.cpu().numpy())
            all_ot.append(batch_OT.cpu().numpy())

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
    """计算并返回一组回归指标 (同 eval_model.py)"""
    return {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': root_mean_squared_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }

def get_classification_metrics(y_true, y_pred, labels):
    """计算并返回一组分类指标 (同 eval_model.py)"""
    # 检查缺失的类别
    present_labels = set(np.unique(np.concatenate([y_true, y_pred])))
    missing_labels = set(labels) - present_labels
    
    if missing_labels:
        print(f"\n*Warning: Labels {missing_labels} are not present in the data for this fold\n")
    
    return {
        'accuracy': accuracy_score(y_true, y_pred) * 100,
        'g_mean': geometric_mean_score(y_true, y_pred, labels=labels, average='multiclass'), # KFold中可能样本少，指定average
        'conf_matrix': confusion_matrix(y_true, y_pred, labels=labels),
        'report': classification_report_imbalanced(
            y_true, y_pred, labels=labels, digits=3, 
            zero_division=0  # 处理除零情况
        )
    }

def plot_scatter(y_true, y_pred, ais_true, title, xlabel, save_path):
    """改进的散点图函数 (同 eval_model.py)"""
    plt.figure(figsize=(8, 7))
    colors = ['blue', 'green', 'yellow', 'orange', 'red', 'darkred']
    
    # 确保 ais_true 中的值不会索引越界
    ais_indices = np.clip(ais_true, 0, 5).astype(int)
    ais_colors = [colors[i] for i in ais_indices]
    
    plt.scatter(y_true, y_pred, c=ais_colors, alpha=0.5, s=40)

    legend_elements = [Patch(facecolor=colors[i], label=f'AIS {i}') for i in range(6) if i in np.unique(ais_true)]
    
    max_val = max(np.max(y_true), np.max(y_pred)) * 1.05
    min_val = min(np.min(y_true), np.min(y_pred))
    min_val = min(0, min_val * 1.05) # 确保从0或更低开始
    
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Ideal Line")
    plt.xlabel(f"Ground Truth ({xlabel})", fontsize=16)
    plt.ylabel(f"Predictions ({xlabel})", fontsize=16)
    plt.title(f"Scatter Plot of Predictions vs Ground Truth\n({title})", fontsize=18)
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    
    first_legend = plt.legend(handles=legend_elements, title='AIS Level', loc='upper left')
    plt.gca().add_artist(first_legend)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(cm, labels, title, save_path):
    """绘制并保存混淆矩阵图 (同 eval_model.py)"""
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, fontsize=12)
    plt.yticks(tick_marks, labels, fontsize=12)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    
    # 处理 cm.max() 为 0 的情况
    thresh = cm.max() / 2. if cm.max() > 0 else 0.5 
    
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=12)
    plt.tight_layout(pad=0.5)  # 减少边距，从默认的 1.08 降低到 0.5
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)  # 添加紧凑保存选项
    plt.close()

def evaluate_and_plot_for_metric(model, model_path, val_loader_k, device, fold, metric_name, fold_run_dir, 
                                  AIS_cal_head, AIS_cal_chest, AIS_cal_neck):
    """
    加载指定指标的最佳模型，执行评估并绘制图表。
    
    返回:
        eval_results (dict): 包含该指标对应模型的详细评估结果。
    """
    if not os.path.exists(model_path):
        print(f"  警告: 未找到 {model_path}，跳过该指标的评估。")
        return None
    
    # 加载模型权重
    model.load_state_dict(torch.load(model_path))
    
    # 执行评估
    predictions, ground_truths = evaluate_fold(model, val_loader_k, device)

    ot = ground_truths['ot']
    pred_hic, pred_dmax, pred_nij = predictions[:, 0], predictions[:, 1], predictions[:, 2]
    true_hic, true_dmax, true_nij = ground_truths['regression'][:, 0], ground_truths['regression'][:, 1], ground_truths['regression'][:, 2]
    
    # 计算 AIS 预测
    ais_head_pred = AIS_cal_head(pred_hic)
    ais_chest_pred = AIS_cal_chest(pred_dmax, ot)
    ais_neck_pred = AIS_cal_neck(pred_nij)
    mais_pred = np.maximum.reduce([ais_head_pred, ais_chest_pred, ais_neck_pred])
    
    # 计算分类指标
    cls_metrics_head = get_classification_metrics(ground_truths['ais_head'], ais_head_pred, list(range(6)))
    cls_metrics_chest = get_classification_metrics(ground_truths['ais_chest'], ais_chest_pred, list(range(6)))
    cls_metrics_neck = get_classification_metrics(ground_truths['ais_neck'], ais_neck_pred, list(range(6)))
    cls_metrics_mais = get_classification_metrics(ground_truths['mais'], mais_pred, list(range(6)))
    
    # 计算回归指标
    reg_metrics_hic = get_regression_metrics(true_hic, pred_hic)
    reg_metrics_dmax = get_regression_metrics(true_dmax, pred_dmax)
    reg_metrics_nij = get_regression_metrics(true_nij, pred_nij)
    
    # 创建该指标专属的子目录
    metric_plot_dir = os.path.join(fold_run_dir, f"eval_{metric_name}")
    os.makedirs(metric_plot_dir, exist_ok=True)
    
    # 绘制散点图
    plot_scatter(true_hic, pred_hic, ground_truths['ais_head'], 
                 f'Fold {fold+1} (Best {metric_name}) - HIC', 'HIC', 
                 os.path.join(metric_plot_dir, "scatter_HIC.png"))
    plot_scatter(true_dmax, pred_dmax, ground_truths['ais_chest'], 
                 f'Fold {fold+1} (Best {metric_name}) - Dmax', 'Dmax (mm)', 
                 os.path.join(metric_plot_dir, "scatter_Dmax.png"))
    plot_scatter(true_nij, pred_nij, ground_truths['ais_neck'], 
                 f'Fold {fold+1} (Best {metric_name}) - Nij', 'Nij', 
                 os.path.join(metric_plot_dir, "scatter_Nij.png"))
    
    # 绘制混淆矩阵
    plot_confusion_matrix(cls_metrics_mais['conf_matrix'], list(range(6)), 
                          f'Fold {fold+1} (Best {metric_name}) - CM MAIS', 
                          os.path.join(metric_plot_dir, "cm_mais.png"))
    plot_confusion_matrix(cls_metrics_head['conf_matrix'], list(range(6)), 
                          f'Fold {fold+1} (Best {metric_name}) - CM Head', 
                          os.path.join(metric_plot_dir, "cm_head.png"))
    plot_confusion_matrix(cls_metrics_chest['conf_matrix'], list(range(6)), 
                          f'Fold {fold+1} (Best {metric_name}) - CM Chest', 
                          os.path.join(metric_plot_dir, "cm_chest.png"))
    plot_confusion_matrix(cls_metrics_neck['conf_matrix'], list(range(6)), 
                          f'Fold {fold+1} (Best {metric_name}) - CM Neck', 
                          os.path.join(metric_plot_dir, "cm_neck.png"))
    
    # 构建评估结果字典
    eval_results = {
        'accu_mais': cls_metrics_mais['accuracy'],
        'accu_head': cls_metrics_head['accuracy'],
        'accu_chest': cls_metrics_chest['accuracy'],
        'accu_neck': cls_metrics_neck['accuracy'],
        'g_mean_mais': cls_metrics_mais['g_mean'],
        'mae_hic': reg_metrics_hic['mae'],
        'rmse_hic': reg_metrics_hic['rmse'],
        'r2_hic': reg_metrics_hic['r2'],
        'mae_dmax': reg_metrics_dmax['mae'],
        'rmse_dmax': reg_metrics_dmax['rmse'],
        'r2_dmax': reg_metrics_dmax['r2'],
        'mae_nij': reg_metrics_nij['mae'],
        'rmse_nij': reg_metrics_nij['rmse'],
        'r2_nij': reg_metrics_nij['r2'],
    }
    
    print(f"    Fold {fold+1} (Best {metric_name}) 评估完成，图表已保存至 {metric_plot_dir}")
    return eval_results

def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, (np.integer, np.int_)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

if __name__ == "__main__":
    
    ############################################################################################
    ############################################################################################
    # ---- 从导入的配置中加载超参数 ----
    # 1. 优化与训练相关
    Epochs = training_params['Epochs']
    Batch_size = training_params['Batch_size']
    Learning_rate = training_params['Learning_rate']
    Learning_rate_min = training_params['Learning_rate_min']
    weight_decay = training_params['weight_decay']
    Patience = training_params['Patience']
    
    # 2. 损失函数相关
    base_loss = loss_params['base_loss']
    weight_factor_classify = loss_params['weight_factor_classify']
    weight_factor_sample = loss_params['weight_factor_sample']
    loss_weights = loss_params['loss_weights']

    # 3. 模型结构相关
    Ksize_init = model_params['Ksize_init']
    Ksize_mid = model_params['Ksize_mid']
    num_blocks_of_tcn = model_params['num_blocks_of_tcn']
    tcn_channels_list = model_params['tcn_channels_list']
    num_layers_of_mlpE = model_params['num_layers_of_mlpE']
    num_layers_of_mlpD = model_params['num_layers_of_mlpD']
    mlpE_hidden = model_params['mlpE_hidden']
    mlpD_hidden = model_params['mlpD_hidden']
    tcn_output_dim = model_params['tcn_output_dim']
    mlp_encoder_output_dim = model_params['mlp_encoder_output_dim']
    mlp_decoder_output_dim = model_params['mlp_decoder_output_dim']
    dropout_MLP = model_params['dropout_MLP']
    dropout_TCN = model_params['dropout_TCN']
    use_channel_attention = model_params['use_channel_attention']
    fixed_channel_weight = model_params['fixed_channel_weight']

    # K-Fold 设置
    K = kfold_params['K']
    val_metrics_to_track = kfold_params['val_metrics_to_track']
    
    # 构建指标跟踪器
    metric_trackers = build_metric_trackers(val_metrics_to_track)
    print(f"将跟踪以下指标: {list(metric_trackers.keys())}")

    ############################################################################################
    ############################################################################################
    
    # --- 2. 创建本次 K-Fold 运行的主目录 ---
    current_time = datetime.now().strftime("%m%d%H%M")
    main_run_dir = os.path.join("./runs", f"InjuryPredictModel_KFold_{current_time}")
    os.makedirs(main_run_dir, exist_ok=True)
    print(f"K-Fold 主运行目录: {main_run_dir}")

    # --- 3. 加载由 dataset_prepare.py 生成的数据 ---
    print("正在加载 pt dataset ...")
    try:
        train_subset_orig = torch.load("./data/train_dataset.pt", weights_only=False)
        val_subset_orig = torch.load("./data/val_dataset.pt", weights_only=False)
        test_subset_orig = torch.load("./data/test_dataset.pt",weights_only=False)
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("请确保 './data/' 目录下存在 train_dataset.pt 和 val_dataset.pt 文件。")
        print("您需要先运行 utils/dataset_prepare.py 来生成这些文件。")
        exit()
        
    # 获取底层的 CrashDataset 实例 (假设两个 Subset 指向同一个实例)
    full_processed_dataset = train_subset_orig.dataset
    
    # 合并训练集和验证集的【索引】用于 K-Fold 划分
    # combined_indices = np.concatenate([train_subset_orig.indices, val_subset_orig.indices])
    combined_indices = np.concatenate([train_subset_orig.indices, val_subset_orig.indices, test_subset_orig.indices])
    
    # 获取用于【分层】的标签 (从底层数据集中按合并后的索引提取)
    combined_labels = full_processed_dataset.mais[combined_indices]
    
    print(f"已加载并合并数据用于 K-Fold。总样本数: {len(combined_indices)}")
    
    # 获取模型所需的 num_classes_of_discrete
    num_classes_of_discrete = full_processed_dataset.num_classes_of_discrete

    # --- 预先实例化模型以获取参数量 ---
    print("正在计算模型参数量...")
    dummy_model = models.InjuryPredictModel(
        Ksize_init=Ksize_init, Ksize_mid=Ksize_mid,
        num_classes_of_discrete=num_classes_of_discrete,
        num_blocks_of_tcn=num_blocks_of_tcn, tcn_channels_list=tcn_channels_list,
        num_layers_of_mlpE=num_layers_of_mlpE, num_layers_of_mlpD=num_layers_of_mlpD,
        mlpE_hidden=mlpE_hidden, mlpD_hidden=mlpD_hidden,
        tcn_output_dim=tcn_output_dim, mlp_encoder_output_dim=mlp_encoder_output_dim, mlp_decoder_output_dim=mlp_decoder_output_dim,
        dropout_MLP=dropout_MLP, dropout_TCN=dropout_TCN,
        use_channel_attention=use_channel_attention, fixed_channel_weight=fixed_channel_weight
    )
    total_params = sum(p.numel() for p in dummy_model.parameters())
    trainable_params = sum(p.numel() for p in dummy_model.parameters() if p.requires_grad)
    print(dummy_model)
    print(f"模型总参数量: {total_params}, 可训练参数量: {trainable_params}")
    del dummy_model # 释放内存    

    # --- 4. 初始化 KFold ---
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=GLOBAL_SEED)
    
    # --- 5. 存储每一折的最佳验证指标 (按指标分组) ---
    all_folds_results = {metric_name: {'best_metrics': [], 'best_epochs': [], 'eval_results': []} 
                         for metric_name in metric_trackers.keys()}

    # --- add：初始保存 K-Fold 配置 ---
    results_path = os.path.join(main_run_dir, "KFold_TrainingRecord.json")
    initial_kfold_record = {
        "model_type": "InjuryPredictModel",
        "model_params_count": {
            "total_params": total_params,
            "trainable_params": trainable_params
        },
        "kfold_info": {
            "total_samples_for_kfold": len(combined_indices),
            "k_value": K,
            "val_metrics_to_track": val_metrics_to_track  # 记录所有跟踪的指标
        },
        "hyperparameters": { # 记录使用的超参数
             "training": {
                "Epochs": Epochs, "Batch_size": Batch_size, "Learning_rate": Learning_rate,
                "Learning_rate_min": Learning_rate_min, "weight_decay": weight_decay,
                "Patience": Patience,
            },
            "loss": {
                "base_loss": base_loss, "weight_factor_classify": weight_factor_classify,
                "weight_factor_sample": weight_factor_sample, "loss_weights": loss_weights,
            },
            "model": {
                "Ksize_init": Ksize_init, "Ksize_mid": Ksize_mid, "num_blocks_of_tcn": num_blocks_of_tcn,
                "tcn_channels_list": tcn_channels_list,
                "num_layers_of_mlpE": num_layers_of_mlpE, "num_layers_of_mlpD": num_layers_of_mlpD,
                "mlpE_hidden": mlpE_hidden, "mlpD_hidden": mlpD_hidden,
                "tcn_output_dim": tcn_output_dim, "mlp_encoder_output_dim": mlp_encoder_output_dim, "mlp_decoder_output_dim": mlp_decoder_output_dim,
                "dropout_MLP": dropout_MLP, "dropout_TCN": dropout_TCN,
                "use_channel_attention": use_channel_attention,
                "fixed_channel_weight": fixed_channel_weight,
                "num_classes_of_discrete": num_classes_of_discrete
            }
        }
    }
    initial_kfold_record = convert_numpy_types(initial_kfold_record)
    with open(results_path, "w") as f:
        json.dump(initial_kfold_record, f, indent=4)
    print(f"K-Fold 初始配置已保存至: {results_path}")

    # --- 6. K-Fold 交叉验证主循环 ---
    for fold, (train_k_indices, val_k_indices) in enumerate(skf.split(combined_indices, combined_labels)):
        
        fold_start_time = time.time()
        print("\n" + "="*50)
        print(f"                 Fold {fold+1}/{K}")
        print("="*50)
        
        # --- 6.1 创建当前 Fold 的运行目录和 TensorBoard Writer ---
        fold_run_dir = os.path.join(main_run_dir, f"Fold_{fold+1}")
        os.makedirs(fold_run_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=fold_run_dir)
        
        # --- 6.2 获取当前 Fold 对应的【原始数据集索引】 ---
        # kf.split 返回的是 combined_indices 数组内部的索引，需要映射回 full_processed_dataset 的索引
        train_orig_indices = combined_indices[train_k_indices]
        val_orig_indices = combined_indices[val_k_indices]
        
        # --- 6.3 创建当前 Fold 的 Subset 和 DataLoader ---
        train_subset_k = Subset(full_processed_dataset, train_orig_indices)
        val_subset_k = Subset(full_processed_dataset, val_orig_indices)
        
        train_loader_k = DataLoader(train_subset_k, batch_size=Batch_size, shuffle=True, num_workers=0)
        val_loader_k = DataLoader(val_subset_k, batch_size=Batch_size, shuffle=False, num_workers=0)
        
        print(f"Fold {fold+1} 数据划分 - Train: {len(train_subset_k)}, Valid: {len(val_subset_k)}")
        
        # --- 6.4 **重新初始化模型、优化器、调度器** ---
        # (确保每折训练的独立性)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = models.InjuryPredictModel(
            Ksize_init=Ksize_init, Ksize_mid=Ksize_mid,
            num_classes_of_discrete=num_classes_of_discrete,
            num_blocks_of_tcn=num_blocks_of_tcn, tcn_channels_list=tcn_channels_list,
            num_layers_of_mlpE=num_layers_of_mlpE, num_layers_of_mlpD=num_layers_of_mlpD,
            mlpE_hidden=mlpE_hidden, mlpD_hidden=mlpD_hidden,
            tcn_output_dim=tcn_output_dim, mlp_encoder_output_dim=mlp_encoder_output_dim, mlp_decoder_output_dim=mlp_decoder_output_dim,
            dropout_MLP=dropout_MLP, dropout_TCN=dropout_TCN,
            use_channel_attention=use_channel_attention, fixed_channel_weight=fixed_channel_weight
        ).to(device)

        # 在第一折打印: 模型各层参数量和总量, 我想知道模型参数量集中在哪里
        if fold == 0:
            print("\n模型结构:")
            print(model)
            print("\n模型各层参数量:")
            for name, param in model.named_parameters():
                print(f"  {name}: {param.numel()} parameters")
            print(f"\n模型参数量统计:")
            # total_params = sum(p.numel() for p in model.parameters())
            # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"模型总参数量: {total_params}, 可训练参数量: {trainable_params}")
            
        
        # 定义损失函数
        criterion = weighted_loss(base_loss, weight_factor_classify, weight_factor_sample, loss_weights)
        # 优化器（参数分组管理）和学习率调度器
        param_groups = get_parameter_groups(model, weight_decay=weight_decay, head_decay_ratio=0.05,head_keywords=('head',))   
        optimizer = optim.AdamW(param_groups, lr=Learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Epochs, eta_min=Learning_rate_min)

        # --- 6.5 初始化当前 Fold 的跟踪变量 ---
        val_loss_history = []
        
        # 为每个跟踪的指标初始化状态
        fold_metric_states = {}
        for metric_name, tracker_info in metric_trackers.items():
            fold_metric_states[metric_name] = {
                'best_value': tracker_info['initial_value'],
                'best_epoch': 0,
                'best_metrics_dict': None,
                'is_better': tracker_info['is_better'],
                'model_filename': tracker_info['model_filename']
            }
        
        # --- 6.6 Epoch 训练循环 (内层循环) ---
        if Patience > Epochs: current_patience = Epochs
        else: current_patience = Patience
            
        for epoch in range(Epochs):
            epoch_start_time = time.time()

            # --- 调用统一函数进行训练 ---
            train_metrics = run_one_epoch(model, train_loader_k, criterion, device, optimizer=optimizer)

            # --- 调用统一函数进行验证 ---
            val_metrics = run_one_epoch(model, val_loader_k, criterion, device, optimizer=None)
            
            val_loss_history.append(val_metrics['loss'])

            # 打印当前 Fold 的 Epoch 信息
            metric_strs = [f"{name}: {val_metrics[name]:.2f}" for name in metric_trackers.keys() if name != 'loss']
            loss_str = f"Val Loss: {val_metrics['loss']:.3f}"
            print(f"  Epoch {epoch+1}/{Epochs} | Train Loss: {train_metrics['loss']:.3f} | {loss_str} | {' | '.join(metric_strs)} | Time: {time.time()-epoch_start_time:.2f}s")
            
            scheduler.step()

            # --- TensorBoard 记录 ---
            # 训练指标
            writer.add_scalar("Loss/Train", train_metrics['loss'], epoch)
            writer.add_scalar("Accuracy_Train/MAIS", train_metrics['accu_mais'], epoch)
            writer.add_scalar("MAE_Train/Train_HIC", train_metrics['mae_hic'], epoch)
            writer.add_scalar("MAE_Train/Train_Dmax", train_metrics['mae_dmax'], epoch)
            writer.add_scalar("MAE_Train/Train_Nij", train_metrics['mae_nij'], epoch)
            writer.add_scalar("R2_Train/HIC", train_metrics['r2_hic'], epoch)
            writer.add_scalar("R2_Train/Dmax", train_metrics['r2_dmax'], epoch)
            writer.add_scalar("R2_Train/Nij", train_metrics['r2_nij'], epoch)

            # 验证指标
            writer.add_scalar("Loss/Val", val_metrics['loss'], epoch)
            writer.add_scalar("Accuracy_Val/MAIS", val_metrics['accu_mais'], epoch)
            writer.add_scalar("Accuracy_Val/Head", val_metrics['accu_head'], epoch)
            writer.add_scalar("Accuracy_Val/Chest", val_metrics['accu_chest'], epoch)
            writer.add_scalar("Accuracy_Val/Neck", val_metrics['accu_neck'], epoch)
            writer.add_scalar("MAE_Val/HIC", val_metrics['mae_hic'], epoch)
            writer.add_scalar("MAE_Val/Dmax", val_metrics['mae_dmax'], epoch)
            writer.add_scalar("MAE_Val/Nij", val_metrics['mae_nij'], epoch)
            writer.add_scalar("R2_Val/HIC", val_metrics['r2_hic'], epoch)
            writer.add_scalar("R2_Val/Dmax", val_metrics['r2_dmax'], epoch)
            writer.add_scalar("R2_Val/Nij", val_metrics['r2_nij'], epoch)

            # --- 跟踪当前 Fold 的最佳模型 (为每个指标) ---
            for metric_name, state in fold_metric_states.items():
                current_value = val_metrics[metric_name]
                if state['is_better'](current_value, state['best_value']):
                    state['best_value'] = current_value
                    state['best_epoch'] = epoch + 1
                    state['best_metrics_dict'] = val_metrics.copy()
                    
                    # 保存当前指标的最佳模型权重
                    torch.save(model.state_dict(), os.path.join(fold_run_dir, state['model_filename']))
                    print(f"    [Fold {fold+1}] Best {metric_name} model saved: {current_value:.3f} at epoch {epoch+1}")

            # --- 早停逻辑 (检查所有跟踪的指标) ---
            if epoch > Epochs * 0.4 and len(val_loss_history) >= current_patience:
                all_stagnant = all(
                    (epoch + 1 - state['best_epoch']) >= current_patience 
                    for state in fold_metric_states.values()
                )
                if all_stagnant:
                    print(f"    Early Stop at epoch {epoch+1} for Fold {fold+1}!")
                    for metric_name, state in fold_metric_states.items():
                        print(f"    Best {metric_name}: {state['best_value']:.3f} (at epoch {state['best_epoch']})")
                    break

        # --- 6.7 当前 Fold 训练结束，为每个指标执行详细评估 ---
        print(f"  Fold {fold+1} 训练完成。正在为每个跟踪指标执行详细评估...")

        for metric_name, state in fold_metric_states.items():
            model_path = os.path.join(fold_run_dir, state['model_filename'])
            eval_results = evaluate_and_plot_for_metric(
                model, model_path, val_loader_k, device, fold, metric_name, fold_run_dir,
                AIS_cal_head, AIS_cal_chest, AIS_cal_neck
            )
            
            # 记录结果
            all_folds_results[metric_name]['best_metrics'].append(state['best_metrics_dict'] or val_metrics)
            all_folds_results[metric_name]['best_epochs'].append(state['best_epoch'])
            if eval_results:
                all_folds_results[metric_name]['eval_results'].append(eval_results)

        writer.close()
        print(f"Fold {fold+1} finished in {time.time() - fold_start_time:.2f}s.")

    # --- 7. K-Fold 循环结束，计算并打印总体结果 ---
    print("\n" + "="*60)
    print("         K-Fold Cross-Validation Summary")
    print("="*60)
    
    kfold_summary = {}
    
    for metric_name in metric_trackers.keys():
        print(f"\n--- Results for Best '{metric_name}' Model ---")
        
        metrics_df = pd.DataFrame(all_folds_results[metric_name]['best_metrics'])
        best_epochs = all_folds_results[metric_name]['best_epochs']
        
        # 计算主要指标的均值和标准差
        summary_for_metric = {
            'mean_best_epoch': np.mean(best_epochs),
            'mean_loss': metrics_df['loss'].mean(),
            'std_loss': metrics_df['loss'].std(),
            'mean_accu_mais': metrics_df['accu_mais'].mean(),
            'std_accu_mais': metrics_df['accu_mais'].std(),
            'mean_accu_head': metrics_df['accu_head'].mean(),
            'std_accu_head': metrics_df['accu_head'].std(),
            'mean_accu_chest': metrics_df['accu_chest'].mean(),
            'std_accu_chest': metrics_df['accu_chest'].std(),
            'mean_accu_neck': metrics_df['accu_neck'].mean(),
            'std_accu_neck': metrics_df['accu_neck'].std(),
            'mean_mae_hic': metrics_df['mae_hic'].mean(),
            'std_mae_hic': metrics_df['mae_hic'].std(),
            'mean_mae_dmax': metrics_df['mae_dmax'].mean(),
            'std_mae_dmax': metrics_df['mae_dmax'].std(),
            'mean_mae_nij': metrics_df['mae_nij'].mean(),
            'std_mae_nij': metrics_df['mae_nij'].std(),
            'mean_r2_hic': metrics_df['r2_hic'].mean(),
            'std_r2_hic': metrics_df['r2_hic'].std(),
            'mean_r2_dmax': metrics_df['r2_dmax'].mean(),
            'std_r2_dmax': metrics_df['r2_dmax'].std(),
            'mean_r2_nij': metrics_df['r2_nij'].mean(),
            'std_r2_nij': metrics_df['r2_nij'].std(),
        }
        
        # 如果有详细评估结果，也计算其统计
        if all_folds_results[metric_name]['eval_results']:
            eval_df = pd.DataFrame(all_folds_results[metric_name]['eval_results'])
            for col in eval_df.columns:
                summary_for_metric[f'eval_mean_{col}'] = eval_df[col].mean()
                summary_for_metric[f'eval_std_{col}'] = eval_df[col].std()
        
        kfold_summary[metric_name] = summary_for_metric
        
        print(f"  Average Best Epoch: {summary_for_metric['mean_best_epoch']:.1f}")
        print(f"  Loss      : {summary_for_metric['mean_loss']:.3f} +/- {summary_for_metric['std_loss']:.3f}")
        print(f"  MAIS Acc  : {summary_for_metric['mean_accu_mais']:.2f}% +/- {summary_for_metric['std_accu_mais']:.2f}%")
        print(f"  Head Acc  : {summary_for_metric['mean_accu_head']:.2f}% +/- {summary_for_metric['std_accu_head']:.2f}%")
        print(f"  Chest Acc : {summary_for_metric['mean_accu_chest']:.2f}% +/- {summary_for_metric['std_accu_chest']:.2f}%")
        print(f"  Neck Acc  : {summary_for_metric['mean_accu_neck']:.2f}% +/- {summary_for_metric['std_accu_neck']:.2f}%")
        # 打印回归指标
        print(f"  HIC MAE   : {summary_for_metric['mean_mae_hic']:.3f} +/- {summary_for_metric['std_mae_hic']:.3f}")
        print(f"  Dmax MAE  : {summary_for_metric['mean_mae_dmax']:.3f} +/- {summary_for_metric['std_mae_dmax']:.3f}")
        print(f"  Nij MAE   : {summary_for_metric['mean_mae_nij']:.3f} +/- {summary_for_metric['std_mae_nij']:.3f}")
        # 打印 R² 指标
        print(f"  HIC R²    : {summary_for_metric['mean_r2_hic']:.4f} +/- {summary_for_metric['std_r2_hic']:.4f}")
        print(f"  Dmax R²   : {summary_for_metric['mean_r2_dmax']:.4f} +/- {summary_for_metric['std_r2_dmax']:.4f}")
        print(f"  Nij R²    : {summary_for_metric['mean_r2_nij']:.4f} +/- {summary_for_metric['std_r2_nij']:.4f}")

    
    print("="*60)
    
    # --- 8. 保存 K-Fold 总体结果 ---
    print("K-Fold 训练完成，正在加载初始记录并添加总结...")

    try:
        with open(results_path, "r") as f:
            final_kfold_record = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"警告: 未找到或无法解析 {results_path}。将创建一个新的记录文件。")
        final_kfold_record = initial_kfold_record

    # 添加按指标分组的结果
    # kfold_summary_by_metric 和 per_fold_results_by_metric 下记录的所有指标，是基于：特定评估指标（val_metrics_to_track中的）最优模型checkpoint的完整评估结果，不是各个指标历史最优值的独立拼凑
    final_kfold_record["kfold_summary_by_metric"] = convert_numpy_types(kfold_summary)
    final_kfold_record["per_fold_results_by_metric"] = convert_numpy_types({
        metric_name: {
            'best_metrics': data['best_metrics'],
            'best_epochs': data['best_epochs'],
            'eval_results': data['eval_results']
        }
        for metric_name, data in all_folds_results.items()
    })

    with open(results_path, "w") as f:
        json.dump(final_kfold_record, f, indent=4)
        
    print(f"\nK-Fold 总体结果已更新至: {results_path}")