# -*- coding: utf-8 -*-
"""
使用 K-Fold 交叉验证训练。
加载由 Injurydata_prepare.py 生成的 train_dataset.pt、val_dataset.pt 和 test_dataset.pt，
将三者的样本索引合并后进行 K-Fold 划分，并在每个 fold 上独立训练和验证模型。
最终报告 K-Fold 的平均性能。
"""

import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = 'T' # 避免部分 Windows 控制台环境中的控制信号处理干扰训练进程。
import warnings
warnings.filterwarnings('ignore')
import json
import time
from datetime import datetime
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from torch.utils.tensorboard import SummaryWriter

from common.metrics.injury_risk import AIS_cal_head, AIS_cal_chest, AIS_cal_neck
from common.tools.seeding import GLOBAL_SEED, set_random_seed
from common.settings import INJURY_PROCESSED_DIR, get_injury_processed_dataset_path

from InjuryPredict.utils import models
from InjuryPredict.Injurydata_prepare import load_processed_subset
from InjuryPredict.utils.loss import InjuryKendallMultiTaskLoss
from InjuryPredict.utils.tools import build_single_metric_trackers, round_float_fields, convert_numpy_types
from InjuryPredict.utils.tools import get_regression_metrics, get_classification_metrics, get_mais_3c_metrics, MAIS_3C_DISPLAY_LABELS, plot_scatter, plot_confusion_matrix
from InjuryPredict.config import RUNS_DIR, curriculum_params, training_params, loss_params, model_params, model_selection_params, kfold_params
from InjuryPredict.utils.training import (
    build_injury_optimizer,
    compute_output_consistency_weights,
    log_injury_tensorboard_metrics,
    get_early_stop_anchor_epoch,
    get_curriculum_state,
    get_metric_selection_start_epoch,
    run_one_epoch,
    save_bn_recalibrated_state,
    should_stop_early,
    validate_curriculum_params,
)

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
            (_batch_x_acc_gt, batch_x_acc_pred, batch_x_att_continuous, batch_x_att_discrete,
             batch_y_HIC, batch_y_Dmax, batch_y_Nij,
             batch_ais_head, batch_ais_chest, batch_ais_neck, batch_y_MAIS, 
             batch_OT) = [d.to(device) for d in batch]
            
            # K-Fold 评估始终采用预测波形源，以匹配真实部署域。
            batch_pred, _, _ = model(batch_x_acc_pred, batch_x_att_continuous, batch_x_att_discrete) # [B, 2, L], [B, C], [B, D] -> [B, 3]

            # 收集回归和分类的标签
            batch_y_true = torch.stack([batch_y_HIC, batch_y_Dmax, batch_y_Nij], dim=1) # [B], [B], [B] -> [B, 3]
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

def evaluate_and_plot_for_metric(model, model_path, val_loader_k, device, fold, metric_name, fold_run_dir):
    """
    加载指定指标的最佳模型，执行评估并绘制图表。
    
    返回:
        eval_results (dict): 包含该指标对应模型的详细评估结果。
    """
    if not os.path.exists(model_path):
        print(f"  警告: 未找到 {model_path}，跳过该指标的评估。")
        return None
    
    # 加载模型权重
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    
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
    cls_metrics_head = get_classification_metrics(ground_truths['ais_head'], ais_head_pred, list(range(6)), context_hint="the fold data")
    cls_metrics_chest = get_classification_metrics(ground_truths['ais_chest'], ais_chest_pred, list(range(6)), context_hint="the fold data")
    cls_metrics_neck = get_classification_metrics(ground_truths['ais_neck'], ais_neck_pred, list(range(6)), context_hint="the fold data")
    cls_metrics_mais = get_classification_metrics(ground_truths['mais'], mais_pred, list(range(6)), context_hint="the fold data")
    cls_metrics_mais_3c = get_mais_3c_metrics(ground_truths['mais'], mais_pred, context_hint="the fold data")
    
    # 计算回归指标
    reg_metrics_hic = get_regression_metrics(true_hic, pred_hic)
    reg_metrics_dmax = get_regression_metrics(true_dmax, pred_dmax)
    reg_metrics_nij = get_regression_metrics(true_nij, pred_nij)
    
    # 创建该指标专属的子目录（与最佳权重文件名严格对应，避免歧义）
    weight_stem = Path(model_path).stem
    metric_plot_dir = os.path.join(fold_run_dir, f"eval_{weight_stem}")
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
    plot_confusion_matrix(cls_metrics_mais_3c['conf_matrix'], MAIS_3C_DISPLAY_LABELS,
                          f'Fold {fold+1} (Best {metric_name}) - CM MAIS 3C',
                          os.path.join(metric_plot_dir, "cm_mais_3c.png"))
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
        'accu_mais_3c': cls_metrics_mais_3c['accuracy'],
        'accu_head': cls_metrics_head['accuracy'],
        'accu_chest': cls_metrics_chest['accuracy'],
        'accu_neck': cls_metrics_neck['accuracy'],
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
    
    print(f"    Fold {fold+1} (Best val/{metric_name}) 评估完成，图表已保存至 {metric_plot_dir}")
    return eval_results

if __name__ == "__main__":
    set_random_seed() # 设置全局随机种子
    ############################################################################################
    ############################################################################################
    # ---- 从导入的配置中加载超参数 ----
    # 1. 优化与训练相关
    Epochs = int(training_params['Epochs'])
    Batch_size = int(training_params['Batch_size'])
    Learning_rate = float(training_params['Learning_rate'])
    Learning_rate_min = float(training_params['Learning_rate_min'])
    weight_decay = float(training_params['weight_decay'])
    early_stop_start_epochs = int(training_params['early_stop_start_epochs'])
    Patience = min(int(training_params['Patience']), Epochs)
    
    phase_epochs = validate_curriculum_params(Epochs, curriculum_params)
    metric_selection_start_epoch = get_metric_selection_start_epoch(phase_epochs)
    early_stop_anchor_epoch = get_early_stop_anchor_epoch(early_stop_start_epochs, phase_epochs)

    # K-Fold 设置
    K = int(kfold_params['K'])
    
    # 构建指标跟踪器
    # K-Fold 汇总逻辑以单指标 best 权重为统计单元，因此这里复用 model_selection_params 中的 single_metric_trackers。
    # metric_trackers 是所有 fold 共用的静态规则表。key 为 run_one_epoch 返回的验证指标名；value 包含 compare_indicator、initial_value、is_better、model_filename、display_name。
    metric_trackers = build_single_metric_trackers(model_selection_params.get("single_metric_trackers", []))
    if not metric_trackers:
        raise ValueError("model_selection_params.single_metric_trackers 不能为空。")
    tracked_metric_names = [tracker['display_name'] for tracker in metric_trackers.values()]
    print(f"将跟踪以下验证指标: {tracked_metric_names}")
    if metric_selection_start_epoch > 1:
        print(f"最佳模型权重将只从 Phase III 开始保存，首个候选 epoch={metric_selection_start_epoch}。")
    print(f"早停 patience 将从 epoch={early_stop_anchor_epoch} 之后开始重新计数。")

    ############################################################################################
    ############################################################################################
    
    # --- 2. 创建本次 K-Fold 运行的主目录 ---
    current_time = datetime.now().strftime("%m%d%H%M")
    main_run_dir = os.path.join(RUNS_DIR, f"InjuryPredictModel_KFold_{current_time}")
    os.makedirs(main_run_dir, exist_ok=True)
    print(f"K-Fold 主运行目录: {main_run_dir}")

    # --- 3. 加载由 Injurydata_prepare.py 生成的数据 ---
    print("正在加载 pt dataset ...")
    train_pt = get_injury_processed_dataset_path("train")
    val_pt = get_injury_processed_dataset_path("val")
    test_pt = get_injury_processed_dataset_path("test")
    missing_pt_files = [path for path in (train_pt, val_pt, test_pt) if not path.exists()]
    if missing_pt_files:
        raise FileNotFoundError(
            f"缺少 K-Fold 所需数据文件: {missing_pt_files}。"
            "请先运行: python -m InjuryPredict.Injurydata_prepare"
        )
    train_subset_orig = load_processed_subset(train_pt)
    val_subset_orig = load_processed_subset(val_pt)
    test_subset_orig = load_processed_subset(test_pt)
        
    # 各个 Subset 共享同一个底层 processed dataset 实例。
    full_processed_dataset = train_subset_orig.dataset
    required_processed_fields = ("x_acc_gt", "x_acc_pred", "x_att_continuous", "x_att_discrete")
    missing_processed_fields = [
        name for name in required_processed_fields
        if getattr(full_processed_dataset, name, None) is None
    ]
    if missing_processed_fields:
        raise RuntimeError(
            f"processed dataset 缺少字段 {missing_processed_fields}。"
            "请重新运行 `python -m InjuryPredict.Injurydata_prepare --overwrite`。"
        )
    if getattr(full_processed_dataset, "pulse_prediction_meta", None) is None:
        raise RuntimeError("processed dataset 缺少 PulsePredict 波形源记录，请使用新版 Injurydata_prepare 重新生成。")
    
    # K-Fold 需要在同一个已处理数据源内重新划分样本，因此这里合并 train/val/test 的原始索引；普通 train.py 仍保持固定 train/val 划分。
    combined_indices = np.concatenate([train_subset_orig.indices, val_subset_orig.indices, test_subset_orig.indices])
    
    # 获取用于【分层】的标签 (从底层数据集中按合并后的索引提取)
    combined_labels = full_processed_dataset.mais[combined_indices]
    
    print(f"已加载并合并数据用于 K-Fold。总样本数: {len(combined_indices)}")
    
    # 获取模型所需的 num_classes_of_discrete
    num_classes_of_discrete = full_processed_dataset.num_classes_of_discrete

    # --- 预先实例化模型以获取参数量 ---
    print("正在计算模型参数量...")
    dummy_model = models.InjuryPredictModel(
        num_classes_of_discrete=num_classes_of_discrete,
        **model_params,
    )
    total_params = sum(p.numel() for p in dummy_model.parameters())
    trainable_params = sum(p.numel() for p in dummy_model.parameters() if p.requires_grad)
    print(dummy_model)
    print(f"模型总参数量: {total_params}, 可训练参数量: {trainable_params}")
    del dummy_model # 释放内存    

    # --- 4. 初始化 KFold ---
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=GLOBAL_SEED)
    
    # --- 5. 存储每一折的最佳验证指标 (按指标分组) ---
    # all_folds_results 是跨 fold 的结果汇总容器。key 与 metric_trackers 一致；value 中 best_values/best_epochs 保存各 fold 的最优跟踪值和对应 epoch，eval_results 保存对应权重的详细评估指标。
    all_folds_results = {metric_name: {'best_values': [], 'best_epochs': [], 'eval_results': []} 
                         for metric_name in metric_trackers.keys()}
    kfold_bn_recalibration = []
    per_fold_training_records = []

    # --- 初始保存 K-Fold 配置 ---
    results_path = os.path.join(main_run_dir, "TrainingRecord.json")
    initial_kfold_record = {
        "model_type": "InjuryPredictModel",
        "model_params_count": {
            "total_params": total_params,
            "trainable_params": trainable_params
        },
        "kfold_info": {
            "INJURY_PROCESSED_DIR": str(INJURY_PROCESSED_DIR),
            "default_entrypoint_rule": (
                "train_KFold.py 在未显式传入 processed_dir 时，默认通过 "
                "common.settings.INJURY_PROCESSED_DIR 读取 processed .pt 子集。"
            ),
            "waveform_fields": {
                "x_acc_gt": "共享归一化空间中的真值 XY 波形。",
                "x_acc_pred": "冻结 PulsePredict 输出的共享归一化 XY 预测波形，验证与部署默认使用该字段。",
            },
            "pulse_prediction": getattr(full_processed_dataset, "pulse_prediction_meta", None),
            "total_samples_for_kfold": len(combined_indices),
            "k_value": K,
            "single_metric_trackers": model_selection_params.get("single_metric_trackers", []),
        },
        "hyperparameters": { # 记录使用的超参数
             "training": {
                **training_params,
                "Patience": Patience,
            },
            "loss": loss_params,
            "curriculum": curriculum_params,
            "model_selection_params": model_selection_params,
            "lr_scheduler": {
                "type": "CosineAnnealingLR",
                "T_max": Epochs,
                "eta_min": Learning_rate_min,
                "restart_on_phase_change": False,
            },
            "model_selection": {
                "phase3_only_best_model": metric_selection_start_epoch > 1,
                "best_model_start_epoch": metric_selection_start_epoch,
                "early_stop_anchor_epoch": early_stop_anchor_epoch,
            },
            "model": {
                **model_params,
                "num_classes_of_discrete": num_classes_of_discrete,
            }
        }
    }
    initial_kfold_record = convert_numpy_types(initial_kfold_record)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(initial_kfold_record, f, ensure_ascii=False, indent=4)
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
            num_classes_of_discrete=num_classes_of_discrete,
            **model_params,
        ).to(device)

        # 仅在第一折打印模型结构和逐层参数量，便于核查参数分布。
        if fold == 0:
            print("\n模型结构:")
            print(model)
            print("\n模型各层参数量:")
            for name, param in model.named_parameters():
                print(f"  {name}: {param.numel()} parameters")
            print(f"\n模型参数量统计:")
            print(f"模型总参数量: {total_params}, 可训练参数量: {trainable_params}")
            
        
        # 定义损失函数。Kendall log_vars 属于 criterion 参数，由优化器单独以 0 weight_decay 管理。
        criterion = InjuryKendallMultiTaskLoss(
            base_loss=loss_params['base_loss'],
            weight_factor_classify=loss_params['weight_factor_classify'],
            weight_factor_sample=loss_params['weight_factor_sample'],
            task_prior_weights=loss_params['task_prior_weights'],
        ).to(device)
        output_consistency_weights = compute_output_consistency_weights(train_subset_k, device)
        optimizer = build_injury_optimizer(model, criterion, Learning_rate, weight_decay)
        # 学习率调度与课程阶段解耦：每个 fold 内只创建一次优化器和余弦调度器，避免 Phase III 出现学习率重启。
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Epochs, eta_min=Learning_rate_min)
        # BN 重校准只用当前 fold 训练集的预测波形源顺序前向刷新 BatchNorm buffer
        bn_loader_k = DataLoader(train_subset_k, batch_size=Batch_size, shuffle=False, num_workers=0)

        # --- 6.5 初始化当前 Fold 的跟踪变量 ---
        # fold_metric_states 是当前 fold 的动态状态表。key 与 metric_trackers 一致；value 包含当前 fold 的 best_value、1-based best_epoch、判优函数 is_better 和对应 model_filename。
        fold_metric_states = {}
        for metric_name, tracker_info in metric_trackers.items():
            fold_metric_states[metric_name] = {
                'best_value': tracker_info['initial_value'],
                'best_epoch': 0,
                'is_better': tracker_info['is_better'],
                'model_filename': tracker_info['model_filename']
            }
        
        # --- 6.6 Epoch 训练循环 (内层循环) ---
        for epoch in range(Epochs):
            epoch_start_time = time.time()

            curriculum_state = get_curriculum_state(epoch, phase_epochs, curriculum_params)

            train_metrics = run_one_epoch(
                model,
                train_loader_k,
                criterion,
                device,
                optimizer=optimizer,
                curriculum_state=curriculum_state,
                output_consistency_weights=output_consistency_weights,
            )

            # 验证阶段不参与课程调度，固定使用预测波形源计算 Kendall 主任务损失，以保证各阶段 val/loss 可直接比较。
            val_metrics = run_one_epoch(model, val_loader_k, criterion, device, optimizer=None)
            missing_metrics = [name for name in metric_trackers.keys() if name not in val_metrics]
            if missing_metrics:
                raise KeyError(f"model_selection_params.single_metric_trackers 中存在无效验证指标: {missing_metrics}")

            # 打印当前 Fold 的 Epoch 信息
            metric_strs = [
                f"{metric_trackers[name]['display_name']}: {val_metrics[name]:.4g}"
                for name in metric_trackers.keys() if name != 'loss'
            ]
            loss_str = f"Val Loss: {val_metrics['loss']:.4g}"
            print(
                f"  Epoch {epoch+1}/{Epochs} | Phase: {curriculum_state['phase']} "
                f"| alpha={curriculum_state['alpha']:.4g} | Train Loss: {train_metrics['loss']:.4g} "
                f"| {loss_str} | {' | '.join(metric_strs)} | Time: {time.time()-epoch_start_time:.2f}s"
            )
            
            # TensorBoard 仅按 Train/Val 与 Loss/InjuryMetrics 两级语义分组，避免入口脚本各自维护不一致的 tag。
            log_injury_tensorboard_metrics(
                writer,
                "Train",
                train_metrics,
                epoch,
                criterion_weight_state=criterion.get_weight_state(),
                optimizer=optimizer,
            )
            # Val/Loss/total_loss 表示预测波形源上的主任务损失，不包含课程学习的一致性正则项。
            log_injury_tensorboard_metrics(writer, "Val", val_metrics, epoch)
            scheduler.step()

            # --- 跟踪当前 Fold 的最佳模型 (为每个指标) ---
            current_epoch = epoch + 1
            if current_epoch >= metric_selection_start_epoch:
                for metric_name, state in fold_metric_states.items():
                    current_value = val_metrics[metric_name]
                    # metric_name 是 val_metrics 字典中的真实 key；fold_metric_states[metric_name] 保存当前 fold 内该指标已达到的最优值和对应 epoch。
                    if state['is_better'](current_value, state['best_value']):
                        state['best_value'] = current_value
                        state['best_epoch'] = current_epoch

                        # 保存当前指标的最佳模型权重
                        torch.save(model.state_dict(), os.path.join(fold_run_dir, state['model_filename']))
                        print(f"    [Fold {fold+1}] Best {metric_trackers[metric_name]['display_name']} model saved: {current_value:.4g} at epoch {current_epoch}")

            # --- 早停逻辑 (检查所有跟踪的指标) ---
            if should_stop_early(fold_metric_states, current_epoch, Patience, early_stop_anchor_epoch):
                print(f"    Early Stop at epoch {current_epoch} for Fold {fold+1}!")
                for metric_name, state in fold_metric_states.items():
                    print(f"    Best {metric_trackers[metric_name]['display_name']}: {state['best_value']:.4g} (at epoch {state['best_epoch']})")
                break

        # 每个 fold 与普通 train.py 一样保存训练结束时的 final_model.pth；K-Fold 汇总仍以 best_val_* 权重为主。
        final_model_path = os.path.join(fold_run_dir, "final_model.pth")
        torch.save(model.state_dict(), final_model_path)
        print(f"    [Fold {fold+1}] Final model saved: {final_model_path}")

        # --- 6.7 当前 Fold 训练结束，为每个指标执行详细评估 ---
        print(f"  Fold {fold+1} 训练完成。正在为每个跟踪指标执行详细评估...")

        fold_bn_candidate_paths = {final_model_path: "final_model"}
        for metric_name, state in fold_metric_states.items():
            model_path = os.path.join(fold_run_dir, state['model_filename'])
            fold_bn_candidate_paths[model_path] = metric_name
            eval_results = evaluate_and_plot_for_metric(
                model, model_path, val_loader_k, device, fold, metric_name, fold_run_dir
            )
            
            # 记录结果
            all_folds_results[metric_name]['best_values'].append(float(state['best_value']))
            all_folds_results[metric_name]['best_epochs'].append(state['best_epoch'])
            if eval_results:
                all_folds_results[metric_name]['eval_results'].append(eval_results)

        if curriculum_params.get("bn_recalibration", True):
            # 对每个 fold 的 final_model 和已保存 best_val_* 权重分别派生重校准权重；K-Fold 汇总指标仍来自原始 best_val_* 评估。
            for model_path, source_weight_label in sorted(fold_bn_candidate_paths.items()):
                if not os.path.exists(model_path):
                    continue
                recal_result = save_bn_recalibrated_state(model, model_path, bn_loader_k, device)
                recal_result["fold"] = fold + 1
                recal_result["source_weight_label"] = source_weight_label
                kfold_bn_recalibration.append(recal_result)
                print(f"    [Fold {fold+1}] BN recalibrated model saved: {recal_result['recalibrated']}")

        writer.close()
        per_fold_training_records.append({
            "fold": fold + 1,
            "final_model": final_model_path,
            "output_consistency_weights": output_consistency_weights.detach().cpu().numpy().reshape(-1).tolist(), # [1, 3] -> [3]
            "kendall_weight_state_final": criterion.get_weight_state(),
        })
        print(f"Fold {fold+1} finished in {time.time() - fold_start_time:.2f}s.")

    # --- 7. K-Fold 循环结束，计算并打印总体结果 ---
    print("\n" + "="*60)
    print("         K-Fold Cross-Validation Summary")
    print("="*60)
    
    kfold_summary = {}
    
    for metric_name in metric_trackers.keys():
        print(f"\n--- Results for Best '{metric_trackers[metric_name]['display_name']}' Model ---")
        
        eval_df = pd.DataFrame(all_folds_results[metric_name]['eval_results'])
        best_epochs = all_folds_results[metric_name]['best_epochs']
        
        # 计算主要指标的均值和标准差
        summary_for_metric = {
            'mean_best_epoch': np.mean(best_epochs),
            'mean_best_value': np.mean(all_folds_results[metric_name]['best_values']),
            'std_best_value': np.std(all_folds_results[metric_name]['best_values'], ddof=1) if len(all_folds_results[metric_name]['best_values']) > 1 else 0.0,
        }
        
        # 使用 eval_results 作为唯一性能统计来源，避免与前文重复字段
        if not eval_df.empty:
            for col in eval_df.columns:
                summary_for_metric[f'mean_{col}'] = eval_df[col].mean()
                summary_for_metric[f'std_{col}'] = eval_df[col].std()
        
        kfold_summary[metric_name] = summary_for_metric
        
        print(f"  Average Best Epoch: {summary_for_metric['mean_best_epoch']:.4g}")
        print(f"  Tracker Best Value : {summary_for_metric['mean_best_value']:.4g} +/- {summary_for_metric['std_best_value']:.4g}")
        if 'mean_accu_mais' in summary_for_metric:
            print(f"  val/accu_mais : {summary_for_metric['mean_accu_mais']:.4g}% +/- {summary_for_metric['std_accu_mais']:.4g}%")
            print(f"  val/accu_mais_3c: {summary_for_metric['mean_accu_mais_3c']:.4g}% +/- {summary_for_metric['std_accu_mais_3c']:.4g}%")
            print(f"  val/accu_head : {summary_for_metric['mean_accu_head']:.4g}% +/- {summary_for_metric['std_accu_head']:.4g}%")
            print(f"  val/accu_chest: {summary_for_metric['mean_accu_chest']:.4g}% +/- {summary_for_metric['std_accu_chest']:.4g}%")
            print(f"  val/accu_neck : {summary_for_metric['mean_accu_neck']:.4g}% +/- {summary_for_metric['std_accu_neck']:.4g}%")
            print(f"  val/mae_hic   : {summary_for_metric['mean_mae_hic']:.4g} +/- {summary_for_metric['std_mae_hic']:.4g}")
            print(f"  val/mae_dmax  : {summary_for_metric['mean_mae_dmax']:.4g} +/- {summary_for_metric['std_mae_dmax']:.4g}")
            print(f"  val/mae_nij   : {summary_for_metric['mean_mae_nij']:.4g} +/- {summary_for_metric['std_mae_nij']:.4g}")
            print(f"  val/r2_hic    : {summary_for_metric['mean_r2_hic']:.4g} +/- {summary_for_metric['std_r2_hic']:.4g}")
            print(f"  val/r2_dmax   : {summary_for_metric['mean_r2_dmax']:.4g} +/- {summary_for_metric['std_r2_dmax']:.4g}")
            print(f"  val/r2_nij    : {summary_for_metric['mean_r2_nij']:.4g} +/- {summary_for_metric['std_r2_nij']:.4g}")

    
    print("="*60)
    
    # --- 8. 保存 K-Fold 总体结果 ---
    print("K-Fold 训练完成，正在加载初始记录并添加总结...")

    try:
        with open(results_path, "r", encoding="utf-8") as f:
            final_kfold_record = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"警告: 未找到或无法解析 {results_path}。将创建一个新的记录文件。")
        final_kfold_record = initial_kfold_record

    # 添加按指标分组的结果
    final_kfold_record["kfold_summary_by_metric"] = round_float_fields(convert_numpy_types(kfold_summary), digits=4)
    # best_metrics_by_tracker 是按静态 tracker 分组的跨 fold 最优记录摘要。key 与 metric_trackers 一致；value 只记录各 fold 的 best_values、best_epochs 和权重文件模式，不再参与训练过程更新。
    final_kfold_record["best_metrics_by_tracker"] = round_float_fields(convert_numpy_types({
        metric_name: {
            'best_values': data['best_values'],
            'best_epochs': data['best_epochs'],
            'model_file_pattern': f"Fold_x/{metric_trackers[metric_name]['model_filename']}"
        }
        for metric_name, data in all_folds_results.items()
    }), digits=4)
    final_kfold_record["per_fold_eval_results_by_metric"] = round_float_fields(convert_numpy_types({
        metric_name: {
            'eval_results': data['eval_results']
        }
        for metric_name, data in all_folds_results.items()
    }), digits=4)
    final_kfold_record["bn_recalibration"] = convert_numpy_types(kfold_bn_recalibration)
    final_kfold_record["per_fold_training_records"] = convert_numpy_types(per_fold_training_records)

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(final_kfold_record, f, ensure_ascii=False, indent=4)
        
    print(f"\nK-Fold 总体结果已更新至: {results_path}")
