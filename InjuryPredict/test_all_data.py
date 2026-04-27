# -*- coding: utf-8 -*-
"""
加载一个已训练好的模型，在【完整】的数据集上运行预测，
并将预测结果、真实标签、误差、数据集划分（train/valid/test）、
以及原始的13个标量工况特征合并到一个CSV文件中，保存到该模型的 run 目录下。

同时，在命令行打印各子集上的核心性能指标（MAE 和 Accuracy）。
"""

import warnings
warnings.filterwarnings('ignore')
import os
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader

from common.settings import FEATURE_ORDER, INJURY_PROCESSED_DIR, RAW_DATA, get_injury_processed_dataset_path
from common.metrics.injury_risk import AIS_cal_head, AIS_cal_chest, AIS_cal_neck
from common.tools.seeding import set_random_seed

from InjuryPredict.utils import models
from InjuryPredict.Injurydata_prepare import load_processed_subset
from InjuryPredict.config import RUNS_DIR
from InjuryPredict.utils.tools import get_mais_3c_metrics

# --- 1. 配置区：请在此处设置您的路径 ---

# 1.1) 要评估的模型所在的运行目录
RUN_DIR = Path(RUNS_DIR) / "InjuryPredictModel_04241327"

# 1.2) 要加载的模型权重文件名（相对于 RUN_DIR）
FOLD = Path("") # 可以指定K折中的某一折，例如 Path("Fold_1")，如果没有K折划分则保持 Path("")
WEIGHT_FILE = FOLD / Path("best_val_loss.pth")

# --- 结束配置 ---

def make_weight_output_label(weight_file: Path) -> str:
    """将权重相对路径转为稳定的导出文件标签，避免不同子目录下的同名权重覆盖结果。"""
    return "_".join(Path(weight_file).with_suffix("").parts)

def load_original_features(raw_packed_path: str) -> pd.DataFrame:
    """从 raw_packed.npz 恢复原始标量特征与 case 映射。"""
    print(f"正在从 {raw_packed_path} 加载原始标量特征...")
    packed = np.load(raw_packed_path)

    required_keys = {"case_ids", "pulse_source_case_ids", "x_att_raw"}
    missing_keys = sorted(required_keys - set(packed.files))
    if missing_keys:
        raise KeyError(f"raw_packed 文件缺少以下必需键: {missing_keys}")

    x_att_raw = np.asarray(packed["x_att_raw"], dtype=np.float32)
    if x_att_raw.ndim != 2 or x_att_raw.shape[1] != len(FEATURE_ORDER):
        raise ValueError(
            f"x_att_raw 形状异常: {x_att_raw.shape}, 期望 (N, {len(FEATURE_ORDER)})"
        )

    original_features_df = pd.DataFrame(x_att_raw, columns=FEATURE_ORDER)
    original_features_df.insert(0, "pulse_source_case_id", packed["pulse_source_case_ids"].astype(np.int64))
    original_features_df.insert(0, "case_id", packed["case_ids"].astype(np.int64))

    # 两个离散特征在 raw_packed 中以数值数组存储，这里恢复为整数列，便于导出结果直接复用。
    original_features_df["is_driver_side"] = original_features_df["is_driver_side"].round().astype(np.int64)
    original_features_df["OT"] = original_features_df["OT"].round().astype(np.int64)

    return original_features_df

def load_model_and_data(run_dir, weight_file):
    """加载模型、完整的数据集对象以及数据集划分的 case_id 映射"""
    print(f"正在加载模型: {os.path.join(run_dir, weight_file)}")
    
    # 1. 加载模型超参数
    record_path = os.path.join(run_dir, "TrainingRecord.json")
    if not os.path.exists(record_path):
        raise FileNotFoundError(f"未找到 TrainingRecord.json 文件于: {run_dir}")
        
    with open(record_path, "r", encoding="utf-8") as f:
        training_record = json.load(f)
    model_params = training_record["hyperparameters"]["model"]
    
    # 2. 加载数据集 .pt 文件
    print(f"评估数据集来源路径: {INJURY_PROCESSED_DIR}")
    train_pt_path = get_injury_processed_dataset_path("train").as_posix()
    val_pt_path = get_injury_processed_dataset_path("val").as_posix()
    test_pt_path = get_injury_processed_dataset_path("test").as_posix()
    
    if not all(os.path.exists(p) for p in [train_pt_path, val_pt_path, test_pt_path]):
        raise FileNotFoundError(f"未在 {INJURY_PROCESSED_DIR.as_posix()} 中找到 train/val/test_dataset.pt。请先运行: python -m InjuryPredict.Injurydata_prepare 来生成数据集文件。")
        
    train_subset = load_processed_subset(train_pt_path)
    val_subset = load_processed_subset(val_pt_path)
    test_subset = load_processed_subset(test_pt_path)
    
    # 拼接 Subset 作为完整的数据集
    full_dataset = torch.utils.data.ConcatDataset([train_subset, val_subset, test_subset])
    all_case_ids = train_subset.dataset.case_ids # 获取所有 case_id 的顺序
    
    print(f"成功加载完整数据集，共 {len(full_dataset)} 个样本。")
    
    # 3. 创建 case_id 到数据集类型的映射
    train_ids = set(all_case_ids[train_subset.indices])
    valid_ids = set(all_case_ids[val_subset.indices])
    test_ids = set(all_case_ids[test_subset.indices])
    
    case_id_map = {}
    for case_id in all_case_ids:
        if case_id in train_ids:
            case_id_map[case_id] = 'train'
        elif case_id in valid_ids:
            case_id_map[case_id] = 'valid'
        elif case_id in test_ids:
            case_id_map[case_id] = 'test'
        else:
            case_id_map[case_id] = 'unassigned' # 理论上不应发生

    # 4. 实例化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = models.InjuryPredictModel(**model_params).to(device)

    # 5. 加载权重
    weight_path = os.path.join(run_dir, weight_file)
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"未找到模型权重文件: {weight_path}")
    model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=False))
    model.eval()
    
    return model, full_dataset, device, case_id_map

def run_inference(model, dataset, device):
    """在完整数据集上运行推理"""
    
    # DataLoader 直接加载通过 Subset 保存的底层 InjuryPackedDataset（shuffle=False 保证顺序）
    data_loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)
    
    all_preds_list = []
    
    print("开始在完整数据集上运行模型推理（仅使用预测波形源）...")
    with torch.no_grad():
        for batch in data_loader:
            # 从底层 Dataset 的 __getitem__ 解包
            (_batch_x_acc_gt, batch_x_acc_pred, batch_x_att_continuous, batch_x_att_discrete,
             _batch_y_HIC, _batch_y_Dmax, _batch_y_Nij,
             _batch_ais_head, _batch_ais_chest, _batch_ais_neck, _batch_y_MAIS, _batch_OT) = [d.to(device) for d in batch]

            # 完整数据集导出默认使用预测波形源，避免把真值波形评估结果误认为部署域表现。
            batch_pred, _, _ = model(batch_x_acc_pred, batch_x_att_continuous, batch_x_att_discrete) # [B, 2, L], [B, C], [B, D] -> [B, 3]
            
            all_preds_list.append(batch_pred.cpu().numpy())
            
    # 合并所有批次的预测结果
    predictions_np = np.concatenate(all_preds_list, axis=0) # 形状 (N, 3)
    
    # 验证预测数量是否与数据集大小一致
    assert len(predictions_np) == len(dataset), \
        f"预测数量 ({len(predictions_np)}) 与数据集大小 ({len(dataset)}) 不匹配!"
        
    print("推理完成。")
    return predictions_np

def create_results_dataframe(dataset, predictions_np, original_features_df, case_id_map):
    """合并所有数据到 DataFrame"""
    
    print("正在创建和合并结果 DataFrame...")
    
    # 1. 从数据集中提取 case_id 和真实标签
    # 支持两种输入类型：
    # - 单一底层 InjuryPackedDataset（具有 .case_ids 等属性）
    # - ConcatDataset([...])（通常由 train/val/test 的 Subset 拼接而成）
    if hasattr(dataset, 'datasets'):
        # dataset 是 ConcatDataset：按传入的子集顺序拼接底层数据
        parts_case, parts_hic, parts_dmax, parts_nij = [], [], [], []
        parts_ais_h, parts_ais_c, parts_ais_n, parts_mais, parts_ot = [], [], [], [], []
        for sub in dataset.datasets:
            # 每个 sub 可能是 Subset（常见）或直接为底层 Dataset
            if isinstance(sub, torch.utils.data.Subset):
                base = sub.dataset
                idxs = np.asarray(sub.indices, dtype=np.int64)
            else:
                base = sub
                idxs = np.arange(len(base), dtype=np.int64)

            parts_case.append(base.case_ids[idxs])
            parts_hic.append(base.y_HIC[idxs])
            parts_dmax.append(base.y_Dmax[idxs])
            parts_nij.append(base.y_Nij[idxs])
            parts_ais_h.append(base.ais_head[idxs])
            parts_ais_c.append(base.ais_chest[idxs])
            parts_ais_n.append(base.ais_neck[idxs])
            parts_mais.append(base.mais[idxs])
            parts_ot.append(base.OT_raw[idxs])

        case_ids = np.concatenate(parts_case)
        hic_true = np.concatenate(parts_hic)
        dmax_true = np.concatenate(parts_dmax)
        nij_true = np.concatenate(parts_nij)
        ais_head_true = np.concatenate(parts_ais_h)
        ais_chest_true = np.concatenate(parts_ais_c)
        ais_neck_true = np.concatenate(parts_ais_n)
        mais_true = np.concatenate(parts_mais)
        ot_raw = np.concatenate(parts_ot)
    else:
        # 直接使用单个 Dataset 的属性
        case_ids = np.asarray(dataset.case_ids)
        hic_true = np.asarray(dataset.y_HIC)
        dmax_true = np.asarray(dataset.y_Dmax)
        nij_true = np.asarray(dataset.y_Nij)
        ais_head_true = np.asarray(dataset.ais_head)
        ais_chest_true = np.asarray(dataset.ais_chest)
        ais_neck_true = np.asarray(dataset.ais_neck)
        mais_true = np.asarray(dataset.mais)
        ot_raw = np.asarray(dataset.OT_raw)

    # 构建基础 DataFrame（顺序与 predictions_np 一致）
    results_df = pd.DataFrame({
        'case_id': case_ids,
        'HIC_true': hic_true,
        'Dmax_true': dmax_true,
        'Nij_true': nij_true,
        'AIS_head_true_raw': ais_head_true,
        'AIS_chest_true_raw': ais_chest_true,
        'AIS_neck_true_raw': ais_neck_true,
        'MAIS_true_raw': mais_true,
    })
    
    # 2. 添加模型预测值
    results_df['HIC_pred'] = predictions_np[:, 0]
    results_df['Dmax_pred'] = predictions_np[:, 1]
    results_df['Nij_pred'] = predictions_np[:, 2]
    
    # 3. 计算预测的AIS等级 (确保返回整数类型)
    results_df['AIS_head_pred'] = AIS_cal_head(results_df['HIC_pred']).astype(int)
    # 使用拼接得到的 OT 值进行胸部 AIS 计算
    results_df['AIS_chest_pred'] = AIS_cal_chest(results_df['Dmax_pred'], ot_raw).astype(int)
    results_df['AIS_neck_pred'] = AIS_cal_neck(results_df['Nij_pred']).astype(int)
    
    # 4. 计算预测的 MAIS 等级 (确保整数)
    results_df['MAIS_pred'] = np.maximum.reduce([
        results_df['AIS_head_pred'], 
        results_df['AIS_chest_pred'], 
        results_df['AIS_neck_pred']
    ]).astype(int)
    
    # 5. 计算误差 (diff = pred - true)
    results_df['HIC_diff'] = results_df['HIC_pred'] - results_df['HIC_true']
    results_df['Dmax_diff'] = results_df['Dmax_pred'] - results_df['Dmax_true']
    results_df['Nij_diff'] = results_df['Nij_pred'] - results_df['Nij_true']
    
    results_df['AIS_head_diff'] = results_df['AIS_head_pred'] - results_df['AIS_head_true_raw']
    results_df['AIS_chest_diff'] = results_df['AIS_chest_pred'] - results_df['AIS_chest_true_raw']
    results_df['AIS_neck_diff'] = results_df['AIS_neck_pred'] - results_df['AIS_neck_true_raw']
    results_df['MAIS_diff'] = results_df['MAIS_pred'] - results_df['MAIS_true_raw'] # MAIS 误差
    
    # 6. 添加数据集类型 (新)
    results_df['dataset_type'] = results_df['case_id'].map(case_id_map)

    # 6.5 增加一列，表示是否三个部位等级全都预测正确
    results_df['all_AIS_correct'] = (
        (results_df['AIS_head_true_raw'] == results_df['AIS_head_pred']) &
        (results_df['AIS_chest_true_raw'] == results_df['AIS_chest_pred']) &
        (results_df['AIS_neck_true_raw'] == results_df['AIS_neck_pred'])
    ).astype(int)  # 1表示全对，0表示有错

    # 7. 合并原始的标量特征
    final_df = pd.merge(results_df, original_features_df, on='case_id', how='left')
    
    # 8. 调整列顺序以满足导出分析需要
    original_feature_names = list(original_features_df.columns.drop('case_id'))
    
    ordered_columns = [
        'case_id',
        'dataset_type',
        'all_AIS_correct',
        # MAIS
        'MAIS_true_raw', 'MAIS_pred', 'MAIS_diff',
        # 头部
        'HIC_true', 'HIC_pred', 
        'AIS_head_true_raw', 'AIS_head_pred', 
        'HIC_diff', 'AIS_head_diff',
        # 胸部
        'Dmax_true', 'Dmax_pred',
        'AIS_chest_true_raw', 'AIS_chest_pred',
        'Dmax_diff', 'AIS_chest_diff',
        # 颈部
        'Nij_true', 'Nij_pred',
        'AIS_neck_true_raw', 'AIS_neck_pred',
        'Nij_diff', 'AIS_neck_diff',
    ] + original_feature_names
    
    # 确保所有列都存在
    final_df = final_df[ordered_columns]
    
    print("DataFrame 创建完毕。")
    return final_df

def print_metrics_summary(df):
    """在命令行打印各子集的 MAE 和 Accuracy 摘要"""
    from sklearn.metrics import mean_absolute_error, accuracy_score
    
    print("\n" + "="*60)
    print("           模型在各子集上的性能摘要")
    print("="*60)
    
    # 定义要评估的子集
    subsets = {
        "Train": df[df['dataset_type'] == 'train'],
        "Valid": df[df['dataset_type'] == 'valid'],
        "Test": df[df['dataset_type'] == 'test'],
        "Valid+Test": df[df['dataset_type'].isin(['valid', 'test'])]
    }
    
    for name, subset_df in subsets.items():
        if len(subset_df) == 0:
            print(f"\n--- {name} Set Metrics (Size: 0) ---")
            print("  (跳过)")
            continue
            
        print(f"\n--- {name} Set Metrics (Size: {len(subset_df)}) ---")
        
        # 1. 计算 MAE (使用原始浮点数)
        try:
            mae_hic = mean_absolute_error(subset_df['HIC_true'], subset_df['HIC_pred'])
            mae_dmax = mean_absolute_error(subset_df['Dmax_true'], subset_df['Dmax_pred'])
            mae_nij = mean_absolute_error(subset_df['Nij_true'], subset_df['Nij_pred'])
            
            print(f"  [标量 MAE]")
            print(f"    HIC MAE : {mae_hic:.4f}")
            print(f"    Dmax MAE: {mae_dmax:.4f}")
            print(f"    Nij MAE : {mae_nij:.4f}")
        except Exception as e:
            print(f"  [标量 MAE] 计算出错: {e}")
        
        # 2. 计算 Accuracy (确保使用整数类型，并添加错误处理)
        try:
            # 确保所有AIS列都是整数类型
            ais_head_true = subset_df['AIS_head_true_raw'].astype(int).values
            ais_head_pred = subset_df['AIS_head_pred'].astype(int).values
            ais_chest_true = subset_df['AIS_chest_true_raw'].astype(int).values
            ais_chest_pred = subset_df['AIS_chest_pred'].astype(int).values
            ais_neck_true = subset_df['AIS_neck_true_raw'].astype(int).values
            ais_neck_pred = subset_df['AIS_neck_pred'].astype(int).values
            mais_true = subset_df['MAIS_true_raw'].astype(int).values
            mais_pred = subset_df['MAIS_pred'].astype(int).values
            mais_metrics_3c = get_mais_3c_metrics(mais_true, mais_pred, context_hint=f"the {name} subset")
            
            # 计算准确率
            acc_head = accuracy_score(ais_head_true, ais_head_pred) * 100
            acc_chest = accuracy_score(ais_chest_true, ais_chest_pred) * 100
            acc_neck = accuracy_score(ais_neck_true, ais_neck_pred) * 100
            acc_mais = accuracy_score(mais_true, mais_pred) * 100
            
            # 计算三个部位全对的准确率
            all_correct = subset_df['all_AIS_correct'].sum()
            all_correct_rate = (all_correct / len(subset_df)) * 100
            
            print(f"  [等级 Accuracy]")
            print(f"    AIS Head Acc : {acc_head:.2f}%")
            print(f"    AIS Chest Acc: {acc_chest:.2f}%")
            print(f"    AIS Neck Acc : {acc_neck:.2f}%")
            print(f"    MAIS Acc 6C  : {acc_mais:.2f}%")
            print(f"    MAIS Acc 3C  : {mais_metrics_3c['accuracy']:.2f}%")
            print(f"    All AIS Correct: {all_correct_rate:.2f}% ({all_correct}/{len(subset_df)})")
            
        except Exception as e:
            print(f"  [等级 Accuracy] 计算出错: {e}")
        
    print("="*60)

if __name__ == "__main__":
    set_random_seed()
    
    # 1. 从项目内 raw_packed 恢复原始13个标量特征
    original_features_df = load_original_features((RAW_DATA).as_posix()) # 从项目内 原始打包数据(.npz文件) 恢复每条样本的原始标量特征值与 pulse_source_case_id。
    
    # 2. 加载模型、完整数据集和 case_id 映射
    model, full_dataset, device, case_id_map = load_model_and_data(RUN_DIR, WEIGHT_FILE)
    
    # 3. 运行推理
    predictions_np = run_inference(model, full_dataset, device)
    
    # 4. 创建并合并结果
    final_results_df = create_results_dataframe(full_dataset, predictions_np, original_features_df, case_id_map)
    
    # 5. 保存到 CSV 文件
    weight_label = make_weight_output_label(WEIGHT_FILE)
    output_filename = f"full_dataset_predictions_{weight_label}.csv"
    output_path = RUN_DIR / output_filename
    
    final_results_df.to_csv(output_path, index=False, float_format='%.4f')
    
    print("\n" + "="*60)
    print("测试完成！")
    print(f"结果已保存至: {output_path}")
    print(f"总计处理 {len(final_results_df)} 条数据。")
    print("="*60)
    
    # 6. 打印性能摘要
    print_metrics_summary(final_results_df)
