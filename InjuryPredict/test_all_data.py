# -*- coding: utf-8 -*-
"""
加载一个已训练好的模型（教师或学生），在【完整】的数据集上运行预测，
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
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, accuracy_score

# 导入项目中的必要模块
from utils import models
# 导入 CrashDataset 即使不直接使用，torch.load 也需要它来反序列化
from utils.dataset_prepare import CrashDataset 
from utils.AIS_cal import AIS_cal_head, AIS_cal_chest, AIS_cal_neck
from utils.set_random_seed import set_random_seed

# --- 1. 配置区：请在此处设置您的路径 ---

# 1.1) 要评估的模型所在的运行目录
RUN_DIR = "./runs/InjuryPredictModel_01281059"  # 示例: "./runs/InjuryPredictModel_XXXXXXXX" 或 "./runs/StudentModel_XXXXXX"

# 1.2) 要加载的模型权重文件名
WEIGHT_FILE = "best_val_loss.pth"

# 1.3) 包含原始13个标量特征的 distribution 文件路径
DISTRIBUTION_FILE = r"E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\仿真数据库相关\distribution\distribution_0123.csv" 

# 1.4) 存放 .pt 数据集的目录
DATA_DIR = "./data"

# --- 结束配置 ---


def load_original_features(dist_file_path):
    """从 distribution 文件加载原始的标量特征"""
    print(f"正在从 {dist_file_path} 加载原始标量特征...")
    if dist_file_path.endswith('.csv'):
        dist_df = pd.read_csv(dist_file_path)
    else:
        # 支持 .npz 格式 (以防万一)
        dist_npz = np.load(dist_file_path, allow_pickle=True)
        dist_df = pd.DataFrame({key: dist_npz[key] for key in dist_npz.files})

    # 定义13个标量特征的列名
    # 连续特征 (0-10): impact_velocity, impact_angle, overlap, LL1, LL2, BTF, LLATTF, AFT, SP, SH, RA
    # 离散特征 (11-12): is_driver_side, OT
    feature_columns = [
        'case_id',
        'impact_velocity', 'impact_angle', 'overlap', 'LL1', 'LL2', 
        'BTF', 'LLATTF', 'AFT', 'SP', 'SH', 'RA', 
        'is_driver_side', 'OT',
    ]
    
    # 确保所有列都存在
    missing_cols = [col for col in feature_columns if col not in dist_df.columns]
    if missing_cols:
        raise ValueError(f"Distribution 文件中缺少以下必需列: {missing_cols}")
        
    original_features_df = dist_df[feature_columns]
    return original_features_df

def load_model_and_data(run_dir, weight_file, data_dir=DATA_DIR):
    """加载模型、完整的数据集对象以及数据集划分的 case_id 映射"""
    print(f"正在加载模型: {os.path.join(run_dir, weight_file)}")
    
    # 1. 加载模型超参数
    record_path = os.path.join(run_dir, "TrainingRecord.json")
    if not os.path.exists(record_path):
        raise FileNotFoundError(f"未找到 TrainingRecord.json 文件于: {run_dir}")
        
    with open(record_path, "r") as f:
        training_record = json.load(f)
    model_params = training_record["hyperparameters"]["model"]
    
    # 2. 加载数据集 .pt 文件
    train_pt_path = os.path.join(data_dir, "train_dataset.pt")
    val_pt_path = os.path.join(data_dir, "val_dataset.pt")
    test_pt_path = os.path.join(data_dir, "test_dataset.pt")
    
    if not all(os.path.exists(p) for p in [train_pt_path, val_pt_path, test_pt_path]):
        raise FileNotFoundError(f"未在 {data_dir} 中找到 train/val/test_dataset.pt。请先运行 utils/dataset_prepare.py。")
        
    train_subset = torch.load(train_pt_path)
    val_subset = torch.load(val_pt_path)
    test_subset = torch.load(test_pt_path)
    
    # 获取底层的、包含所有样本的 CrashDataset 实例
    full_dataset = train_subset.dataset
    all_case_ids = full_dataset.case_ids # 获取所有 case_id 的顺序
    
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
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    
    return model, full_dataset, device, case_id_map

def run_inference(model, dataset, device):
    """在完整数据集上运行推理"""
    
    # DataLoader 直接加载 CrashDataset 实例，以保证顺序
    data_loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)
    
    all_preds_list = []
    
    print("开始在完整数据集上运行模型推理...")
    with torch.no_grad():
        for batch in data_loader:
            # 从 CrashDataset 的 __getitem__ 解包
            (batch_x_acc, batch_x_att_continuous, batch_x_att_discrete,
             batch_y_HIC, batch_y_Dmax, batch_y_Nij,
             batch_ais_head, batch_ais_chest, batch_ais_neck, batch_y_MAIS, batch_OT) = [d.to(device) for d in batch]

            batch_pred, _, _ = model(batch_x_acc, batch_x_att_continuous, batch_x_att_discrete)
            
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
    # 由于 DataLoader(shuffle=False)，顺序是完全一致的
    results_df = pd.DataFrame({
        'case_id': dataset.case_ids,
        'HIC_true': dataset.y_HIC,
        'Dmax_true': dataset.y_Dmax,
        'Nij_true': dataset.y_Nij,
        'AIS_head_true_raw': dataset.ais_head,
        'AIS_chest_true_raw': dataset.ais_chest,
        'AIS_neck_true_raw': dataset.ais_neck,
        'MAIS_true_raw': dataset.mais, # MAIS 真值
    })
    
    # 2. 添加模型预测值
    results_df['HIC_pred'] = predictions_np[:, 0]
    results_df['Dmax_pred'] = predictions_np[:, 1]
    results_df['Nij_pred'] = predictions_np[:, 2]
    
    # 3. 计算预测的AIS等级 (确保返回整数类型)
    results_df['AIS_head_pred'] = AIS_cal_head(results_df['HIC_pred']).astype(int)
    results_df['AIS_chest_pred'] = AIS_cal_chest(results_df['Dmax_pred'], dataset.OT_raw).astype(int)
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
    
    # 8. 调整列顺序以满足您的要求
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
            print(f"    MAIS Acc     : {acc_mais:.2f}%")
            print(f"    All AIS Correct: {all_correct_rate:.2f}% ({all_correct}/{len(subset_df)})")
            
        except Exception as e:
            print(f"  [等级 Accuracy] 计算出错: {e}")
        
    print("="*60)

if __name__ == "__main__":
    set_random_seed()
    
    # 1. 加载原始13个标量特征
    original_features_df = load_original_features(DISTRIBUTION_FILE)
    
    # 2. 加载模型、完整数据集和 case_id 映射
    model, full_dataset, device, case_id_map = load_model_and_data(RUN_DIR, WEIGHT_FILE, DATA_DIR)
    
    # 3. 运行推理
    predictions_np = run_inference(model, full_dataset, device)
    
    # 4. 创建并合并结果
    final_results_df = create_results_dataframe(full_dataset, predictions_np, original_features_df, case_id_map)
    
    # 5. 保存到 CSV
    output_filename = f"full_dataset_predictions_{WEIGHT_FILE.replace('.pth', '.csv')}"
    output_path = os.path.join(RUN_DIR, output_filename)
    
    final_results_df.to_csv(output_path, index=False, float_format='%.4f')
    
    print("\n" + "="*60)
    print("测试完成！")
    print(f"结果已保存至: {output_path}")
    print(f"总计处理 {len(final_results_df)} 条数据。")
    print("="*60)
    
    # 6. 打印性能摘要
    print_metrics_summary(final_results_df)