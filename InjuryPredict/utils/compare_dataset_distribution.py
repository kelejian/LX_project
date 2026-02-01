# -*- coding: utf-8 -*-
"""
compare_data_distribution.py

加载 train_dataset.pt, val_dataset.pt, test_dataset.pt 文件，
比较训练集与 (验证集+测试集) 之间，指定的输入特征和标签值的分布。
生成并保存对比图 (KDE 或 百分比直方图) 到 ./data/distribution_comparison 目录。
"""

import warnings
warnings.filterwarnings('ignore')
import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats # 用于计算 KS 统计量

# --- 从项目中导入 CrashDataset 和 AIS 计算函数 ---
# 确保此脚本位于 utils 文件夹内
try:
    from dataset_prepare import CrashDataset # 需要能够导入同级目录的 CrashDataset
    from AIS_cal import AIS_cal_head, AIS_cal_chest, AIS_cal_neck
except ImportError as e:
    print(f"错误：无法导入必要的模块 ({e})。")
    print("请确保此脚本位于 'utils' 文件夹内，并且可以访问 'dataset_prepare.py' 和 'AIS_cal.py'。")
    exit()

# --- 1. 配置区 ---

# 获取项目根目录（更可靠的方式）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # utils 目录
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # LX_model_injurypredict 目录

# 1.1) 存放 .pt 数据集的目录
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# 1.2) 保存对比图表的输出目录
OUTPUT_DIR = os.path.join(DATA_DIR, "distribution_comparison_2640+466_123")

# --- 结束配置 ---

def load_data_from_subsets(data_dir=DATA_DIR):
    """加载 .pt 文件并提取原始特征和标签"""
    print("正在加载 .pt 数据集文件...")
    
    train_pt_path = os.path.join(data_dir, "train_dataset.pt")
    val_pt_path = os.path.join(data_dir, "val_dataset.pt")
    test_pt_path = os.path.join(data_dir, "test_dataset.pt")
    
    if not all(os.path.exists(p) for p in [train_pt_path, val_pt_path, test_pt_path]):
        raise FileNotFoundError(f"未在 {os.path.abspath(data_dir)} 中找到 train/val/test_dataset.pt 文件。请先运行 utils/dataset_prepare.py。")

    try:
        train_subset = torch.load(train_pt_path, map_location='cpu')
        val_subset = torch.load(val_pt_path, map_location='cpu')
        test_subset = torch.load(test_pt_path, map_location='cpu')
    except Exception as e:
        print(f"加载 .pt 文件时出错: {e}")
        print("这可能是由于保存文件的环境与当前环境不兼容（例如 Python 或 PyTorch 版本）。")
        print("如果 CrashDataset 定义已更改，也可能导致此错误。")
        exit()
        
    # 访问底层的 CrashDataset
    if not hasattr(train_subset, 'dataset') or not isinstance(train_subset.dataset, CrashDataset):
         print("错误：加载的 train_dataset.pt 文件似乎不包含有效的 CrashDataset 实例。")
         exit()
         
    full_dataset = train_subset.dataset
    # 特征数据 (x_att_raw) 说明：形状 (N, 13)
    # 连续特征 (0-10): impact_velocity, impact_angle, overlap, LL1, LL2, BTF, LLATTF, AFT, SP, SH, RA
    # 离散特征 (11-12): is_driver_side, OT
    # 定义特征名称映射 (基于 CrashDataset 和 DataProcessor)
    feature_names = {
        0: "impact_velocity", 1: "impact_angle", 2: "overlap", 3: "LL1", 4: "LL2",
        5: "BTF", 6: "LLATTF", 7: "AFT", 8: "SP", 9: "SH", 10: "RA",
        11: "is_driver_side", 12: "OT"
    }

    # -- 提取数据 --
    
    # 训练集
    train_indices = train_subset.indices
    df_train = pd.DataFrame(full_dataset.x_att_raw[train_indices], columns=feature_names.values())
    df_train['HIC'] = full_dataset.y_HIC[train_indices]
    df_train['Dmax'] = full_dataset.y_Dmax[train_indices]
    df_train['Nij'] = full_dataset.y_Nij[train_indices]
    df_train['subset'] = 'Train'
    
    # 验证集+测试集
    vt_indices = np.concatenate([val_subset.indices, test_subset.indices])
    df_vt = pd.DataFrame(full_dataset.x_att_raw[vt_indices], columns=feature_names.values())
    df_vt['HIC'] = full_dataset.y_HIC[vt_indices]
    df_vt['Dmax'] = full_dataset.y_Dmax[vt_indices]
    df_vt['Nij'] = full_dataset.y_Nij[vt_indices]
    df_vt['subset'] = 'Valid+Test'
    
    # 合并
    combined_df = pd.concat([df_train, df_vt], ignore_index=True)
    
    print(f"数据加载完成。训练集: {len(df_train)} 条, 验证+测试集: {len(df_vt)} 条。")
    
    return combined_df

def calculate_ais_levels(df):
    """计算 AIS 和 MAIS 等级"""
    if df is None:
        return None
        
    print("正在计算 AIS/MAIS 等级...")
    
    # 注意：这里使用 HIC 列， CrashDataset 中存储的就是 HIC
    df['AIS_head'] = AIS_cal_head(df['HIC'])
    df['AIS_chest'] = AIS_cal_chest(df['Dmax'], df['OT'])
    df['AIS_neck'] = AIS_cal_neck(df['Nij'])
    
    df['MAIS'] = np.maximum.reduce([
        df['AIS_head'], 
        df['AIS_chest'], 
        df['AIS_neck']
    ])
    
    return df

def plot_continuous_distributions(combined_df, columns, output_dir):
    """为连续标量特征绘制 KDE 分布对比图，并添加统计量"""
    print(f"\n正在绘制 {len(columns)} 个连续特征的分布图...")
    output_subdir = os.path.join(output_dir, "1_Continuous_Distributions")
    os.makedirs(output_subdir, exist_ok=True)
    
    for col in tqdm(columns, desc="Plotting Continuous"):
        plt.figure(figsize=(12, 7))
        
        # 绘制 KDE 图
        sns.kdeplot(data=combined_df, x=col, hue='subset', 
                    fill=True, common_norm=False, alpha=0.5, warn_singular=False)
        
        # 计算统计量
        stats_text = ""
        ks_stat, ks_pvalue = -1, -1 # 初始化
        try:
            group1 = combined_df[combined_df['subset'] == 'Train'][col].dropna()
            group2 = combined_df[combined_df['subset'] == 'Valid+Test'][col].dropna()

            if len(group1) > 1 and len(group2) > 1: # KS 测试需要至少2个样本
                stats_text += f"Train: Mean={group1.mean():.2f}, Std={group1.std():.2f}, N={len(group1)}\n"
                stats_text += f"Valid+Test: Mean={group2.mean():.2f}, Std={group2.std():.2f}, N={len(group2)}\n"
                # 执行 Kolmogorov-Smirnov 测试
                ks_result = stats.ks_2samp(group1, group2)
                ks_stat = ks_result.statistic
                ks_pvalue = ks_result.pvalue
                stats_text += f"KS Stat={ks_stat:.3f}, p-value={ks_pvalue:.3g}" # 使用 .3g 格式化 p 值
            else:
                 stats_text += f"Train: N={len(group1)}\n"
                 stats_text += f"Valid+Test: N={len(group2)}\n"
                 stats_text += "KS test skipped (insufficient samples)"

        except Exception as e:
            stats_text = f"Error calculating stats: {e}"
            print(f"  警告: 计算 '{col}' 的统计量时出错: {e}")

        # 将统计量添加到图表右上角
        plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes, 
                 fontsize=10, verticalalignment='top', horizontalalignment='right', 
                 bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

        plt.title(f"Distribution Comparison for '{col}' (Continuous)")
        plt.xlabel(col)
        plt.ylabel("Density")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(output_subdir, f"{col}_distribution.png"))
        plt.close()

def plot_categorical_distributions(combined_df, columns, output_dir):
    """为离散/等级特征绘制百分比直方图,并添加计数"""
    print(f"\n正在绘制 {len(columns)} 个离散/等级特征的分布图...")
    output_subdir = os.path.join(output_dir, "2_Categorical_Distributions")
    os.makedirs(output_subdir, exist_ok=True)

    # 定义每个特征的完整类别，以确保X轴完整性
    category_orders = {
        'AIS_head': [0, 1, 2, 3, 4, 5],
        'AIS_chest': [0, 1, 2, 3, 4, 5],
        'AIS_neck': [0, 1, 2, 3, 4, 5],
        'MAIS': [0, 1, 2, 3, 4, 5]
    }

    for col in tqdm(columns, desc="Plotting Categorical"):
        plt.figure(figsize=(14, 8))

        # --- 1. 数据预处理 ---
        order = category_orders.get(col, sorted(combined_df[col].unique()))
        subset_totals = combined_df['subset'].value_counts()
        
        plot_data = combined_df.groupby(['subset', col]).size().reset_index(name='count')
        plot_data['percentage'] = plot_data.apply(
            lambda row: (row['count'] / subset_totals[row['subset']]) * 100, axis=1
        )
        
        # 确保所有类别都存在
        all_cats = pd.MultiIndex.from_product(
            [subset_totals.index, order], names=['subset', col]
        )
        plot_data = plot_data.set_index(['subset', col]).reindex(all_cats, fill_value=0).reset_index()
        
        # --- 2. 使用 barplot 绘图 ---
        ax = sns.barplot(
            data=plot_data,
            x=col,
            y='percentage',
            hue='subset',
            order=order,
            palette='tab10',
            alpha=0.8,
            edgecolor='black'
        )

        # --- 3. 添加标签 (使用字典查找) ---
        # 为每个子集和类别创建一个查找字典
        lookup = {}
        for _, row in plot_data.iterrows():
            key = (row['subset'], row[col])
            lookup[key] = (row['percentage'], int(row['count']))
        
        # 遍历 containers 并添加标签
        for i, container in enumerate(ax.containers):
            # 获取当前容器对应的子集名称
            hue_name = ax.get_legend_handles_labels()[1][i]
            
            # 为每个柱子添加标签
            labels = []
            for j, bar in enumerate(container):
                category = order[j]
                percentage, count = lookup.get((hue_name, category), (0, 0))
                
                # 只有当计数大于0时才显示标签
                if count > 0:
                    labels.append(f"{percentage:.1f}%\n(N={count})")
                else:
                    labels.append("")
            
            ax.bar_label(container, labels=labels, label_type='edge', fontsize=9, padding=3)

        # --- 4. 优化图例和坐标轴 ---
        handles, labels_legend = ax.get_legend_handles_labels()
        new_labels = [f'{lbl} (N={subset_totals.get(lbl, 0)})' for lbl in labels_legend]
        ax.legend(handles, new_labels, title='Subset', loc='upper right')

        plt.title(f"Distribution Comparison for '{col}' (Categorical/Levels)", fontsize=16)
        plt.xlabel(col, fontsize=12)
        plt.ylabel("Percentage (%) within each Subset", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6, axis='y')
        
        # 动态调整Y轴上限
        max_height = plot_data['percentage'].max()
        plt.ylim(0, max(10, max_height * 1.2))

        plt.tight_layout()
        plt.savefig(os.path.join(output_subdir, f"{col}_distribution.png"), dpi=100)
        plt.close()

if __name__ == "__main__":
    
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. 加载和合并数据
    combined_df = load_data_from_subsets()
    
    if combined_df is None:
        print("\n错误：数据加载失败，无法继续。")
        exit()
        
    # 2. 计算AIS等级
    combined_df = calculate_ais_levels(combined_df)
    
    # 3. 定义要绘图的列
    # 连续标量 (输入特征)
    continuous_features = [
        'impact_velocity', 'impact_angle', 'overlap', 'LL1', 'LL2', 
        'BTF', 'LLATTF', 'AFT', 'SP', 'SH', 'RA'
    ]
    
    # 离散标量 (输入特征)
    discrete_features = [
        'OT', 'is_driver_side'
    ]
    
    # 损伤标量 (标签)
    injury_scalars = ['HIC', 'Dmax', 'Nij']
    
    # 损伤等级 (计算得到)
    injury_levels = ['AIS_head', 'AIS_chest', 'AIS_neck', 'MAIS']
    
    # 4. 执行绘图
    plot_continuous_distributions(combined_df, continuous_features + injury_scalars, OUTPUT_DIR)
    plot_categorical_distributions(combined_df, discrete_features + injury_levels, OUTPUT_DIR)
    
    print("\n" + "="*50)
    print("对比完成！")
    print(f"所有对比图表已保存至: {OUTPUT_DIR}")
    print("="*50)