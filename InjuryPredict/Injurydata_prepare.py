"""Injurydata_prepare.py

- 读取由根目录 `prepare_data.py` 生成的打包文件（默认：data/raw_packed/raw_data_packed.npz）和划分索引（data/split_indices/）
- 使用 `common.data_utils.processor.UnifiedDataProcessor` 统一生成/加载归一化配置
- 生成并保存标准的 PyTorch `.pt` 子集文件：
    data/processed/injury/train_dataset.pt
    data/processed/injury/val_dataset.pt
    data/processed/injury/test_dataset.pt
- 生成统计摘要 JSON 及散点图（velocity vs HIC/Dmax/Nij）、AIS 分布图（保存在 figs/）
- 默认不覆盖已存在的输出，除非显式传入 `--overwrite`

用法（示例）：
    python -m InjuryPredict.Injurydata_prepare --out-dir data/processed/injury

"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import os
from typing import Dict, Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
import matplotlib.pyplot as plt

from common.settings import (
    RAW_DATA_DIR, SPLIT_INDICES_DIR, NORMALIZATION_CONFIG_PATH,
    INJURY_PROCESSED_DIR, ensure_dirs
)
from common.data_utils.processor import UnifiedDataProcessor
from common.data_utils.split_io import load_int_vector_csv
from common.metrics.injury_risk import AIS_cal_head, AIS_cal_chest, AIS_cal_neck

# --------------------- 辅助 Dataset（轻量） ---------------------
class InjuryPackedDataset(Dataset):
    """轻量 Dataset：包装 raw_packed.npz 的 arrays 并保存 processor 引用。

    返回项顺序与训练/评估流水线保持一致：
        (x_acc, x_att_continuous, x_att_discrete,
         y_HIC, y_Dmax, y_Nij,
         ais_head, ais_chest, ais_neck, mais, OT_raw)
    说明：保留必要的字段与元信息 (`processor`, `num_classes_of_discrete`)。
    """
    def __init__(self, raw_npz: Path, processor: Optional[UnifiedDataProcessor] = None):
        data = np.load(raw_npz)
        required_keys = {
            "case_ids", "x_att_raw", "x_acc_xy",
            "y_HIC", "y_Dmax", "y_Nij",
            "ais_head", "ais_chest", "ais_neck", "mais"
        }
        missing_keys = sorted(required_keys - set(data.files))
        if missing_keys:
            raise KeyError(f"raw_packed 数据缺少必要键: {missing_keys}. 请先使用最新 prepare_data.py 重新打包。")
        # 原始字段（来自 prepare_data 的命名约定）
        self.case_ids = data["case_ids"].astype(np.int32)
        # x_att_raw: shape (N, len(FEATURE_ORDER))
        self.x_att_raw = data["x_att_raw"].astype(np.float32)
        # 波形：优先使用 x_acc_xy / x_acc_xyz 的可用项
        if "x_acc_xy" in data:
            self.x_acc_raw = data["x_acc_xy"].astype(np.float32)  # (N, C=2, T)
        else:
            self.x_acc_raw = data["x_acc_xyz"][:, :2, :].astype(np.float32)
        # labels（容许存在缺失以保证运行; 因为存在缺失的case不会被包含在train/val/test数据划分中）
        self.y_HIC = data.get("y_HIC", np.full((len(self.case_ids),), np.nan)).astype(np.float32)
        self.y_Dmax = data.get("y_Dmax", np.full((len(self.case_ids),), np.nan)).astype(np.float32)
        self.y_Nij = data.get("y_Nij", np.full((len(self.case_ids),), np.nan)).astype(np.float32)
        self.ais_head = data.get("ais_head", np.full((len(self.case_ids),), -1)).astype(np.int32)
        self.ais_chest = data.get("ais_chest", np.full((len(self.case_ids),), -1)).astype(np.int32)
        self.ais_neck = data.get("ais_neck", np.full((len(self.case_ids),), -1)).astype(np.int32)
        self.mais = data.get("mais", np.full((len(self.case_ids),), -1)).astype(np.int32)
        self.OT_raw = self.x_att_raw[:, -1].astype(np.int32)  # OT 在 FEATURE_ORDER 的末尾

        # processed fields (在 prepare 阶段填充)
        self.x_acc = None
        self.x_att_continuous = None
        self.x_att_discrete = None

        # meta
        self.processor = processor
        self.num_classes_of_discrete = None

    def __len__(self):
        return int(self.case_ids.shape[0])

    def __getitem__(self, idx):
        if self.x_acc is None or self.x_att_continuous is None or self.x_att_discrete is None:
            raise RuntimeError("Dataset 尚未被处理 —— 请先运行 Injurydata_prepare 生成 .pt 文件")
        return (
            torch.tensor(self.x_acc[idx], dtype=torch.float32),
            torch.tensor(self.x_att_continuous[idx], dtype=torch.float32),
            torch.tensor(self.x_att_discrete[idx], dtype=torch.int),
            torch.tensor(self.y_HIC[idx], dtype=torch.float32),
            torch.tensor(self.y_Dmax[idx], dtype=torch.float32),
            torch.tensor(self.y_Nij[idx], dtype=torch.float32),
            torch.tensor(self.ais_head[idx], dtype=torch.int),
            torch.tensor(self.ais_chest[idx], dtype=torch.int),
            torch.tensor(self.ais_neck[idx], dtype=torch.int),
            torch.tensor(self.mais[idx], dtype=torch.int),
            torch.tensor(self.OT_raw[idx], dtype=torch.int),
        )


def load_processed_subset(pt_path: Path):
    """严格加载 Injury processed .pt：不做任何历史兼容兜底。"""
    pt_path = Path(pt_path)
    if not pt_path.exists():
        raise FileNotFoundError(f"processed dataset 文件不存在: {pt_path}")

    subset = torch.load(pt_path.as_posix(), weights_only=False)
    if not isinstance(subset, Subset):
        raise TypeError(f"{pt_path} 不是 torch.utils.data.Subset，实际类型: {type(subset)}")
    if not isinstance(subset.dataset, InjuryPackedDataset):
        raise TypeError(
            f"{pt_path} 的底层数据集类型非法，期望 InjuryPackedDataset，实际: {type(subset.dataset)}"
        )
    return subset

# --------------------- 主流程函数 ---------------------
def build_and_save_splits(
    raw_packed: Path,
    norm_config: Path,
    split_dir: Path,
    out_dir: Path,
    overwrite: bool = False,
):
    """基于已有的 raw_packed 与 split indices 生成并保存 .pt 与统计图。

    调用前必须已运行根目录的 `prepare_data.py` 来生成
    - `data/raw_packed/raw_data_packed.npz`
    - `data/split_indices/*_indices.csv`
    - `data/normalization_config.json`

    """
    ensure_dirs([out_dir])
    out_dir = Path(out_dir)
    figs_dir = out_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    # 1) 校验输入文件存在
    if not raw_packed.exists():
        raise FileNotFoundError(f"raw_packed 文件未找到: {raw_packed} — 请先运行 prepare_data.py")

    # 2) 读取划分索引（优先使用 injury 前缀）
    train_idx_path = split_dir / "injury_train_indices.csv"
    val_idx_path = split_dir / "injury_val_indices.csv"
    test_idx_path = split_dir / "injury_test_indices.csv"
    if not (train_idx_path.exists() and val_idx_path.exists() and test_idx_path.exists()):
        raise FileNotFoundError(
            f"缺少划分索引（injury_*_indices.csv）。请先运行 prepare_data.py 或检查 {split_dir}" 
        )

    train_idx = load_int_vector_csv(train_idx_path)
    val_idx = load_int_vector_csv(val_idx_path)
    test_idx = load_int_vector_csv(test_idx_path)

    # 3) 构建 Dataset 实例
    dataset = InjuryPackedDataset(raw_packed)

    # 4) 构建或加载 UnifiedDataProcessor
    processor = UnifiedDataProcessor(config_path=norm_config)

    # 强校验：不允许在此处自动生成或拟合归一化配置——必须由 prepare_data.py 离线完成
    if not norm_config.exists():
        raise FileNotFoundError(
            f"缺少归一化配置: {norm_config}。\n请先在项目根目录运行: `python -m prepare_data` 来生成 raw_packed / split_indices / normalization_config.json" 
        )

    # 显式加载配置（若非法会抛出异常）
    processor.load_config()
    dataset.processor = processor

    # 5) 使用 processor 对整个数据集进行转换（严格校验处理结果）
    # waveform: scale; features: continuous/discrete
    x_acc_processed = processor.process_waveform(dataset.x_acc_raw, inverse=False)  # same shape
    x_cont, x_disc = processor.process_all_features(dataset.x_att_raw, inverse=False)

    # 严格校验处理结果的完整性与形状
    if x_acc_processed is None or x_cont is None or x_disc is None:
        raise RuntimeError("归一化器返回空结果；请确认 data/normalization_config.json 与 raw_packed 数据是否匹配。")
    if not (hasattr(x_acc_processed, 'ndim') and x_acc_processed.ndim == 3 and x_acc_processed.shape[0] == len(dataset)):
        raise ValueError(f"处理后的波形维度异常: got {getattr(x_acc_processed, 'shape', None)}, expected (N, C, T) with N={len(dataset)}")
    if x_cont.shape[0] != len(dataset) or x_disc.shape[0] != len(dataset):
        raise ValueError("处理后的标量特征维度与样本数不匹配")
    # 填回 dataset
    dataset.x_acc = x_acc_processed.astype(np.float32)
    dataset.x_att_continuous = x_cont.astype(np.float32)
    dataset.x_att_discrete = x_disc.astype(np.int32)
    # 将离散特征类别数从 dict 转为按 processor 内部离散特征顺序的 list，
    # 以便后续构建 nn.Embedding 时能按位置索引使用（如: 期望形式: [is_driver_side_num, OT_num]）
    discrete_map = processor.get_discrete_num_classes()
    dataset.num_classes_of_discrete = [int(discrete_map[name]) for name in processor.discrete_feature_names]

    # 6) 基于索引构造 Subset 并保存为 .pt
    train_subset = Subset(dataset, train_idx.tolist())
    val_subset = Subset(dataset, val_idx.tolist())
    test_subset = Subset(dataset, test_idx.tolist())

    paths = {
        "train": out_dir / "train_dataset.pt",
        "val": out_dir / "val_dataset.pt",
        "test": out_dir / "test_dataset.pt",
        "summary": out_dir / "split_summary.json",
        "figs": figs_dir,
        "norm_config": norm_config,
    }

    # 如果已存在且不允许覆盖则报错
    for k in ("train", "val", "test"):
        if paths[k].exists() and not overwrite:
            raise FileExistsError(f"处理后的数据集.pt文件已存在: {paths[k]}, 如需覆盖请使用 --overwrite 选项。")

    torch.save(train_subset, paths["train"].as_posix())
    torch.save(val_subset, paths["val"].as_posix())
    torch.save(test_subset, paths["test"].as_posix())
    print(f"已生成并保存：\n  - {paths['train']}\n  - {paths['val']}\n  - {paths['test']}\n")
    
    # 7) 计算并保存统计信息 + 绘图
    summary = _compute_and_save_statistics(dataset, train_idx, val_idx, test_idx, paths["figs"])
    with open(paths["summary"], "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"统计与图像保存在: {paths['figs']} (摘要文件: {paths['summary']})")

    return paths


def _compute_and_save_statistics(dataset: InjuryPackedDataset, train_idx, val_idx, test_idx, figs_dir: Path) -> Dict[str, Any]:
    """打印并保存若干常用统计与散点图（velocity vs HIC/Dmax/Nij，AIS 分布）。"""
    os.makedirs(figs_dir, exist_ok=True)

    # 从原始未归一化的数据中读取用于工程统计的原始量（仅使用 train/val/test 并集，排除无效 case）
    raw_params_all = dataset.x_att_raw  # [N, D]
    # 使用 train/val/test 的并集作为有效样本集合
    union_idx = np.unique(np.concatenate([np.asarray(train_idx), np.asarray(val_idx), np.asarray(test_idx)]))
    raw_params = raw_params_all[union_idx]
    vel = raw_params[:, 0]

    # 只取并集对应的标签/波形/OT
    hic_all = dataset.y_HIC
    dmax_all = dataset.y_Dmax
    nij_all = dataset.y_Nij
    ot_all = dataset.OT_raw

    hic = hic_all[union_idx]
    dmax = dmax_all[union_idx]
    nij = nij_all[union_idx]
    ot = ot_all[union_idx]

    # AIS（优先使用打包时已有值，否则基于子集重新计算）
    ais_head_all = dataset.ais_head
    ais_chest_all = dataset.ais_chest
    ais_neck_all = dataset.ais_neck

    ais_head = ais_head_all[union_idx]
    ais_chest = ais_chest_all[union_idx]
    ais_neck = ais_neck_all[union_idx]

    if ais_head.min() < 0:
        ais_head = AIS_cal_head(hic)
    if ais_chest.min() < 0:
        ais_chest = AIS_cal_chest(dmax, ot)
    if ais_neck.min() < 0:
        ais_neck = AIS_cal_neck(nij)
    mais = np.maximum.reduce([ais_head, ais_chest, ais_neck])

    # 将 AIS 等级计数写入 summary
    unique, counts = np.unique(ais_head, return_counts=True)
    head_counts = dict(zip(unique.tolist(), counts.tolist()))
    unique, counts = np.unique(ais_chest, return_counts=True)
    chest_counts = dict(zip(unique.tolist(), counts.tolist()))
    unique, counts = np.unique(ais_neck, return_counts=True)
    neck_counts = dict(zip(unique.tolist(), counts.tolist()))
    unique, counts = np.unique(mais, return_counts=True)
    mais_counts = dict(zip(unique.tolist(), counts.tolist()))

    # velocity vs HIC / Nij （使用 train/val/test 并集）
    def _scatter(x, y, color_lbl, fname, xlabel, ylabel, legend_label: str = "MAIS"):
        """散点图：当 color_lbl 为整数 AIS 值（例如 AIS0..AIS5）时使用离散配色并绘制图例，
        否则退回连续 colormap。`legend_label` 用于图例/颜色条标签。
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        # 尝试将 color_lbl 转为 numpy 数组并判定是否为整数标签序列
        try:
            vals = np.asarray(color_lbl)
            is_integer_labels = np.issubdtype(vals.dtype, np.integer)
            unique_vals = np.unique(vals)
        except Exception:
            is_integer_labels = False
            unique_vals = []

        if is_integer_labels and np.all((unique_vals >= 0) & (unique_vals <= 5)):
            # 离散颜色表（AIS0..AIS5）
            ais_colors = ['#1f77b4', '#2ca02c', '#fff7a3', '#ff7f0e', '#d62728', '#8c564b']
            mapped = [ais_colors[int(v)] if (0 <= int(v) <= 5) else '#777777' for v in vals]
            sc = ax.scatter(x, y, c=mapped, alpha=0.85, s=40, edgecolor='k', linewidth=0.2)
            # 添加图例（AIS 标签）
            from matplotlib.patches import Patch
            legend_elems = [Patch(facecolor=ais_colors[i], edgecolor='k', label=f'AIS{i}') for i in range(6)]
            ax.legend(handles=legend_elems, title=legend_label, bbox_to_anchor=(1.02, 1), loc='upper left')
        else:
            sc = ax.scatter(x, y, c=color_lbl, cmap='viridis', alpha=0.7, s=40)
            cbar = fig.colorbar(sc, ax=ax)
            cbar.set_label(legend_label)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle='--', alpha=0.3)
        p = figs_dir / fname
        fig.savefig(p, dpi=200, bbox_inches='tight')
        plt.close(fig)
        return p

    # 按部位 AIS 着色：HIC -> 头部 AIS；Nij -> 颈部 AIS；Dmax -> 胸部 AIS
    _scatter(vel, hic, ais_head, "vel_vs_HIC.png", "impact_velocity (km/h)", "HIC15", legend_label="AIS (head)")
    _scatter(vel, nij, ais_neck, "vel_vs_Nij.png", "impact_velocity (km/h)", "Nij", legend_label="AIS (neck)")

    # velocity vs Dmax: overall + 按 OT 值分别绘图（仅并集样本），按胸部 AIS 着色
    _scatter(vel, dmax, ais_chest, "vel_vs_Dmax_all.png", "impact_velocity (km/h)", "Dmax (mm)", legend_label="AIS (chest)")
    ot_values = np.unique(ot)
    for ot_val in ot_values:
        mask = (ot == ot_val)
        if np.sum(mask) == 0:
            continue
        fname = f"vel_vs_Dmax_OT_{int(ot_val)}.png"
        _scatter(vel[mask], dmax[mask], ais_chest[mask], fname, "impact_velocity (km/h)", f"Dmax (mm) — OT={int(ot_val)}", legend_label=f"AIS (chest) OT={int(ot_val)}")

    summary = {
        "n_total": int(len(union_idx)),
        "train_count": int(len(train_idx)),
        "val_count": int(len(val_idx)),
        "test_count": int(len(test_idx)),
        "mais_counts": convert_numpy_for_json(mais_counts),
        "ais_head_counts": convert_numpy_for_json(head_counts),
        "ais_chest_counts": convert_numpy_for_json(chest_counts),
        "ais_neck_counts": convert_numpy_for_json(neck_counts),
    }
    return summary


def convert_numpy_for_json(obj):
    if isinstance(obj, dict):
        return {int(k): int(v) for k, v in obj.items()}
    return obj


# --------------------- CLI ---------------------
def main(argv=None):
    p = argparse.ArgumentParser(description="生成 InjuryPredict 所需的 .pt 数据集并输出统计图（使用 common.UnifiedDataProcessor）。\n注意：此脚本严格依赖由根目录的 prepare_data.py 预先生成的 raw_packed、split_indices 与 normalization_config.json；若缺失将直接报错。")
    p.add_argument("--raw-npz", default=(RAW_DATA_DIR / "raw_data_packed.npz"), type=Path,
                   help="由 prepare_data.py 生成的原始打包文件（默认来自 common settings）")
    p.add_argument("--norm-config", default=NORMALIZATION_CONFIG_PATH, type=Path, help="归一化配置文件路径（默认来自 common settings）")
    p.add_argument("--split-dir", default=SPLIT_INDICES_DIR, type=Path, help="split indices 目录（默认来自 common settings）")
    p.add_argument("--out-dir", default=INJURY_PROCESSED_DIR, type=Path, help="输出目录（默认来自 common settings）")
    p.add_argument("--overwrite", action="store_true", help="覆盖已存在的train/val/test.pt 输出文件; 不设置则默认保护现有文件不被覆盖")

    args = p.parse_args(argv)

    paths = build_and_save_splits(
        raw_packed=Path(args.raw_npz),
        norm_config=Path(args.norm_config),
        split_dir=Path(args.split_dir),
        out_dir=Path(args.out_dir),
        overwrite=args.overwrite,
    )
    return paths


if __name__ == '__main__':
    main()
