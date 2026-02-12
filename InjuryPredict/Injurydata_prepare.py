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
from common.metrics.injury_risk import AIS_cal_head, AIS_cal_chest, AIS_cal_neck

# --------------------- 辅助 Dataset（轻量） ---------------------
class InjuryPackedDataset(Dataset):
    """轻量 Dataset：包装 raw_packed.npz 的 arrays 并保存 processor 引用。

    返回项顺序与训练/评估流水线保持一致：
        (x_acc, x_att_continuous, x_att_discrete,
         y_HIC, y_Dmax, y_Nij,
         ais_head, ais_chest, ais_neck, mais, OT_raw)
    说明：保留必要的字段与元信息 (`processor`, `num_classes_of_discrete`)，不再包含额外的历史兼容分支。
    """
    def __init__(self, raw_npz: Path, processor: Optional[UnifiedDataProcessor] = None):
        data = np.load(raw_npz)
        # 原始字段（来自 prepare_data 的命名约定）
        self.case_ids = data["case_ids"].astype(np.int64)
        # x_att_raw: shape (N, len(FEATURE_ORDER))
        self.x_att_raw = data["x_att_raw"].astype(np.float32)
        # 波形：优先使用 x_acc_xy / x_acc_xyz 的可用项
        if "x_acc_xy" in data:
            self.x_acc_raw = data["x_acc_xy"].astype(np.float32)  # (N, C=2, T)
        else:
            self.x_acc_raw = data["x_acc_xyz"][:, :2, :].astype(np.float32)
        # labels（可能部分缺失）
        self.y_HIC = data.get("y_HIC", np.full((len(self.case_ids),), np.nan)).astype(np.float32)
        self.y_Dmax = data.get("y_Dmax", np.full((len(self.case_ids),), np.nan)).astype(np.float32)
        self.y_Nij = data.get("y_Nij", np.full((len(self.case_ids),), np.nan)).astype(np.float32)
        self.ais_head = data.get("ais_head", np.full((len(self.case_ids),), -1)).astype(np.int64)
        self.ais_chest = data.get("ais_chest", np.full((len(self.case_ids),), -1)).astype(np.int64)
        self.ais_neck = data.get("ais_neck", np.full((len(self.case_ids),), -1)).astype(np.int64)
        self.mais = data.get("mais", np.full((len(self.case_ids),), -1)).astype(np.int64)
        self.OT_raw = self.x_att_raw[:, -1].astype(np.int64)  # OT 在 FEATURE_ORDER 的末尾

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
        ) # 与旧有dataset_prepare.py 中的数据集类的返回项保持一致

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
    - `data/split_indices/*_indices.npy`
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
    train_idx_path = split_dir / "injury_train_indices.npy"
    val_idx_path = split_dir / "injury_val_indices.npy"
    test_idx_path = split_dir / "injury_test_indices.npy"
    if not (train_idx_path.exists() and val_idx_path.exists() and test_idx_path.exists()):
        raise FileNotFoundError(
            f"缺少划分索引（injury_*_indices.npy）。请先运行 prepare_data.py 或检查 {split_dir}" 
        )

    train_idx = np.load(train_idx_path)
    val_idx = np.load(val_idx_path)
    test_idx = np.load(test_idx_path)

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
    if np.isnan(x_acc_processed).any() or np.isnan(x_cont).any():
        raise ValueError("处理后数据包含 NaN —— 请检查原始数据与 normalization_config.json 的一致性。")

    # 填回 dataset
    dataset.x_acc = x_acc_processed.astype(np.float32)
    dataset.x_att_continuous = x_cont.astype(np.float32)
    dataset.x_att_discrete = x_disc.astype(np.int64)
    dataset.num_classes_of_discrete = processor.get_discrete_num_classes()

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

    # 从原始未归一化的数据中读取用于工程统计的原始量
    raw_params = dataset.x_att_raw  # [N, D]
    vel = raw_params[:, 0]

    hic = dataset.y_HIC
    dmax = dataset.y_Dmax
    nij = dataset.y_Nij
    ot = dataset.OT_raw

    # 确保 AIS 已计算（优先使用已有，否则用函数计算）
    ais_head = dataset.ais_head
    ais_chest = dataset.ais_chest
    ais_neck = dataset.ais_neck
    if ais_head.min() < 0:
        ais_head = AIS_cal_head(hic)
    if ais_chest.min() < 0:
        ais_chest = AIS_cal_chest(dmax, ot)
    if ais_neck.min() < 0:
        ais_neck = AIS_cal_neck(nij)
    mais = np.maximum.reduce([ais_head, ais_chest, ais_neck])

    def _save_bar(counts, name):
        fig, ax = plt.subplots(figsize=(6, 4))
        keys = sorted(list(counts.keys()))
        vals = [counts[k] for k in keys]
        ax.bar([str(k) for k in keys], vals, color="C0", alpha=0.8)
        ax.set_title(name)
        ax.set_xlabel("AIS")
        ax.set_ylabel("count")
        p = figs_dir / f"{name.replace(' ', '_')}.png"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return p

    # AIS distributions (overall + by OT for chest)
    unique, counts = np.unique(ais_head, return_counts=True)
    head_counts = dict(zip(unique.tolist(), counts.tolist()))
    _save_bar(head_counts, "AIS_head_distribution")

    unique, counts = np.unique(mais, return_counts=True)
    mais_counts = dict(zip(unique.tolist(), counts.tolist()))
    _save_bar(mais_counts, "MAIS_distribution")

    # velocity vs HIC/Dmax/Nij scatter (colored by MAIS)
    def _scatter(x, y, color_lbl, fname, xlabel, ylabel):
        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(x, y, c=color_lbl, cmap="viridis", alpha=0.7, s=40)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.3)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("MAIS")
        p = figs_dir / fname
        fig.savefig(p, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return p

    _scatter(vel, hic, mais, "vel_vs_HIC.png", "impact_velocity (km/h)", "HIC15")
    _scatter(vel, dmax, mais, "vel_vs_Dmax.png", "impact_velocity (km/h)", "Dmax (mm)")
    _scatter(vel, nij, mais, "vel_vs_Nij.png", "impact_velocity (km/h)", "Nij")

    summary = {
        "n_total": int(len(dataset)),
        "train_count": int(len(train_idx)),
        "val_count": int(len(val_idx)),
        "test_count": int(len(test_idx)),
        "mais_counts": convert_numpy_for_json(mais_counts),
        "head_counts": convert_numpy_for_json(head_counts),
    }
    return summary


def convert_numpy_for_json(obj):
    if isinstance(obj, dict):
        return {int(k): int(v) for k, v in obj.items()}
    return obj


# --------------------- CLI ---------------------
def cli_main(argv=None):
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
    cli_main()
