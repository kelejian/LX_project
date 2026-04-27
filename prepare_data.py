import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = 'T'
import warnings
warnings.filterwarnings('ignore')
import argparse
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd

from common.tools.seeding import GLOBAL_SEED, set_random_seed
from common.settings import (
    FEATURE_ORDER,
    NORMALIZATION_CONFIG_PATH,
    RAW_DATA,
    REQUIRED_COLUMNS_FOR_PACKING,
    SPLIT_ROOT_DIR,
    WAVEFORM_CHANNELS_XY,
    WAVEFORM_CHANNELS_XYZ,
    WAVEFORM_LENGTH,
    ensure_dirs,
    get_injury_split_dir,
    get_split_case_ids_path,
    get_split_indices_path,
    get_pulse_split_dir,
)
from common.data_utils.splitter import stratified_split_case_ids, case_ids_to_indices
from common.data_utils.processor import UnifiedDataProcessor
from common.data_utils.split_io import save_int_vector_csv, load_int_vector_csv
from common.metrics.injury_risk import AIS_cal_head, AIS_cal_chest, AIS_cal_neck

from tqdm import tqdm



def _read_distribution(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() == ".npz":
        npz = np.load(path, allow_pickle=True)
        df = pd.DataFrame({k: npz[k] for k in npz.files})
    else:
        raise ValueError("distribution 文件必须是 .csv 或 .npz")

    if "case_id" not in df.columns:
        raise ValueError("distribution 缺少 case_id 列")

    df["case_id"] = df["case_id"].astype(int)
    df = df.set_index("case_id", drop=False)
    return df


def _downsample_waveform(csv_path: Path) -> np.ndarray:
    # 读取时间列，推断dt并降采样；返回长度 WAVEFORM_LENGTH
    time = pd.read_csv(csv_path, sep="\t", header=None, usecols=[0]).values.flatten()
    if time.size < 3:
        raise ValueError(f"时间序列过短: {csv_path}")

    dt = float(np.mean(np.diff(time)))
    total_length = len(time)

    if np.isclose(dt, 1e-5, atol=1e-7):
        downsample_indices = np.arange(100, total_length, 100)
    elif np.isclose(dt, 5e-6, atol=5e-8):
        downsample_indices = np.arange(200, total_length, 200)
    else:
        raise ValueError(f"时间步长 {dt} 不符合预期: {csv_path}")

    sig = pd.read_csv(csv_path, sep="\t", header=None, usecols=[1]).values.flatten()
    sampled = sig[downsample_indices]
    sampled = sampled[:WAVEFORM_LENGTH]
    if sampled.shape[0] != WAVEFORM_LENGTH:
        raise ValueError(f"降采样后长度不足{WAVEFORM_LENGTH}: {csv_path}")
    return sampled


def _load_xyz_waveforms(pulse_dir: Path, pulse_source_case_id: int, case_id: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    # 返回 (xyz[WAVEFORM_CHANNELS_XYZ, WAVEFORM_LENGTH], xy[WAVEFORM_CHANNELS_XY, WAVEFORM_LENGTH])
    pulse_case_id = int(pulse_source_case_id)

    x_path = pulse_dir / f"x{pulse_case_id}.csv"
    y_path = pulse_dir / f"y{pulse_case_id}.csv"
    z_path = pulse_dir / f"z{pulse_case_id}.csv"

    if not x_path.exists() or not y_path.exists():
        missing = [str(p) for p in [x_path, y_path] if not p.exists()]
        if case_id is None:
            raise FileNotFoundError(f"波形文件缺失(pulse_source_case_id={pulse_case_id}): {missing}")
        raise FileNotFoundError(f"波形文件缺失(case_id={case_id}, pulse_source_case_id={pulse_case_id}): {missing}")

    ax = _downsample_waveform(x_path)
    ay = _downsample_waveform(y_path)
    if z_path.exists():
        az = _downsample_waveform(z_path)
    else:
        az = np.zeros_like(ax)

    xyz = np.stack([ax, ay, az], axis=0).astype(np.float32)  # (WAVEFORM_CHANNELS_XYZ, WAVEFORM_LENGTH)
    xy = xyz[:WAVEFORM_CHANNELS_XY, :].astype(np.float32)  # (WAVEFORM_CHANNELS_XY, WAVEFORM_LENGTH)
    return xyz, xy


def _load_waveforms_batch(
    pulse_dir: Path,
    case_ids: np.ndarray,
    pulse_source_case_ids: np.ndarray,
    strict: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """批量读取波形。

    注意：波形存储在独立 CSV 文件中，逐文件 I/O 无法完全向量化。
    本函数按 pulse_source_case_id 去重读取波形，并将同一波形复用到共享该来源的样本。
    这里做的是批量封装，减少循环体中的 pandas 操作，并可在 non-strict 模式下跳过异常样本。
    Args:
        pulse_dir: 波形 CSV 文件目录
        case_ids: (N,) 待读取的 case_id 列表（包含主/副驾）. 本文件中指 is_pulse_ok==True 的 case_ids
        pulse_source_case_ids: (N,) 每个 case 对应的波形来源 case_id（用于定位 x/y/z*.csv）
        strict: 是否严格模式（遇到缺失波形/异常 case 则报错退出）

    Returns:
        x_acc_xyz: (M, WAVEFORM_CHANNELS_XYZ, WAVEFORM_LENGTH)
        x_acc_xy:  (M, WAVEFORM_CHANNELS_XY, WAVEFORM_LENGTH)
        ok_mask:   (N,) 表示输入 case_ids 中哪些成功读取。在 strict=True 且函数“正常返回”（不抛出异常）的前提下，返回的 ok_mask 对应的输入行要么全部为 True(对于非空输入), 要么为空数组(输入长度为0)    """
    if case_ids.ndim != 1:
        raise ValueError("case_ids 必须是一维数组")
    if pulse_source_case_ids.shape[0] != case_ids.shape[0]:
        raise ValueError("pulse_source_case_ids 长度必须与 case_ids 一致")

    pulse_source_to_indices: Dict[int, list] = {}
    pulse_source_order = []
    for idx, src in enumerate(pulse_source_case_ids.tolist()):
        src_int = int(src)
        if src_int not in pulse_source_to_indices:
            pulse_source_to_indices[src_int] = []
            pulse_source_order.append(src_int)
        pulse_source_to_indices[src_int].append(idx)

    n = case_ids.shape[0] # 输入的 case_ids 数量（包含主/副驾）
    x_acc_xyz = np.empty((n, WAVEFORM_CHANNELS_XYZ, WAVEFORM_LENGTH), dtype=np.float32)
    x_acc_xy = np.empty((n, WAVEFORM_CHANNELS_XY, WAVEFORM_LENGTH), dtype=np.float32)
    ok_mask = np.zeros((n,), dtype=bool) # 初始化全部为 False

    for pulse_source_case_id in tqdm(pulse_source_order, total=len(pulse_source_order), desc="读取波形", unit="case"):
        try:
            # 按 pulse_source_case_id 读取一份波形并复用到所有共享来源的样本
            first_idx = pulse_source_to_indices[int(pulse_source_case_id)][0]
            xyz, xy = _load_xyz_waveforms(
                pulse_dir,
                int(pulse_source_case_id),
                case_id=int(case_ids[first_idx]),
            )
            for idx in pulse_source_to_indices[int(pulse_source_case_id)]:
                x_acc_xyz[idx] = xyz
                x_acc_xy[idx] = xy
                ok_mask[idx] = True
        except Exception:
            if strict:
                raise
            # non-strict: 该来源及其对应样本全部标记失败
            for idx in pulse_source_to_indices[int(pulse_source_case_id)]:
                ok_mask[idx] = False

    # 过滤失败样本（non-strict）
    if np.any(ok_mask):
        x_acc_xyz = x_acc_xyz[ok_mask]
        x_acc_xy = x_acc_xy[ok_mask]
    else:
        x_acc_xyz = np.empty((0, WAVEFORM_CHANNELS_XYZ, WAVEFORM_LENGTH), dtype=np.float32)
        x_acc_xy = np.empty((0, WAVEFORM_CHANNELS_XY, WAVEFORM_LENGTH), dtype=np.float32)

    return x_acc_xyz, x_acc_xy, ok_mask


def _normalize_side_series(side_series: pd.Series, scope: str) -> pd.Series:
    """把 is_driver_side 统一校验并转为 0/1 整数。"""
    side_numeric = pd.to_numeric(side_series, errors="coerce")
    invalid_mask = ~side_numeric.isin([0, 1])
    if invalid_mask.any():
        invalid_values = side_series.loc[invalid_mask].head(10).tolist()
        raise ValueError(
            f"{scope} 中存在非法 is_driver_side 取值，必须为 0/1。"
            f" 示例值: {invalid_values}"
        )
    return side_numeric.astype(np.int64)


def _split_case_ids_or_empty(
    case_ids: np.ndarray,
    stratify_labels: np.ndarray,
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    rule: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """在允许空集合的前提下执行 case_id 划分。"""
    case_ids = np.asarray(case_ids, dtype=np.int64)
    stratify_labels = np.asarray(stratify_labels)
    if case_ids.ndim != 1:
        raise ValueError("case_ids 必须为一维数组")
    if stratify_labels.ndim != 1 or stratify_labels.shape[0] != case_ids.shape[0]:
        raise ValueError("stratify_labels 必须与 case_ids 一一对应")

    if case_ids.size == 0:
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            {
                "rule": rule,
                "total_final": 0,
                "train_final": 0,
                "val_final": 0,
                "test_final": 0,
                "total_ori": 0,
                "forced": {"train": 0, "valid": 0, "test": 0, "exclude": 0},
                "date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
        )

    train_ids, val_ids, test_ids, summary = stratified_split_case_ids(
        case_ids=case_ids,
        stratify_labels=stratify_labels,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        special_case_assignments=None,
        seed=seed,
    )
    summary = dict(summary)
    summary.update({
        "rule": rule,
        "date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    return (
        np.asarray(train_ids, dtype=np.int64),
        np.asarray(val_ids, dtype=np.int64),
        np.asarray(test_ids, dtype=np.int64),
        summary,
    )


def _merge_split_triplets(
    split_a: Dict[str, np.ndarray],
    split_b: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """按 train/val/test 维度把两套互斥划分合并。"""
    split_names = ("train", "val", "test")
    for split_name_a in split_names:
        arr_a = np.asarray(split_a[split_name_a], dtype=np.int64)
        for split_name_b in split_names:
            arr_b = np.asarray(split_b[split_name_b], dtype=np.int64)
            overlap = np.intersect1d(arr_a, arr_b, assume_unique=False)
            if overlap.size == 0:
                continue
            raise ValueError(
                f"合并 split 时检测到跨侧冲突：{split_name_a} vs {split_name_b} 存在重复 ID，共 {overlap.size} 个，"
                f"示例: {overlap[:10].tolist()}"
            )

    merged = {}
    for split_name in split_names:
        arr_a = np.asarray(split_a[split_name], dtype=np.int64)
        arr_b = np.asarray(split_b[split_name], dtype=np.int64)
        merged[split_name] = np.unique(np.concatenate([arr_a, arr_b])).astype(np.int64, copy=False)
    return merged


def package_raw_packed(
    distribution_path: Path,
    pulse_dir: Path,
    output_npz: Path,
    strict: bool = True,
) -> Path:
    """打包原始数据到 raw_packed.npz。
    """
    df = _read_distribution(distribution_path)

    for col in REQUIRED_COLUMNS_FOR_PACKING:
        if col not in df.columns:
            raise ValueError(f"distribution 缺少必要列: {col}")

    print(f"✅️ distribution 文件已读取: {distribution_path}")

    # 只打包 is_pulse_ok==True 的 case
    pulse_ok_mask = df["is_pulse_ok"].fillna(False).astype(bool)
    pulse_df = df.loc[pulse_ok_mask].copy() # 仅包含 is_pulse_ok==True 的样本，后续如果 strict=False 则会在读取波形时进一步过滤掉那些缺失波形的 case
    if pulse_df.shape[0] == 0:
        raise RuntimeError("distribution 中没有 is_pulse_ok==True 的样本，无法打包")
    if "is_driver_side" not in pulse_df.columns:
        raise ValueError("distribution 缺少 is_driver_side 列，无法生成主驾/副驾/合并三套划分")
    pulse_df["is_driver_side"] = _normalize_side_series(
        pulse_df["is_driver_side"],
        scope="raw_packed 待打包样本",
    )
    print(f"✅️ 已筛选出 {pulse_df.shape[0]} 个 is_pulse_ok==True 的样本")
    # ---------------------------
    # 1) 向量化：case_ids / params / 标志位
    # ---------------------------
    case_ids_all = pulse_df["case_id"].astype(int).to_numpy(dtype=np.int64)
    pulse_source_case_ids_all = pd.to_numeric(pulse_df["pulse_source_case_id"], errors="raise").to_numpy(dtype=np.int64)
    x_att_raw_all = pulse_df[FEATURE_ORDER].to_numpy(dtype=np.float32) # (N, len(FEATURE_ORDER)), 无列名，纯数值, 因此后续如果需要知道每一列的含义必须依赖 FEATURE_ORDER 的顺序！
    is_pulse_ok_all = pulse_df["is_pulse_ok"].fillna(False).astype(bool).to_numpy(dtype=bool) # 原始 distribution 中的缺失值会变成 False；以及 能被解释为 False 的值（例如 False、0、空字符串）会变成 False
    is_injury_ok_all = pulse_df["is_injury_ok"].fillna(False).astype(bool).to_numpy(dtype=bool) # 原始 distribution 中的缺失值会变成 False；以及 能被解释为 False 的值（例如 False、0、空字符串）会变成 False

    hic15_all = pulse_df["HIC15"].to_numpy(dtype=np.float32)
    dmax_all = pulse_df["Dmax"].to_numpy(dtype=np.float32)
    nij_all = pulse_df["Nij"].to_numpy(dtype=np.float32)
    ot_all = pulse_df["OT"].astype(int).to_numpy(dtype=np.int64)

    print(f"✅️ 标量参数已打包，准备打包波形数据 (strict={strict})")

    # ---------------------------
    # 2) 波形读取：无法彻底向量化（逐文件 I/O），但可批量封装
    # ---------------------------
    x_acc_xyz, x_acc_xy, ok_mask = _load_waveforms_batch(
        pulse_dir=pulse_dir,
        case_ids=case_ids_all, # 仅 is_pulse_ok==True 的 case_ids
        pulse_source_case_ids=pulse_source_case_ids_all,
        strict=strict, # 如果 strict=True 则遇到缺失波形/异常 case 直接报错退出；如果 strict=False 则跳过这些 case，继续打包剩余数据
    )

    # 如果 strict 模式（默认，即未设置 --non-strict），则要么全部成功（ok_mask 全为 True），要么函数直接抛出异常退出；如果 non-strict 模式，则会过滤掉那些缺失波形的 case，剩余 case_ids 以及对应的参数/标签等数据只包含成功读取波形的样本。
    case_ids = case_ids_all[ok_mask]
    pulse_source_case_ids = pulse_source_case_ids_all[ok_mask]
    x_att_raw = x_att_raw_all[ok_mask]
    is_pulse_ok = is_pulse_ok_all[ok_mask]
    is_injury_ok = is_injury_ok_all[ok_mask]
    hic15 = hic15_all[ok_mask]
    dmax = dmax_all[ok_mask]
    nij = nij_all[ok_mask]
    ot = ot_all[ok_mask]

    if case_ids.shape[0] == 0:
        raise RuntimeError("没有成功打包任何数据")
    print(f"✅️ 成功打包波形数据 (strict={strict})")

    # ---------------------------
    # 3) 向量化：labels / AIS / MAIS
    # ---------------------------
    n = int(case_ids.shape[0])
    y_hic = np.full((n,), np.nan, dtype=np.float32)
    y_dmax = np.full((n,), np.nan, dtype=np.float32)
    y_nij = np.full((n,), np.nan, dtype=np.float32)

    ais_head = np.full((n,), -1, dtype=np.int64)
    ais_chest = np.full((n,), -1, dtype=np.int64)
    ais_neck = np.full((n,), -1, dtype=np.int64)
    mais = np.full((n,), -1, dtype=np.int64)

    inj_mask = is_injury_ok
    if np.any(inj_mask):
        y_hic[inj_mask] = hic15[inj_mask].astype(np.float32)
        y_dmax[inj_mask] = dmax[inj_mask].astype(np.float32)
        y_nij[inj_mask] = nij[inj_mask].astype(np.float32)

        ais_head[inj_mask] = np.asarray(AIS_cal_head(y_hic[inj_mask]), dtype=np.int64)
        ais_chest[inj_mask] = np.asarray(AIS_cal_chest(y_dmax[inj_mask], ot[inj_mask]), dtype=np.int64)
        ais_neck[inj_mask] = np.asarray(AIS_cal_neck(y_nij[inj_mask]), dtype=np.int64)
        mais[inj_mask] = np.maximum.reduce([ais_head[inj_mask], ais_chest[inj_mask], ais_neck[inj_mask]]).astype(np.int64)

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    print(f"✅️ 标签计算完成并打包")

    # 包含成功读取波形的 case_ids 以及对应的参数/标签等数据只包含成功读取波形的样本（如果 strict=False 则会过滤掉那些缺失波形的 case）
    np.savez(
        output_npz,
        case_ids=case_ids.astype(np.int64), # (n,)
        pulse_source_case_ids=pulse_source_case_ids.astype(np.int64), # (n,)
        x_att_raw=x_att_raw.astype(np.float32), # ndarray (n,13) 顺序与 FEATURE_ORDER 保持一致
        x_acc_xyz=x_acc_xyz.astype(np.float32), # ndarray (n,3,150)
        x_acc_xy=x_acc_xy.astype(np.float32), # ndarray (n,2,150)
        is_pulse_ok=is_pulse_ok.astype(bool), # (n,)
        is_injury_ok=is_injury_ok.astype(bool), # (n,)
        y_HIC=y_hic.astype(np.float32), # (n,)
        y_Dmax=y_dmax.astype(np.float32), # (n,)
        y_Nij=y_nij.astype(np.float32), # (n,)
        ais_head=ais_head.astype(np.int32), # (n,)
        ais_chest=ais_chest.astype(np.int32), # (n,)
        ais_neck=ais_neck.astype(np.int32), # (n,)
        mais=mais.astype(np.int32) # (n,)
    )

    return output_npz


def _save_split(out_dir: Path, prefix: str, case_ids_all: np.ndarray,
                train_case_ids: np.ndarray, val_case_ids: np.ndarray, test_case_ids: np.ndarray,
                summary: Dict[str, Any]):
    '''保存划分结果到指定目录。目前主要用于 injury split 的 driver/passenger/combined 三套划分
    Args:
        out_dir: 输出目录
        prefix: 文件名前缀
        case_ids_all: 全量 case_ids 数组
        train_case_ids: 训练集 case_ids 数组
        val_case_ids: 验证集 case_ids 数组
        test_case_ids: 测试集 case_ids 数组
        summary: 划分结果的汇总信息字典
    '''
    out_dir.mkdir(parents=True, exist_ok=True)

    save_int_vector_csv(get_split_case_ids_path(prefix, "train", out_dir), train_case_ids)
    save_int_vector_csv(get_split_case_ids_path(prefix, "val", out_dir), val_case_ids)
    save_int_vector_csv(get_split_case_ids_path(prefix, "test", out_dir), test_case_ids)

    train_idx = case_ids_to_indices(case_ids_all, train_case_ids) # 将训练集 case_ids 转为对应的索引
    val_idx = case_ids_to_indices(case_ids_all, val_case_ids)   # 将验证集 case_ids 转为对应的索引
    test_idx = case_ids_to_indices(case_ids_all, test_case_ids) # 将测试集 case_ids 转为对应的索引

    save_int_vector_csv(get_split_indices_path(prefix, "train", out_dir), train_idx)
    save_int_vector_csv(get_split_indices_path(prefix, "val", out_dir), val_idx)
    save_int_vector_csv(get_split_indices_path(prefix, "test", out_dir), test_idx)

    with open(out_dir / f"{prefix}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def _save_pulse_split_with_first_occurrence_indices(
    out_dir: Path,
    pulse_source_case_ids_all: np.ndarray,
    train_source_ids: np.ndarray,
    val_source_ids: np.ndarray,
    test_source_ids: np.ndarray,
    summary: Dict[str, Any],
):
    """保存 pulse 划分结果。

    - case_ids 文件保存 pulse_source_case_id；
    - indices 文件保存 pulse_source_case_id 在 raw_packed 中“首次出现”的原始行索引，直接对应 raw_data_packed.npz 各数组的行。
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    save_int_vector_csv(get_split_case_ids_path("pulse", "train", out_dir), train_source_ids)
    save_int_vector_csv(get_split_case_ids_path("pulse", "val", out_dir), val_source_ids)
    save_int_vector_csv(get_split_case_ids_path("pulse", "test", out_dir), test_source_ids)

    source_to_first_row: Dict[int, int] = {}
    for row_idx, src in enumerate(pulse_source_case_ids_all.tolist()):
        src_int = int(src)
        if src_int not in source_to_first_row:
            source_to_first_row[src_int] = int(row_idx)

    def _map_sources_to_first_rows(source_ids: np.ndarray, split_name: str) -> np.ndarray:
        mapped = []
        missing = []
        for src in source_ids.tolist():
            src_int = int(src)
            first_row = source_to_first_row.get(src_int)
            if first_row is None:
                missing.append(src_int)
            else:
                mapped.append(first_row)
        if missing:
            raise ValueError(
                f"pulse {split_name} split 中存在未匹配到 raw_packed 首次出现行的 pulse_source_case_id，共 {len(missing)} 个，示例: {missing[:10]}"
            )
        return np.asarray(mapped, dtype=np.int64)

    save_int_vector_csv(get_split_indices_path("pulse", "train", out_dir), _map_sources_to_first_rows(train_source_ids, "train"))
    save_int_vector_csv(get_split_indices_path("pulse", "val", out_dir), _map_sources_to_first_rows(val_source_ids, "val"))
    save_int_vector_csv(get_split_indices_path("pulse", "test", out_dir), _map_sources_to_first_rows(test_source_ids, "test"))

    with open(out_dir / "pulse_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def generate_splits(
    raw_npz_path: Path,
    out_dir: Path,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float
):
    """基于 raw_packed 生成 pulse/injury/ 两个任务的三套划分。

    输出结构：
    <out_dir>/pulse/
    <out_dir>/injury/{driver,passenger,combined}/

    其中pulse 仅保留一套完整数据划分。injury 的 combined 严格由 driver 与 passenger 对应 split 的并集构成，不会把某一侧的 val/test 样本混入另一侧的 train；
    
    """
    split_root = Path(out_dir)
    data = np.load(raw_npz_path)
    case_ids_all = data["case_ids"].astype(np.int64)
    if "pulse_source_case_ids" not in data.files:
        raise KeyError("raw_packed.npz 缺少必要键 'pulse_source_case_ids'，请先使用最新 prepare_data.py 重新打包")
    pulse_source_case_ids_all = data["pulse_source_case_ids"].astype(np.int64)
    x_att_raw = data["x_att_raw"].astype(np.float32)
    is_injury_ok = data["is_injury_ok"].astype(bool)
    mais = data["mais"].astype(np.int64)

    side_idx = FEATURE_ORDER.index("is_driver_side")
    side_values_raw = x_att_raw[:, side_idx]
    side_values_rounded = np.rint(side_values_raw).astype(np.int64)
    if not np.allclose(side_values_raw, side_values_rounded.astype(np.float32), atol=1e-6):
        raise ValueError("raw_packed 中 is_driver_side 列存在非整数值，无法生成主驾/副驾划分")
    if not np.isin(side_values_rounded, [0, 1]).all():
        bad_values = np.unique(side_values_rounded[~np.isin(side_values_rounded, [0, 1])]).tolist()
        raise ValueError(f"raw_packed 中 is_driver_side 只能取 0/1，当前检测到非法值: {bad_values}")

    side_masks = {
        "driver": side_values_rounded == 1,
        "passenger": side_values_rounded == 0,
    }
    side_labels = {
        "driver": "主驾",
        "passenger": "副驾",
        "combined": "主副驾合并",
    }

    # ------------------------------------------------------------------
    # 1) Pulse split：仅基于完整 pulse_source_case_id 全集独立划分，不区分主/副驾。
    # ------------------------------------------------------------------
    pulse_source_ids = np.unique(pulse_source_case_ids_all).astype(np.int64)
    pulse_train, pulse_val, pulse_test, pulse_summary = _split_case_ids_or_empty(
        case_ids=pulse_source_ids,
        stratify_labels=np.zeros(pulse_source_ids.shape[0], dtype=np.int64), # pulse_source_case_id 划分不进行分层，传入全零标签，效果等同于 random_split
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        rule="pulse_source_case_ids_random_split_independent_from_injury",
    )

    universe_set = set(np.unique(pulse_source_case_ids_all).astype(np.int64).tolist())
    for split_name, split_ids in {
        "train": pulse_train,
        "val": pulse_val,
        "test": pulse_test,
    }.items():
        unknown = sorted(set(split_ids.tolist()) - universe_set)
        if unknown:
            raise ValueError(
                f"pulse {split_name} split 中存在不在 pulse_source_case_ids 全集中的 source_id，共 {len(unknown)} 个，"
                f"示例: {unknown[:10]}"
            )

    pulse_summary.update({
        "task": "pulse",
        "variant": "all",
        "variant_label": "完整 pulse 数据",
        "raw_packed_path": str(Path(raw_npz_path).resolve()),
        "split_dir": str(get_pulse_split_dir(split_root).resolve()),
    })
    _save_pulse_split_with_first_occurrence_indices(
        out_dir=get_pulse_split_dir(split_root),
        pulse_source_case_ids_all=pulse_source_case_ids_all,
        train_source_ids=pulse_train,
        val_source_ids=pulse_val,
        test_source_ids=pulse_test,
        summary=pulse_summary,
    )

    # ------------------------------------------------------------------
    # 2) Injury split：主驾 / 副驾独立分层，combined = 两侧同名 split 的并集
    # ------------------------------------------------------------------
    # 先按照仅主驾和仅副驾去划分，然后按同名 split 做并集得到 combined
    
    injury_side_splits: Dict[str, Dict[str, np.ndarray]] = {}
    for variant in ("driver", "passenger"):
        variant_mask = is_injury_ok & side_masks[variant]
        variant_case_ids = case_ids_all[variant_mask]
        variant_labels = mais[variant_mask]
        train_ids, val_ids, test_ids, summary = _split_case_ids_or_empty(
            case_ids=variant_case_ids,
            stratify_labels=variant_labels,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            rule="injury_ok_only_stratify_by_MAIS_per_side",
        )
        summary.update({
            "task": "injury",
            "variant": variant,
            "variant_label": side_labels[variant],
            "raw_packed_path": str(Path(raw_npz_path).resolve()),
            "split_dir": str(get_injury_split_dir(variant, split_root).resolve()),
        })
        injury_side_splits[variant] = {
            "train": train_ids,
            "val": val_ids,
            "test": test_ids,
        }
        _save_split(
            out_dir=get_injury_split_dir(variant, split_root),
            prefix="injury",
            case_ids_all=case_ids_all,
            train_case_ids=train_ids,
            val_case_ids=val_ids,
            test_case_ids=test_ids,
            summary=summary,
        )

    injury_combined = _merge_split_triplets(
        injury_side_splits["driver"],
        injury_side_splits["passenger"],
    )
    summary_injury_combined = {
        "task": "injury",
        "variant": "combined",
        "variant_label": side_labels["combined"],
        "rule": "combined_split_equals_driver_split_union_passenger_split",
        "raw_packed_path": str(Path(raw_npz_path).resolve()),
        "split_dir": str(get_injury_split_dir("combined", split_root).resolve()),
        "component_split_dirs": {
            "driver": str(get_injury_split_dir("driver", split_root).resolve()),
            "passenger": str(get_injury_split_dir("passenger", split_root).resolve()),
        },
        "total_final": int(
            injury_combined["train"].size + injury_combined["val"].size + injury_combined["test"].size
        ),
        "train_final": int(injury_combined["train"].size),
        "val_final": int(injury_combined["val"].size),
        "test_final": int(injury_combined["test"].size),
        "driver_component_counts": {
            split_name: int(injury_side_splits["driver"][split_name].size)
            for split_name in ("train", "val", "test")
        },
        "passenger_component_counts": {
            split_name: int(injury_side_splits["passenger"][split_name].size)
            for split_name in ("train", "val", "test")
        },
        "date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    _save_split(
        out_dir=get_injury_split_dir("combined", split_root),
        prefix="injury",
        case_ids_all=case_ids_all,
        train_case_ids=injury_combined["train"],
        val_case_ids=injury_combined["val"],
        test_case_ids=injury_combined["test"],
        summary=summary_injury_combined,
    )

def main():
    parser = argparse.ArgumentParser(description="准备数据：raw_packed 打包 + injury 三套索引划分（driver/passenger/combined）+ pulse 单套完整划分")
    parser.add_argument(
        "--distribution",
        type=str,
        default=r"E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\仿真数据库相关\distribution\distribution_0424_del.csv",
        help="distribution 源文件路径，支持 .csv / .npz；绝对路径或相对路径均可",
    )
    parser.add_argument(
        "--pulse-dir",
        type=str,
        default=r"G:\VCS_acc_data\acc_data_before1111_6134",
        help="波形 CSV 目录路径，目录内应包含 x*.csv / y*.csv / z*.csv；绝对路径或相对路径均可",
    )
    
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)

    parser.add_argument("--out-raw",type=str,default=str(RAW_DATA),help="raw_packed 输出文件路径；绝对路径或相对路径均可",)
    parser.add_argument("--out-splits",type=str,default=str(SPLIT_ROOT_DIR),help="split 输出根目录；绝对路径或相对路径均可，目录下会自动创建 injury/{driver,passenger,combined} 与 pulse 子目录")
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED, help="随机种子; 默认值为common/tools/seeding.py中的GLOBAL_SEED")
    parser.add_argument("--non-strict", action="store_true", help="非严格模式：遇到缺失波形/异常case则跳过; 若无此标志则严格模式报错退出")

    args = parser.parse_args()
    set_random_seed(args.seed)

    ensure_dirs()

    distribution_path = Path(args.distribution)
    pulse_dir = Path(args.pulse_dir)
    out_raw = Path(args.out_raw)
    out_splits = Path(args.out_splits)
    print(f"⭐ distribution_path: {distribution_path}")
    print(f"⭐ pulse_dir: {pulse_dir}\n")
    # ========================================================== 
    package_raw_packed(
        distribution_path=distribution_path,
        pulse_dir=pulse_dir,
        output_npz=out_raw,
        strict=(not args.non_strict),
    )
    # ========================================================== 
    generate_splits(
        raw_npz_path=out_raw,
        out_dir=out_splits,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )

    print(f"\n✅️ raw_packed (原始数值尺度, 未归一化) 已生成: {out_raw}")
    print(f"✅️ injury combined split 已生成: {get_injury_split_dir('combined', out_splits)}")
    print(f"✅️ pulse split 已生成: {get_pulse_split_dir(out_splits)}")

    # ================================================================
    # 归一化配置生成逻辑
    # ================================================================
    print("\n" + "="*60)
    print("正在处理归一化配置...")
    print("="*60)
    
    # 加载打包数据和训练集索引
    raw_data = np.load(out_raw)
    train_indices_path = get_split_indices_path("injury", "train", get_injury_split_dir("combined", out_splits))
    
    if train_indices_path.exists():
        train_indices = load_int_vector_csv(train_indices_path)
        if train_indices.size == 0:
            raise ValueError("injury_train_indices.csv 为空，无法基于空训练集拟合 normalization_config。")
        # 构建训练集数据字典（仅用于统计量计算）
        train_data = {
            'x_att_raw': raw_data['x_att_raw'][train_indices], # shape: (N, len(FEATURE_ORDER))
            'x_acc_xy': raw_data['x_acc_xy'][train_indices] # shape: (N, WAVEFORM_CHANNELS_XY, WAVEFORM_LENGTH)
        }
        
        processor = UnifiedDataProcessor(config_path=NORMALIZATION_CONFIG_PATH)
        
        generated = processor.generate_config_if_absent(
            dataset_dict=train_data,
            top_k_waveform=50,
            dataset_id=str(out_raw.name),
            fit_split=str(train_indices_path.resolve())
        )
        if generated:
            print(f"✅️ 请检查并根据需要可手动编辑配置文件中的数值！")
        else:
            # 配置文件已存在，仅打印当前数据的统计量供用户参考
            print(f"[prepare_data] 配置文件已存在: {NORMALIZATION_CONFIG_PATH}")
            print(f"[prepare_data] 不会覆盖现有配置，以下为当前训练集的统计量（仅供参考）:")
            processor.print_computed_stats(dataset_dict=train_data, top_k_waveform=50)
    else:
        print(f"[prepare_data] 警告: 未找到训练集索引文件 {train_indices_path}，跳过归一化配置生成")
    
    print("="*60)
    print("✅️ [prepare_data] 数据准备完成！")
    print("="*60)


if __name__ == "__main__":
    main()
