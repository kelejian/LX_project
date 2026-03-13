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

from common.tools.seeding import GLOBAL_SEED
from common.settings import REQUIRED_COLUMNS_FOR_PACKING, FEATURE_ORDER
from common.settings import RAW_DATA_DIR, SPLIT_INDICES_DIR, NORMALIZATION_CONFIG_PATH, ensure_dirs
from common.settings import WAVEFORM_LENGTH, WAVEFORM_CHANNELS_XY, WAVEFORM_CHANNELS_XYZ
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


def package_raw_packed(
    distribution_path: Path,
    pulse_dir: Path,
    output_npz: Path,
    strict: bool = True,
    side: str = "both"
) -> Path:
    """打包原始数据到 raw_packed.npz。

    `side` 只控制当前打包时保留哪些乘员侧样本：
    - `DS`: 仅主驾 (`is_driver_side==1`)
    - `PS`: 仅副驾 (`is_driver_side==0`)
    - `both`: 主副驾全部保留

    该筛选发生在波形读取前，后续所有标量特征、标签与索引都只针对筛选后的样本。
    """
    df = _read_distribution(distribution_path)

    for col in REQUIRED_COLUMNS_FOR_PACKING:
        if col not in df.columns:
            raise ValueError(f"distribution 缺少必要列: {col}")

    side_normalized = str(side).strip().upper()
    if side_normalized not in {"PS", "DS", "BOTH"}:
        raise ValueError(f"side 参数无效: {side}, 必须是 'PS'（副驾）、'DS'（主驾）或 'both'（主/副驾）")

    if "is_driver_side" not in df.columns:
        raise ValueError("distribution 缺少 is_driver_side 列，无法按主/副驾筛选")

    df["is_driver_side"] = pd.to_numeric(df["is_driver_side"], errors="raise").astype(np.int64)
    invalid_side_mask = ~df["is_driver_side"].isin([0, 1])
    if invalid_side_mask.any():
        invalid_values = sorted(df.loc[invalid_side_mask, "is_driver_side"].unique().tolist())
        raise ValueError(f"distribution 中 is_driver_side 存在非法取值: {invalid_values}，仅允许 0(副驾) 或 1(主驾)")

    if side_normalized == "PS":
        df = df.loc[df["is_driver_side"] == 0].copy()
    elif side_normalized == "DS":
        df = df.loc[df["is_driver_side"] == 1].copy()

    if df.shape[0] == 0:
        raise RuntimeError(f"side={side} 筛选后没有剩余样本")
    print(f"⭐ side={side_normalized}，侧向筛选后剩余样本数: {df.shape[0]}")

    # 只打包 is_pulse_ok==True 的 case（包含主/副驾）
    pulse_ok_mask = df["is_pulse_ok"].fillna(False).astype(bool)
    pulse_df = df.loc[pulse_ok_mask].copy() # 仅包含 is_pulse_ok==True 的样本，后续如果 strict=False 则会在读取波形时进一步过滤掉那些缺失波形的 case
    if pulse_df.shape[0] == 0:
        raise RuntimeError(f"side={side_normalized} 筛选后没有 is_pulse_ok==True 的样本，无法打包")

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
    '''保存划分结果到指定目录。
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

    save_int_vector_csv(out_dir / f"{prefix}_train_case_ids.csv", train_case_ids)
    save_int_vector_csv(out_dir / f"{prefix}_val_case_ids.csv", val_case_ids)
    save_int_vector_csv(out_dir / f"{prefix}_test_case_ids.csv", test_case_ids)

    train_idx = case_ids_to_indices(case_ids_all, train_case_ids) # 将训练集 case_ids 转为对应的索引
    val_idx = case_ids_to_indices(case_ids_all, val_case_ids)   # 将验证集 case_ids 转为对应的索引
    test_idx = case_ids_to_indices(case_ids_all, test_case_ids) # 将测试集 case_ids 转为对应的索引

    save_int_vector_csv(out_dir / f"{prefix}_train_indices.csv", train_idx)
    save_int_vector_csv(out_dir / f"{prefix}_val_indices.csv", val_idx)
    save_int_vector_csv(out_dir / f"{prefix}_test_indices.csv", test_idx)

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

    - case_ids 文件仍保存 pulse_source_case_id（不改变语义）；
    - indices 文件保存 pulse_source_case_id 在 raw_packed 中“首次出现”的原始行索引，
      直接对应 raw_data_packed.npz 各数组的行。
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    save_int_vector_csv(out_dir / "pulse_train_case_ids.csv", train_source_ids)
    save_int_vector_csv(out_dir / "pulse_val_case_ids.csv", val_source_ids)
    save_int_vector_csv(out_dir / "pulse_test_case_ids.csv", test_source_ids)

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

    save_int_vector_csv(out_dir / "pulse_train_indices.csv", _map_sources_to_first_rows(train_source_ids, "train"))
    save_int_vector_csv(out_dir / "pulse_val_indices.csv", _map_sources_to_first_rows(val_source_ids, "val"))
    save_int_vector_csv(out_dir / "pulse_test_indices.csv", _map_sources_to_first_rows(test_source_ids, "test"))

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
    """
    基于打包数据.npz文件 生成 injury/pulse 两套划分结果。
    生成的./data 下保存的所有数据集索引(_indices.csv 文件)，严格对应 raw_data_packed.npz 中各个 np.ndarray 的行索引. raw_data_packed.npz 目前只包含 is_pulse_ok==True 的样本。
    """
    data = np.load(raw_npz_path)
    case_ids_all = data["case_ids"].astype(np.int64) # (N,), 全量 pulse_ok==True 的 case_ids
    if "pulse_source_case_ids" not in data.files:
        raise KeyError("raw_packed.npz 缺少必要键 'pulse_source_case_ids'，请先使用最新 prepare_data.py 重新打包")
    pulse_source_case_ids_all = data["pulse_source_case_ids"].astype(np.int64) # (N,), 与 case_ids_all 对齐
    x_att_raw = data["x_att_raw"].astype(np.float32)  # (N, len(FEATURE_ORDER))
    is_injury_ok = data["is_injury_ok"].astype(bool) # (N,), 打包时取值已统一布尔化，此处取值仅有 True/False
    mais = data["mais"].astype(np.int64)

    # 1) injury split：仅基于 injury_ok==True 的子集，按 MAIS 分层
    injury_mask = is_injury_ok
    injury_case_ids = case_ids_all[injury_mask]
    injury_labels = mais[injury_mask]

    train_inj, val_inj, test_inj, summary_inj = stratified_split_case_ids(
        injury_case_ids, # 仅 injury_ok==True 的 case_ids
        injury_labels,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        special_case_assignments=None,
        seed=seed
    ) # 返回的是划分后的 case_ids 列表

    summary_inj.update({
        "rule": "injury_ok_only_stratify_by_MAIS",
        "date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    })

    # 此处的 case_ids_all 是全量 pulse_ok==True 的 case_ids; train_inj/val_inj/test_inj 为划分后的 case_ids 列表
    _save_split(out_dir, "injury", case_ids_all, train_inj, val_inj, test_inj, summary_inj)

    # 2) pulse split：按 pulse_source_case_id 进行继承与分配
    # ------------------------------------------------------------------
    rng = np.random.default_rng(seed)
    split_ratio_map = {
        "train": float(train_ratio),
        "val": float(val_ratio),
        "test": float(test_ratio),
    }

    train_inj_set = set(train_inj.tolist())
    val_inj_set = set(val_inj.tolist())
    test_inj_set = set(test_inj.tolist())
    case_to_split = {}
    case_to_split.update({cid: "train" for cid in train_inj_set})
    case_to_split.update({cid: "val" for cid in val_inj_set})
    case_to_split.update({cid: "test" for cid in test_inj_set})
    # e.g.,case_to_split == {100:'train', 103:'train', 101:'val', 105:'test', 50106:'val'}

    # 继承 injury split 中的 pulse_source_case_id（去重）
    injury_case_id_to_source = dict(zip(case_ids_all[injury_mask].tolist(), pulse_source_case_ids_all[injury_mask].tolist()))
    # e.g., injury_case_id_to_source == {100:100, 101:100, 103:101, 105:105, 50106:106}，其中 key 是 injury_case_id，value 是对应的 pulse_source_case_id
    source_membership = {}
    for cid, src in injury_case_id_to_source.items():
        if cid not in case_to_split:
            raise ValueError(f"case_id {cid} 在 injury split 中但不在 case_to_split 映射中")
        src_int = int(src)
        source_membership.setdefault(src_int, set()).add(case_to_split[cid])
        # e.g., source_membership == {100:{'train','val'}, 101:{'train'}, 105:{'test'}, 106:{'val'}}，其中 key 是 pulse_source_case_id，value 是一个集合，表示该 pulse_source_case_id 对应的 case_id 在 injury split 中分别属于哪些划分（train/val/test）

    inherited_sources = {"train": set(), "val": set(), "test": set()}
    for src, memberships in source_membership.items():
        members = sorted(list(memberships))
        if len(members) == 1: # 比如 source_membership 中的 101 只出现在 train_inj 中，那么它的 pulse_source_case_id=101 就直接被 train_inj 继承，无需按比例分配
            inherited_sources[members[0]].add(src)
            continue
        # print(f"⚠️ 冲突：pulse_source_case_id={src} 对应的 case_id 在 injury split 中同时出现在 {members} 中，将按 train/val/test_ratio 进行分配")
        # 同一 pulse_source 出现在多个 injury 划分：按 train/val/test_ratio（在冲突成员内归一化）分配
        weights = np.array([split_ratio_map[m] for m in members], dtype=np.float64)
        weight_sum = float(weights.sum())
        if weight_sum <= 0:
            raise ValueError(f"划分比例之和必须大于0，当前为 {weight_sum}，请检查 train/val/test_ratio 的设置")
        else:
            probs = weights / weight_sum
        chosen_split = members[int(rng.choice(np.arange(len(members)), p=probs))] # e.g., 对于 source_membership 中的 100，members=['train','val']，则chosen_split 可能是 'train' 或 'val'，概率分别为 train_ratio/(train_ratio+val_ratio) 和 val_ratio/(train_ratio+val_ratio)
        inherited_sources[chosen_split].add(src)
    # e.g., inherited_sources == {'train': {100, 101}, 'val': {106}, 'test': {105}}, 其中 key 是划分（train/val/test），value 是一个集合，表示该划分继承了哪些 pulse_source_case_id (而非 case_id！)

    inherited_union = set().union(*inherited_sources.values()) # 将字典的值合并在集合中，e.g. inherited_union == {100,101,105,106}, 均为 pulse_source_case_id

    # pulse-only: pulse_ok==True 且 is_injury_ok!=True 且其 pulse_source 尚未被继承
    pulse_only_mask = (~is_injury_ok) & (~np.isin(pulse_source_case_ids_all, list(inherited_union)))
    # pulse_only_sources 是 pulse_only_mask 中对应的 pulse_source_case_id 的唯一值列表（整数类型）
    pulse_only_sources = np.unique(pulse_source_case_ids_all[pulse_only_mask]).astype(np.int64)
    shuffled_sources = pulse_only_sources.copy()
    rng.shuffle(shuffled_sources)

    n_total_src = int(shuffled_sources.shape[0])
    n_train_src = int(round(n_total_src * train_ratio))
    n_val_src = int(round(n_total_src * val_ratio))
    extra_train_sources = set(shuffled_sources[:n_train_src].tolist())
    extra_val_sources = set(shuffled_sources[n_train_src:n_train_src + n_val_src].tolist())
    extra_test_sources = set(shuffled_sources[n_train_src + n_val_src:].tolist())

    final_sources = {
        "train": inherited_sources["train"] | extra_train_sources,
        "val": inherited_sources["val"] | extra_val_sources,
        "test": inherited_sources["test"] | extra_test_sources,
    }

    # 波形预测数据集直接使用 pulse_source_case_ids 作为 case_ids（确保唯一性）
    pulse_train = np.asarray(sorted(final_sources["train"]), dtype=np.int64)
    pulse_val = np.asarray(sorted(final_sources["val"]), dtype=np.int64)
    pulse_test = np.asarray(sorted(final_sources["test"]), dtype=np.int64)

    # 冲突检查：同一case不能同时出现在多个集合
    if (set(pulse_train.tolist()) & set(pulse_val.tolist())) or \
       (set(pulse_train.tolist()) & set(pulse_test.tolist())) or \
       (set(pulse_val.tolist()) & set(pulse_test.tolist())):
        raise ValueError("pulse split 生成后检测到集合交叉")

    summary_pulse = {
        "rule": "inherit_injury_pulse_source_ids + resolve_cross_split_conflicts_by_ratio + add_pulse_only_sources_by_ratio",
        "total_final": int(pulse_train.shape[0] + pulse_val.shape[0] + pulse_test.shape[0]),
        "train_final": int(pulse_train.shape[0]),
        "val_final": int(pulse_val.shape[0]),
        "test_final": int(pulse_test.shape[0]),
        "inherited_source_counts": {
            "train": int(len(inherited_sources["train"])),
            "val": int(len(inherited_sources["val"])),
            "test": int(len(inherited_sources["test"])),
        },
        "pulse_only_source_counts": {
            "train": int(len(extra_train_sources)),
            "val": int(len(extra_val_sources)),
            "test": int(len(extra_test_sources)),
            "total": int(n_total_src),
        },
        "date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # 严格校验：所有 split 的 source_id 必须都来自 raw_packed 中出现过的 pulse_source_case_id
    universe_set = set(np.unique(pulse_source_case_ids_all).astype(np.int64).tolist())
    for split_name, split_ids in {
        "train": pulse_train,
        "val": pulse_val,
        "test": pulse_test,
    }.items():
        unknown = sorted(set(split_ids.tolist()) - universe_set)
        if unknown:
            raise ValueError(
                f"pulse {split_name} split 中存在不在 pulse_source_case_ids 全集中的ID，共 {len(unknown)} 个，示例: {unknown[:10]}"
            )

    _save_pulse_split_with_first_occurrence_indices(
        out_dir=out_dir,
        pulse_source_case_ids_all=pulse_source_case_ids_all,
        train_source_ids=pulse_train,
        val_source_ids=pulse_val,
        test_source_ids=pulse_test,
        summary=summary_pulse,
    )


def main():
    parser = argparse.ArgumentParser(description="准备数据：raw_packed打包 + injury/pulse两套索引划分")
    parser.add_argument("--distribution", type=str, 
                        default=r'E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\仿真数据库相关\distribution\distribution_0311.csv',  
                        help="distribution .csv/.npz 路径")
    parser.add_argument("--pulse-dir", type=str, 
                        default=r'G:\VCS_acc_data\acc_data_before1111_6134', 
                        help="波形CSV目录（包含x*.csv/y*.csv/z*.csv）")
    #--------------------------------------------------------------------------------------
    parser.add_argument("--out-raw", type=str, 
                        default=str(RAW_DATA_DIR / "raw_data_packed.npz"), 
                        help="输出raw_packed npz路径")
    parser.add_argument("--out-splits", type=str, default=str(SPLIT_INDICES_DIR), help="输出split_indices目录")
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED, help="随机种子; 默认值为common/utils/seeding.py中的GLOBAL_SEED")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--non-strict", action="store_true", help="非严格模式：遇到缺失波形/异常case则跳过; 若无此标志则严格模式报错退出")
    parser.add_argument("--side", type=str, default="both", help="选择打包哪一侧样本: DS(主驾) | PS(副驾) | both(主副驾)")

    args = parser.parse_args()

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
        side=args.side,
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
    print(f"✅️ split_indices 已生成: {out_splits}")

    # ================================================================
    # 归一化配置生成逻辑
    # ================================================================
    print("\n" + "="*60)
    print("正在处理归一化配置...")
    print("="*60)
    
    # 加载打包数据和训练集索引
    raw_data = np.load(out_raw)
    train_indices_path = out_splits / "injury_train_indices.csv"
    
    if train_indices_path.exists():
        train_indices = load_int_vector_csv(train_indices_path)
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
            fit_split=str(train_indices_path.name)
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
