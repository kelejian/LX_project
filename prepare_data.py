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

from common.utils.seeding import GLOBAL_SEED
from common.settings import CASE_ID_OFFSET_DEFAULT, REQUIRED_COLUMNS_FOR_PACKING, FEATURE_ORDER
from common.settings import RAW_DATA_DIR, SPLIT_INDICES_DIR, NORMALIZATION_CONFIG_PATH, ensure_dirs
from common.settings import WAVEFORM_LENGTH, WAVEFORM_CHANNELS_XY, WAVEFORM_CHANNELS_XYZ, DISCRETE_INDICES
from common.data_utils.splitter import stratified_split_case_ids, case_ids_to_indices
from common.data_utils.processor import UnifiedDataProcessor
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


def _load_xyz_waveforms(pulse_dir: Path, case_id: int, is_driver_side: int, case_id_offset: int) -> Tuple[np.ndarray, np.ndarray]:
    # 返回 (xyz[WAVEFORM_CHANNELS_XYZ, WAVEFORM_LENGTH], xy[WAVEFORM_CHANNELS_XY, WAVEFORM_LENGTH])
    driver_case_id = int(case_id) if int(is_driver_side) == 1 else int(case_id) - int(case_id_offset)

    x_path = pulse_dir / f"x{driver_case_id}.csv"
    y_path = pulse_dir / f"y{driver_case_id}.csv"
    z_path = pulse_dir / f"z{driver_case_id}.csv"

    if not x_path.exists() or not y_path.exists():
        missing = [str(p) for p in [x_path, y_path] if not p.exists()]
        raise FileNotFoundError(f"波形文件缺失(case_id={case_id}): {missing}")

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
    is_driver_side: np.ndarray,
    case_id_offset: int,
    strict: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """批量读取波形。

    注意：波形存储在独立 CSV 文件中，逐文件 I/O 无法完全向量化。
    由于主副驾数据 case_id 相差 case_id_offset，因此只读取主驾波形并复用到副驾。
    这里做的是批量封装，减少循环体中的 pandas 操作，并可在 non-strict 模式下跳过异常样本。
    Args:
        pulse_dir: 波形 CSV 文件目录
        case_ids: (N,) 待读取的 case_id 列表（包含主/副驾）. 本文件中指is_pulse_ok==True的case_ids
        is_driver_side: (N,) 对应 case_ids 的主副驾标志位数组（1=主驾，0=副驾）
        case_id_offset: 主副驾 case_id 差值
        strict: 是否严格模式（遇到缺失波形/异常 case 则报错退出）

    Returns:
        x_acc_xyz: (M, WAVEFORM_CHANNELS_XYZ, WAVEFORM_LENGTH)
        x_acc_xy:  (M, WAVEFORM_CHANNELS_XY, WAVEFORM_LENGTH)
        ok_mask:   (N,) 表示输入 case_ids 中哪些成功读取
    """
    if case_ids.ndim != 1:
        raise ValueError("case_ids 必须是一维数组")
    if is_driver_side.shape[0] != case_ids.shape[0]:
        raise ValueError("is_driver_side 长度必须与 case_ids 一致")

    # 将 case_id 映射到对应的主驾 case_id（副驾 case_id - offset）
    driver_case_ids = np.where(is_driver_side.astype(int) == 1, case_ids, case_ids - int(case_id_offset))

    driver_to_indices: Dict[int, list] = {}
    driver_order = []
    for idx, d in enumerate(driver_case_ids.tolist()):
        d_int = int(d)
        if d_int not in driver_to_indices:
            driver_to_indices[d_int] = []
            driver_order.append(d_int)
        driver_to_indices[d_int].append(idx)

    n = case_ids.shape[0]
    x_acc_xyz = np.empty((n, WAVEFORM_CHANNELS_XYZ, WAVEFORM_LENGTH), dtype=np.float32)
    x_acc_xy = np.empty((n, WAVEFORM_CHANNELS_XY, WAVEFORM_LENGTH), dtype=np.float32)
    ok_mask = np.zeros((n,), dtype=bool)

    for driver_id in  tqdm(driver_order, total=len(driver_order), desc="读取主驾波形", unit="case"):
        try:
            # 只读取主驾波形（driver_id 本身就是主驾 case_id）
            xyz, xy = _load_xyz_waveforms(pulse_dir, int(driver_id), 1, case_id_offset)
            for idx in driver_to_indices[int(driver_id)]:
                x_acc_xyz[idx] = xyz
                x_acc_xy[idx] = xy
                ok_mask[idx] = True
        except Exception:
            if strict:
                raise
            # non-strict: 该主驾及其对应副驾全部标记失败
            for idx in driver_to_indices[int(driver_id)]:
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
    case_id_offset: int = CASE_ID_OFFSET_DEFAULT,
    strict: bool = True
) -> Path:
    df = _read_distribution(distribution_path)

    for col in REQUIRED_COLUMNS_FOR_PACKING:
        if col not in df.columns:
            raise ValueError(f"distribution 缺少必要列: {col}")

    # 只打包 is_pulse_ok==True 的 case（包含主/副驾）
    pulse_ok_mask = df["is_pulse_ok"].fillna(False).astype(bool)
    pulse_df = df.loc[pulse_ok_mask].copy()
    if pulse_df.shape[0] == 0:
        raise RuntimeError("没有成功打包任何数据")

    # ---------------------------
    # 1) 向量化：case_ids / params / 标志位
    # ---------------------------
    case_ids_all = pulse_df["case_id"].astype(int).to_numpy(dtype=np.int64)
    x_att_raw_all = pulse_df[FEATURE_ORDER].to_numpy(dtype=np.float32) # (N, len(FEATURE_ORDER)), 无列名，纯数值, 因此后续如果需要知道每一列的含义必须依赖 FEATURE_ORDER 的顺序！
    is_pulse_ok_all = pulse_df["is_pulse_ok"].fillna(False).astype(bool).to_numpy(dtype=bool) # 原始 distribution 中的缺失值会变成 False；以及 能被解释为 False 的值（例如 False、0、空字符串）会变成 False
    is_injury_ok_all = pulse_df["is_injury_ok"].fillna(False).astype(bool).to_numpy(dtype=bool) # 原始 distribution 中的缺失值会变成 False；以及 能被解释为 False 的值（例如 False、0、空字符串）会变成 False

    hic15_all = pulse_df["HIC15"].to_numpy(dtype=np.float32)
    dmax_all = pulse_df["Dmax"].to_numpy(dtype=np.float32)
    nij_all = pulse_df["Nij"].to_numpy(dtype=np.float32)
    ot_all = pulse_df["OT"].astype(int).to_numpy(dtype=np.int64)

    is_driver_side_all = x_att_raw_all[:, DISCRETE_INDICES[0]].astype(np.int64)  # is_driver_side

    print(f"✅️ 标量参数已打包，准备打包波形数据 (strict={strict})")

    # ---------------------------
    # 2) 波形读取：无法彻底向量化（逐文件 I/O），但可批量封装
    # ---------------------------
    x_acc_xyz, x_acc_xy, ok_mask = _load_waveforms_batch(
        pulse_dir=pulse_dir,
        case_ids=case_ids_all, # 仅 is_pulse_ok==True 的 case_ids
        is_driver_side=is_driver_side_all,
        case_id_offset=case_id_offset,
        strict=strict,
    )

    # 过滤掉 non-strict 下失败的样本
    # 如果 strict 模式（默认，即未设置 --non-strict），则上面会直接报错退出
    case_ids = case_ids_all[ok_mask]
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

    np.savez(
        output_npz,
        case_ids=case_ids.astype(np.int64), # (n,)
        x_att_raw=x_att_raw.astype(np.float32), # ndarray (n,13)
        x_acc_xyz=x_acc_xyz.astype(np.float32), # ndarray (n,3,150)
        x_acc_xy=x_acc_xy.astype(np.float32), # ndarray (n,2,150)
        is_pulse_ok=is_pulse_ok.astype(bool), # (n,)
        is_injury_ok=is_injury_ok.astype(bool), # (n,)
        y_HIC=y_hic.astype(np.float32), # (n,)
        y_Dmax=y_dmax.astype(np.float32), # (n,)
        y_Nij=y_nij.astype(np.float32), # (n,)
        ais_head=ais_head.astype(np.int64), # (n,)
        ais_chest=ais_chest.astype(np.int64), # (n,)
        ais_neck=ais_neck.astype(np.int64), # (n,)
        mais=mais.astype(np.int64) # (n,)
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

    np.save(out_dir / f"{prefix}_train_case_ids.npy", train_case_ids.astype(np.int64))
    np.save(out_dir / f"{prefix}_val_case_ids.npy", val_case_ids.astype(np.int64))
    np.save(out_dir / f"{prefix}_test_case_ids.npy", test_case_ids.astype(np.int64))

    train_idx = case_ids_to_indices(case_ids_all, train_case_ids) # 将训练集 case_ids 转为对应的索引
    val_idx = case_ids_to_indices(case_ids_all, val_case_ids)   # 将验证集 case_ids 转为对应的索引
    test_idx = case_ids_to_indices(case_ids_all, test_case_ids) # 将测试集 case_ids 转为对应的索引

    np.save(out_dir / f"{prefix}_train_indices.npy", train_idx)
    np.save(out_dir / f"{prefix}_val_indices.npy", val_idx)
    np.save(out_dir / f"{prefix}_test_indices.npy", test_idx)

    with open(out_dir / f"{prefix}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def generate_splits(
    raw_npz_path: Path,
    out_dir: Path,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float
):
    """基于打包数据.npz文件 生成 injury/pulse 两套划分结果。"""
    data = np.load(raw_npz_path)
    case_ids_all = data["case_ids"].astype(np.int64) # (N,), 全量 pulse_ok==True 的 case_ids
    x_att_raw = data["x_att_raw"].astype(np.float32)  # (N, len(FEATURE_ORDER))
    is_injury_ok = data["is_injury_ok"].astype(bool) # (N,), 打包时取值已统一布尔化，此处取值仅有 True/False
    mais = data["mais"].astype(np.int64)

    is_driver_side = x_att_raw[:, DISCRETE_INDICES[0]].astype(int)  # is_driver_side

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

    # 2) pulse split：主驾侧(is_driver_side==1)
    driver_mask = (is_driver_side == 1) # 仅主驾侧

    injury_driver_set = set(injury_case_ids[driver_mask[injury_mask]].tolist())  # injury_ok==True 且为主驾侧的 case_ids

    train_inj_set = set(train_inj.tolist())  # 继承 injury split 的训练集（仅取主驾侧）的 case_ids
    val_inj_set = set(val_inj.tolist())      # 继承 injury split 的验证集（仅取主驾侧）的 case_ids
    test_inj_set = set(test_inj.tolist())    # 继承 injury split 的测试集（仅取主驾侧）的 case_ids

    # 损伤预测数据集中的主驾侧 case_ids（is_injury_ok==True 且 is_driver_side==1）, 直接继承到波形预测数据集中
    pulse_train = [cid for cid in train_inj_set if cid in injury_driver_set]
    pulse_val = [cid for cid in val_inj_set if cid in injury_driver_set]
    pulse_test = [cid for cid in test_inj_set if cid in injury_driver_set]

    inherit_counts = {
        "train": len(pulse_train),
        "val": len(pulse_val),
        "test": len(pulse_test)
    }

    # 将pulse-only的case_ids，按比例随机分配到 train/val/test 中
    # pulse-only：pulse_ok==True(已经是全量raw) 其 is_driver_side==1 且 is_injury_ok!=True
    pulse_only_mask = driver_mask & (~is_injury_ok) # 为主驾侧 且 injury_ok!=True
    pulse_only_case_ids = case_ids_all[pulse_only_mask] # case_ids_all 是所有 pulse_ok==True 的 case_ids

    rng = np.random.default_rng(seed)
    shuffled = pulse_only_case_ids.copy() # 注意：这里必须 copy 一份，避免修改原始数组
    rng.shuffle(shuffled)

    n_total = int(shuffled.shape[0])
    n_train = int(round(n_total * train_ratio))
    n_val = int(round(n_total * val_ratio))

    extra_train = shuffled[:n_train]
    extra_val = shuffled[n_train:n_train + n_val]
    extra_test = shuffled[n_train + n_val:]

    # 合并继承的 injury_driver case_ids 和 新分配的 pulse-only case_ids, 作为最终的 pulse split 结果
    pulse_train = np.asarray(sorted(set(pulse_train) | set(extra_train.tolist())), dtype=np.int64)
    pulse_val = np.asarray(sorted(set(pulse_val) | set(extra_val.tolist())), dtype=np.int64)
    pulse_test = np.asarray(sorted(set(pulse_test) | set(extra_test.tolist())), dtype=np.int64)

    # 冲突检查：同一case不能同时出现在多个集合
    if (set(pulse_train.tolist()) & set(pulse_val.tolist())) or \
       (set(pulse_train.tolist()) & set(pulse_test.tolist())) or \
       (set(pulse_val.tolist()) & set(pulse_test.tolist())):
        raise ValueError("pulse split 生成后检测到集合交叉")

    summary_pulse = {
        "rule": "inherit_injury_driver + add_pulse_only_driver_by_ratio",
        "total_final": int(pulse_only_case_ids.shape[0] + len(injury_driver_set)), # pulse_ok==True且为主驾侧的case总数, 即为 pulse split 的总数
        "train_final": int(pulse_train.shape[0]),
        "val_final": int(pulse_val.shape[0]),
        "test_final": int(pulse_test.shape[0]),
        "inherited_from_injury_driver": inherit_counts,
        "date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # case_ids_all 仍为全量 pulse_ok==True 的 case_ids; pulse_train/val/test 为划分后的 case_ids 列表
    _save_split(out_dir, "pulse", case_ids_all, pulse_train, pulse_val, pulse_test, summary_pulse)


def main():
    parser = argparse.ArgumentParser(description="准备数据：raw_packed打包 + injury/pulse两套索引划分")
    parser.add_argument("--distribution", type=str, 
                        default=r'E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\仿真数据库相关\distribution\distribution_0123.csv',  
                        help="distribution .csv/.npz 路径")
    parser.add_argument("--pulse-dir", type=str, 
                        default=r'G:\VCS_acc_data\acc_data_before1111_6134', 
                        help="波形CSV目录（包含x*.csv/y*.csv/z*.csv）")
    parser.add_argument("--out-raw", type=str, 
                        default=str(RAW_DATA_DIR / "raw_data_packed.npz"), 
                        help="输出raw_packed npz路径")
    parser.add_argument("--out-splits", type=str, default=str(SPLIT_INDICES_DIR), help="输出split_indices目录")
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED, help="随机种子; 默认值为common/utils/seeding.py中的GLOBAL_SEED")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--case-id-offset", type=int, default=CASE_ID_OFFSET_DEFAULT)
    parser.add_argument("--non-strict", action="store_true", help="非严格模式：遇到缺失波形/异常case则跳过; 若无此标志则严格模式报错退出")

    args = parser.parse_args()

    ensure_dirs()

    distribution_path = Path(args.distribution)
    pulse_dir = Path(args.pulse_dir)
    out_raw = Path(args.out_raw)
    out_splits = Path(args.out_splits)
    print(f"⭐ distribution_path: {distribution_path}")
    print(f"⭐ pulse_dir: {pulse_dir}\n")
    # ========================================================== 
    # package_raw_packed(
    #     distribution_path=distribution_path,
    #     pulse_dir=pulse_dir,
    #     output_npz=out_raw,
    #     case_id_offset=args.case_id_offset,
    #     strict=(not args.non_strict)
    # )
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
    train_indices_path = out_splits / "injury_train_indices.npy"
    
    if train_indices_path.exists():
        train_indices = np.load(train_indices_path)
        # 构建训练集数据字典（仅用于统计量计算）
        train_data = {
            'x_att_raw': raw_data['x_att_raw'][train_indices], # shape: (N, len(FEATURE_ORDER))
            'x_acc_xy': raw_data['x_acc_xy'][train_indices] # shape: (N, WAVEFORM_CHANNELS_XY, WAVEFORM_LENGTH)
        }
        
        processor = UnifiedDataProcessor(config_path=NORMALIZATION_CONFIG_PATH)
        
        if not NORMALIZATION_CONFIG_PATH.exists():
            # 配置文件不存在，基于训练集计算统计量并生成
            print(f"[prepare_data] 配置文件不存在，正在基于{out_raw.name}的训练集:{train_indices_path.name} 生成...")
            generated = processor.generate_config_if_absent(
                dataset_dict=train_data,
                top_k_waveform=50,
                dataset_id=str(out_raw.name),
                fit_split=str(train_indices_path.name)
            )
            if generated:
                print(f"\n⚠️  归一化配置已生成: {NORMALIZATION_CONFIG_PATH}")
                print(f"⚠️  请检查并根据需要手动编辑配置文件中的数值！")
        else:
            # 配置文件已存在，仅打印当前数据的统计量供用户参考
            print(f"[prepare_data] 配置文件已存在: {NORMALIZATION_CONFIG_PATH}")
            print(f"[prepare_data] 不会覆盖现有配置，以下为当前训练集的统计量（仅供参考）:")
            processor.print_computed_stats(dataset_dict=train_data, top_k_waveform=50)
    else:
        print(f"[prepare_data] 警告: 未找到训练集索引文件 {train_indices_path}，跳过归一化配置生成")
    
    print("="*60)
    print("✅️ 数据准备完成！")
    print("="*60)


if __name__ == "__main__":
    main()
