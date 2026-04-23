from pathlib import Path
from typing import Iterable, Optional

"""项目级共享常量与路径约定。

设计原则：
1. 集中维护跨子项目共享的特征顺序、目录结构和命名规则，避免路径硬编码分散到各脚本。
2. 路径对象统一返回 `pathlib.Path`，由调用方决定是否转换为绝对路径字符串。

路径命名约定：
- `*_ROOT_DIR` 表示“容器目录”，其下通常还有 combined / driver / passenger 等子目录。
- 不带 `ROOT` 的 `*_DIR` 表示当前默认入口，供未显式传入目录的训练、评估或读取函数使用。
"""


# ================================================================
# 特征定义
# ================================================================

# FEATURE_ORDER 是全项目统一的完整工况特征顺序；raw_packed、normalization_config、PulsePredict、InjuryPredict 和 ARS_optim 的接口都依赖这份顺序，因此这里一旦修改，必须同步重建数据与模型输入接口。
FEATURE_ORDER = [
    "impact_velocity",
    "impact_angle",
    "overlap",
    "LL1",
    "LL2",
    "BTF",
    "LLATTF",
    "AFT",
    "SP",
    "SH",
    "RA",
    "is_driver_side",
    "OT",
]

# 前 11 列为连续特征，后 2 列为离散特征。
CONTINUOUS_INDICES = list(range(11))
DISCRETE_INDICES = [11, 12]

# 连续子向量中的列划分：`impact_angle` 和 `overlap` 采用 max-abs 归一化，其余连续特征采用 min-max 归一化。
MAXABS_INDICES_IN_CONTINUOUS = [1, 2]
MINMAX_INDICES_IN_CONTINUOUS = [0, 3, 4, 5, 6, 7, 8, 9, 10]

# 离散特征的固定取值映射；这不是训练时动态拟合的编码，而是项目约定的稳定映射。
DISCRETE_VALUE_TO_INDEX = {
    "is_driver_side": {"0": 0, "1": 1},
    "OT": {"1": 0, "2": 1, "3": 2},
}

# prepare_data 在打包 raw_packed 时要求的输入列；这里既包含完整工况参数，也包含波形和损伤任务所需的元信息与标签列。
REQUIRED_COLUMNS_FOR_PACKING = set(
    FEATURE_ORDER
    + [
        "case_id",
        "pulse_source_case_id",
        "is_pulse_ok",
        "is_injury_ok",
        "HIC15",
        "Dmax",
        "Nij",
    ]
)


# ================================================================
# 波形常量
# ================================================================

# 降采样后统一使用 150 个时间步。
WAVEFORM_LENGTH = 150

# XY 双轴波形供 InjuryPredict 与 ARS_optim 使用，XYZ 三轴波形保留在 raw_packed 中便于其他脚本按需扩展。
WAVEFORM_CHANNELS_XY = 2
WAVEFORM_CHANNELS_XYZ = 3


# ================================================================
# 目录结构
# ================================================================

# `settings.py` 位于 <repo>/common/ 下，因此父目录的父目录就是项目根目录。
ROOT_DIR = Path(__file__).resolve().parent.parent

# 项目共享数据目录。
DATA_DIR = ROOT_DIR / "data"

# prepare_data 生成的统一打包数据。
RAW_DATA_DIR = DATA_DIR / "raw_packed"
RAW_DATA = RAW_DATA_DIR / "raw_data_packed.npz"

# 全项目共享的一份归一化配置；当前约定是主副驾共用这一份配置，因此不再按 split 单独保存。
NORMALIZATION_CONFIG_PATH = DATA_DIR / "normalization_config.json"


# ================================================================
# 划分目录
# ================================================================

# split 目录固定为：data/split_indices/injury/<variant>/ 和 data/split_indices/pulse/。
SPLIT_ROOT_DIR = DATA_DIR / "split_indices"
# injury split 的容器目录，本身不直接存放 injury_train_indices.csv。
INJURY_SPLIT_ROOT_DIR = SPLIT_ROOT_DIR / "injury"
PULSE_SPLIT_DIR = SPLIT_ROOT_DIR / "pulse"

# 当前项目只承认这三种 injury 划分视角。
INJURY_SPLIT_VARIANTS = ("combined", "driver", "passenger")

# 当前项目只承认这三种数据划分名称。
SPLIT_PARTITIONS = ("train", "val", "test")

# 当前默认 injury 视角。做主副驾消融时，优先只改这里：
# - combined: 主副驾合训 / 合并评估入口
# - driver: 仅主驾训练评估入口
# - passenger: 仅副驾训练评估入口
DEFAULT_INJURY_VARIANT = "combined"
if DEFAULT_INJURY_VARIANT not in INJURY_SPLIT_VARIANTS:
    raise ValueError(
        f"DEFAULT_INJURY_VARIANT={DEFAULT_INJURY_VARIANT!r} is invalid, "
        f"expected one of {INJURY_SPLIT_VARIANTS}"
    )

# 当前默认 injury split 目录，只包含一套视角下的 train/val/test 索引文件。
# 需要非默认 split 时，推荐显式调用 get_injury_split_dir("driver"/"passenger"/"combined")。
INJURY_SPLIT_DIR = INJURY_SPLIT_ROOT_DIR / DEFAULT_INJURY_VARIANT


def _validate_split_name(split_name: str) -> str:
    split_name = str(split_name)
    if split_name not in SPLIT_PARTITIONS:
        raise ValueError(
            f"invalid split name: {split_name}, expected one of {SPLIT_PARTITIONS}"
        )
    return split_name


def _validate_injury_split_variant(variant: str) -> str:
    variant = str(variant)
    if variant not in INJURY_SPLIT_VARIANTS:
        raise ValueError(
            f"invalid injury split variant: {variant}, expected one of {INJURY_SPLIT_VARIANTS}"
        )
    return variant


def get_injury_split_dir(
    variant: str = DEFAULT_INJURY_VARIANT,
    split_root: Optional[Path] = None,
) -> Path:
    """返回 injury 任务某一套划分结果所在目录。

    参数说明：
    - `variant` 只能是 `combined / driver / passenger`。
    - `split_root` 若为 None，则使用默认 `data/split_indices`；若显式传入，则会在其下自动拼出 `injury/<variant>`。
    """
    variant = _validate_injury_split_variant(variant)
    base_root = (
        INJURY_SPLIT_ROOT_DIR
        if split_root is None
        else Path(split_root) / "injury"
    )
    return base_root / variant


def get_pulse_split_dir(split_root: Optional[Path] = None) -> Path:
    """返回 pulse 任务划分目录。PulsePredict 当前只有一套完整划分，因此这里不再带 `variant` 参数。"""
    return PULSE_SPLIT_DIR if split_root is None else Path(split_root) / "pulse"


def _default_split_dir_from_prefix(prefix: str) -> Path:
    prefix = str(prefix)
    if prefix == "injury":
        # get_split_indices_path("injury", ...) 的无参默认入口。
        # 非默认视角请传入 split_dir=get_injury_split_dir(<variant>)，避免隐式读取错目录。
        return INJURY_SPLIT_DIR
    if prefix == "pulse":
        return PULSE_SPLIT_DIR
    raise ValueError("prefix 只能是 'injury' 或 'pulse'")


def get_split_indices_path(
    prefix: str,
    split_name: str,
    split_dir: Optional[Path] = None,
) -> Path:
    """返回某个 split 索引文件路径。

    命名约定：
    - injury_train_indices.csv
    - pulse_val_indices.csv
    """
    split_name = _validate_split_name(split_name)
    base_dir = (
        _default_split_dir_from_prefix(prefix)
        if split_dir is None
        else Path(split_dir)
    )
    return base_dir / f"{prefix}_{split_name}_indices.csv"


def get_split_case_ids_path(
    prefix: str,
    split_name: str,
    split_dir: Optional[Path] = None,
) -> Path:
    """返回某个 split 的 case_id 文件路径。

    该文件与索引文件一一对应，只是保存 case_id 而非 raw_packed 行号。
    """
    split_name = _validate_split_name(split_name)
    base_dir = (
        _default_split_dir_from_prefix(prefix)
        if split_dir is None
        else Path(split_dir)
    )
    return base_dir / f"{prefix}_{split_name}_case_ids.csv"


# ================================================================
# InjuryPredict 处理后数据目录
# ================================================================

PROCESSED_DATA_DIR = DATA_DIR / "processed"
# InjuryPredict processed 数据的容器目录，本身不直接存放 train_dataset.pt。
INJURY_PROCESSED_ROOT_DIR = PROCESSED_DATA_DIR / "injury"
# InjuryPredict 的所有训练评估脚本，在没有显式传入 processed_dir 时，都会默认从这里读取 `.pt` 子集文件。
# 它与 DEFAULT_INJURY_VARIANT 同步，避免 split 默认视角和 processed 默认视角不一致。
INJURY_PROCESSED_DIR = INJURY_PROCESSED_ROOT_DIR / DEFAULT_INJURY_VARIANT


def get_injury_processed_dir(
    variant: str = DEFAULT_INJURY_VARIANT,
    processed_root: Optional[Path] = None,
) -> Path:
    """返回 InjuryPredict 在指定视角下的处理后数据目录。

    参数说明：
    - `variant` 只能是 `combined / driver / passenger`。
    - `processed_root` 若为 None，则使用默认 `data/processed/injury`；若显式传入，则直接把该目录视为 injury processed 根目录。
    """
    variant = _validate_injury_split_variant(variant)
    base_root = (
        INJURY_PROCESSED_ROOT_DIR
        if processed_root is None
        else Path(processed_root)
    )
    return base_root / variant


def get_injury_processed_dataset_path(
    split_name: str,
    processed_dir: Optional[Path] = None,
) -> Path:
    """返回 InjuryPredict 某个 split 对应的 `.pt` 文件路径。

    训练和评估入口通常不传 `processed_dir`，此时使用当前默认的
    `INJURY_PROCESSED_DIR`；需要临时读取其他视角时再显式传入目录。

    命名约定：
    - train_dataset.pt
    - val_dataset.pt
    - test_dataset.pt
    """
    split_name = _validate_split_name(split_name)
    base_dir = INJURY_PROCESSED_DIR if processed_dir is None else Path(processed_dir)
    return base_dir / f"{split_name}_dataset.pt"


# ================================================================
# 子项目根目录
# ================================================================

PULSE_PREDICT_DIR = ROOT_DIR / "PulsePredict"
INJURY_PREDICT_DIR = ROOT_DIR / "InjuryPredict"
ARS_OPTIM_DIR = ROOT_DIR / "ARS_optim"


def ensure_dirs(paths: Optional[Iterable[Path]] = None) -> None:
    """显式创建当前项目约定的关键目录。

    这里不在 import 时自动创建目录，避免普通读配置动作产生副作用；只有 prepare_data / Injurydata_prepare 这类入口脚本在真正需要落盘时再调用它。
    """
    required_dirs = [
        DATA_DIR,
        RAW_DATA_DIR,
        SPLIT_ROOT_DIR,
        INJURY_SPLIT_ROOT_DIR,
        PULSE_SPLIT_DIR,
        PROCESSED_DATA_DIR,
        INJURY_PROCESSED_ROOT_DIR,
    ]
    required_dirs.extend(get_injury_split_dir(variant) for variant in INJURY_SPLIT_VARIANTS)
    required_dirs.extend(get_injury_processed_dir(variant) for variant in INJURY_SPLIT_VARIANTS)
    if paths:
        required_dirs.extend(Path(path) for path in paths)

    for directory in {Path(path) for path in required_dirs}:
        directory.mkdir(parents=True, exist_ok=True)
