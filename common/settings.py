from pathlib import Path
from typing import Optional
''' 设置数据集特征相关的常量, 规范数据接口 '''
FEATURE_ORDER = [
    "impact_velocity", "impact_angle", "overlap",
    "LL1", "LL2", "BTF", "LLATTF", "AFT", "SP", "SH", "RA",
    "is_driver_side", "OT"
] # 共13个特征列，11个连续值+1个二分类标志位+1个整数OT; 顺序不可更改！严格依赖此顺序读取和存储数据，与损伤预测模型输入对应！

CONTINUOUS_INDICES = list(range(11)) # 在特征向量中的索引（对应前11个连续特征）

MAXABS_INDICES_IN_CONTINUOUS = [1, 2] # 在连续子向量(11维)中的索引（对应 impact_angle, overlap）

MINMAX_INDICES_IN_CONTINUOUS = [0, 3, 4, 5, 6, 7, 8, 9, 10] # 在连续子向量(11维)中的索引（对应 impact_velocity + 其余8个连续特征）

DISCRETE_INDICES = [11, 12] # 在特征向量中的索引（对应 is_driver_side, OT）

DISCRETE_VALUE_TO_INDEX = { # 按照 FEATURE_ORDER 中离散特征的顺序，定义每个离散特征的取值到索引的映射（从0开始）
    FEATURE_ORDER[DISCRETE_INDICES[0]]: {"0": 0, "1": 1}, # is_driver_side: 0=副驾, 1=主驾
    FEATURE_ORDER[DISCRETE_INDICES[1]]: {"1": 0, "2": 1, "3": 2} # OT: 1,2,3 分别映射为 0,1,2（共3类）
} # 固定离散映射（等价于LabelEncoder在这些取值上的编码）

REQUIRED_COLUMNS_FOR_PACKING = set(FEATURE_ORDER + [
    "case_id", "pulse_source_case_id", "is_pulse_ok", "is_injury_ok",
    "HIC15", "Dmax", "Nij"
])

# 波形相关常量
WAVEFORM_LENGTH = 150  # 降采样后波形长度
WAVEFORM_CHANNELS_XY = 2  # XY 双轴
WAVEFORM_CHANNELS_XYZ = 3  # XYZ 三轴

# ================================================================
''' 设置项目中的关键路径 '''
# 默认settings.py所在路径为项目根目录下的 common/，因此 ROOT_DIR 定义为 settings.py 的父目录的父目录，即项目根目录。
ROOT_DIR = Path(__file__).resolve().parent.parent 

# 数据目录
DATA_DIR = ROOT_DIR / "data_DS" # 数据目录路径，供全局使用；子项目内也可通过 common.settings.DATA_DIR 访问
RAW_DATA_DIR = DATA_DIR / "raw_packed"
RAW_DATA = RAW_DATA_DIR / "raw_data_packed.npz" # 打包后的原始数据文件路径
SPLIT_INDICES_DIR = DATA_DIR / "split_indices"

# 配置文件（全局共用）
NORMALIZATION_CONFIG_PATH = DATA_DIR / "normalization_config.json"

# 处理后数据（processed）目录 — 供子项目共享
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INJURY_PROCESSED_DIR = PROCESSED_DATA_DIR / "injury" # 专门为 InjuryPredict 处理后数据.pt 文件设置的子目录

# 子项目目录
PULSE_PREDICT_DIR = ROOT_DIR / "PulsePredict"
INJURY_PREDICT_DIR = ROOT_DIR / "InjuryPredict"
ARS_OPTIM_DIR = ROOT_DIR / "ARS_optim"

SPLIT_PARTITIONS = ("train", "val", "test")


def _validate_split_name(split_name: str) -> str:
    split_name = str(split_name)
    if split_name not in SPLIT_PARTITIONS:
        raise ValueError(f"invalid split name: {split_name}, expected one of {SPLIT_PARTITIONS}")
    return split_name


def get_split_indices_path(prefix: str, split_name: str, split_dir: Optional[Path] = None) -> Path:
    """集中管理 split 索引文件命名，避免各脚本散落硬编码。"""
    split_name = _validate_split_name(split_name)
    base_dir = SPLIT_INDICES_DIR if split_dir is None else Path(split_dir)
    return base_dir / f"{prefix}_{split_name}_indices.csv"


def get_split_case_ids_path(prefix: str, split_name: str, split_dir: Optional[Path] = None) -> Path:
    """集中管理 split case_id 文件命名。"""
    split_name = _validate_split_name(split_name)
    base_dir = SPLIT_INDICES_DIR if split_dir is None else Path(split_dir)
    return base_dir / f"{prefix}_{split_name}_case_ids.csv"


def get_injury_processed_dataset_path(split_name: str, processed_dir: Optional[Path] = None) -> Path:
    """统一 InjuryPredict 处理后数据集文件命名。"""
    split_name = _validate_split_name(split_name)
    base_dir = INJURY_PROCESSED_DIR if processed_dir is None else Path(processed_dir)
    return base_dir / f"{split_name}_dataset.pt"

def get_paths():
    """获取路径字典。"""
    return {
        "root": ROOT_DIR,
        "data": DATA_DIR,
        "raw_packed_dir": RAW_DATA_DIR,
        "raw_packed_data": RAW_DATA,
        "split_indices": SPLIT_INDICES_DIR,
        "processed": PROCESSED_DATA_DIR,
        "injury_processed": INJURY_PROCESSED_DIR,
        "normalization_config": NORMALIZATION_CONFIG_PATH,
        "pulse_predict": PULSE_PREDICT_DIR,
        "injury_predict": INJURY_PREDICT_DIR,
        "ars_optim": ARS_OPTIM_DIR
    }

def ensure_dirs(paths=None):
    """显式创建关键目录，避免 import 时产生副作用。"""
    required_dirs = [DATA_DIR, RAW_DATA_DIR, SPLIT_INDICES_DIR, PROCESSED_DATA_DIR]
    if paths:
        required_dirs = list({Path(p) for p in required_dirs + list(paths)})
    for d in required_dirs:
        if not d.exists():
            d.mkdir(parents=True, exist_ok=True)
