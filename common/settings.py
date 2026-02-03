from pathlib import Path
import os

''' 设置数据集特征相关的常量, 规范数据接口 '''
CASE_ID_OFFSET_DEFAULT = 50000 # 默认的主副驾 case_id 差量，可用于生成识别原始碰撞波形csv文件的唯一case_id

FEATURE_ORDER = [
    "impact_velocity", "impact_angle", "overlap",
    "LL1", "LL2", "BTF", "LLATTF", "AFT", "SP", "SH", "RA",
    "is_driver_side", "OT"
] # 共12个特征列，11个连续值+1个二分类标志位+1个整数OT; 顺序不可更改！严格依赖此顺序读取和存储数据，与损伤预测模型输入对应！

CONTINUOUS_INDICES = list(range(11)) # 在特征向量中的索引（对应前11个连续特征）

DISCRETE_INDICES = [11, 12] # 在特征向量中的索引（对应 is_driver_side, OT）

MAXABS_INDICES_IN_CONTINUOUS = [1, 2] # 在连续子向量(11维)中的索引（对应 impact_angle, overlap）

MINMAX_INDICES_IN_CONTINUOUS = [0, 3, 4, 5, 6, 7, 8, 9, 10] # 在连续子向量(11维)中的索引（对应 impact_velocity + 其余8个连续特征）

DISCRETE_VALUE_TO_INDEX = {
    "is_driver_side": {"0": 0, "1": 1},
    "OT": {"1": 0, "2": 1, "3": 2}
} # 固定离散映射（等价于LabelEncoder在这些取值上的编码）

REQUIRED_COLUMNS_FOR_PACKING = set(FEATURE_ORDER + [
    "case_id", "is_pulse_ok", "is_injury_ok",
    "HIC15", "Dmax", "Nij"
])

# ================================================================
''' 设置项目中的关键路径 '''
# Project Root (LX_project/)
# 保证此文件位于 LX_project/common/settings.py
_DEFAULT_ROOT = Path(__file__).resolve().parent.parent

# 允许使用环境变量覆盖（便于多机/多目录部署）
ROOT_DIR = Path(os.environ.get("LX_PROJECT_ROOT", _DEFAULT_ROOT))

# 数据目录
DATA_DIR = Path(os.environ.get("LX_DATA_DIR", ROOT_DIR / "data"))
RAW_DATA_DIR = DATA_DIR / "raw_packed"
SPLIT_INDICES_DIR = DATA_DIR / "split_indices"

# 配置文件（全局共用）
NORMALIZATION_CONFIG_PATH = DATA_DIR / "normalization.json"

def get_paths():
    """获取路径字典（不产生副作用）。"""
    return {
        "root": ROOT_DIR,
        "data": DATA_DIR,
        "raw_packed": RAW_DATA_DIR,
        "split_indices": SPLIT_INDICES_DIR,
        "normalization_config": NORMALIZATION_CONFIG_PATH
    }

def ensure_dirs(paths=None):
    """显式创建关键目录，避免 import 时产生副作用。"""
    required_dirs = [DATA_DIR, RAW_DATA_DIR, SPLIT_INDICES_DIR]
    if paths:
        required_dirs = list({Path(p) for p in required_dirs + list(paths)})
    for d in required_dirs:
        if not d.exists():
            d.mkdir(parents=True, exist_ok=True)
