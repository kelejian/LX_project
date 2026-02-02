from pathlib import Path
import os
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
