from pathlib import Path
import csv
import numpy as np


def save_int_vector_csv(path: Path, values: np.ndarray) -> None:
    """保存一维整数数组为无表头单列 CSV。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    arr = np.asarray(values, dtype=np.int64).reshape(-1)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for v in arr.tolist():
            writer.writerow([int(v)])


def load_int_vector_csv(path: Path) -> np.ndarray:
    """读取无表头单列 CSV 为一维 int64 数组。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"split csv 文件不存在: {path}")

    values = []
    with open(path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            values.append(int(str(row[0]).strip()))

    return np.asarray(values, dtype=np.int64)
