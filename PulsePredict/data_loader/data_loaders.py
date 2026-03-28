import os
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset

from PulsePredict.base import BaseDataLoader
from common.data_utils.processor import UnifiedDataProcessor
from common.data_utils.split_io import load_int_vector_csv
from common.settings import (
    FEATURE_ORDER,
    NORMALIZATION_CONFIG_PATH,
    PULSE_SPLIT_DIR,
    RAW_DATA,
    get_split_indices_path,
)
#==========================================================================================
# 定制的 Dataset 类
#==========================================================================================
class PulseDataset(Dataset):
    def __init__(self, packaged_data_path, processor_config_path):
        # config.json 中的路径写为 null 会被解析成 Python 的 None。这里会自动使用 common.settings 里的统一路径
        packaged_data_path = RAW_DATA if packaged_data_path in (None, "") else Path(packaged_data_path)
        processor_config_path = NORMALIZATION_CONFIG_PATH if processor_config_path in (None, "") else Path(processor_config_path)
        if not os.path.exists(packaged_data_path):
            raise FileNotFoundError(f"Packaged data not found: {packaged_data_path}")
        
        # 加载全量数据
        data = np.load(packaged_data_path, allow_pickle=True)
        print(f"[PulseDataset] Loaded raw scale data from {packaged_data_path} with keys: {list(data.keys())}")
        required_keys = {"case_ids", "pulse_source_case_ids", "x_att_raw", "x_acc_xyz"}
        missing_keys = sorted(required_keys - set(data.files))
        if missing_keys:
            raise KeyError(f"[PulseDataset] Packaged data 缺少必要键: {missing_keys}. 请先使用最新 prepare_data.py 重新打包。")

        case_ids_all = data['case_ids'].astype(np.int64)
        pulse_source_case_ids_all = data['pulse_source_case_ids'].astype(np.int64)
        att_raw_all = data['x_att_raw']
        acc_raw_all = data['x_acc_xyz']

        # 保留 raw_packed 的原始行顺序；pulse split 的 indices 直接使用 raw 行索引（首次出现 source 对应行）。
        self.case_ids = case_ids_all
        self.pulse_source_case_ids = pulse_source_case_ids_all
        self.att_raw = att_raw_all
        self.acc_raw = acc_raw_all
        print(f"[PulseDataset] Raw rows: {len(self.case_ids)}, unique pulse sources: {len(np.unique(self.pulse_source_case_ids))}")
        
        # 初始化公共处理器（保留 config 路径）
        self._processor_config_path = processor_config_path
        self.processor = UnifiedDataProcessor(config_path=self._processor_config_path)
        if not self.processor.load_config():
            raise RuntimeError(f"Failed to load processor config: {processor_config_path}")
        if not self.processor.validate_config():
            raise ValueError(f"Invalid processor config: {processor_config_path}")
        # 打印processor信息
        print(f"[PulseDataset] Loaded UnifiedDataProcessor with config: {processor_config_path}")
        self.impact_feat_names = ["impact_velocity", "impact_angle", "overlap"]
        self.impact_feat_indices = [FEATURE_ORDER.index(name) for name in self.impact_feat_names]

        # -----------------------------
        # 预先向量化归一化（内存缓存，避免每个样本的 Python 开销）
        # - 使用 processor 的批量接口（注意参数名与返回 dtype）
        # - 处理时使用 float64 做计算，最终存为 float32 以减少内存和拷贝开销
        # -----------------------------
        # impact features: (N, 3)
        self.impact_feats_raw = self.att_raw[:, self.impact_feat_indices]
        self.impact_feats_norm = self.processor.process_by_name(
            values=self.impact_feats_raw,
            feature_names=self.impact_feat_names,
            inverse=False
        ).astype(np.float32)
        print(f"[PulseDataset] Preprocessed impact features with shape {self.impact_feats_norm.shape} and dtype {self.impact_feats_norm.dtype}")

        # 波形支持批量输入 (N, C, T)
        self.acc_norm = self.processor.process_waveform(self.acc_raw.astype(np.float64), inverse=False).astype(np.float32)
        print(f"[PulseDataset] Preprocessed waveforms with shape {self.acc_norm.shape} and dtype {self.acc_norm.dtype}")

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx):
        # 返回预先归一化并缓存的 numpy -> tensor（零计算开销）
        return (
            torch.from_numpy(self.impact_feats_norm[idx]),
            torch.from_numpy(self.acc_norm[idx]),
            self.pulse_source_case_ids[idx]  # 传递波形主键ID，避免与case_id语义混淆
        )
    
#==========================================================================================
#  DataLoader 类
#==========================================================================================
class PulseDataLoader(BaseDataLoader):
    def __init__(self, packaged_data_path, split_indices_dir, processor_config, batch_size, num_workers=0, training=True):
        self.split_dir = PULSE_SPLIT_DIR if split_indices_dir in (None, "") else Path(split_indices_dir)
        packaged_data_path = RAW_DATA if packaged_data_path in (None, "") else Path(packaged_data_path)
        processor_config = NORMALIZATION_CONFIG_PATH if processor_config in (None, "") else Path(processor_config)
        
        # 1. 实例化全量数据集
        self.dataset = PulseDataset(
            packaged_data_path=packaged_data_path,
            processor_config_path=processor_config
        )
        self.processor = self.dataset.processor  # 共享处理器实例（如果需要在外部访问）

        # 2. 准备索引 (Strict Mode)
        self.train_test_indices = None
        self.val_indices = None
        
        if training:
            # --- 训练模式 ---
            t_idx_path = get_split_indices_path("pulse", "train", self.split_dir)
            v_idx_path = get_split_indices_path("pulse", "val", self.split_dir)

            if not t_idx_path.exists():
                raise FileNotFoundError(f"[PulseDataLoader] Train split missing: {t_idx_path}")
            if not v_idx_path.exists():
                raise FileNotFoundError(f"[PulseDataLoader] Val split missing: {v_idx_path}")

            self.train_test_indices = self._load_and_validate_indices(load_int_vector_csv(t_idx_path), "train")
            self.val_indices = self._load_and_validate_indices(load_int_vector_csv(v_idx_path), "val")
            
        else:
            # --- 测试模式 ---
            test_idx_path = get_split_indices_path("pulse", "test", self.split_dir)

            if not test_idx_path.exists():
                raise FileNotFoundError(f"[PulseDataLoader] Test split missing: {test_idx_path}")

            self.train_test_indices = self._load_and_validate_indices(load_int_vector_csv(test_idx_path), "test")
            
            # 测试模式下强制无验证集
            self.val_indices = None

        if self.train_test_indices is None or len(self.train_test_indices) == 0:
            split_name = "train" if training else "test"
            raise ValueError(f"[PulseDataLoader] {split_name} split is empty: {self.split_dir}")

        # 3. 初始化基类
        super().__init__(
            dataset=self.dataset, 
            batch_size=batch_size, 
            num_workers=num_workers,
            training=training,
            train_test_indices=self.train_test_indices,  # 显式传入，不可为None
            val_indices=self.val_indices                 # 显式传入
        )

    def _load_and_validate_indices(self, indices: np.ndarray, split_name: str) -> np.ndarray:
        if indices.ndim != 1:
            raise ValueError(f"[PulseDataLoader] {split_name} split 索引必须是一维数组，实际形状: {indices.shape}")
        if indices.size == 0:
            return np.asarray(indices, dtype=np.int64)

        min_idx = int(indices.min())
        max_idx = int(indices.max())
        n = len(self.dataset)
        if min_idx < 0 or max_idx >= n:
            raise ValueError(
                f"[PulseDataLoader] {split_name} split 索引越界：min={min_idx}, max={max_idx}, 数据集大小={n}"
            )
        return np.asarray(indices, dtype=np.int64)
