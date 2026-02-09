import os
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset

from PulsePredict.base import BaseDataLoader
from common.data_utils.processor import UnifiedDataProcessor
from common.settings import FEATURE_ORDER
#==========================================================================================
# 定制的 Dataset 类
#==========================================================================================
class PulseDataset(Dataset):
    def __init__(self, packaged_data_path, processor_config_path):
        if not os.path.exists(packaged_data_path):
            raise FileNotFoundError(f"Packaged data not found: {packaged_data_path}")
        
        # 加载全量数据
        data = np.load(packaged_data_path, allow_pickle=True)
        self.case_ids = data['case_ids']      
        self.att_raw = data['x_att_raw']    # (N, 13)
        self.acc_raw = data['x_acc_xyz']    # (N, 3, 150)
        
        # 初始化公共处理器（保留 config 路径）
        self._processor_config_path = processor_config_path
        self.processor = UnifiedDataProcessor(config_path=self._processor_config_path)
        if not self.processor.load_config():
            raise RuntimeError(f"Failed to load processor config: {processor_config_path}")
        if not self.processor.validate_config():
            raise ValueError(f"Invalid processor config: {processor_config_path}")

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

        # 波形支持批量输入 (N, C, T)
        self.acc_norm = self.processor.process_waveform(self.acc_raw.astype(np.float64), inverse=False).astype(np.float32)
        
    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx):
        # 返回预先归一化并缓存的 numpy -> tensor（零计算开销）
        return (
            torch.from_numpy(self.impact_feats_norm[idx]),
            torch.from_numpy(self.acc_norm[idx]),
            self.case_ids[idx]  # 传递ID以便调试或追踪
        )
    
#==========================================================================================
#  DataLoader 类
#==========================================================================================
class PulseDataLoader(BaseDataLoader):
    def __init__(self, packaged_data_path, split_indices_dir, processor_config, 
                 batch_size, num_workers=0, training=True):

        self.split_dir = Path(split_indices_dir)
        
        # 1. 实例化全量数据集
        self.dataset = PulseDataset(
            packaged_data_path=packaged_data_path,
            processor_config_path=processor_config
        )
        self.processor = self.dataset.processor  # 共享处理器实例（如果需要在外部访问）

        # 2. 准备索引 (Strict Mode)
        train_test_indices = None
        val_indices = None
        
        if training:
            # --- 训练模式 ---
            t_path = self.split_dir / "pulse_train_indices.npy"
            v_path = self.split_dir / "pulse_val_indices.npy"
            
            # 严格检查训练索引
            if not t_path.exists():
                raise FileNotFoundError(f"[data_loader] Train indices missing: {t_path}")
            train_test_indices = np.load(t_path)
            
            # 加载验证索引 (如果存在)
            if v_path.exists():
                val_indices = np.load(v_path)
            # 注意：如果验证集文件不存在，val_indices 为 None，BaseDataLoader 将不创建验证集 loader
            
        else:
            # --- 测试模式 ---
            test_path = self.split_dir / "pulse_test_indices.npy"
            
            # 严格检查测试索引
            if not test_path.exists():
                raise FileNotFoundError(f"[data_loader] Test indices missing: {test_path}")
            
            # 将测试集索引传给 train_test_indices，作为 Loader 的主迭代对象
            train_test_indices = np.load(test_path) 
            
            # 测试模式下强制无验证集
            val_indices = None

        # 3. 初始化基类
        super().__init__(
            dataset=self.dataset, 
            batch_size=batch_size, 
            num_workers=num_workers,
            training=training,
            train_test_indices=train_test_indices,  # 显式传入，不可为None
            val_indices=val_indices                 # 显式传入
        )