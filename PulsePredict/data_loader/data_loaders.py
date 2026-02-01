import os
import torch
import numpy as np
import joblib
from torch.utils.data import Dataset
from base import BaseDataLoader # 继承自项目基类
from sklearn.preprocessing import MinMaxScaler
from utils import InputScaler, PulseAbsMaxScaler

#==========================================================================================
# 定制的 Dataset 类
#==========================================================================================
class CollisionDataset(Dataset):
    """
    用于加载已打包的碰撞数据的自定义数据集。
    """
    def __init__(self, packaged_data_path, target_scaler=None, physics_bounds=None):
        """
        :param packaged_data_path: 包含已打包的 case_id, params, waveforms 的 .npz 文件路径。
        :param target_scaler: 可选的目标缩放器，用于对波形进行缩放。
        :param physics_bounds: 用于初始化 InputScaler 的物理参数边界字典
        """
        self.target_scaler = target_scaler

        # --- 1. 加载打包好的数据 ---
        data = np.load(packaged_data_path)
        self.case_ids = data['case_ids']
        raw_params = data['params']         # 原始物理尺度的参数
        raw_waveforms = data['waveforms']   # 原始物理尺度的波形

        # --- 2. 归一化输入参数 ---
        if physics_bounds is None:
             raise ValueError("CollisionDataset 必须提供 physics_bounds 参数。")
             
        self.input_scaler = InputScaler(**physics_bounds) # 使用传入的参数初始化
        norm_velocities, norm_angles, norm_overlaps = self.input_scaler.transform(
            raw_params[:, 0], raw_params[:, 1], raw_params[:, 2]
        )
        self.features = torch.tensor(
            np.stack([norm_velocities, norm_angles, norm_overlaps], axis=1),
            dtype=torch.float32
        ) # 形状 (N, 3)

        # --- 3. （可选）归一化输出波形 ---
        if self.target_scaler:
            original_shape = raw_waveforms.shape
            # (N, 3, 150) -> (N * 3 * 150, 1)
            waveforms_reshaped = raw_waveforms.reshape(-1, 1)
            waveforms_scaled = self.target_scaler.transform(waveforms_reshaped)
            # (N * 3 * 150, 1) -> (N, 3, 150)
            waveforms_np = waveforms_scaled.reshape(original_shape)
        else:
            waveforms_np = raw_waveforms
        
        self.waveforms = torch.tensor(waveforms_np, dtype=torch.float32)

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx):
        """
        根据索引 idx 获取一个样本。
        返回顺序为 (特征, 目标波形, 案例ID)，与 Trainer 逻辑保持一致。
        """
        input_features = self.features[idx]
        target_waveforms = self.waveforms[idx]
        case_id = self.case_ids[idx]
        
        return input_features, target_waveforms, case_id

#==========================================================================================
#  DataLoader 类
#==========================================================================================
class CollisionDataLoader(BaseDataLoader):
    """
    用于加载碰撞波形数据的 DataLoader 类。
    """
    def __init__(self, packaged_data_path, batch_size, pulse_norm_mode='none', scaler_path=None, shuffle=True, validation_split=0.1, num_workers=0, training=True, physics_bounds=None):
        """
        :param packaged_data_path: 包含打包数据的 .npz 文件路径。
        :param batch_size: 批量大小。
        :param pulse_norm_mode: 归一化模式，'none', 'minmax', 'absmax'之一。
        :param scaler_path: 保存或加载Scaler的文件路径 (e.g., 'saved/scalers/pulse_scaler.joblib')。
        :param shuffle: 是否打乱数据。在非训练模式下会被强制设为 False。
        :param validation_split: 验证集比例。在非训练模式下会被强制设为 0。
        :param num_workers: 数据加载的工作线程数。
        :param training: 是否为训练模式。这会影响Scaler的加载/保存行为。
        :param physics_bounds: 用于初始化 InputScaler 的物理参数边界字典, 从config.json传入
        """
        if not os.path.exists(packaged_data_path):
            raise FileNotFoundError(f"打包好的数据文件未找到: {packaged_data_path}。")
        
        target_scaler = None
        # --- Scaler 处理 ---
        if pulse_norm_mode != 'none':
            if scaler_path is None:
                raise ValueError("当 pulse_norm_mode 不为 'none' 时, 必须提供 'scaler_path'。")

            # 如果是训练模式，则拟合并保存Scaler
            if training:
                print(f"训练模式：正在从 '{packaged_data_path}' 为目标波形拟合Scaler (模式: {pulse_norm_mode})...")
                
                # 直接从打包文件中加载波形数据进行拟合
                with np.load(packaged_data_path) as data:
                    all_waveforms_data = data['waveforms']
                
                if all_waveforms_data.size == 0: 
                    raise ValueError(f"未能从 {packaged_data_path} 加载任何波形数据。")
                
                if pulse_norm_mode == 'minmax':
                    scaler = MinMaxScaler(feature_range=(-1, 1))
                    target_scaler = scaler.fit(all_waveforms_data.reshape(-1, 1))
                    print(f"MinMaxScaler 拟合完毕。Min: {target_scaler.data_min_[0]:.4f}, Max: {target_scaler.data_max_[0]:.4f}")
                
                elif pulse_norm_mode == 'absmax':
                    target_scaler = PulseAbsMaxScaler().fit(all_waveforms_data)
                    print(f"PulseAbsMaxScaler 拟合完毕。全局绝对值Max: {target_scaler.data_abs_max_:.4f}")
                else:
                    raise ValueError(f"未知的 pulse_norm_mode: {pulse_norm_mode}")
                
                # 创建目录并保存Scaler
                os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
                joblib.dump(target_scaler, scaler_path)
                print(f"Scaler 已保存至: {scaler_path}")

            # 如果是测试模式，则直接加载Scaler
            else:
                print(f"测试模式：正在从 {scaler_path} 加载Scaler...")
                try:
                    target_scaler = joblib.load(scaler_path)
                    print("Scaler 加载成功。")
                except FileNotFoundError:
                    raise FileNotFoundError(f"Scaler文件未找到: {scaler_path}。请先在训练模式下运行以生成Scaler文件。")
        
        # --- 根据模式调整参数 ---
        if not training:
            shuffle = False
            validation_split = 0.0

        # --- 实例化Dataset ---
        self.dataset = CollisionDataset(packaged_data_path, target_scaler, physics_bounds=physics_bounds)
        
        # 保存一些有用的属性
        self.pulse_norm_mode = pulse_norm_mode
        self.training = training
        self.target_scaler = target_scaler

        # --- 调用父类构造函数 ---
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)