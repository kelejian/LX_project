import json
import torch
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt', encoding='utf-8') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt', encoding='utf-8') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

def get_case_ids_from_directory(waveform_dir, axis=None):
    """
    从波形文件目录中提取 case_id 列表
    
    :param waveform_dir: 存放波形 CSV 文件的目录路径
    :param axis: 用于提取 case_id 的轴向文件前缀 ('x', 'y', 或 'z')
                 如果为 None，则获取同时拥有 x, y, z 三个方向波形文件的完整 case_id 列表
    :return: 排序后的 case_id 列表
    """
    import os
    import re
    
    if not os.path.exists(waveform_dir):
        raise FileNotFoundError(f"目录不存在: {waveform_dir}")
    
    # 如果 axis 为 None，获取完整的 case_id 列表
    if axis is None:
        # 分别获取三个轴向的 case_id
        x_case_ids = set()
        y_case_ids = set()
        z_case_ids = set()
        
        for axis_name, case_set in [('x', x_case_ids), ('y', y_case_ids), ('z', z_case_ids)]:
            pattern = rf'^{axis_name}(\d+)\.csv$'
            for filename in os.listdir(waveform_dir):
                match = re.match(pattern, filename)
                if match:
                    case_id = int(match.group(1))
                    case_set.add(case_id)
        
        # 取交集，确保每个 case_id 都有完整的三个文件
        complete_case_ids = sorted(list(x_case_ids & y_case_ids & z_case_ids))
        
        print(f"找到 {len(complete_case_ids)} 个完整的 case_id（同时拥有 x, y, z 文件）")
        
        # 检查缺失的文件
        all_case_ids = x_case_ids | y_case_ids | z_case_ids
        incomplete_case_ids = all_case_ids - set(complete_case_ids)
        
        if incomplete_case_ids:
            print(f"警告：以下 case_id 的文件不完整: {sorted(incomplete_case_ids)}")
        
        return complete_case_ids
    
    # 如果指定了特定轴向，获取该轴向的 case_id 列表
    else:
        if axis not in ['x', 'y', 'z']:
            raise ValueError(f"axis 参数必须是 'x', 'y', 'z' 中的一个，当前值: {axis}")
        
        case_ids = []
        pattern = rf'^{axis}(\d+)\.csv$'  # 匹配格式如 x10.csv, y46.csv 等
        
        # 遍历目录中的所有文件
        for filename in os.listdir(waveform_dir):
            match = re.match(pattern, filename)
            if match:
                case_id = int(match.group(1))  # 提取数字部分
                case_ids.append(case_id)
        
        # 排序并去重
        case_ids = sorted(list(set(case_ids)))
        
        print(f"在目录 {waveform_dir} 中找到 {len(case_ids)} 个 {axis} 轴的 case_id")
        return case_ids

def inverse_transform(pred_tensor, target_tensor, scaler):
    """
    对预测和目标张量进行逆变换，以还原到原始物理尺度。

    :param pred_tensor: 模型预测的归一化张量。
    :param target_tensor: 真实的归一化目标张量。
    :param scaler: 用于逆变换的scaler对象 (e.g., from scikit-learn)。
    :return: 包含逆变换后的 (pred_orig, target_orig) 的元组。
    """
    if scaler is None:
        return pred_tensor, target_tensor

    original_shape = pred_tensor.shape
    
    # 使用 .detach() 确保此操作不影响计算图
    pred_numpy = pred_tensor.detach().cpu().numpy().reshape(-1, 1)
    pred_orig = scaler.inverse_transform(pred_numpy).reshape(original_shape)
    pred_orig = torch.from_numpy(pred_orig).to(pred_tensor.device)

    target_numpy = target_tensor.detach().cpu().numpy().reshape(-1, 1)
    target_orig = scaler.inverse_transform(target_numpy).reshape(original_shape)
    target_orig = torch.from_numpy(target_orig).to(target_tensor.device)
    
    return pred_orig, target_orig

class InputScaler:
    def __init__(self, v_min, v_max, a_abs_max, o_abs_max):
        '''
        :param v_min: 速度的最小值
        :param v_max: 速度的最大值
        :param a_abs_max: 角度的绝对值最大值
        :param o_abs_max: 重叠率的绝对值最大值
        '''

        self.v_min = v_min
        self.v_max = v_max
        self.a_abs_max = a_abs_max
        self.o_abs_max = o_abs_max

    def transform(self, velocity, angle, overlap):
        norm_velocity = (velocity - self.v_min) / (self.v_max - self.v_min) # 归一化到[0, 1]
        norm_angle = angle / self.a_abs_max  # 归一化到[-1, 1]
        norm_overlap = overlap / self.o_abs_max  # 归一化到[-1, 1]
        return norm_velocity, norm_angle, norm_overlap

    def inverse_transform(self, norm_velocity, norm_angle, norm_overlap):
        velocity = norm_velocity * (self.v_max - self.v_min) + self.v_min
        angle = norm_angle * self.a_abs_max
        overlap = norm_overlap * self.o_abs_max
        return velocity, angle, overlap
    
class PulseAbsMaxScaler:
    """
    一个自定义的Scaler，执行 x / max(abs(x)) 的归一化。
    它的接口模仿了sklearn的scaler，以便于替换。
    """
    def __init__(self):
        self.data_abs_max_ = None

    def fit(self, data, num=50):
        """
        计算并存储数据中的绝对值最大的 num 个值的平均值。
        :param data: 一个Numpy数组。
        :param num: 要考虑的最大值数量。
        """
        self.data_abs_max_ = np.mean(np.sort(np.abs(data).flatten())[-num:])
        return self

    def transform(self, data):
        """
        使用存储的绝对值最大值对数据进行归一化。
        :param data: 一个Numpy数组。
        """
        if self.data_abs_max_ is None:
            raise RuntimeError("Scaler has not been fitted yet. Call a.fit(data) first.")
        if self.data_abs_max_ == 0:
            return data # 避免除以零
        return data / self.data_abs_max_

    def inverse_transform(self, data):
        """
        进行反归一化操作。
        :param data: 一个归一化后的Numpy数组。
        """
        if self.data_abs_max_ is None:
            raise RuntimeError("Scaler has not been fitted yet. Call a.fit(data) first.")
        return data * self.data_abs_max_

def get_parameter_groups(model, weight_decay=1e-2, head_decay_ratio=0.1, head_keywords=('head',)):
    """
    精细化参数分组策略：
    1. Body Group (高 WD): 维持骨干网络的强正则化，促进平坦解搜索。
    2. Head Group (低 WD): 降低输出头的正则化惩罚，允许其在训练末期精细拟合目标值。
    3. No Decay Group (0 WD): Bias 和 Normalization 层参数，保持数值稳定性。

    :param model: 模型实例
    :param weight_decay: 全局(Body)的权重衰减系数
    :param head_decay_ratio: Head 部分的 WD 缩放比例 (e.g., 0.1 表示 Head_WD = 0.1 * Body_WD)
    :param head_keywords: 识别 Head 参数的关键词元组
    """
    decay_body_params = []
    decay_head_params = []
    no_decay_params = []
    
    # 遍历所有需要梯度的参数
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # 1. 不衰减组: 维度 < 2 的参数 (覆盖所有 Bias, BN/LN 的 weight/bias)
        if param.ndim < 2:
            no_decay_params.append(param)
        else:
            # 2. 衰减组: 根据名称区分 Body 和 Head
            # HybridPulseCNN 中输出头命名均包含 'head' (e.g., s1_head, s3_branches.x.head)
            if any(k in name for k in head_keywords):
                decay_head_params.append(param)
            else:
                decay_body_params.append(param)

    return [
        # Group 1: Body weights (High Decay)
        {'params': decay_body_params, 'weight_decay': weight_decay},
        
        # Group 2: Head weights (Low Decay)
        {'params': decay_head_params, 'weight_decay': weight_decay * head_decay_ratio},
        
        # Group 3: Bias/Norm (No Decay)
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]

def plot_waveform_comparison(pred_wave, true_wave, params, case_id, epoch, batch_idx, sample_idx, save_dir, iso_ratings=None):
    """
    绘制单样本的预测与真实波形对比图，并在标题中显示工况参数。

    :param pred_wave: 单个样本的预测波形 (numpy array, shape: (3, 150))
    :param true_wave: 单个样本的真实波形 (numpy array, shape: (3, 150))
    :param params: 一个包含原始工况参数的字典, e.g., {'vel': 50.0, 'ang': 30.0, 'ov': 0.5}
    :param case_id: 样本的原始Case ID。
    :param epoch: 当前的epoch数或'test'字符串。
    :param batch_idx: 当前的批次索引。
    :param sample_idx: 样本在批次中的索引。
    :param save_dir: 图片保存的根目录 (仅在训练时使用)。
    :param iso_ratings: (可选) 包含ISO评级分数的字典, e.g., {'x': 0.85, 'y': 0.92, 'z': 0.77}
    """
    # 根据是训练阶段还是测试阶段，决定图片保存目录
    if epoch == 'test':
        plot_dir = os.path.join(save_dir, 'fig')
    else:
        plot_dir = os.path.join(save_dir, 'fig', f'epoch_{epoch}')
    
    # 确保保存图片的目录存在
    os.makedirs(plot_dir, exist_ok=True)

    # 将PyTorch Tensors转换为Numpy arrays (如果尚未转换)
    if not isinstance(pred_wave, np.ndarray):
        pred_wave = pred_wave.detach().cpu().numpy()
    if not isinstance(true_wave, np.ndarray):
        true_wave = true_wave.detach().cpu().numpy()

    # 创建时间轴
    time = np.arange(1, len(pred_wave[0]) + 1)  # 时间单位为ms
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    
    # --- 创建包含工况参数的新标题 ---
    vel = params['vel']
    ang = params['ang']
    ov = params['ov']
    title_line1 = (f'Case ID: {case_id}, Epoch: {epoch}, Batch: {batch_idx}, Sample: {sample_idx}\n'
                   f'Velocity: {vel:.1f} km/h, Angle: {ang:.1f}°, Overlap: {ov:.2f}')
    
    title_line2 = ""
    # 如果 iso_ratings 参数被提供，则创建第二行标题用于显示分数
    if iso_ratings:
        title_line2 = (f'\nISO Ratings -> X: {iso_ratings["x"]:.3f}, Y: {iso_ratings["y"]:.3f}, Z: {iso_ratings["z"]:.3f}')
    
    # 组合标题
    title = title_line1 + title_line2

    fig.suptitle(title, fontsize=15, fontweight='bold')
    # --------------------------------

    # X方向加速度
    axes[0].plot(time, true_wave[0, :], 'b-', linewidth=2, label='Ground Truth')
    axes[0].plot(time, pred_wave[0, :], 'r--', linewidth=1.5, label='Prediction')
    axes[0].set_ylabel('Acceleration (m/s²)', fontsize=12)
    axes[0].set_title('X-direction Acceleration', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Y方向加速度
    axes[1].plot(time, true_wave[1, :], 'b-', linewidth=2, label='Ground Truth')
    axes[1].plot(time, pred_wave[1, :], 'r--', linewidth=1.5, label='Prediction')
    axes[1].set_ylabel('Acceleration (m/s²)', fontsize=12)
    axes[1].set_title('Y-direction Acceleration', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Z方向旋转加速度
    axes[2].plot(time, true_wave[2, :], 'b-', linewidth=2, label='Ground Truth')
    axes[2].plot(time, pred_wave[2, :], 'r--', linewidth=1.5, label='Prediction')
    axes[2].set_xlabel('Time (ms)', fontsize=12)
    axes[2].set_ylabel('Angular Acceleration (rad/s²)', fontsize=12)
    axes[2].set_title('Z-direction Rotational Acceleration', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.99])
    
    if epoch == 'test':
        plot_filename = f'test_batch_{batch_idx}_sample_{sample_idx}_case_{case_id}.png'
    else:
        plot_filename = f'epoch_{epoch}_batch_{batch_idx}_sample_{sample_idx}_case_{case_id}.png'

    save_path = os.path.join(plot_dir, plot_filename)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        # 使用 loc 进行赋值，避免链式赋值警告
        self._data.loc[key, 'total'] += value * n
        self._data.loc[key, 'counts'] += n
        self._data.loc[key, 'average'] = self._data.loc[key, 'total'] / self._data.loc[key, 'counts']

    def avg(self, key):
        return self._data.loc[key, 'average']

    def result(self):
        return dict(self._data['average'])