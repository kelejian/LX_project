import json
import torch
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import os

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
    """将有限 DataLoader 包装为无限迭代器。"""
    for loader in repeat(data_loader):
        yield from loader

def prepare_device(n_gpu_use):
    """
    根据配置选择训练设备，并返回 DataParallel 需要的设备编号列表。
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("警告：当前环境未检测到 GPU，将使用 CPU 训练。")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"警告：配置要求使用 {n_gpu_use} 张 GPU，但当前仅检测到 {n_gpu} 张，将自动下调。")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

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
    :param params: 一个包含原始工况参数的字典，例如 {'vel': 50.0, 'ang': 30.0, 'ov': 0.5}
    :param case_id: 样本的原始工况编号。
    :param epoch: 当前 epoch 编号或 'test' 字符串。
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

    # 将 PyTorch Tensor 转为 NumPy，便于后续绘图。
    if not isinstance(pred_wave, np.ndarray):
        pred_wave = pred_wave.detach().cpu().numpy()
    if not isinstance(true_wave, np.ndarray):
        true_wave = true_wave.detach().cpu().numpy()

    # 创建时间轴，单位为毫秒。
    time = np.arange(1, len(pred_wave[0]) + 1)
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
