import numpy as np
import torch
from typing import Union, List, Optional
from scipy.stats import norm

# -----------------------------------------------------------------------------
# 跨框架兼容辅助函数
# -----------------------------------------------------------------------------

def _to_tensor_or_array(x, backend='numpy'):
    """
    并将输入转换为指定后端的格式（Tensor 或 ndarray）。
    """
    if backend == 'torch':
        if not isinstance(x, torch.Tensor):
            return torch.tensor(x, dtype=torch.float32)
        return x
    else:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.atleast_1d(x)

def _log(x, backend='numpy'):
    return torch.log(x) if backend == 'torch' else np.log(x)

def _exp(x, backend='numpy'):
    return torch.exp(x) if backend == 'torch' else np.exp(x)

def _power(x, p, backend='numpy'):
    return torch.pow(x, p) if backend == 'torch' else np.power(x, p)

def _where(condition, x, y, backend='numpy'):
    return torch.where(condition, x, y) if backend == 'torch' else np.where(condition, x, y)

def _clip(x, min_val, max_val, backend='numpy'):
    if backend == 'torch':
        return torch.clamp(x, min=min_val, max=max_val)
    return np.clip(x, min_val, max_val)

def _norm_cdf(x, backend='numpy'):
    """标准正态分布的累积分布函数。"""
    if backend == 'torch':
        # PyTorch 等价于 norm.cdf 的实现
        return 0.5 * (1 + torch.erf(x / 1.41421356)) # 1.414... = sqrt(2)
    return norm.cdf(x)

# -----------------------------------------------------------------------------
# 损伤指标计算
# -----------------------------------------------------------------------------

def AIS_cal_head(
    HIC15: Union[float, np.ndarray, torch.Tensor], 
    prob_thresholds: list = [0.01, 0.05, 0.1, 0.2, 0.3]
) -> Union[np.ndarray, torch.Tensor]:
    """
    根据 HIC15 计算头部 AIS 等级。
    公式: P_head(AIS3+) = Φ((ln(HIC15) - 7.45231) / 0.73998)
    
    Args:
        HIC15: HIC15 值。可以是标量、numpy 数组或 torch 张量。
        prob_thresholds: AIS 等级 1-5 的概率阈值。
    
    Returns:
        与输入形状一致的 AIS 等级 (int/long tensor)。
    """
    backend = 'torch' if isinstance(HIC15, torch.Tensor) else 'numpy'
    HIC15_proc = _to_tensor_or_array(HIC15, backend)
    
    # 截断至有效范围以避免对数错误
    HIC15_proc = _clip(HIC15_proc, 1.0, 2500.0, backend)

    # P_head(AIS3+) = Φ((ln(HIC15) - 7.45231) / 0.73998)
    z = (_log(HIC15_proc, backend) - 7.45231) / 0.73998
    prob = _norm_cdf(z, backend)

    if backend == 'torch':
        AIS = torch.zeros_like(HIC15_proc, dtype=torch.long)
        for i, thresh in enumerate(prob_thresholds):
            AIS = torch.where(prob >= thresh, torch.tensor(i + 1, device=AIS.device), AIS)
    else:
        AIS = np.zeros_like(HIC15_proc, dtype=int)
        for i, thresh in enumerate(prob_thresholds):
            AIS = np.where(prob >= thresh, i + 1, AIS)
            
    if np.isscalar(HIC15) and backend == 'numpy':
        return AIS.item()
    return AIS


def AIS_cal_chest(
    Dmax: Union[float, np.ndarray, torch.Tensor], 
    OT: Union[int, np.ndarray, torch.Tensor],
    prob_thresholds: list = [0.02, 0.06, 0.15, 0.25, 0.4]
) -> Union[np.ndarray, torch.Tensor]:
    """
    根据 Dmax (mm) 计算胸部 AIS 等级。
    公式: P_chest_defl(AIS3+) = 1 / (1 + e^(10.5456 - 1.568 * ChestDefl^0.4612))
    
    Args:
        Dmax: 胸部压缩量 (mm)。
        OT: 假人类型 (1=5th 女性, 2=50th 男性, 3=95th 男性)。
    
    Returns:
        AIS 等级。
    """
    backend = 'torch' if isinstance(Dmax, torch.Tensor) else 'numpy'
    Dmax_proc = _to_tensor_or_array(Dmax, backend)
    OT_proc = _to_tensor_or_array(OT, backend)
    
    Dmax_proc = _clip(Dmax_proc, 0.0, 500.0, backend)

    # OT=1: 221/182.9; OT=2: 1.0; OT=3: 221/246.38
    # 使用具体数值以避免 tensor 操作中的除法问题（如果需要），但直接除法通常也是行的
    sf_1 = 221.0 / 182.9
    sf_2 = 1.0
    sf_3 = 221.0 / 246.38
    
    if backend == 'torch':
        Scaling_Factor = torch.ones_like(Dmax_proc)
        Scaling_Factor = torch.where(OT_proc == 1, torch.tensor(sf_1, device=Dmax_proc.device), Scaling_Factor)
        Scaling_Factor = torch.where(OT_proc == 3, torch.tensor(sf_3, device=Dmax_proc.device), Scaling_Factor)
        # OT=2 保持 1.0
    else:
        Scaling_Factor = np.where(OT_proc == 1, sf_1, 
                               np.where(OT_proc == 2, sf_2,
                                        np.where(OT_proc == 3, sf_3, 1.0)))

    Dmax_eq = Dmax_proc * Scaling_Factor
        
    # P_chest_defl(AIS3+) = 1 / (1 + e^(10.5456 - 1.568 * ChestDefl^0.4612))
    exponent = 10.5456 - 1.568 * _power(Dmax_eq, 0.4612, backend)
    prob = 1.0 / (1.0 + _exp(exponent, backend))
    
    if backend == 'torch':
        AIS = torch.zeros_like(Dmax_proc, dtype=torch.long)
        for i, thresh in enumerate(prob_thresholds):
            AIS = torch.where(prob >= thresh, torch.tensor(i + 1, device=AIS.device), AIS)
    else:
        AIS = np.zeros_like(Dmax_proc, dtype=int)
        for i, thresh in enumerate(prob_thresholds):
            AIS = np.where(prob >= thresh, i + 1, AIS)

    if np.isscalar(Dmax) and backend == 'numpy':
        return AIS.item()
    return AIS


def AIS_cal_neck(
    Nij: Union[float, np.ndarray, torch.Tensor], 
    prob_thresholds: list = [0.06, 0.1, 0.2, 0.3, 0.4]
) -> Union[np.ndarray, torch.Tensor]:
    """
    根据 Nij 计算颈部 AIS 等级。
    公式: P_neck_Nij(AIS3+) = 1 / (1 + e^(3.2269 - 1.9688 * Nij))
    """
    backend = 'torch' if isinstance(Nij, torch.Tensor) else 'numpy'
    Nij_proc = _to_tensor_or_array(Nij, backend)
    
    prob = 1.0 / (1.0 + _exp(3.2269 - 1.9688 * Nij_proc, backend))

    if backend == 'torch':
        AIS = torch.zeros_like(Nij_proc, dtype=torch.long)
        for i, thresh in enumerate(prob_thresholds):
            AIS = torch.where(prob >= thresh, torch.tensor(i + 1, device=AIS.device), AIS)
    else:
        AIS = np.zeros_like(Nij_proc, dtype=int)
        for i, thresh in enumerate(prob_thresholds):
            AIS = np.where(prob >= thresh, i + 1, AIS)

    if np.isscalar(Nij) and backend == 'numpy':
        return AIS.item()
    return AIS
