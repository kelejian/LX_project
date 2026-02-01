import numpy as np
from typing import Union
from scipy.stats import norm

def AIS_cal_head(
    HIC15: Union[float, np.ndarray], 
    prob_thresholds: list = [0.01, 0.05, 0.1, 0.2, 0.3]
) -> np.ndarray:
    """
    根据头部 HIC15 值计算头部 AIS 等级。
    使用公式: P_head(AIS3+) = Φ((ln(HIC15) - 7.45231) / 0.73998)
    其中 Φ 是累积正态分布函数
    
    Args:
        HIC15 (Union[float, np.ndarray]): HIC15 值。
        prob_thresholds (list): 概率阈值列表，用于划分 AIS 等级。
            例如 [0.02, 0.05, 0.15, 0.4, 0.75] 表示:
            P < 0.02: AIS=0, 0.02≤P<0.05: AIS=1, 0.05≤P<0.15: AIS=2,
            0.15≤P<0.4: AIS=3, 0.4≤P<0.75: AIS=4, P≥0.75: AIS=5

    Returns:
        float or np.ndarray: 计算出的 AIS 等级, 和输入的损伤值形状一致。
    """
    if np.issubdtype(type(HIC15), np.number):
        is_single_value = True
    else:
        is_single_value = False 
    HIC15 = np.atleast_1d(HIC15).astype(float)
    HIC15 = np.clip(HIC15, 1, 2500)

    # P_head(AIS3+) = Φ((ln(HIC15) - 7.45231) / 0.73998)
    z = (np.log(HIC15) - 7.45231) / 0.73998
    prob = norm.cdf(z)

    # 根据概率阈值确定 AIS 等级
    AIS = np.zeros_like(HIC15, dtype=int)
    for i, thresh in enumerate(prob_thresholds):
        AIS = np.where(prob >= thresh, i + 1, AIS)

    if is_single_value:
        return AIS[0]
    return AIS


def AIS_cal_chest(
    Dmax: Union[float, np.ndarray], 
    OT: Union[int, np.ndarray],
    prob_thresholds: list = [0.02, 0.06, 0.15, 0.25, 0.4]
) -> np.ndarray:
    """
    根据胸部压缩量 Dmax (mm) 计算胸部 AIS 等级。
    使用公式: P_chest_defl(AIS3+) = 1 / (1 + e^(10.5456 - 1.568 * ChestDefl^0.4612))

    Args:
        Dmax (Union[float, np.ndarray]): 胸部最大压缩量 (mm)。
        OT (Union[int, np.ndarray],) : 假人类别, 1为5th 女性, 2为50th男性, 3为95th男性
        prob_thresholds (list): 概率阈值列表，用于划分 AIS 等级。
            例如 [0.06, 0.1, 0.2, 0.3, 0.4] 表示:
            P < 0.06: AIS=0, 0.06≤P<0.1: AIS=1, 0.1≤P<0.2: AIS=2,
            0.2≤P<0.3: AIS=3, 0.3≤P<0.4: AIS=4, P≥0.4: AIS=5

    Returns:
        float or np.ndarray: 计算出的 AIS 等级, 和输入的损伤值形状一致。
    """
    if np.issubdtype(type(Dmax), np.number):
        is_single_value = True
    else:
        is_single_value = False
    Dmax = np.atleast_1d(Dmax).astype(float)
    Dmax = np.clip(Dmax, 0.0, 500.0)

    # OT= 1时，Scaling_Factor=221/182.9; OT=2时，Scaling_Factor=1.0; OT=3时，Scaling_Factor=221/246.38
    Scaling_Factor = np.where(OT == 1, 221.0 / 182.9, 
                              np.where(OT == 2, 1.0,
                                       np.where(OT == 3, 221.0 / 246.38, 1.0)))
    # 根据缩放因子调整 Dmax
    Dmax_eq = Dmax * Scaling_Factor
        
    # P_chest_defl(AIS3+) = 1 / (1 + e^(10.5456 - 1.568 * ChestDefl^0.4612))
    prob = 1.0 / (1.0 + np.exp(10.5456 - 1.568 * np.power(Dmax_eq, 0.4612)))
    # 根据概率阈值确定 AIS 等级
    AIS = np.zeros_like(Dmax_eq, dtype=int)
    for i, thresh in enumerate(prob_thresholds):
        AIS = np.where(prob >= thresh, i + 1, AIS)

    if is_single_value:
        return AIS[0]
    return AIS


def AIS_cal_neck(
    Nij: Union[float, np.ndarray], 
    prob_thresholds: list = [0.06, 0.1, 0.2, 0.3, 0.4]
) -> np.ndarray:
    """
    根据颈部伤害指数 Nij 计算颈部 AIS 等级。
    使用公式: P_neck_Nij(AIS3+) = 1 / (1 + e^(3.2269 - 1.9688 * Nij))

    Args:
        Nij (Union[float, np.ndarray]): Nij 值。
        prob_thresholds (list): 概率阈值列表，用于划分 AIS 等级。
            例如 [0.06, 0.1, 0.2, 0.3, 0.4] 表示:
            P < 0.06: AIS=0, 0.06≤P<0.1: AIS=1, 0.1≤P<0.2: AIS=2,
            0.2≤P<0.3: AIS=3, 0.3≤P<0.4: AIS=4, P≥0.4: AIS=5

    Returns:
        float or np.ndarray: 计算出的 AIS 等级, 和输入的损伤值形状一致。
    """
    if np.issubdtype(type(Nij), np.number):
        is_single_value = True
    else:
        is_single_value = False
    Nij = np.atleast_1d(Nij).astype(float)
    Nij = np.clip(Nij, 0, 50.0)

    # P_neck_Nij(AIS3+) = 1 / (1 + e^(3.2269 - 1.9688 * Nij))
    prob = 1.0 / (1.0 + np.exp(3.2269 - 1.9688 * Nij))

    # 根据概率阈值确定 AIS 等级
    AIS = np.zeros_like(Nij, dtype=int)
    for i, thresh in enumerate(prob_thresholds):
        AIS = np.where(prob >= thresh, i + 1, AIS)

    if is_single_value:
        return AIS[0]
    return AIS


# def AIS_cal_head(
#     HIC15: Union[float, np.ndarray], 
#     ais_level: int = 3,
#     prob_thresholds: list = [0.02, 0.05, 0.15, 0.4, 0.75]
# ) -> np.ndarray:
#     """
#     根据头 HIC15 值计算头部 AIS 等级。
    
#     Args:
#         HIC15 (Union[float, np.ndarray]): HIC15 值。
#         ais_level (int): 使用的风险曲线 P(AIS≥n) 中的 n，范围 1-5。
#         prob_thresholds (list): 概率阈值列表，用于划分 AIS 等级。
#             例如 [0.02, 0.05, 0.15, 0.4, 0.75] 表示:
#             P < 0.02: AIS=0, 0.02≤P<0.05: AIS=1, 0.05≤P<0.15: AIS=2,
#             0.15≤P<0.4: AIS=3, 0.4≤P<0.75: AIS=4, P≥0.75: AIS=5

#     Returns:
#         float or np.ndarray: 计算出的 AIS 等级, 和输入的损伤值形状一致。
#     """
#     if np.issubdtype(type(HIC15), np.number):
#         is_single_value = True
#     else:
#         is_single_value = False 
#     HIC15 = np.atleast_1d(HIC15)
#     HIC15 = np.clip(HIC15, 1, 2500)

#     # 定义系数
#     coefficients = np.array([
#         [1.54, 0.00650],  # P(AIS≥1)
#         [2.49, 0.00483],  # P(AIS≥2)
#         [3.39, 0.00372],  # P(AIS≥3)
#         [4.90, 0.00351],  # P(AIS≥4)
#         [7.82, 0.00429]   # P(AIS≥5)
#     ])
    
#     if ais_level < 1 or ais_level > 5:
#         raise ValueError("头部 AIS 等级应在 1-5 之间")
    
#     c1, c2 = coefficients[ais_level - 1]
#     HIC_inv = 200 / HIC15
#     prob = 1.0 / (1.0 + np.exp(c1 + HIC_inv - c2 * HIC15))

#     # 根据概率阈值确定 AIS 等级
#     AIS = np.zeros_like(HIC15, dtype=int)
#     for i, thresh in enumerate(prob_thresholds):
#         AIS = np.where(prob >= thresh, i + 1, AIS)

#     if is_single_value:
#         return AIS[0]
#     return AIS

# def AIS_cal_chest(
#     Dmax: Union[float, np.ndarray], 
#     ais_level: int = 3,
#     prob_thresholds: list = [0.06, 0.1, 0.2, 0.3, 0.4]
# ) -> np.ndarray:
#     """
#     根据胸部压缩量 Dmax (mm) 计算胸部 AIS 等级。

#     Args:
#         Dmax (Union[float, np.ndarray]): 胸部最大压缩量 (mm)。
#         ais_level (int): 使用的风险曲线 P(AIS≥n) 中的 n，范围 2-5。
#         prob_thresholds (list): 概率阈值列表，用于划分 AIS 等级。
#             例如 [0.06, 0.1, 0.2, 0.3, 0.4] 表示:
#             P < 0.05: AIS=0, 0.05≤P<0.1: AIS=1, 0.1≤P<0.2: AIS=2,
#             0.2≤P<0.3: AIS=3, 0.3≤P<0.4: AIS=4, P≥0.4: AIS=5

#     Returns:
#         float or np.ndarray: 计算出的 AIS 等级, 和输入的损伤值形状一致。
#     """
#     if np.issubdtype(type(Dmax), np.number):
#         is_single_value = True
#     else:
#         is_single_value = False
#     Dmax = np.atleast_1d(Dmax)
#     Dmax = np.clip(Dmax, 0.0, 500.0)

#     coefficients = np.array([
#         [1.8706, 0.04439],  # P(AIS≥2)
#         [3.7124, 0.04750],  # P(AIS≥3)
#         [5.0952, 0.04750],  # P(AIS≥4)
#         [8.8274, 0.04590]   # P(AIS≥5)
#     ])
    
#     if ais_level < 2 or ais_level > 5:
#         raise ValueError("胸部 AIS 等级应在 2-5 之间")
    
#     c1, c2 = coefficients[ais_level - 2]
#     prob = 1.0 / (1.0 + np.exp(c1 - c2 * Dmax))

#     # 根据概率阈值确定 AIS 等级
#     AIS = np.zeros_like(Dmax, dtype=int)
#     for i, thresh in enumerate(prob_thresholds):
#         AIS = np.where(prob >= thresh, i + 1, AIS)

#     if is_single_value:
#         return AIS[0]
#     return AIS

# def AIS_cal_neck(
#     Nij: Union[float, np.ndarray], 
#     ais_level: int = 3,
#     prob_thresholds: list = [0.06, 0.1, 0.2, 0.3, 0.4]
# ) -> np.ndarray:
#     """
#     根据颈部伤害指数 Nij 计算颈部 AIS 等级。

#     Args:
#         Nij (Union[float, np.ndarray]): Nij 值。
#         ais_level (int): 使用的风险曲线 P(AIS≥n) 中的 n，范围 2-5。
#         prob_thresholds (list): 概率阈值列表，用于划分 AIS 等级。
#             例如 [0.06, 0.1, 0.2, 0.3, 0.4] 表示:
#             P < 0.05: AIS=0, 0.05≤P<0.1: AIS=1, 0.1≤P<0.2: AIS=2,
#             0.2≤P<0.3: AIS=3, 0.3≤P<0.4: AIS=4, P≥0.4: AIS=5

#     Returns:
#         float or np.ndarray: 计算出的 AIS 等级, 和输入的损伤值形状一致。
#     """
#     if np.issubdtype(type(Nij), np.number):
#         is_single_value = True
#     else:
#         is_single_value = False
#     Nij = np.atleast_1d(Nij)
#     Nij = np.clip(Nij, 0, 50.0)

#     coefficients = np.array([
#         [2.054, 1.195],  # P(AIS≥2)
#         [3.227, 1.969],  # P(AIS≥3)
#         [2.693, 1.195],  # P(AIS≥4)
#         [3.817, 1.195]   # P(AIS≥5)
#     ])
    
#     if ais_level < 2 or ais_level > 5:
#         raise ValueError("颈部 AIS 等级应在 2-5 之间")
    
#     c1, c2 = coefficients[ais_level - 2]
#     prob = 1.0 / (1.0 + np.exp(c1 - c2 * Nij))

#     # 根据概率阈值确定 AIS 等级
#     AIS = np.zeros_like(Nij, dtype=int)
#     for i, thresh in enumerate(prob_thresholds):
#         AIS = np.where(prob >= thresh, i + 1, AIS)

#     if is_single_value:
#         return AIS[0]
#     return AIS

