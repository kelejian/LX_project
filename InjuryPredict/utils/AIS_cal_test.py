# %% old
import numpy as np
import matplotlib.pyplot as plt
import math
hic_threshold = 0.15 # qf:0.2; 推荐0.15
Dmax_threshold = 0.2 # qf:0.15;推荐0.2
Nij_threshold = 0.17 # qf:0.17;推荐0.17

HIC = np.array([i for i in range(1, 1600)])
Dmax = np.array([i*0.01 for i in range(0, 17001)])
Nij = np.array([i * 0.001 for i in range(0, 2001)])

def AIS_3_cal_head(HIC):
    HIC = np.clip(HIC, 1, 2500)
    coefficients = np.array([
        [1.54, 0.00650],  # P(AIS≥1)
        [3.39, 0.00372]   # P(AIS≥3)
    ])
    threshold = 0.2
    c1 = coefficients[:, 0].reshape(-1, 1)
    c2 = coefficients[:, 1].reshape(-1, 1)
    HIC_inv = 200 / HIC
    hic_prob = 1.0 / (1.0 + np.exp(c1 + HIC_inv - c2 * HIC))
    AIS_31 = np.sum(hic_prob.T >= threshold, axis=1)
    # 映射到0，1，3
    AIS_31 = np.where(AIS_31 == 2, 3, AIS_31)
    AIS_32 = np.max(np.where(hic_prob.T >= threshold, np.array([1, 3]), 0), axis=1)
    return AIS_31, AIS_32

AIS_31, AIS_32 = AIS_3_cal_head(HIC)
print((AIS_31==AIS_32).all())

def AIS_cal_head(HIC):
    # 限制 HIC 范围，防止数值不稳定
    HIC = np.clip(HIC, 1, 2500)

    # 定义常量和系数
    coefficients = np.array([
        [1.54, 0.00650],  # P(AIS≥1)
        [2.49, 0.00483],  # P(AIS≥2)
        [3.39, 0.00372],  # P(AIS≥3)
        [4.90, 0.00351],  # P(AIS≥4)
        [7.82, 0.00429]   # P(AIS≥5)
    ])
    threshold = hic_threshold  # 经验概率阈值

    # 计算 P(AIS≥n) 的概率（向量化计算）
    c1 = coefficients[:, 0].reshape(-1, 1)  # 系数1
    c2 = coefficients[:, 1].reshape(-1, 1)  # 系数2
    HIC_inv = 200 / HIC  # HIC 的倒数部分

    # 计算所有 P(AIS≥n)
    hic_prob = 1.0 / (1.0 + np.exp(c1 + HIC_inv - c2 * HIC))
    # 确定 AIS 等级
    AIS1 = np.sum(hic_prob.T >= threshold, axis=1)
    AIS2 = np.max(np.where(hic_prob.T >= threshold, np.arange(1, 6), 0), axis=1)   
    return hic_prob, AIS1, AIS2

def AIS_cal_chest(C_disp):
    """
    Calculate AIS level from Chest Displacement (C_disp).
    Sets AIS to 0 for results less than AIS 2.
    """
    # ... (此处为您提供的 AIS_cal_chest 函数) ...
    # Clip C_disp range to prevent numerical instability
    C_disp = np.clip(C_disp, 0.0, 500.0)

    # Define coefficients [c1, c2] for P(AIS>=n) = 1 / (1 + exp(c1 - c2 * C_disp))
    # Based on document rev_criteria2.pdf, page 73, eq. 4.4
    coefficients = np.array([
        [1.8706, 0.04439],  # P(AIS≥2)
        [3.7124, 0.04750],  # P(AIS≥3)
        [5.0952, 0.04750],  # P(AIS≥4)
        [8.8274, 0.04590]   # P(AIS≥5)
    ])
    threshold = Dmax_threshold  # Empirical probability threshold for chest

    # Calculate P(AIS≥n) probabilities (vectorized)
    c1 = coefficients[:, 0].reshape(-1, 1)  # Intercepts
    c2 = coefficients[:, 1].reshape(-1, 1)  # Slopes
    
    # Calculate all P(AIS≥n) for n=2,3,4,5
    cdisp_prob = 1.0 / (1.0 + np.exp(c1 - c2 * C_disp))
    # 确定 AIS 等级
    raw_ais = np.sum(cdisp_prob.T >= threshold, axis=1)
    AIS1 = np.where(raw_ais > 0, raw_ais + 1, 0)
    AIS2 = np.max(np.where(cdisp_prob.T >= threshold, np.arange(2, 6), 0), axis=1)

    return cdisp_prob, AIS1, AIS2

def AIS_cal_neck(Nij):
    """
    Calculate AIS level from Neck Injury Criterion (Nij).
    Sets AIS to 0 for results less than AIS 2.
    """
    # ... (此处为您提供的 AIS_cal_neck 函数) ...
    # Clip Nij range to prevent numerical instability
    Nij = np.clip(Nij, 0, 50.0)

    # Define coefficients [c1, c2] for P(AIS>=n) = 1 / (1 + exp(c1 - c2 * Nij))
    # Based on document rev_criteria2.pdf, page 46, eq. 3.2
    coefficients = np.array([
        [2.054, 1.195],  # P(AIS≥2)
        [3.227, 1.969],  # P(AIS≥3)
        [2.693, 1.195],  # P(AIS≥4)
        [3.817, 1.195]   # P(AIS≥5)
    ])
    threshold = Nij_threshold  # Empirical probability threshold for neck

    # Calculate P(AIS≥n) probabilities (vectorized)
    c1 = coefficients[:, 0].reshape(-1, 1)  # Intercepts
    c2 = coefficients[:, 1].reshape(-1, 1)  # Slopes

    # Calculate all P(AIS≥n) for n=2,3,4,5
    nij_prob = 1.0 / (1.0 + np.exp(c1 - c2 * Nij))
    # 确定 AIS 等级
    raw_ais = np.sum(nij_prob.T >= threshold, axis=1)
    AIS1 = np.where(raw_ais > 0, raw_ais + 1, 0)
    AIS2 = np.max(np.where(nij_prob.T >= threshold, np.arange(2, 6), 0), axis=1)

    return nij_prob, AIS1, AIS2



# 三个子图画出，标题处不留空白；三张子图之间要留空白; 并标出阈值概率线及与各条曲线的交点
plt.figure(figsize=(6, 12))
# 绘制HIC VS HIC_prob曲线，几条曲线放在一起
HIC_prob, AIS1_h, AIS2_h = AIS_cal_head(HIC)
plt.subplot(3, 1, 1)
for i in range(1, 6):
    plt.plot(HIC, HIC_prob[i-1, :], label=f'P(AIS≥{i})')
plt.title('HIC vs P(AIS≥n)')
plt.xlabel('HIC')
plt.ylabel('Probability')
plt.legend()
plt.grid()
# 标出阈值概率线及与各条曲线的交点
plt.axhline(y=hic_threshold, color='r', linestyle='--', label='Threshold')
for i in range(1, 6):
    # 找到与阈值线的交点
    crossing_indices = np.where(np.diff(np.sign(HIC_prob[i-1, :] - hic_threshold)))[0]
    for idx in crossing_indices:
        plt.plot(HIC[idx], HIC_prob[i-1, idx], 'ro')  # 标出交点
        plt.text(HIC[idx], HIC_prob[i-1, idx], f'({HIC[idx]}, {HIC_prob[i-1, idx]:.2f})', fontsize=12, verticalalignment='bottom')

print((AIS1_h==AIS2_h).all())

# 绘制Dmax VS Dmax_prob曲线
Dmax_prob, AIS1_d, AIS2_d = AIS_cal_chest(Dmax)
plt.subplot(3, 1, 2)
for i in range(2, 6):
    plt.plot(Dmax, Dmax_prob[i-2, :], label=f'P(AIS≥{i})')
plt.title('Dmax vs P(AIS≥n)')
plt.xlabel('Dmax (mm)')
plt.ylabel('Probability')
plt.legend()
plt.grid()
# 标出阈值概率线及与各条曲线的交点
plt.axhline(y=Dmax_threshold, color='r', linestyle='--', label='Threshold')
for i in range(2, 6):
    # 找到与阈值线的交点
    crossing_indices = np.where(np.diff(np.sign(Dmax_prob[i-2, :] - Dmax_threshold)))[0]
    for idx in crossing_indices:
        plt.plot(Dmax[idx], Dmax_prob[i-2, idx], 'ro')  # 标出交点
        plt.text(Dmax[idx], Dmax_prob[i-2, idx], f'({Dmax[idx]:.2f}, {Dmax_prob[i-2, idx]:.2f})', fontsize=12, verticalalignment='bottom')

print((AIS1_d==AIS2_d).all())

# 绘制Nij VS Nij_prob曲线
Nij_prob, AIS1, AIS2 = AIS_cal_neck(Nij)
plt.subplot(3, 1, 3)
for i in range(2, 6):
    plt.plot(Nij, Nij_prob[i-2, :], label=f'P(AIS≥{i})')
plt.title('Nij vs P(AIS≥n)')
plt.xlabel('Nij')
plt.ylabel('Probability')
plt.legend()
plt.grid()
# 标出阈值概率线及与各条曲线的交点
plt.axhline(y=Nij_threshold, color='r', linestyle='--', label='Threshold')
for i in range(2, 6):
    # 找到与阈值线的交点
    crossing_indices = np.where(np.diff(np.sign(Nij_prob[i-2, :] - Nij_threshold)))[0]
    for idx in crossing_indices:
        plt.plot(Nij[idx], Nij_prob[i-2, idx], 'ro')  # 标出交点
        plt.text(Nij[idx], Nij_prob[i-2, idx], f'({Nij[idx]:.3f}, {Nij_prob[i-2, idx]:.2f})', fontsize=12, verticalalignment='bottom')

print((AIS1==AIS2).all())
# 调整布局：减少顶部空白，增加子图间距
plt.subplots_adjust(
    top=0.975,      # 减少顶部空白（标题处不留空白）
    bottom=0.05,   # 底部边距
    hspace=0.25    # 增加子图之间的垂直间距
)

# plt.show()

# 绘制HIC VS AIS1,AIS2
plt.figure(figsize=(6, 6))
plt.plot(HIC, AIS1_h, label='AIS from sum method', linestyle='--')
plt.plot(HIC, AIS2_h, label='AIS from max method', linestyle=':')
plt.title('HIC vs AIS Level')
plt.xlabel('HIC')
plt.ylabel('AIS Level')
plt.ylim(0, 6)
plt.legend()
plt.grid()
# 标出每个等级的分界点，也就是AIS每次首个增大的HIC值
for i in range(1, 6):
    hic_thres = HIC[np.where(AIS1_h == i)[0][0]]
    plt.axvline(x=hic_thres, color='gray', linestyle='--', linewidth=0.5)
    plt.text(hic_thres, 0.5, f'AIS {i} threshold: {hic_thres}', rotation=90, verticalalignment='bottom', fontsize=12)


# Dmax VS AIS1,AIS2
plt.figure(figsize=(6, 6))
plt.plot(Dmax, AIS1_d, label='AIS from sum method', linestyle='--')
plt.plot(Dmax, AIS2_d, label='AIS from max method', linestyle=':')
plt.title('Dmax vs AIS Level')
plt.xlabel('Dmax (mm)')
plt.ylabel('AIS Level')
plt.ylim(0, 6)
plt.legend()
plt.grid()
# 标出每个等级的分界点，也就是AIS每次首个增大的Dmax值
for i in range(2, 6):
    dmax_thres = Dmax[np.where(AIS1_d == i)[0][0]]
    plt.axvline(x=dmax_thres, color='gray', linestyle='--', linewidth=0.5)
    plt.text(dmax_thres, 0.5, f'AIS {i} threshold: {dmax_thres:.2f}', rotation=90, verticalalignment='bottom', fontsize=12)


# Nij VS AIS1,AIS2
plt.figure(figsize=(6, 6))
plt.plot(Nij, AIS1, label='AIS from sum method', linestyle='--')
plt.plot(Nij, AIS2, label='AIS from max method', linestyle=':')
plt.title('Nij vs AIS Level')
plt.xlabel('Nij')
plt.ylabel('AIS Level')
plt.ylim(0, 6)
plt.legend()
plt.grid()
# 标出每个等级的分界点，也就是AIS每次首个增大的Nij值
for i in range(2, 6):
    nij_thres = Nij[np.where(AIS1 == i)[0][0]]
    plt.axvline(x=nij_thres, color='gray', linestyle='--', linewidth=0.5)
    plt.text(nij_thres, 0.5, f'AIS {i} threshold: {nij_thres:.3f}', rotation=90, verticalalignment='bottom', fontsize=12)

plt.show()

# %% new 20260121
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from typing import Union

HIC15 = np.array([i for i in range(1, 1500)])
Dmax = np.array([i*0.01 for i in range(0, 8000)])
Nij = np.array([i * 0.001 for i in range(0, 1500)])


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


def plot_ais_risk_curve(
    body_part: str,
    prob_thresholds: list,
    figsize: tuple = (10, 7),
    save_path: str = None
):
    """
    绘制指定部位的 AIS 风险概率曲线 P(AIS≥3+)，并标注指定概率值的交点。

    Args:
        body_part (str): 部位名称，可选 'head', 'chest', 'neck'
        prob_thresholds (list): 概率阈值列表，如 [0.05, 0.1, 0.2, 0.3, 0.5]
        figsize (tuple): 图形大小
        save_path (str): 保存路径，若为 None 则不保存
    """
    # 根据部位选择参数
    if body_part.lower() == 'head':
        x_values = HIC15
        x_label = 'HIC15'
        title = 'Head Injury Risk Curve: P(AIS≥3+)'
        # P_head(AIS3+) = Φ((ln(HIC15) - 7.45231) / 0.73998)
        z = (np.log(x_values) - 7.45231) / 0.73998
        prob = norm.cdf(z)
        
        # 创建图形
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(x_values, prob, 'b-', linewidth=2, label='Head')
        
        # 绘制概率阈值水平线和交点
        colors = plt.cm.tab10(np.linspace(0, 1, len(prob_thresholds)))
        for i, p_thresh in enumerate(prob_thresholds):
            ax.axhline(y=p_thresh, color=colors[i], linestyle='--', alpha=0.7)
            # 在虚线左侧标注概率值
            ax.annotate(f'{p_thresh:.3f}', 
                        xy=(x_values[0], p_thresh),
                        xytext=(-5, 0), textcoords='offset points',
                        fontsize=11, fontweight='bold',
                        color=colors[i], ha='right', va='center')
            
            idx = np.where(prob >= p_thresh)[0]
            if len(idx) > 0:
                first_idx = idx[0]
                if first_idx > 0:
                    x_cross = np.interp(p_thresh, [prob[first_idx-1], prob[first_idx]], 
                                        [x_values[first_idx-1], x_values[first_idx]])
                else:
                    x_cross = x_values[first_idx]
                ax.plot(x_cross, p_thresh, 'o', color=colors[i], markersize=10)
                ax.annotate(f'{x_cross:.1f}', 
                            xy=(x_cross, p_thresh),
                            xytext=(5, 10), textcoords='offset points',
                            fontsize=11, fontweight='bold',
                            color=colors[i],
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        ax.legend(fontsize=12, loc='lower right')
        
    elif body_part.lower() == 'chest':
        x_values = Dmax
        x_label = 'Dmax (mm)'
        title = 'Chest Injury Risk Curve: P(AIS≥3+)'
        
        # 三种 OT 的缩放因子
        ot_configs = [
            {'OT': 1, 'name': '5th Female', 'scale': 221.0 / 182.9, 'color': 'r', 'linestyle': '-'},
            {'OT': 2, 'name': '50th Male', 'scale': 1.0, 'color': 'b', 'linestyle': '-'},
            {'OT': 3, 'name': '95th Male', 'scale': 221.0 / 246.38, 'color': 'g', 'linestyle': '-'}
        ]
        
        # 创建图形
        fig, ax = plt.subplots(figsize=figsize)
        
        # 概率阈值颜色
        thresh_colors = plt.cm.Greys(np.linspace(0.4, 0.7, len(prob_thresholds)))
        
        # 先绘制所有水平阈值线并标注概率值
        for i, p_thresh in enumerate(prob_thresholds):
            ax.axhline(y=p_thresh, color=thresh_colors[i], linestyle='--', alpha=0.7, linewidth=1)
            # 在虚线左侧标注概率值
            ax.annotate(f'{p_thresh:.2f}', 
                        xy=(x_values[0], p_thresh),
                        xytext=(-5, 0), textcoords='offset points',
                        fontsize=11, fontweight='bold',
                        color=thresh_colors[i], ha='right', va='center')
        
        # 绘制三条曲线及交点
        for ot_cfg in ot_configs:
            scale = ot_cfg['scale']
            # 等效 Dmax
            Dmax_eq = x_values * scale
            prob = 1.0 / (1.0 + np.exp(10.5456 - 1.568 * np.power(Dmax_eq, 0.4612)))
            
            # 绘制曲线
            ax.plot(x_values, prob, color=ot_cfg['color'], linestyle=ot_cfg['linestyle'], 
                    linewidth=2, label=f"{ot_cfg['name']}")
            
            # 绘制与阈值的交点
            for i, p_thresh in enumerate(prob_thresholds):
                idx = np.where(prob >= p_thresh)[0]
                if len(idx) > 0:
                    first_idx = idx[0]
                    if first_idx > 0:
                        x_cross = np.interp(p_thresh, [prob[first_idx-1], prob[first_idx]], 
                                            [x_values[first_idx-1], x_values[first_idx]])
                    else:
                        x_cross = x_values[first_idx]
                    
                    # 绘制交点
                    ax.plot(x_cross, p_thresh, 'o', color=ot_cfg['color'], markersize=8)
                    
                    # 标注横坐标
                    offset_y = 5
                    ax.annotate(f'{x_cross:.1f}', 
                                xy=(x_cross, p_thresh),
                                xytext=(3, offset_y), textcoords='offset points',
                                fontsize=11, fontweight='bold',
                                color=ot_cfg['color'],
                                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
        
        ax.legend(fontsize=12, loc='upper left')
        
    elif body_part.lower() == 'neck':
        x_values = Nij
        x_label = 'Nij'
        title = 'Neck Injury Risk Curve: P(AIS≥3+)'
        # P_neck_Nij(AIS3+) = 1 / (1 + e^(3.2269 - 1.9688 * Nij))
        prob = 1.0 / (1.0 + np.exp(3.2269 - 1.9688 * x_values))
        
        # 创建图形
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(x_values, prob, 'b-', linewidth=2, label='Neck')
        
        # 绘制概率阈值水平线和交点
        colors = plt.cm.tab10(np.linspace(0, 1, len(prob_thresholds)))
        for i, p_thresh in enumerate(prob_thresholds):
            ax.axhline(y=p_thresh, color=colors[i], linestyle='--', alpha=0.7)
            # 在虚线左侧标注概率值
            ax.annotate(f'{p_thresh:.2f}', 
                        xy=(x_values[0], p_thresh),
                        xytext=(-5, 0), textcoords='offset points',
                        fontsize=11, fontweight='bold',
                        color=colors[i], ha='right', va='center')
            
            idx = np.where(prob >= p_thresh)[0]
            if len(idx) > 0:
                first_idx = idx[0]
                if first_idx > 0:
                    x_cross = np.interp(p_thresh, [prob[first_idx-1], prob[first_idx]], 
                                        [x_values[first_idx-1], x_values[first_idx]])
                else:
                    x_cross = x_values[first_idx]
                ax.plot(x_cross, p_thresh, 'o', color=colors[i], markersize=10)
                ax.annotate(f'{x_cross:.3f}', 
                            xy=(x_cross, p_thresh),
                            xytext=(5, 10), textcoords='offset points',
                            fontsize=11, fontweight='bold',
                            color=colors[i],
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        ax.legend(fontsize=12, loc='lower right')
        
    else:
        raise ValueError("body_part 必须是 'head', 'chest' 或 'neck'")

    # 设置图形属性
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel('Probability', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig, ax


# 使用示例
if __name__ == "__main__":
    # 绘制头部 P(AIS>=3+) 曲线
    plot_ais_risk_curve('head', prob_thresholds=[0.01, 0.05, 0.1, 0.2, 0.3])
    
    # 绘制胸部 P(AIS>=3+) 曲线
    plot_ais_risk_curve('chest', prob_thresholds=[0.02, 0.06, 0.15, 0.25, 0.4])
    
    # 绘制颈部 P(AIS>=3+) 曲线
    plot_ais_risk_curve('neck', prob_thresholds=[0.06, 0.1, 0.2, 0.3, 0.4])
