import torch
import numpy as np
from dtw import dtw

class ISORating:
    """
    根据 ISO/TS 18571:2024 标准，计算两条时间历史曲线的客观评级分数。

    该类实现了 Corridor, Phase, Magnitude, 和 Slope 四个子分数的计算，
    并最终给出一个加权的综合ISO评级分数。
    """
    def __init__(self, analyzed_signal, reference_signal, dt=0.001):
        """
        初始化ISORating计算器。

        参数:
        analyzed_signal (array-like): 待评估信号 (Analyzed Signal, C(t))。
                                      通常是计算机仿真(CAE)数据。本项目中，为 output 或 pred_wave
        reference_signal (array-like): 参考信号 (Reference Signal, T(t))。
                                       通常是物理试验(Test)数据。本项目中，为 target 或 true_wave
        dt (float): 信号的时间间隔，单位为秒。默认为 0.001s (对应1kHz采样率)。
        """
        if len(analyzed_signal) != len(reference_signal):
            raise ValueError("输入信号 'analyzed_signal' 和 'reference_signal' 的长度必须相同。")
            
        self.analyzed_signal = np.asarray(analyzed_signal)
        self.reference_signal = np.asarray(reference_signal)
        self.dt = dt

        # 初始化用于存储中间和最终结果的成员变量
        self.iso_rating = None
        self.corridor_score = None
        self.phase_score = None
        self.magnitude_score = None
        self.slope_score = None
        
        # 用于存储相位校正后的曲线，供后续方法使用
        self._shifted_analyzed = None
        self._shifted_reference = None

    def calculate(self):
        """
        执行完整的ISO评级计算，并返回最终的综合分数。

        该方法会依次计算所有四个子分数，并将它们存储在类的成员变量中，
        然后根据标准权重计算最终的综合分数。

        返回:
        float: 最终的ISO综合评级分数 (R)。
        """
        # 优化点：如果已经计算过，直接返回结果，避免重复计算
        if self.iso_rating is not None:
            return self.iso_rating

        # 1. 定义权重因子 (依据标准 Table 6.1)
        w_corridor = 0.4
        w_phase = 0.2
        w_magnitude = 0.2
        w_slope = 0.2

        # 2. 依次调用私有方法计算各子分数，并将结果存储在成员变量中
        #    注意：调用顺序很重要，因为幅值和斜率计算依赖于相位计算的结果
        self.corridor_score = self._calculate_corridor_score()
        self.phase_score = self._calculate_phase_score()
        self.magnitude_score = self._calculate_magnitude_score()
        self.slope_score = self._calculate_slope_score()

        # 3. 根据标准公式(6.1)计算最终的综合分数
        self.iso_rating = (w_corridor * self.corridor_score +
                           w_phase * self.phase_score +
                           w_magnitude * self.magnitude_score +
                           w_slope * self.slope_score)

        return self.iso_rating

    def _calculate_corridor_score(self):
        """计算廊道分数 (Z)。非对称。"""
        a0 = 0.05
        b0 = 0.50
        kz = 2
        
        t_norm = np.max(np.abs(self.reference_signal))
        delta_i = a0 * t_norm
        delta_o = b0 * t_norm
        
        abs_diff = np.abs(self.reference_signal - self.analyzed_signal)
        
        mark = np.zeros_like(abs_diff, dtype=float)

        # 情况1：在内走廊内 (得分=1)
        inner_mask = abs_diff <= delta_i
        mark[inner_mask] = 1.0

        # 情况3：在内外走廊之间
        between_mask = (abs_diff > delta_i) & (abs_diff <= delta_o)
        denominator = delta_o - delta_i
        if denominator > 1e-9: # 避免除以零
            mark[between_mask] = ((delta_o - abs_diff[between_mask]) / denominator) ** kz
        
        return np.mean(mark)

    def _calculate_phase_score(self):
        """计算相位分数 (Ep)。对称。同时生成并存储移位后的曲线。"""
        max_shift_percent = 0.2
        exponent_factor = 1
        
        n = len(self.reference_signal)
        max_shift_steps = int(np.floor(max_shift_percent * n))

        # 计算所有可能的互相关值
        # m=0 (不平移)
        corr_orig = np.corrcoef(self.reference_signal, self.analyzed_signal)[0, 1]
        
        # m > 0 (左移和右移)
        rou_L = np.zeros(max_shift_steps)
        rou_R = np.zeros(max_shift_steps)
        for m in range(1, max_shift_steps + 1):
            # 左移 m 步
            rou_L[m-1] = np.corrcoef(self.analyzed_signal[m:n], self.reference_signal[0:n-m])[0, 1]
            # 右移 m 步
            rou_R[m-1] = np.corrcoef(self.analyzed_signal[0:n-m], self.reference_signal[m:n])[0, 1]

        # 找出左、右、中三个方向上的最佳候选者
        pos_L = np.argmax(rou_L) + 1 if len(rou_L) > 0 else 0
        max_L_corr = rou_L[pos_L - 1] if len(rou_L) > 0 else -np.inf
        
        pos_R = np.argmax(rou_R) + 1 if len(rou_R) > 0 else 0
        max_R_corr = rou_R[pos_R - 1] if len(rou_R) > 0 else -np.inf

        # 初始化最佳选择为不平移
        best_corr = corr_orig
        n_E = 0
        is_left_shift = False # 标记最终方向

        # --- 严格按ISO标准三层优先级进行决策 ---
        # 1. 比较左移与当前最佳
        if max_L_corr > best_corr:
            best_corr = max_L_corr
            n_E = pos_L
            is_left_shift = True
        elif max_L_corr == best_corr:
            if pos_L < n_E: # 步数少者优先
                n_E = pos_L
                is_left_shift = True
            # 如果步数也相等(只能是n_E=0时)，不处理，因为原始位置优先于有位移的位置

        # 2. 比较右移与当前最佳 (此时最佳可能是原始或左移)
        if max_R_corr > best_corr:
            best_corr = max_R_corr
            n_E = pos_R
            is_left_shift = False
        elif max_R_corr == best_corr:
            if pos_R < n_E: # 步数少者优先
                n_E = pos_R
                is_left_shift = False
            elif pos_R == n_E and not is_left_shift: # 步数也相等，且当前最佳不是左移
                # 标准规定左移优先于右移，但这里当前最佳是右移，所以什么都不做
                pass
            elif pos_R == n_E and is_left_shift: # 步数也相等，且当前最佳是左移
                # 标准规定左移优先于右移，所以什么都不做
                pass

        # 根据最终确定的 n_E 和方向，生成并存储移位后的曲线
        if n_E == 0:
            self._shifted_analyzed = self.analyzed_signal
            self._shifted_reference = self.reference_signal
        elif is_left_shift:
            self._shifted_analyzed = self.analyzed_signal[n_E:n]
            self._shifted_reference = self.reference_signal[0:n-n_E]
        else: # Right shift
            self._shifted_analyzed = self.analyzed_signal[0:n-n_E]
            self._shifted_reference = self.reference_signal[n_E:n]

        # 根据标准公式(6.12)计算分数
        if n_E >= max_shift_steps:
            return 0.0
        else:
            return ((max_shift_steps - n_E) / max_shift_steps) ** exponent_factor

    def _calculate_magnitude_score(self):
        """计算幅值分数 (EM)。非对称。"""
        max_allowed_error = 0.5
        exponent_factor = 1
        
        n_shifted = len(self._shifted_reference)
        window_size = int(np.ceil(0.1 * n_shifted))

        alignment = dtw(self._shifted_reference, self._shifted_analyzed,
                        dist_method='sqeuclidean',
                        window_type='sakoechiba',
                        window_args={'window_size': window_size})

        ref_warped = self._shifted_reference[alignment.index1]
        ana_warped = self._shifted_analyzed[alignment.index2]

        epsilon = 1e-9
        norm_diff = np.linalg.norm(ana_warped - ref_warped)
        norm_ref = np.linalg.norm(ref_warped)
        
        magnitude_error = norm_diff / (norm_ref + epsilon)

        if magnitude_error >= max_allowed_error:
            return 0.0
        else:
            return ((max_allowed_error - magnitude_error) / max_allowed_error) ** exponent_factor

    def _calculate_slope_score(self):
        """计算斜率分数 (Es)。非对称。"""
        max_allowed_error = 2.0
        exponent_factor = 1
        
        T_ts = self._shifted_reference
        C_ts = self._shifted_analyzed
        n = len(T_ts)

        def calculate_raw_derivative(curve, dt):
            if len(curve) < 2: return np.zeros_like(curve)
            deriv = np.zeros_like(curve)
            deriv[0] = (curve[1] - curve[0]) / dt
            if len(curve) > 2: deriv[1:-1] = (curve[2:] - curve[:-2]) / (2 * dt)
            deriv[-1] = (curve[-1] - curve[-2]) / dt
            return deriv

        def smooth_derivative(raw_deriv):
            # 如果信号长度小于窗口尺寸，直接返回原始导数，避免错误
            if len(raw_deriv) < 9:
                return raw_deriv
            
            # 使用 'same' 模式，输出数组与输入数组长度相同
            # 卷积核是一个权重均为1/9的九点窗口
            smoothed = np.convolve(raw_deriv, np.ones(9) / 9, mode='same')
            
            return smoothed

        T_d = smooth_derivative(calculate_raw_derivative(T_ts, self.dt))
        C_d = smooth_derivative(calculate_raw_derivative(C_ts, self.dt))

        epsilon = 1e-9
        norm_diff = np.linalg.norm(C_d - T_d)
        norm_ref = np.linalg.norm(T_d)
        
        slope_error = norm_diff / (norm_ref + epsilon)

        if slope_error >= max_allowed_error:
            return 0.0
        else:
            return ((max_allowed_error - slope_error) / max_allowed_error) ** exponent_factor

def _calculate_iso_rating_for_channel(output, target, channel_idx, dt=0.001): # dt单位为秒
    """
    内部帮助函数，用于计算指定通道的平均ISO-rating。
    """
    with torch.no_grad():
        # 将PyTorch张量转换为Numpy数组
        # output 和 target 的形状: (batch_size, 3, 150)
        pred_waves = output.cpu().numpy()
        true_waves = target.cpu().numpy()

        batch_size = pred_waves.shape[0]
        total_score = 0.0

        for i in range(batch_size):
            # 提取指定通道的单条波形
            pred_wave = pred_waves[i, channel_idx, :]
            true_wave = true_waves[i, channel_idx, :]
            
            # 实例化并计算得分
            iso_calculator = ISORating(analyzed_signal=pred_wave, reference_signal=true_wave, dt=dt)
            total_score += iso_calculator.calculate()

        return total_score / batch_size if batch_size > 0 else 0.0

def iso_rating_x(output, target, dt=0.001):
    """
    计算 X 方向波形的平均 ISO-rating。
    """
    return _calculate_iso_rating_for_channel(output, target, channel_idx=0, dt=dt)

def iso_rating_y(output, target, dt=0.001):
    """
    计算 Y 方向波形的平均 ISO-rating。
    """
    return _calculate_iso_rating_for_channel(output, target, channel_idx=1, dt=dt)

def iso_rating_z(output, target, dt=0.001):
    """
    计算 Z 方向波形的平均 ISO-rating。
    """
    return _calculate_iso_rating_for_channel(output, target, channel_idx=2, dt=dt)

def _rmse_for_channel(output, target, channel_idx):
    """
    内部帮助函数，用于计算指定通道的均方根误差 (RMSE)。
    
    :param output: 模型的预测输出张量，形状为 (batch_size, 3, 150)。
    :param target: 真实的标签张量，形状为 (batch_size, 3, 150)。
    :param channel_idx: 要计算的通道索引 (0 for x, 1 for y, 2 for z)。
    :return: 指定通道上的RMSE标量值。
    """
    with torch.no_grad():
        # 从输出和目标张量中选取特定通道的数据
        output_channel = output[:, channel_idx, :]
        target_channel = target[:, channel_idx, :]
        
        # 计算该通道的RMSE
        loss = torch.sqrt(torch.mean((output_channel - target_channel)**2))
        
    return loss.item()

def rmse_x(output, target):
    """
    计算 X 方向波形的均方根误差 (RMSE)。
    """
    return _rmse_for_channel(output, target, channel_idx=0)

def rmse_y(output, target):
    """
    计算 Y 方向波形的均方根误差 (RMSE)。
    """
    return _rmse_for_channel(output, target, channel_idx=1)

def rmse_z(output, target):
    """
    计算 Z 方向波形的均方根误差 (RMSE)。
    """
    return _rmse_for_channel(output, target, channel_idx=2)


if __name__ == "__main__":
    # 测试代码
    import pandas as pd
    pulse1_np = pd.read_csv(r'E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\仿真数据库相关\old模型_全宽正碰结果\x1.csv', sep='\t', header=None, usecols=[1]).values
    pulse2_np = pd.read_csv(r'E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\仿真数据库相关\old模型_全宽正碰结果\x2.csv', sep='\t', header=None, usecols=[1]).values
    pulse3_np = pd.read_csv(r'E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\仿真数据库相关\old模型_全宽正碰结果\x3.csv', sep='\t', header=None, usecols=[1]).values
    pulse4_np = pd.read_csv(r'E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\仿真数据库相关\old模型_全宽正碰结果\x4.csv', sep='\t', header=None, usecols=[1]).values
    pulse5_np = pd.read_csv(r'E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\仿真数据库相关\old模型_全宽正碰结果\x5.csv', sep='\t', header=None, usecols=[1]).values
    pulse6_np = pd.read_csv(r'E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\仿真数据库相关\old模型_全宽正碰结果\x6.csv', sep='\t', header=None, usecols=[1]).values

    downsample_indices = np.arange(100, 20001, 100)

    pulse1_np = pulse1_np[downsample_indices].squeeze()
    pulse2_np = pulse2_np[downsample_indices].squeeze()
    pulse3_np = pulse3_np[downsample_indices].squeeze()
    pulse4_np = pulse4_np[downsample_indices].squeeze()
    pulse5_np = pulse5_np[downsample_indices].squeeze()
    pulse6_np = pulse6_np[downsample_indices].squeeze()

    # 上述6个波形，计算出corridor，phase, magnitude, slope，isorating的分数矩阵（6*6）并打印
    iso_scores = np.zeros((6, 6, 5))  # 6*6*5的矩阵，最后一维分别对应corridor, Phase, Magnitude, Slope, ISO-Rating

    for i in range(6):
        for j in range(6):
            iso_calculator = ISORating(eval(f'pulse{i+1}_np'), eval(f'pulse{j+1}_np'))
            iso_scores[i, j, 4] = iso_calculator.calculate()
            iso_scores[i, j, 0] = iso_calculator.corridor_score
            iso_scores[i, j, 1] = iso_calculator.phase_score
            iso_scores[i, j, 2] = iso_calculator.magnitude_score
            iso_scores[i, j, 3] = iso_calculator.slope_score

    print("\n 终版版本:ISO-Rating Matrix:")
    print(np.round(iso_scores[..., 4], 6))
    # print("\n 终版版本:corridor_score Matrix:")
    # print(np.round(iso_scores[..., 0], 6))
    # print("\n 终版版本:Phase Score Matrix:")
    # print(np.round(iso_scores[..., 1], 6))
    # print("\n 终版版本:Magnitude Score Matrix:")
    # print(np.round(iso_scores[..., 2], 6))
    # print("\n 终版版本:Slope Score Matrix:")
    # print(np.round(iso_scores[..., 3], 6))

