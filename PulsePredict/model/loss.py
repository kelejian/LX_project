import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================================================================
# 1. 核心调度器 (Manager)
# ==========================================================================================

class AutoWeightedLoss(nn.Module):
    """
    [自动加权总损失容器] - 项目唯一的 Loss 入口
    
    职责：
    1. 解析配置列表，实例化所有子 Loss。
    2. 管理可学习的权重参数 (Homoscedastic Uncertainty)，实现多任务自动平衡。
    3. 结合人工先验权重 (Prior Weight) 计算最终总损失。
    4. 监控：向 TensorBoard 报告原始的、具有物理意义的 Loss 值 (未被 s_i 缩放)。
    """
    def __init__(self, loss_configs):
        """
        :param loss_configs: 配置列表，例如:
            [
                {'type': 'MultiScaleLoss', 'prior_weight': 1.0, 'args': {...}},
                {'type': 'CorridorLoss', 'prior_weight': 2.0, 'args': {...}}
            ]
        """
        super().__init__()
        self.losses = nn.ModuleList()
        self.prior_weights = []  # 人工先验权重 (lambda_i)
        self.loss_names = []

        for config in loss_configs: # 逐个解析各项子 Loss 配置
            loss_type = config['type']
            prior_w = config.get('prior_weight', 1.0)
            loss_args = config.get('args', {})

            # 动态实例化 Loss 类 (从当前模块 globals 中查找类)
            if loss_type not in globals():
                raise ValueError(f"未知的 Loss 类型: {loss_type}")
            loss_cls = globals()[loss_type] # 类引用
            
            self.losses.append(loss_cls(**loss_args)) # 实例化并添加到 ModuleList
            self.prior_weights.append(prior_w)
            self.loss_names.append(loss_type)

        # 初始化可学习参数 s_i = log(sigma^2)
        # 初始化为0，意味着初始 sigma=1，保证训练初期梯度平稳
        self.log_vars = nn.Parameter(torch.zeros(len(self.losses)))
        
        # 注册先验权重为 buffer (不参与梯度更新，但随模型保存)
        self.register_buffer('priors', torch.tensor(self.prior_weights))

    def forward(self, model_output, target):
        """
        :param:
            model_output: 模型输出 (可能是列表、元组或张量)
            target: 真实标签 (B, C, L)
        :return: 
            total_loss: 用于反向传播的加权总损失
            loss_components: 用于监控的原始损失值字典 (对应旧版 tensorboard 曲线)
        """
        total_loss = 0
        loss_components = {}

        for i, loss_fn in enumerate(self.losses):
            # 1. 计算子 Loss (返回的是物理尺度的标量，即加权平均后的 Loss)
            raw_loss = loss_fn(model_output, target)
            
            # 2. 自动加权逻辑 (Kendall et al.): L_final = lambda * (0.5 * exp(-s) * L_raw + 0.5 * s)
            s_i = self.log_vars[i]
            precision = torch.exp(-s_i)
            
            # 结合先验权重和不确定性权重
            weighted_loss = self.priors[i] * (0.5 * precision * raw_loss + 0.5 * s_i)
            
            total_loss += weighted_loss
            
            # 3. 记录原始 loss 值 (Raw Physical Loss)用于 TensorBoard 监控
            loss_components[self.loss_names[i]] = raw_loss.item()

        return total_loss, loss_components


# ==========================================================================================
# 2. 统一基类 (Base Class)
# ==========================================================================================

class BaseSingleScaleLoss(nn.Module):
    """
    [单尺度损失基类] - 所有物理/形态 Loss 的父类
    
    核心功能：
    1. 统一接口：自动从复杂的 model_output (多尺度列表/高斯元组) 中提取最后阶段的物理均值。
    2. 统一加权：自动处理通道加权 (Channel Weighting) 和 Reduction (Mean)。
    
    子类开发指南：
    只需实现 `forward_step(pred, target)`，计算原始的 Loss 张量 (B, C, L) 或 (B, C)，
    无需关心 reduction 和 weighting。
    """
    def __init__(self, channel_weights=[1.0, 1.0, 1.0]):
        super().__init__()
        # 转换为 (C,) 的 1D Tensor，方便后续计算
        self.register_buffer('channel_weights', torch.tensor(channel_weights, dtype=torch.float32))

    def _get_prediction(self, model_output):
        """解析模型输出，提取最后阶段的均值预测"""
        pred = model_output
        if isinstance(pred, (list, tuple)):
            pred = pred[-1] # 取最后一个尺度
        if isinstance(pred, (list, tuple)):
            pred = pred[0]  # 取均值 (如果是 GauNLL 输出)
        return pred

    def forward_step(self, pred, target):
        """
        [抽象方法] 子类必须实现。
        计算原始损失，不要进行 mean/sum reduction。
        :return: Tensor, 形状 (B, C, L) 或 (B, C)
        """
        raise NotImplementedError

    def forward(self, model_output, target):
        # 1. 解析与对齐
        pred = self._get_prediction(model_output)
        
        # 2. 形状对齐 (防守性编程)
        # 物理 Loss 通常在最高分辨率计算，如果 Target 分辨率不匹配则进行插值
        if pred.shape[-1] != target.shape[-1]:
            target = F.interpolate(target, size=pred.shape[-1], mode='linear', align_corners=False)
            
        # 2. 计算原始损失,reduction='none' (B, C, L) 或 (B, C)
        raw_loss = self.forward_step(pred, target)
        
        # 3. 通道加权平均逻辑

        # 步骤 A: 对 Batch 和 Time 维度求均值，保留 Channel 维度 -> (C,)
        dims_to_reduce = [0] + list(range(2, raw_loss.ndim)) # e.g.(B, C, L) -> reduce over B and L, remain C
        loss_per_channel = raw_loss.mean(dim=dims_to_reduce) # (C,)
        
        # 步骤 B: 确保权重设备一致
        weights = self.channel_weights.to(loss_per_channel.device)
        
        # 步骤 C: 通道加权平均 (Weighted Average)
        # Formula: Sum(Loss_c * w_c) / Sum(w_c)
        weighted_sum = (loss_per_channel * weights).sum()
        weight_sum = weights.sum()
        
        if weight_sum > 0:
            return weighted_sum / weight_sum
        else:
            return torch.tensor(0.0, device=raw_loss.device)


# ==========================================================================================
# 3. 多尺度损失 (Specialized Loss)
# ==========================================================================================

class MultiScaleLoss(nn.Module):
    """
    [多尺度回归损失]
    
    职责：对模型输出的所有尺度进行监督。
    特点：
    1. 支持 GaussianNLLLoss (同时监督 Mean 和 Var) 或 普通 MSE/L1 (只监督 Mean)。
    2. 内置 尺度加权 (Scale Weights) 和 通道加权 (Channel Weights)。
    """
    def __init__(self, scale_weights=[0.1, 0.2, 1.0], channel_weights=[1.0, 1.0, 1.0], 
                 base_loss_type='L1Loss', base_loss_args=None):
        super().__init__()
        self.scale_weights = scale_weights
        self.register_buffer('channel_weights', torch.tensor(channel_weights, dtype=torch.float32))
        
        # 实例化基础损失函数 (强制 reduction='none' 以便手动加权)
        base_loss_args = base_loss_args or {}
        base_loss_args['reduction'] = 'none'
        
        if base_loss_type == 'GaussianNLLLoss':
            self.base_criterion = nn.GaussianNLLLoss(**base_loss_args)
            self.is_gauss = True
        else:
            self.base_criterion = getattr(nn, base_loss_type)(**base_loss_args)
            self.is_gauss = False

    def forward(self, model_output, target):
        # 兼容性处理: 确保 model_output 是列表
        if not isinstance(model_output, (list, tuple)):
            preds_list = [model_output]
        else:
            preds_list = model_output

        total_loss_sum = 0.0
        total_scale_weight_sum = 0.0
        
        # 遍历每个尺度
        for i, pred_item in enumerate(preds_list):
            # 如果超出权重配置或权重为0，跳过
            if i >= len(self.scale_weights) or self.scale_weights[i] == 0:
                continue

            current_scale_w = self.scale_weights[i]

            # 1. 解析当前尺度的预测
            if self.is_gauss:
                # 期望格式tuple (mean, var)
                if not isinstance(pred_item, (list, tuple)):
                    # 如果模型没开启 GauNll 但 Loss 配置了 GauNll，这里会报错，属于配置错误
                    raise ValueError(f"MultiScaleLoss 配置为 GaussianNLLLoss，但模型输出尺度 {i} 不是 (mean, var) 元组。")
                pred, var = pred_item
            else:
                # 期望格式tensor: mean (or (mean, var) 但只取 mean)
                pred = pred_item[0] if isinstance(pred_item, (list, tuple)) else pred_item

            # 2. 对齐目标 (Interpolate Target) 到当前尺度长度
            if pred.shape[-1] != target.shape[-1]:
                curr_target = F.interpolate(target, size=pred.shape[-1], mode='linear', align_corners=False)
            else:
                curr_target = target

            # 3. 计算基础 Loss , reduction默认为none, 因此输出不为标量
            if self.is_gauss:
                loss = self.base_criterion(pred, curr_target, var) # nn.GaussianNLLLoss(input, target, var)
            else:
                loss = self.base_criterion(pred, curr_target) # e.g., nn.MSELoss(input, target)

            # 4. 通道加权平均
            dims_to_reduce = [0] + list(range(2, loss.ndim)) # e.g.(B, C, L) -> reduce over B and L, remain C
            loss_per_channel = loss.mean(dim=dims_to_reduce) # (C,)
            
            weights = self.channel_weights.to(loss.device)
            channel_weighted_mean = (loss_per_channel * weights).sum() / weights.sum() #  Sum(L_c * w_c) / Sum(w_c); 标量

            # 5. 累加尺度损失
            total_loss_sum += current_scale_w * channel_weighted_mean # 各尺度loss值加权累加
            total_scale_weight_sum += current_scale_w # 各尺度对应权重累加

        # 6. 尺度加权平均 total_loss / sum(scale_loss_weights)
        if total_scale_weight_sum > 0:
            return total_loss_sum / total_scale_weight_sum
        else:
            return torch.tensor(0.0, device=target.device)


# ==========================================================================================
# 4. 物理与形态损失实现 (Inheriting BaseSingleScaleLoss)
# ==========================================================================================

class RegressionLoss(BaseSingleScaleLoss):
    """
    [通用单尺度回归损失]
    用于替代 MultiScaleLoss，仅对最终输出进行 L1/MSE 监督。
    """
    def __init__(self, loss_type='L1Loss', **kwargs):
        super().__init__(**kwargs)
        self.criterion = getattr(nn, loss_type)(reduction='none')

    def forward_step(self, pred, target):
        return self.criterion(pred, target)

class CorridorLoss(BaseSingleScaleLoss):
    """
    [廊道损失] ISO-18571
    核心逻辑：定义一个内廊道 (Inner Corridor)，只惩罚超出该范围的误差。
    """
    def __init__(self, inner_corridor_width=0.05, exponent=2.0, **kwargs):
        super().__init__(**kwargs)
        self.inner_corridor_width = inner_corridor_width
        self.exponent = exponent

    def forward_step(self, pred, target):
        '''
        :param pred: 模型的预测输出张量, 形状 (B, C, L)。
        :param target: 真实的目标张量, 形状 (B, C, L)。
        :return: 计算得到的廊道损失张量, 形状 (B, C, L), 未经聚合。
        '''
        # 计算每个样本的幅值基准 t_norm: (B, C, 1)
        t_norm = torch.max(torch.abs(target), dim=-1, keepdim=True)[0]
        delta_i = self.inner_corridor_width * (t_norm + 1e-9)

        abs_diff = torch.abs(pred - target)
        # ReLU 截断：只保留 |error| > delta_i 的部分
        exceeded_error = F.relu(abs_diff - delta_i)
        
        # 指数惩罚, 返回 (B, C, L)
        return torch.pow(exceeded_error, self.exponent)

class SlopeLoss(BaseSingleScaleLoss):
    """
    [斜率损失] ISO-18571
    核心逻辑：惩罚预测波形与真实波形在一阶导数（斜率）上的差异。
    """
    def __init__(self, apply_smoothing=True, smoothing_window_size=9, **kwargs):
        super().__init__(**kwargs)
        self.apply_smoothing = apply_smoothing
        if apply_smoothing:
            # 定义平滑卷积核 (1, 1, K)
            kernel = torch.ones(1, 1, smoothing_window_size) / smoothing_window_size
            self.register_buffer('kernel', kernel)

    def forward_step(self, pred, target):
        # 1. 计算一阶差分 (B, C, L-1)
        pred_slope = pred[..., 1:] - pred[..., :-1]
        target_slope = target[..., 1:] - target[..., :-1]

        # 2. (可选) 对斜率进行平滑处理 (模拟 ISO 标准流程)
        if self.apply_smoothing:
            # 融合 B, C 维度以便使用 conv1d: (B*C, 1, L-1)
            B, C, L = pred_slope.shape
            p_s = pred_slope.view(B*C, 1, L)
            t_s = target_slope.view(B*C, 1, L)
            
            # Padding='same' 保持长度不变
            p_s = F.conv1d(p_s, self.kernel, padding='same')
            t_s = F.conv1d(t_s, self.kernel, padding='same')
            
            pred_slope = p_s.view(B, C, L)
            target_slope = t_s.view(B, C, L)

        # 3. 计算 MSE 差异 (B, C, L-1)
        return F.mse_loss(pred_slope, target_slope, reduction='none')

class PhaseLoss(BaseSingleScaleLoss):
    """
    [相位损失]
    核心逻辑：使用 STFT 将信号变换到频域，计算复数谱图的欧氏距离，强制相位和频率一致。
    """
    def __init__(self, n_fft=64, hop_length=16, win_length=64, **kwargs):
        super().__init__(**kwargs)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer('window', torch.hann_window(win_length))

    def forward_step(self, pred, target):
        # STFT 需要输入为 (Batch, Time)，因此合并 B, C 维度
        B, C, L = pred.shape
        pred_flat = pred.view(B*C, L)
        target_flat = target.view(B*C, L)

        # 计算 STFT: return (B*C, Freq, Frames) Complex Tensor
        pred_stft = torch.stft(pred_flat, self.n_fft, self.hop_length, self.win_length, 
                               window=self.window, return_complex=True, center=True)
        target_stft = torch.stft(target_flat, self.n_fft, self.hop_length, self.win_length, 
                               window=self.window, return_complex=True, center=True)

        # 计算复数距离的平方: |z1 - z2|^2
        diff_sq = (pred_stft - target_stft).abs().pow(2) # (B*C, Freq, Frames)
        
        # 在频域和时间帧上求均值 -> 得到每个通道的 Loss (B*C,)
        loss_flat = diff_sq.mean(dim=(1, 2))
        
        # 还原形状 -> (B, C) 以便基类进行通道加权
        return loss_flat.view(B, C)

class VelocityLoss(BaseSingleScaleLoss):
    """
    [速度损失]
    核心逻辑：对加速度积分得到速度曲线，计算速度曲线的 MSE
    """
    def __init__(self, dt=0.001, loss_type='MSELoss', **kwargs):
        super().__init__(**kwargs)
        self.dt = dt
        self.criterion = getattr(nn, loss_type)(reduction='none')

    def forward_step(self, pred, target):
        # 积分计算速度: (B, C, L)
        pred_vel = torch.cumsum(pred, dim=-1) * self.dt
        target_vel = torch.cumsum(target, dim=-1) * self.dt
        
        return self.criterion(pred_vel, target_vel)

class DeltaVLoss(BaseSingleScaleLoss):
    """
    [速度变化量损失]
    核心逻辑：计算完整碰撞响应后的速度变化量，惩罚预测与真实的速度变化量差异。
    """
    def __init__(self, dt=0.001, loss_type='MSELoss', **kwargs):
        super().__init__(**kwargs)
        self.dt = dt
        self.criterion = getattr(nn, loss_type)(reduction='none')

    def forward_step(self, pred, target):
        # 计算速度变化量: delta_v = ∫ a dt = sum(a) * dt
        pred_delta_v = torch.sum(pred, dim=-1) * self.dt  # (B, C)
        target_delta_v = torch.sum(target, dim=-1) * self.dt  # (B, C)
        
        return self.criterion(pred_delta_v, target_delta_v)  # (B, C)

class InitialLoss(BaseSingleScaleLoss):
    """
    [初始段约束损失]
    核心逻辑：约束波形前 5% 的数据点，使其趋向于 0 或目标值，保证波形平稳启动。
    """
    def __init__(self, percentage=0.05, weight_target=0.0, loss_type='mae', **kwargs):
        super().__init__(**kwargs)

        if not 0 < percentage <= 1:
            raise ValueError("`percentage` 必须在 (0, 1] 范围内。")
        if loss_type not in ['mae', 'mse']:
            raise ValueError("`loss_type` 必须是 'mae' 或 'mse'。")
            
        self.percentage = percentage
        self.weight_target = weight_target
        self.loss_type = loss_type
        self.criterion = nn.L1Loss(reduction='none') if loss_type == 'mae' else nn.MSELoss(reduction='none')

    def forward_step(self, pred, target):
        seq_len = pred.shape[-1]
        n_points = int(seq_len * self.percentage) # 计算初始段点数
        if n_points == 0:
            return torch.zeros_like(pred) # 避免切片为空

        # 提取初始段
        seg_pred = pred[..., :n_points]
        seg_target = target[..., :n_points]

        # 惩罚初始段与0的差异 (即希望接近0) + 与真值的差异
        loss = self.criterion(seg_pred, torch.zeros_like(seg_pred))
        if self.weight_target > 0:
            loss += self.weight_target * self.criterion(seg_pred, seg_target)
            
        return loss # (B, C, n_points)

class TerminalLoss(BaseSingleScaleLoss):
    """
    [终端段约束损失]
    核心逻辑：约束波形最后 5% 的数据点，抑制末端飞逸现象。
    """
    def __init__(self, percentage=0.05, weight_target=1.0, loss_type='mae', **kwargs):
        super().__init__(**kwargs)
        self.percentage = percentage
        self.weight_target = weight_target
        if not 0 < percentage <= 1:
            raise ValueError("`percentage` 必须在 (0, 1] 范围内。")
        if loss_type not in ['mae', 'mse']:
            raise ValueError("`loss_type` 必须是 'mae' 或 'mse'。")
        self.loss_type = loss_type
        self.criterion = nn.L1Loss(reduction='none') if loss_type == 'mae' else nn.MSELoss(reduction='none')

    def forward_step(self, pred, target):
        seq_len = pred.shape[-1]
        n_points = int(seq_len * self.percentage)
        if n_points == 0:
            return torch.zeros_like(pred)

        # 提取末尾段
        seg_pred = pred[..., -n_points:]
        seg_target = target[..., -n_points:]

        # 惩罚末尾段与0的差异 + 与真值的差异
        loss = self.criterion(seg_pred, torch.zeros_like(seg_pred))
        if self.weight_target > 0:
            loss += self.weight_target * self.criterion(seg_pred, seg_target)
            
        return loss # (B, C, n_points)

