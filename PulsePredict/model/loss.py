import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoWeightedLoss(nn.Module):
    """
    多任务损失调度器。

    该类负责两件事：
    1. 按配置实例化各个子损失。
    2. 在任务层面使用 Kendall 同方差不确定性加权，将人工先验权重与可学习任务权重结合起来。

    总损失形式为：
        L_total = sum_i lambda_i * (0.5 * exp(-s_i) * L_i + 0.5 * s_i)

    其中：
    - lambda_i 是配置文件中的人工先验权重，用于表达主任务/辅助任务的重要性偏好。
    - s_i = log(sigma_i^2) 是可学习参数，用于根据任务当前的损失水平自动调整权重。
    - L_i 是各子损失返回的标量原始损失。
    """

    def __init__(self, loss_configs):
        super().__init__()
        if not loss_configs:
            raise ValueError("`loss_configs` 不能为空。")

        self.losses = nn.ModuleList()
        self.loss_names = []

        prior_weights = []
        for config in loss_configs:
            loss_type = config["type"]
            prior_weight = config.get("prior_weight", 1.0)
            loss_args = config.get("args", {})

            if prior_weight < 0:
                raise ValueError(f"损失项 `{loss_type}` 的 `prior_weight` 不能为负数。")
            if loss_type not in globals():
                raise ValueError(f"未知的 Loss 类型：{loss_type}")

            loss_cls = globals()[loss_type]
            self.losses.append(loss_cls(**loss_args))
            self.loss_names.append(loss_type)
            prior_weights.append(prior_weight)
        
        # 初始化可学习参数 s_i = log(sigma^2)
        # 初始化为0，意味着初始 sigma=1，保证训练初期梯度平稳
        self.log_vars = nn.Parameter(torch.zeros(len(self.losses), dtype=torch.float32))
        # 注册先验权重为 buffer (不参与梯度更新，但随模型保存)
        self.register_buffer("priors", torch.tensor(prior_weights, dtype=torch.float32))

    def forward(self, model_output, target):
        total_loss = self.log_vars.new_tensor(0.0)
        loss_components = {}

        for idx, loss_fn in enumerate(self.losses):
            # `raw_loss` 是子损失本身返回的原始标量值，其数值仅由该任务的物理/形态定义决定，尚未乘入 Kendall 自适应缩放项 `exp(-log_var)`，也尚未加入 `0.5 * log_var` 不确定性正则项。
            raw_loss = loss_fn(model_output, target)
            log_var = self.log_vars[idx]
            precision = torch.exp(-log_var)
            weighted_loss = self.priors[idx] * (0.5 * precision * raw_loss + 0.5 * log_var)

            total_loss = total_loss + weighted_loss
            # `loss_components` 专门用于监控原始子损失曲线；因此 TensorBoard 中诸如 `MultiScaleLoss`的标量曲线表示未经自适应加权修正的原始任务损失，不带有 Kendall 中项的影响。
            loss_components[self.loss_names[idx]] = raw_loss.item()

        return total_loss, loss_components

    def get_weight_state(self):
        """
        返回当前任务权重状态。

        该接口只用于训练监控，不参与前向传播。
        `effective_loss_weight = 0.5 * lambda_i * exp(-s_i)` 反映了当前任务
        在原始损失前面的实际缩放系数。
        """
        state = {}
        with torch.no_grad():
            for idx, loss_name in enumerate(self.loss_names):
                log_var = self.log_vars[idx]
                precision = torch.exp(-log_var)
                sigma = torch.exp(0.5 * log_var)
                prior = self.priors[idx]

                state[loss_name] = {
                    # `prior_weight = lambda_i`：人工先验权重，用于表达任务重要性的外部偏好；
                    # 其作用是在自适应平衡之外保留人为约束，从而限制辅助项或正则项的相对影响上界。
                    "prior_weight": prior.item(),
                    # `log_var = s_i = log(sigma_i^2)`：Kendall 框架中的可学习对数方差参数；
                    # 训练过程中直接优化该量比直接优化 `sigma` 更稳定，也便于通过指数映射保证方差为正。
                    "log_var": log_var.item(),
                    # `sigma = sigma_i`：由 `log_var` 还原得到的任务级标准差估计；
                    # 数值越大，表示模型倾向于将该任务视为更难拟合或噪声更强，进而降低其相对权重。
                    "sigma": sigma.item(),
                    # `precision = 1 / sigma_i^2 = exp(-log_var)`：任务级精度项；
                    # 它直接控制原始子损失前的自适应缩放强度，数值越大，对应任务在当前阶段被赋予更高权重。
                    "precision": precision.item(),
                    # `effective_loss_weight = 0.5 * lambda_i * exp(-log_var)`：原始子损失前的实际系数；
                    # 这是同时综合人工先验与自适应不确定性之后，对梯度贡献最直接的监控量。
                    "effective_loss_weight": (0.5 * prior * precision).item(),
                }
        return state


class BaseSingleScaleLoss(nn.Module):
    """
    单尺度损失基类。

    该基类约定子类只负责实现 `forward_step(pred, target)`，
    即返回未做 batch/time 聚合的逐点或逐通道损失张量。
    通道加权平均、长度对齐等公共逻辑统一在父类中处理。
    """

    def __init__(self, channel_weights=[1.0, 1.0, 1.0]):
        super().__init__()
        weights = torch.as_tensor(channel_weights, dtype=torch.float32)
        if weights.ndim != 1 or weights.numel() == 0:
            raise ValueError("`channel_weights` 必须是一维非空序列。")
        if torch.any(weights < 0):
            raise ValueError("`channel_weights` 不能包含负数。")
        if weights.sum().item() <= 0:
            raise ValueError("`channel_weights` 的和必须大于 0。")

        self.register_buffer("channel_weights", weights)

    def _get_prediction(self, model_output):
        """
        解析模型输出，提取最终阶段的预测波形。
        """
        pred = model_output[-1] if isinstance(model_output, (list, tuple)) else model_output
        if not torch.is_tensor(pred):
            raise TypeError("模型输出必须是张量，或由张量组成的列表/元组。")
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
        # 对 Batch 和 Time 维度求均值，保留 Channel 维度 -> (C,)
        dims_to_reduce = [0] + list(range(2, raw_loss.ndim)) # e.g.(B, C, L) -> reduce over B and L, remain C
        loss_per_channel = raw_loss.mean(dim=dims_to_reduce) # (C,)

        return (loss_per_channel * self.channel_weights).sum() / self.channel_weights.sum()


class MultiScaleLoss(nn.Module):
    """
    多尺度主回归损失。

    这是波形重建的主损失项，只负责对不同尺度的预测波形做确定性回归监督。
    任务级的自动平衡由外层 `AutoWeightedLoss` 统一负责

    """

    def __init__(
        self,
        scale_weights=[0.1, 0.2, 1.0],
        channel_weights=[1.0, 1.0, 1.0],
        base_loss_type="MSELoss",
        base_loss_args=None,
    ):
        super().__init__()

        scale_weights_tensor = torch.as_tensor(scale_weights, dtype=torch.float32)
        channel_weights_tensor = torch.as_tensor(channel_weights, dtype=torch.float32)

        if scale_weights_tensor.ndim != 1 or scale_weights_tensor.numel() == 0:
            raise ValueError("`scale_weights` 必须是一维非空序列。")
        if torch.any(scale_weights_tensor < 0):
            raise ValueError("`scale_weights` 不能包含负数。")
        if scale_weights_tensor.sum().item() <= 0:
            raise ValueError("`scale_weights` 的和必须大于 0。")

        if channel_weights_tensor.ndim != 1 or channel_weights_tensor.numel() == 0:
            raise ValueError("`channel_weights` 必须是一维非空序列。")
        if torch.any(channel_weights_tensor < 0):
            raise ValueError("`channel_weights` 不能包含负数。")
        if channel_weights_tensor.sum().item() <= 0:
            raise ValueError("`channel_weights` 的和必须大于 0。")

        if base_loss_type == "GaussianNLLLoss":
            raise ValueError("当前版本已移除 GaussianNLLLoss，请改用确定性回归损失。")
        if not hasattr(nn, base_loss_type):
            raise ValueError(f"未知的基础回归损失类型：{base_loss_type}")

        base_loss_args = {} if base_loss_args is None else dict(base_loss_args)
        base_loss_args["reduction"] = "none"

        self.scale_weights = scale_weights_tensor.tolist()
        self.register_buffer("channel_weights", channel_weights_tensor)
        self.base_criterion = getattr(nn, base_loss_type)(**base_loss_args)

    def forward(self, model_output, target):
        preds_list = model_output if isinstance(model_output, (list, tuple)) else [model_output]

        total_loss_sum = target.new_tensor(0.0)
        total_scale_weight_sum = 0.0

        for idx, pred in enumerate(preds_list):
            if idx >= len(self.scale_weights) or self.scale_weights[idx] == 0:
                continue
            if not torch.is_tensor(pred):
                raise TypeError("MultiScaleLoss 期望模型输出为张量列表。")

            current_scale_weight = self.scale_weights[idx]
            curr_target = (
                F.interpolate(target, size=pred.shape[-1], mode="linear", align_corners=False)
                if pred.shape[-1] != target.shape[-1]
                else target
            )

            loss = self.base_criterion(pred, curr_target)
            dims_to_reduce = [0] + list(range(2, loss.ndim))
            loss_per_channel = loss.mean(dim=dims_to_reduce)
            channel_weighted_mean = (loss_per_channel * self.channel_weights).sum() / self.channel_weights.sum()

            total_loss_sum = total_loss_sum + current_scale_weight * channel_weighted_mean
            total_scale_weight_sum += current_scale_weight

        return total_loss_sum / total_scale_weight_sum


class RegressionLoss(BaseSingleScaleLoss):
    """最终阶段尺度的回归损失。"""

    def __init__(self, loss_type="L1Loss", **kwargs):
        super().__init__(**kwargs)
        if not hasattr(nn, loss_type):
            raise ValueError(f"未知的回归损失类型：{loss_type}")
        self.criterion = getattr(nn, loss_type)(reduction="none")

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
        # 计算每个样本的幅值基准 : (B, C, 1)
        target_peak = torch.max(torch.abs(target), dim=-1, keepdim=True)[0]
        corridor_width = self.inner_corridor_width * (target_peak + 1e-9)
        # ReLU 截断：只保留 超出廊道宽度的部分误差
        exceeded_error = F.relu(torch.abs(pred - target) - corridor_width)
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
            kernel = torch.ones(1, 1, smoothing_window_size, dtype=torch.float32) / smoothing_window_size
            self.register_buffer("kernel", kernel)

    def forward_step(self, pred, target):
        # 1. 计算一阶差分 (B, C, L-1)
        pred_slope = pred[..., 1:] - pred[..., :-1]
        target_slope = target[..., 1:] - target[..., :-1]

        # 2. (可选) 对斜率进行平滑处理 (模拟 ISO 标准流程)
        if self.apply_smoothing:
            # 融合 B, C 维度以便使用 conv1d: (B*C, 1, L-1)
            batch_size, channels, length = pred_slope.shape

            pred_slope = pred_slope.reshape(batch_size * channels, 1, length)
            target_slope = target_slope.reshape(batch_size * channels, 1, length)
            # Padding='same' 保持长度不变

            pred_slope = F.conv1d(pred_slope, self.kernel, padding="same")
            target_slope = F.conv1d(target_slope, self.kernel, padding="same")

            pred_slope = pred_slope.reshape(batch_size, channels, length)
            target_slope = target_slope.reshape(batch_size, channels, length)
            
        # 3. 计算 MSE 差异 (B, C, L-1)
        return F.mse_loss(pred_slope, target_slope, reduction="none")


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
        self.register_buffer("window", torch.hann_window(win_length))

    def forward_step(self, pred, target):
        # STFT 需要输入为 (Batch, Time)，因此合并 B, C 维度
        batch_size, channels, length = pred.shape
        pred_flat = pred.reshape(batch_size * channels, length)
        target_flat = target.reshape(batch_size * channels, length)
        # 计算 STFT: return (B*C, Freq, Frames) Complex Tensor
        pred_stft = torch.stft(
            pred_flat,
            self.n_fft,
            self.hop_length,
            self.win_length,
            window=self.window,
            return_complex=True,
            center=True,
        )
        target_stft = torch.stft(
            target_flat,
            self.n_fft,
            self.hop_length,
            self.win_length,
            window=self.window,
            return_complex=True,
            center=True,
        )
        # 计算复数距离的平方: |z1 - z2|^2
        diff_sq = (pred_stft - target_stft).abs().pow(2) # (B*C, Freq, Frames)
        # 在频域和时间帧上求均值 -> 得到每个通道的 Loss (B*C,)
        loss_flat = diff_sq.mean(dim=(1, 2))
        # 还原形状 -> (B, C) 以便基类进行通道加权
        return loss_flat.reshape(batch_size, channels)


class VelocityLoss(BaseSingleScaleLoss):
    """
    [速度损失]
    核心逻辑：对加速度积分得到速度曲线，计算速度曲线的 MSE
    """

    def __init__(self, dt=0.001, loss_type="MSELoss", **kwargs):
        super().__init__(**kwargs)
        if not hasattr(nn, loss_type):
            raise ValueError(f"未知的速度损失类型：{loss_type}")
        self.dt = dt
        self.criterion = getattr(nn, loss_type)(reduction="none")

    def forward_step(self, pred, target):
        pred_velocity = torch.cumsum(pred, dim=-1) * self.dt
        target_velocity = torch.cumsum(target, dim=-1) * self.dt
        return self.criterion(pred_velocity, target_velocity)


class DeltaVLoss(BaseSingleScaleLoss):
    """
    [速度变化量损失]
    核心逻辑：计算完整碰撞响应后的速度变化量，惩罚预测与真实的速度变化量差异。
    """

    def __init__(self, dt=0.001, loss_type="MSELoss", **kwargs):
        super().__init__(**kwargs)
        if not hasattr(nn, loss_type):
            raise ValueError(f"未知的 DeltaV 损失类型：{loss_type}")
        self.dt = dt
        self.criterion = getattr(nn, loss_type)(reduction="none")

    def forward_step(self, pred, target):
        pred_delta_v = torch.sum(pred, dim=-1) * self.dt # (B, C)
        target_delta_v = torch.sum(target, dim=-1) * self.dt # (B, C)
        return self.criterion(pred_delta_v, target_delta_v) # (B, C)


class InitialLoss(BaseSingleScaleLoss):
    """
    起始段约束损失。

    用于限制波形开头一小段的非物理振荡。主惩罚项是“接近 0”，
    可选再叠加一项“接近真实目标”，以避免把开头段简单压平到完全忽略目标结构。
    """

    def __init__(self, percentage=0.05, weight_target=0.0, loss_type="mae", **kwargs):
        super().__init__(**kwargs)
        if not 0 < percentage <= 1:
            raise ValueError("`percentage` 必须位于 (0, 1] 区间。")
        if loss_type not in ["mae", "mse"]:
            raise ValueError("`loss_type` 只能是 `mae` 或 `mse`。")

        self.percentage = percentage
        self.weight_target = weight_target
        self.criterion = nn.L1Loss(reduction="none") if loss_type == "mae" else nn.MSELoss(reduction="none")

    def forward_step(self, pred, target):
        n_points = int(pred.shape[-1] * self.percentage) # 计算初始段点数
        if n_points == 0:
            return torch.zeros_like(pred) # 避免切片为空

        # 提取初始段
        seg_pred = pred[..., :n_points]
        seg_target = target[..., :n_points]

        # 惩罚初始段与0的差异 (即希望接近0) + 与真值的差异
        loss = self.criterion(seg_pred, torch.zeros_like(seg_pred))
        if self.weight_target > 0:
            loss = loss + self.weight_target * self.criterion(seg_pred, seg_target)
        return loss # (B, C, n_points)


class TerminalLoss(BaseSingleScaleLoss):
    """
    [终端段约束损失]
    核心逻辑：约束波形最后 5% 的数据点，抑制末端飞逸现象。
    """

    def __init__(self, percentage=0.05, weight_target=1.0, loss_type="mae", **kwargs):
        super().__init__(**kwargs)
        if not 0 < percentage <= 1:
            raise ValueError("`percentage` 必须位于 (0, 1] 区间。")
        if loss_type not in ["mae", "mse"]:
            raise ValueError("`loss_type` 只能是 `mae` 或 `mse`。")

        self.percentage = percentage
        self.weight_target = weight_target
        self.criterion = nn.L1Loss(reduction="none") if loss_type == "mae" else nn.MSELoss(reduction="none")

    def forward_step(self, pred, target):
        n_points = int(pred.shape[-1] * self.percentage)
        if n_points == 0:
            return torch.zeros_like(pred)

        # 提取末尾段
        seg_pred = pred[..., -n_points:]
        seg_target = target[..., -n_points:]

        # 惩罚末尾段与0的差异 + 与真值的差异
        loss = self.criterion(seg_pred, torch.zeros_like(seg_pred))
        if self.weight_target > 0:
            loss = loss + self.weight_target * self.criterion(seg_pred, seg_target)
        return loss # (B, C, n_points)
