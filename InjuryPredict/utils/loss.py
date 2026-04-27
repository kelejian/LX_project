"""InjuryPredict 训练损失项。

本模块统一定义 HIC、Dmax、Nij 三个主任务监督损失，以及课程学习中使用的输出一致性、
特征一致性损失。Kendall 同质不确定性加权只作用于三个主任务；一致性损失仅作为外层
课程正则项由训练流程按阶段系数相加。
"""
import numpy as np
import torch
import torch.nn as nn

from common.metrics.injury_risk import AIS_cal_head, AIS_cal_chest, AIS_cal_neck


TASK_NAMES = ("HIC", "Dmax", "Nij")


def Piecewise_linear(y_true, y_pred, params, weight_add_mid=1.0):
    """
    通用分段线性权重增加量计算函数。

    参数:
        y_true (torch.Tensor): 真实标签，形状为 (B,)。
        y_pred (torch.Tensor): 预测值，形状为 (B,)。
        params (dict): 包含 'a', 'b', 'c', 'd', 't' 的超参数字典。
        weight_add_mid (float): 中间区间的权重增加量。

    返回:
        torch.Tensor: 权重增加量，形状为 (B,)。
    """
    a, b, c, d, t = params['a'], params['b'], params['c'], params['d'], params['t']

    weight_adds = torch.zeros_like(y_true)

    # 区间 1: 0 <= y < a，线性递增至 weight_add_mid。
    mask = (y_true >= 0) & (y_true < a)
    if a > 0:
        weight_adds[mask] = (weight_add_mid / a) * y_true[mask]

    # 区间 2: a <= y <= b，权重增加量为 weight_add_mid。
    mask = (y_true >= a) & (y_true <= b)
    weight_adds[mask] = weight_add_mid

    # 区间 3: b < y < c，线性递减至 0。
    mask = (y_true > b) & (y_true < c)
    if c > b:
        weight_adds[mask] = weight_add_mid - (weight_add_mid / (c - b)) * (y_true[mask] - b)

    # 区间 4: c <= y <= d，不额外增加样本权重。
    mask = (y_true >= c) & (y_true <= d)
    weight_adds[mask] = 0

    # 区间 5: y > d，指数衰减至负权重增量下界，用于避免极端大值样本长期主导训练。
    mask = y_true > d
    if d > 0:
        if t <= -1:
            raise ValueError("Piecewise_linear 中的 t 必须大于 -1，以保证指数衰减系数可定义。")
        k = -np.log(t + 1) / d
        weight_adds[mask] = -1 + torch.exp(-k * (y_true[mask] - d))

    # 对负预测值施加额外惩罚，因为 HIC、Dmax、Nij 在物理语义上均不应为负。
    mask_pred_neg = y_pred < 0
    weight_adds[mask_pred_neg] += weight_add_mid

    return weight_adds


class InjuryKendallMultiTaskLoss(nn.Module):
    """
    HIC、Dmax、Nij 三个主任务的 Kendall 同质不确定性加权损失。

    该损失分两层计算：

    1. 每个任务先计算带样本权重的原始回归损失，样本权重由 AIS 分类误差和损伤值区间共同决定：
    1a.对每个任务分别计算 HIC / Dmax / Nij 的逐样本 base_loss；该逐样本项不单独写入 TensorBoard。
    1b.对每个任务分别计算各自的 weighted_function 样本权重；该样本权重不单独写入 TensorBoard。
    1c.对每个任务分别做 mean(base_loss * sample_weight)，得到三个任务的原始加权子损失 L_i；return_components=True 时返回为 hic_loss、dmax_loss、nij_loss，训练日志再按波形源前缀记录为 gt_*_loss 或 pred_*_loss。

    2. 三个任务再按 `prior_i * (0.5 * exp(-s_i) * L_i + 0.5 * s_i)` 合成为主损失:
    2a.对每个任务分别乘 Kendall 的自适应系数 0.5 * exp(-s_i), 再分别加 Kendall 的正则项 0.5 * s_i；
    2b.对每个任务分别乘人工先验任务权重 task_prior_weights[i]；
    2c.最终三个任务加和，得到完整的 Kendall 主任务损失 total_loss；return_components=True 时返回为 main_loss，训练流程再按波形源记录为 main_gt_loss 或 main_pred_loss。

    参数:
        base_loss: 基础回归损失类型，只支持 'mse'、'mae' 或 'huber'。
        weight_factor_classify: AIS 分类等级误差对应的指数权重系数。
        weight_factor_sample: 分段区间权重的中间增加量。
        task_prior_weights: 三个主任务的人工先验权重，顺序为 HIC、Dmax、Nij。
        huber_deltas: Huber 损失阈值，顺序为 HIC、Dmax、Nij，仅当 base_loss='huber' 时使用。
    """

    def __init__(
        self,
        base_loss="huber",
        weight_factor_classify=1.1,
        weight_factor_sample=1.0,
        task_prior_weights=(1.0, 1.0, 1.0),
        huber_deltas=(50.0, 2.0, 0.05),
    ):
        super().__init__()
        if len(task_prior_weights) != 3:
            raise ValueError("task_prior_weights 必须包含 HIC、Dmax、Nij 三个权重。")

        priors = torch.as_tensor(task_prior_weights, dtype=torch.float32)
        if torch.any(priors < 0):
            raise ValueError("task_prior_weights 不能包含负数。")
        if priors.sum().item() <= 0:
            raise ValueError("task_prior_weights 的总和必须大于 0。")

        if base_loss == "mse":
            self.loss_funcs = nn.ModuleList([nn.MSELoss(reduction='none') for _ in range(3)])
        elif base_loss == "mae":
            self.loss_funcs = nn.ModuleList([nn.L1Loss(reduction='none') for _ in range(3)])
        elif base_loss == "huber":
            self.loss_funcs = nn.ModuleList([
                nn.HuberLoss(reduction='none', delta=huber_deltas[0]),
                nn.HuberLoss(reduction='none', delta=huber_deltas[1]),
                nn.HuberLoss(reduction='none', delta=huber_deltas[2]),
            ])
        else:
            raise ValueError("base_loss 只能为 'mse'、'mae' 或 'huber'。")

        self.base_loss_name = base_loss
        self.weight_factor_classify = float(weight_factor_classify)
        self.weight_factor_sample = float(weight_factor_sample)
        self.log_vars = nn.Parameter(torch.zeros(3, dtype=torch.float32))
        self.register_buffer("task_priors", priors)

        self.params_head = {'a': 80, 'b': 1500, 'c': 1750, 'd': 2000, 't': -0.5}
        self.params_chest = {'a': 10.0, 'b': 75, 'c': 85, 'd': 100, 't': -0.5}
        self.params_neck = {'a': 0.15, 'b': 1.5, 'c': 1.7, 'd': 1.9, 't': -0.5}

    def weighted_function(self, pred, true, injury_type, ot=None):
        """
        根据 AIS 分类误差和损伤值区间计算样本权重。

        分类等级由无梯度的工程规则计算，因此该权重仅作为样本重加权因子。
        """
        pred_np = pred.detach().cpu().numpy() # pred: (B,)
        true_np = true.detach().cpu().numpy() # true: (B,)

        if injury_type == 'head':
            pred_ais_np = AIS_cal_head(pred_np) # (B,)
            true_ais_np = AIS_cal_head(true_np) # (B,)
            weights_mid = 1.0 + Piecewise_linear(true, pred, self.params_head, self.weight_factor_sample) # (B,)
        elif injury_type == 'chest':
            if ot is None:
                raise ValueError("计算 Dmax 主任务损失时必须提供 OT。")
            ot_np = ot.detach().cpu().numpy() if hasattr(ot, 'detach') else ot
            pred_ais_np = AIS_cal_chest(pred_np, ot_np)
            true_ais_np = AIS_cal_chest(true_np, ot_np)
            weights_mid = 1.0 + Piecewise_linear(true, pred, self.params_chest, self.weight_factor_sample)
        elif injury_type == 'neck':
            pred_ais_np = AIS_cal_neck(pred_np)
            true_ais_np = AIS_cal_neck(true_np)
            weights_mid = 1.0 + Piecewise_linear(true, pred, self.params_neck, self.weight_factor_sample)
        else:
            raise ValueError(f"未知损伤类型: {injury_type}")

        pred_ais = torch.as_tensor(pred_ais_np, device=pred.device)
        true_ais = torch.as_tensor(true_ais_np, device=true.device)
        weights_classify = self.weight_factor_classify ** torch.abs(pred_ais.float() - true_ais.float()) # (B,)
        return weights_classify * weights_mid # (B,)

    def compute_task_losses(self, pred, true, ot):
        """
        计算三个主任务的原始加权子损失。

        参数:
            pred: 模型预测值，形状为 (B, 3)，列顺序为 HIC、Dmax、Nij。
            true: 真实标签，形状为 (B, 3)，列顺序为 HIC、Dmax、Nij。
            ot: 乘员体型类别，形状为 (B,)，用于 Dmax 对应 AIS 规则。
        """
        if pred.ndim != 2 or pred.shape[1] != 3:
            raise ValueError(f"pred 形状必须为 (B, 3)，实际为 {tuple(pred.shape)}")
        if true.ndim != 2 or true.shape[1] != 3:
            raise ValueError(f"true 形状必须为 (B, 3)，实际为 {tuple(true.shape)}")
        if ot is None:
            raise ValueError("ot 不能为空。")

        pred_hic, pred_dmax, pred_nij = pred[:, 0], pred[:, 1], pred[:, 2] # 均为 (B,)
        true_hic, true_dmax, true_nij = true[:, 0], true[:, 1], true[:, 2]

        weights_hic = self.weighted_function(pred_hic, true_hic, 'head') # (B,)
        hic_loss = (self.loss_funcs[0](pred_hic, true_hic) * weights_hic).mean() # 标量

        weights_dmax = self.weighted_function(pred_dmax, true_dmax, 'chest', ot) # (B,)
        dmax_loss = (self.loss_funcs[1](pred_dmax, true_dmax) * weights_dmax).mean() # 标量

        weights_nij = self.weighted_function(pred_nij, true_nij, 'neck') # (B,)
        nij_loss = (self.loss_funcs[2](pred_nij, true_nij) * weights_nij).mean() # 标量

        # 以下三个标量对应第 1c 层的 L_i，尚未经过 Kendall 自适应权重和人工先验权重。
        return {
            "hic_loss": hic_loss,
            "dmax_loss": dmax_loss,
            "nij_loss": nij_loss,
        }

    def _apply_kendall(self, task_losses):
        raw_losses = torch.stack([
            task_losses["hic_loss"], # 标量
            task_losses["dmax_loss"],
            task_losses["nij_loss"],
        ]) # [标量, 标量, 标量] -> [3]
        precision = torch.exp(-self.log_vars) # (3,)，Kendall 的自适应系数: 方差越小，精度越高/该任务损失权重越大，即 precision 越大
        return torch.sum(self.task_priors * (0.5 * precision * raw_losses + 0.5 * self.log_vars)) # 标量

    def forward(self, pred, true, ot=None, return_components=False):
        """
        计算 Kendall 加权后的三任务主损失。
        pred 和 true 的形状必须为 (B, 3)，列顺序为 HIC、Dmax、Nij。ot 的形状必须为 (B,)，且不能为 None。

        返回:
            当 return_components=False 时返回标量主损失；
            当 return_components=True 时返回 `(main_loss, components)`，其中 components 为脱离计算图的监控值。
        """
        task_losses = self.compute_task_losses(pred, true, ot)
        # total_loss 对应第 2c 层的 Kendall 主任务损失，训练流程会按波形源记录为 main_gt_loss 或 main_pred_loss。
        total_loss = self._apply_kendall(task_losses)

        if not return_components:
            return total_loss

        components = {
            # main_loss 是对应第 2c 层的已完成 Kendall 聚合的主任务标量；*_loss 是第 1c 层的原始加权子损失。
            "main_loss": float(total_loss.detach().item()),
            "hic_loss": float(task_losses["hic_loss"].detach().item()),
            "dmax_loss": float(task_losses["dmax_loss"].detach().item()),
            "nij_loss": float(task_losses["nij_loss"].detach().item()),
        }
        return total_loss, components

    def get_weight_state(self):
        """
        返回 Kendall 任务权重状态，用于训练记录与 TensorBoard 监控。
        """
        state = {}
        with torch.no_grad():
            for idx, name in enumerate(TASK_NAMES):
                log_var = self.log_vars[idx]
                precision = torch.exp(-log_var)
                sigma = torch.exp(0.5 * log_var)
                prior = self.task_priors[idx]
                state[name] = {
                    "prior_weight": float(prior.item()),
                    "log_var": float(log_var.item()),
                    "sigma": float(sigma.item()),
                    "precision": float(precision.item()),
                    "effective_loss_weight": float((0.5 * prior * precision).item()),
                }
        return state


class OutputConsistencyLoss(nn.Module):
    """
    预测波形源与真值波形源之间的输出一致性损失。

    输入应由训练流程显式提供：
    - `reference_pred`: 参考输出，通常为真值波形分支预测且已在调用侧执行 stop-gradient。
    - `target_pred`: 需要被约束的目标输出，通常为预测波形分支预测。
    - `output_weights`: 三个损伤输出的全局尺度归一化权重，通常由训练集标签标准差的倒数给出，形状可广播到 `(B, 3)`。

    `output_weights` 用于消除 HIC、Dmax、Nij 物理量纲和数值尺度差异，避免 HIC 等大数值任务在输出一致性正则中天然占优。
    该类不直接读取模型结构或切片特征，避免损失定义与 InjuryPredict 的内部表示耦合。
    """

    def forward(self, reference_pred, target_pred, output_weights):
        if reference_pred.shape != target_pred.shape:
            raise ValueError(
                f"输出一致性损失要求两侧预测形状一致，实际为 {tuple(reference_pred.shape)} 与 {tuple(target_pred.shape)}。"
            )
        if reference_pred.ndim != 2 or reference_pred.shape[1] != 3:
            raise ValueError(f"输出一致性损失期望输入形状为 (B, 3)，实际为 {tuple(reference_pred.shape)}。")
        if output_weights is None:
            raise ValueError("output_weights 不能为空，需由训练集标签尺度统计得到。")
        return torch.mean(torch.abs((reference_pred - target_pred) * output_weights))


class FeatureConsistencyLoss(nn.Module):
    """
    两个波形源分支的波形编码器特征一致性损失。

    调用侧负责传入已经选定的特征张量，例如只传入 TCN 波形编码器输出，而不是融合后特征或解码器特征。
    这样可以把“对齐哪一层”的模型结构决策保留在训练流程中，损失项本身只表达受控的 L1 一致性度量。

    normalize='sample_layernorm' 表示对每个样本的特征向量在特征维上做无可学习参数的标准化，再计算两路特征的 L1 距离。
    该归一化只作用于特征一致性正则，不改变模型前向输出；其目的在于约束两种波形源的相对激活模式，而不是强制隐藏特征的绝对幅值完全相同。
    这与 OutputConsistencyLoss 中按标签标准差构造的输出尺度归一化不同：前者是样本内隐藏特征归一化，后者是跨任务物理输出尺度归一化。
    """

    def __init__(self, normalize: str | None = None, eps: float = 1e-6):
        super().__init__()
        self.normalize = "none" if normalize is None else str(normalize).lower().strip()
        self.eps = float(eps)
        if self.normalize not in ("none", "sample_layernorm"):
            raise ValueError("FeatureConsistencyLoss.normalize 仅支持 None/'none' 或 'sample_layernorm'。")

    def _normalize_feature(self, feature):
        if self.normalize == "none":
            return feature
        # sample_layernorm 在每个样本内部沿特征维计算均值和标准差，因此不会引入跨样本统计量或可学习参数。
        mean = feature.mean(dim=1, keepdim=True) # [B, F] -> [B, 1]
        std = feature.std(dim=1, keepdim=True, unbiased=False).clamp_min(self.eps) # [B, F] -> [B, 1]
        return (feature - mean) / std # [B, F] -> [B, F]

    def forward(self, reference_feature, target_feature):
        if reference_feature.shape != target_feature.shape:
            raise ValueError(
                f"特征一致性损失要求两侧特征形状一致，实际为 {tuple(reference_feature.shape)} 与 {tuple(target_feature.shape)}。"
            )
        if reference_feature.ndim != 2:
            raise ValueError(f"特征一致性损失期望二维特征张量，实际为 {tuple(reference_feature.shape)}。")
        reference_feature = self._normalize_feature(reference_feature)
        target_feature = self._normalize_feature(target_feature)
        return torch.mean(torch.abs(reference_feature - target_feature))


if __name__ == '__main__':
    pred = torch.tensor([
        [100.0, 10.0, 0.8],
        [1800.0, 90.0, 3.0],
        [-50.0, -5.0, -0.2],
        [900.0, 120.0, 1.5],
    ], dtype=torch.float32)
    true = torch.tensor([
        [50.0, 5.0, 0.5],
        [1700.0, 110.0, 2.5],
        [10.0, 2.0, 0.1],
        [800.0, 100.0, 1.0],
    ], dtype=torch.float32)
    ot = torch.tensor([2, 3, 1, 1], dtype=torch.int32)

    criterion = InjuryKendallMultiTaskLoss(
        base_loss='mae',
        weight_factor_classify=1.5,
        weight_factor_sample=2.0,
        task_prior_weights=(1.0, 0.8, 1.2),
    )
    loss, components = criterion(pred, true, ot, return_components=True)
    print("\nTotal Kendall Weighted Loss:", loss.item())
    print("Components:", components)

    output_consistency = OutputConsistencyLoss()
    feature_consistency = FeatureConsistencyLoss()
    print("Output Consistency:", output_consistency(pred.detach(), pred + 1.0, torch.ones(1, 3)).item())
    print("Feature Consistency:", feature_consistency(torch.zeros(2, 4), torch.ones(2, 4)).item())
