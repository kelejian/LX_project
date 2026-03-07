import torch
import torch.nn as nn
import logging
from typing import Dict, Tuple

# 引入全局损伤风险计算模块与特征顺序常量
from common.metrics import injury_risk
from common.settings import FEATURE_ORDER, CONTINUOUS_INDICES, DISCRETE_INDICES
from ARS_optim.src.core.param_manager import ParamManager

# 引入物理约束管理器
from ARS_optim.src.core.constraints import PhysicalConstraintManager
from ARS_optim.src.core.distribution_penalty import DistributionPenalty

class SurrogateAdapter(nn.Module):
    """
    代理模型级联适配器。

    作用：
    - 管理“波形预测 + 损伤预测 + 总损失计算”的统一接口；
    - 训练/评估阶段都可复用同一套输入拼接与归一化规则；
    - 支持先生成并缓存 pulse，再多次评估不同动作（避免重复推理波形模型）。
    """
    def __init__(
        self,
        pulse_model: nn.Module,
        injury_model: nn.Module,
        param_manager: ParamManager,
        constraint_manager: PhysicalConstraintManager,
        config: dict,
        data_processor,
    ):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.pulse_model = pulse_model
        self.injury_model = injury_model
        self.param_manager = param_manager

        # 约束管理器由调用方统一创建并注入，避免系统内出现重复实例。
        if constraint_manager is None:
            raise ValueError("[致命错误] SurrogateAdapter 必须传入有效的 constraint_manager 实例！")
        self.constraint_manager = constraint_manager
        
        if data_processor is None:
            raise ValueError("[致命错误] SurrogateAdapter 必须传入有效的 data_processor 实例！")
        self.data_processor = data_processor
        if not self.data_processor.load_config():
            raise RuntimeError("无法加载统一归一化配置 normalization_config.json")
        self.data_processor.validate_config(raise_on_error=True)
        
        # 代理模型在 ARS_optim 中视为冻结环境：不参与权重更新
        self.pulse_model.eval()
        self.injury_model.eval()
        for param in self.pulse_model.parameters():
            param.requires_grad = False
        for param in self.injury_model.parameters():
            param.requires_grad = False
            
        # 损失权重来自配置：
        # loss = risk + weight_penalty * constraint
        obj_cfg = config.get('optimization', {}).get('objectives', {})
        self.w_head = float(obj_cfg.get('weight_hic', 1.0))
        self.w_chest = float(obj_cfg.get('weight_dmax', 1.0))
        self.w_neck = float(obj_cfg.get('weight_nij', 1.0))
        
        if self.w_head <= 0 or self.w_chest <= 0 or self.w_neck <= 0:
            raise ValueError("[致命配置错误] weight_hic, weight_dmax, weight_nij 必须全部为大于 0 的正值！")
            
        # 物理边界惩罚系数
        self.weight_penalty = float(obj_cfg.get('weight_penalty', 10.0))

        # 训练分布偏离惩罚项：用于降低策略落入经验池稀疏区域的概率。
        dist_cfg = config.get('optimization', {}).get('distribution_penalty', {})
        self.weight_distribution = float(dist_cfg.get('weight', 0.0))
        self.distribution_penalty = DistributionPenalty(config)

    def _ensure_finite(self, name: str, tensor: torch.Tensor) -> None:
        if tensor is not None and (torch.isnan(tensor).any() or torch.isinf(tensor).any()):
            raise ValueError(f"{name} 包含 NaN 或 Inf")

    def fit_distribution_reference(self, reference_features: torch.Tensor) -> None:
        """
        拟合训练分布参考统计，用于后续计算分布偏离惩罚。
        """
        if self.distribution_penalty.enabled:
            self.distribution_penalty.fit(reference_features)

    def _prepare_normalized_inputs(self, context_params: torch.Tensor, control_trainable: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        组装完整特征并做归一化。

        输入：
        - context_params: [Batch, D_context]
        - control_trainable: [Batch, D_trainable]（可选）

        输出：
        - combined_phys: [Batch, D_total]（物理尺度）
        - model_input_norm: [Batch, D_total]（归一化尺度）
        """
        batch_size = context_params.shape[0]
        device = context_params.device
        total_dim = self.param_manager.get_total_feature_dim()
        
        # 1. 组装输入张量 (物理尺度)
        combined_phys = self.param_manager.get_default_feature_matrix(batch_size=batch_size, device=device)
        combined_phys[:, self.param_manager.get_context_indices()] = context_params
        
        if control_trainable is not None:
            # 只覆盖可调控制参数列；其余列仍保留 context/default 语义
            combined_phys[:, self.param_manager.get_control_trainable_indices()] = control_trainable
            
        # 2. 归一化特征（保持梯度链）
        # [Batch, Total_Dim] -> [Batch, Total_Dim]
        model_input_norm = self.data_processor.process_by_name(
            values=combined_phys, 
            feature_names=FEATURE_ORDER, 
            inverse=False
        )
        return combined_phys, model_input_norm

    def predict_injury_and_loss(self, context_params: torch.Tensor, control_trainable: torch.Tensor, pulse_norm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        损伤预测与总损失计算。
        该接口是策略训练和局部精调的核心计算入口。
        
        参数:
            context_params: [Batch, D_context] 物理尺度上下文参数（state + fixed-control）
            control_trainable: [Batch, D_Trainable] 当前优化的物理尺度约束参数
            pulse_norm: [Batch, 2, Seq_Len] 从 generate_pulse 获取并缓存的归一化波形
        """
        device = context_params.device
        self._ensure_finite("context_params", context_params)
        self._ensure_finite("control_trainable", control_trainable)
        self._ensure_finite("pulse_norm", pulse_norm)

        if control_trainable is not None:
            exp = self.param_manager.get_trainable_dim()
            if control_trainable.size(1) != exp:
                raise ValueError(f"control_trainable has {control_trainable.size(1)} cols, expected {exp}")
        combined_phys, model_input_norm = self._prepare_normalized_inputs(context_params, control_trainable)
        
        x_att_continuous = model_input_norm[:, CONTINUOUS_INDICES]
        x_att_discrete = model_input_norm[:, DISCRETE_INDICES].to(torch.long)
        
        predictions_phys, _, _ = self.injury_model(pulse_norm, x_att_continuous, x_att_discrete)

        hic15 = predictions_phys[:, 0]
        dmax = predictions_phys[:, 1]
        nij = predictions_phys[:, 2]
        
        ot_tensor = combined_phys[:, 12]
        
        p_head = injury_risk.Injury_prob_cal_head(hic15)
        p_chest = injury_risk.Injury_prob_cal_chest(dmax, OT=ot_tensor)
        p_neck = injury_risk.Injury_prob_cal_neck(nij)
        
        p_head = torch.clamp(p_head, 1e-6, 1.0 - 1e-6)
        p_chest = torch.clamp(p_chest, 1e-6, 1.0 - 1e-6)
        p_neck = torch.clamp(p_neck, 1e-6, 1.0 - 1e-6)
        
        term_head = torch.pow(1.0 - p_head, self.w_head)
        term_chest = torch.pow(1.0 - p_chest, self.w_chest)
        term_neck = torch.pow(1.0 - p_neck, self.w_neck)
        
        loss_risk = 1.0 - (term_head * term_chest * term_neck) 

        joint_risk = 1.0 - ((1.0 - p_head) * (1.0 - p_chest) * (1.0 - p_neck))
        
        norm_actions = self._normalize_control(control_trainable, device=device)
        norm_min, norm_max = self._get_normalized_bounds(device=device)
        exceed_max = torch.relu(norm_actions - norm_max.unsqueeze(0))
        exceed_min = torch.relu(norm_min.unsqueeze(0) - norm_actions)
        abs_penalty = (exceed_max + exceed_min).sum(dim=1)
        
        rel_penalty = self.constraint_manager.compute_soft_penalty(control_trainable, context_params)
        
        loss_constraint = abs_penalty + rel_penalty # [Batch]

        loss_distribution = self.distribution_penalty.compute(
            context_params=context_params,
            control_trainable=control_trainable,
        )
        
        total_loss = (
            loss_risk
            + self.weight_penalty * loss_constraint
            + self.weight_distribution * loss_distribution
        ) # [Batch]
        
        info = {
            "loss_risk": loss_risk.detach(),
            "loss_constraint": loss_constraint.detach(),
            "loss_distribution": loss_distribution.detach(),
            "abs_penalty": abs_penalty.detach(),
            "rel_penalty": rel_penalty.detach(),
            "p_head": p_head.detach(),
            "p_chest": p_chest.detach(),
            "p_neck": p_neck.detach(),
            "joint_risk": joint_risk.detach(),
            "hic15": hic15.detach(),
            "dmax": dmax.detach(),
            "nij": nij.detach()
        }
        
        return total_loss, predictions_phys, info

    # ------------------------------------------------------------------
    def _normalize_control(self, control_trainable: torch.Tensor, device=None) -> torch.Tensor:
        """
        将物理尺度的可训练控制参数归一化到 [0,1]。
        采用 trainable 参数自身的 min/max 做解析归一化，避免构造伪造离散状态导致映射异常。
        用于在归一化空间中计算边界惩罚。
        """
        if device is None:
            device = control_trainable.device
        min_phys, max_phys = self.param_manager.get_trainable_bounds(device=device)
        span = torch.clamp(max_phys - min_phys, min=1e-12)
        return (control_trainable - min_phys.unsqueeze(0)) / span.unsqueeze(0)

    def _get_normalized_bounds(self, device=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算可训练参数在归一化空间的最小/最大值。
        对应 _normalize_control 的解析归一化定义，边界恒为 [0,1]。
        """
        if device is None:
            device = torch.device('cpu')
        d_train = self.param_manager.get_trainable_dim()
        norm_min = torch.zeros(d_train, device=device, dtype=torch.float32)
        norm_max = torch.ones(d_train, device=device, dtype=torch.float32)
        return norm_min, norm_max

    def generate_pulse(self, context_params: torch.Tensor) -> torch.Tensor:
        """
        利用波形代理模型生成归一化后的 XY 轴碰撞波形。
        输入: 物理尺度的上下文参数 tensor [B, D_context]
        返回: pulse_norm [B, 2, Seq_Len]
        """
        device = context_params.device
        batch = context_params.shape[0]

        total_dim = self.param_manager.get_total_feature_dim()
        combined = self.param_manager.get_default_feature_matrix(batch_size=batch, device=device)
        combined[:, self.param_manager.get_context_indices()] = context_params

        impact_names = ["impact_velocity", "impact_angle", "overlap"]
        impact_indices = [FEATURE_ORDER.index(name) for name in impact_names]
        impact_phys = combined[:, impact_indices]
        impact_norm = self.data_processor.process_by_name(
            values=impact_phys, feature_names=impact_names, inverse=False
        )
        pulse_output_raw = self.pulse_model(impact_norm)
        if hasattr(self.pulse_model, 'get_metrics_output'):
            waveform_norm = self.pulse_model.get_metrics_output(pulse_output_raw)
        else:
            waveform_norm = pulse_output_raw[-1][0] if isinstance(pulse_output_raw, (list, tuple)) else pulse_output_raw
        x_acc_xy = waveform_norm[:, 0:2, :]
        from common.settings import WAVEFORM_LENGTH
        assert x_acc_xy.dim() == 3 and x_acc_xy.size(1) == 2, \
            f"generate_pulse returned wrong channel count {x_acc_xy.size(1)}"
        assert x_acc_xy.size(2) == WAVEFORM_LENGTH, \
            f"expected waveform length {WAVEFORM_LENGTH}, got {x_acc_xy.size(2)}"
        return x_acc_xy

    def forward(self, context_params: torch.Tensor, control_trainable: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        兼容性前向端到端通道 (End-to-End Pipeline)
        用于简单的基线测试或不支持波形缓存的场景，内部隐式调用解耦后的管线。
        """
        # [Batch, 2, Seq_Len]
        pulse_norm = self.generate_pulse(context_params)
        return self.predict_injury_and_loss(context_params, control_trainable, pulse_norm)