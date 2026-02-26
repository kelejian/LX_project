import torch
import torch.nn as nn
import logging
from typing import Dict, Tuple

# 引入全局损伤风险计算模块与特征顺序常量
from common.metrics import injury_risk
from common.settings import FEATURE_ORDER
from ARS_optim.src.core.param_manager import ParamManager

# 引入物理约束管理器
from ARS_optim.src.core.constraints import PhysicalConstraintManager

class SurrogateAdapter(nn.Module):
    """
    代理模型级联适配器 (Cascaded Surrogate Adapter) - 物理因果解耦版
    
    架构设计:
    本模块将物理环境严格解耦为两阶马尔可夫链: State -> Pulse -> Actions -> Injury。
    通过拆分波形预测与损伤预测接口，允许优化器在局部寻优（Local Refinement）时
    只计算一次波形并驻留显存缓存，从而在梯度迭代中仅调用轻量的损伤预测模型
    """
    def __init__(self, pulse_model: nn.Module, injury_model: nn.Module, param_manager: ParamManager, config: dict, data_processor):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.pulse_model = pulse_model
        self.injury_model = injury_model
        self.param_manager = param_manager
        
        # 实例化物理约束管理器，用于后续计算耦合关系的软惩罚
        self.constraint_manager = PhysicalConstraintManager(param_manager)
        
        if data_processor is None:
            raise ValueError("[致命错误] SurrogateAdapter 必须传入有效的 data_processor 实例！")
        self.data_processor = data_processor
        
        # 严苛约束：代理模型仅作为提供物理梯度的环境模拟器，严禁在寻优中发生权重更新
        self.pulse_model.eval()
        self.injury_model.eval()
        for param in self.pulse_model.parameters():
            param.requires_grad = False
        for param in self.injury_model.parameters():
            param.requires_grad = False
            
        # 解析风险优化目标权重
        obj_cfg = config.get('optimization', {}).get('objectives', {})
        self.w_head = float(obj_cfg.get('weight_hic', 1.0))
        self.w_chest = float(obj_cfg.get('weight_dmax', 1.0))
        self.w_neck = float(obj_cfg.get('weight_nij', 1.0))
        
        if self.w_head <= 0 or self.w_chest <= 0 or self.w_neck <= 0:
            raise ValueError("[致命配置错误] weight_hic, weight_dmax, weight_nij 必须全部为大于 0 的正值！")
            
        # 物理边界惩罚系数
        self.weight_penalty = float(obj_cfg.get('weight_penalty', 10.0))

    def _prepare_normalized_inputs(self, state_params: torch.Tensor, control_trainable: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        内部特征组装与归一化对齐器。
        严格确保按照 FEATURE_ORDER 顺序拼装物理尺度的特征，并调用统一处理器进行归一化。
        """
        batch_size = state_params.shape[0]
        device = state_params.device
        total_dim = self.param_manager.get_total_feature_dim()
        
        # 1. 组装输入张量 (物理尺度)
        combined_phys = torch.zeros((batch_size, total_dim), device=device, dtype=torch.float32)
        combined_phys[:, self.param_manager.get_state_indices()] = state_params
        
        if control_trainable is not None:
            combined_phys[:, self.param_manager.get_control_trainable_indices()] = control_trainable
            
        fixed_indices, fixed_defaults = self.param_manager.get_control_fixed_defaults(device=device)
        if len(fixed_indices) > 0:
            combined_phys[:, fixed_indices] = fixed_defaults.unsqueeze(0).expand(batch_size, -1)
            
        # 2. 归一化特征 (保持梯度链)
        # [Batch, Total_Dim] -> [Batch, Total_Dim]
        model_input_norm = self.data_processor.process_by_name(
            values=combined_phys, 
            feature_names=FEATURE_ORDER, 
            inverse=False
        )
        # 检查归一化输出是否存在异常
        if torch.isnan(model_input_norm).any() or torch.isinf(model_input_norm).any():
            raise ValueError("归一化后输入包含 NaN/Inf——请检查 normalization_config.json 是否完整且 param_space 与其一致。")

        # 输出: 物理尺度合并张量及其归一化版本
        return combined_phys, model_input_norm

    def predict_injury_and_loss(self, state_params: torch.Tensor, control_trainable: torch.Tensor, pulse_norm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        独立物理接口 2: 损伤预测与风险计算器 (Injury Predictor & Loss Evaluator)
        融合缓存的波形特征与当前的决策参数，评估整体物理风险。这是梯度下降的核心运算域。
        
        参数:
            state_params: [Batch, D_State] 物理尺度的状态参数
            control_trainable: [Batch, D_Trainable] 当前优化的物理尺度约束参数
            pulse_norm: [Batch, 2, Seq_Len] 从 generate_pulse 获取并缓存的归一化波形
        """
        device = state_params.device
        # 检查输入是否包含 NaN/Inf，尽早排除异常
        if torch.isnan(state_params).any() or torch.isinf(state_params).any():
            raise ValueError("state_params 包含 NaN 或 Inf")
        if control_trainable is not None and (torch.isnan(control_trainable).any() or torch.isinf(control_trainable).any()):
            raise ValueError("control_trainable 包含 NaN 或 Inf")

        # 检查 control_trainable 维度与 param_manager 一致
        if control_trainable is not None:
            exp = self.param_manager.get_trainable_dim()
            assert control_trainable.size(1) == exp, \
                f"control_trainable has {control_trainable.size(1)} cols, expected {exp}"
        combined_phys, model_input_norm = self._prepare_normalized_inputs(state_params, control_trainable)
        
        # 1. 切片提取 InjuryPredict 需要的特征结构
        # [Batch, 11] (连续特征) & [Batch, 2] (离散类别)
        x_att_continuous = model_input_norm[:, 0:11]
        x_att_discrete = model_input_norm[:, 11:13].to(torch.long)
        
        # 2. 预测损伤物理值 (梯度将从这里回流至 control_trainable)
        predictions_phys, _, _ = self.injury_model(pulse_norm, x_att_continuous, x_att_discrete)

        # 3. 摊销寻优计算：非线性联合风险项 L_risk
        hic15 = predictions_phys[:, 0]
        dmax = predictions_phys[:, 1]
        nij = predictions_phys[:, 2]
        
        # 提取物理尺度的乘员体型，用于胸压风险评定
        ot_tensor = combined_phys[:, 12]
        
        p_head = injury_risk.Injury_prob_cal_head(hic15)
        p_chest = injury_risk.Injury_prob_cal_chest(dmax, OT=ot_tensor)
        p_neck = injury_risk.Injury_prob_cal_neck(nij)
        
        # 极值保护，防止导数在 0 或 1 处崩溃
        p_head = torch.clamp(p_head, 1e-6, 1.0 - 1e-6)
        p_chest = torch.clamp(p_chest, 1e-6, 1.0 - 1e-6)
        p_neck = torch.clamp(p_neck, 1e-6, 1.0 - 1e-6)
        
        term_head = torch.pow(1.0 - p_head, self.w_head)
        term_chest = torch.pow(1.0 - p_chest, self.w_chest)
        term_neck = torch.pow(1.0 - p_neck, self.w_neck)
        
        # 联合存活概率补集作为风险值 [Batch]
        loss_risk = 1.0 - (term_head * term_chest * term_neck) 
        
        # 4. 摊销寻优计算：物理边界双重惩罚项 L_constraint
        # 4.1 绝对极值违规惩罚 (Min-Max Out-of-Bound) 在归一化空间中计算
        norm_actions = self._normalize_control(control_trainable, device=device)
        norm_min, norm_max = self._get_normalized_bounds(device=device)
        exceed_max = torch.relu(norm_actions - norm_max.unsqueeze(0))
        exceed_min = torch.relu(norm_min.unsqueeze(0) - norm_actions)
        abs_penalty = (exceed_max + exceed_min).sum(dim=1)
        
        # 4.2 相对耦合违规惩罚 (依赖于第二步升级的混合状态约束上下文)
        rel_penalty = self.constraint_manager.compute_soft_penalty(control_trainable, state_params)
        
        loss_constraint = abs_penalty + rel_penalty # [Batch]
        
        # 5. 总损失融合
        total_loss = loss_risk + self.weight_penalty * loss_constraint # [Batch]
        
        info = {
            "loss_risk": loss_risk.detach(),
            "loss_constraint": loss_constraint.detach(),
            "abs_penalty": abs_penalty.detach(),
            "rel_penalty": rel_penalty.detach(),
            "p_head": p_head.detach(),
            "p_chest": p_chest.detach(),
            "p_neck": p_neck.detach(),
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

    def generate_pulse(self, state_params: torch.Tensor) -> torch.Tensor:
        """
        利用波形代理模型生成归一化后的 XY 轴碰撞波形。
        输入: 物理尺度的状态参数 tensor [B, D_state]
        返回: pulse_norm [B, 2, Seq_Len]
        """
        device = state_params.device
        batch = state_params.shape[0]

        # 组装完整物理张量并填充默认控制参数
        total_dim = self.param_manager.get_total_feature_dim()
        combined = torch.zeros((batch, total_dim), device=device, dtype=torch.float32)
        combined[:, self.param_manager.get_state_indices()] = state_params
        fixed_idxs, fixed_defs = self.param_manager.get_control_fixed_defaults(device=device)
        if fixed_defs.numel() > 0:
            combined[:, fixed_idxs] = fixed_defs.unsqueeze(0).expand(batch, -1)

        # 仅取关键工况、并归一化
        impact_names = ["impact_velocity", "impact_angle", "overlap"]
        impact_indices = [FEATURE_ORDER.index(name) for name in impact_names]
        impact_phys = combined[:, impact_indices]
        impact_norm = self.data_processor.process_by_name(
            values=impact_phys, feature_names=impact_names, inverse=False
        )
        # 推理波形
        pulse_output_raw = self.pulse_model(impact_norm)
        if hasattr(self.pulse_model, 'get_metrics_output'):
            waveform_norm = self.pulse_model.get_metrics_output(pulse_output_raw)
        else:
            waveform_norm = pulse_output_raw[-1][0] if isinstance(pulse_output_raw, (list, tuple)) else pulse_output_raw
        # 提取XY轴
        x_acc_xy = waveform_norm[:, 0:2, :]
        from common.settings import WAVEFORM_LENGTH
        assert x_acc_xy.dim() == 3 and x_acc_xy.size(1) == 2, \
            f"generate_pulse returned wrong channel count {x_acc_xy.size(1)}"
        assert x_acc_xy.size(2) == WAVEFORM_LENGTH, \
            f"expected waveform length {WAVEFORM_LENGTH}, got {x_acc_xy.size(2)}"
        return x_acc_xy

    def forward(self, state_params: torch.Tensor, control_trainable: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        兼容性前向端到端通道 (End-to-End Pipeline)
        用于简单的基线测试或不支持波形缓存的场景，内部隐式调用解耦后的管线。
        """
        # [Batch, 2, Seq_Len]
        pulse_norm = self.generate_pulse(state_params)
        return self.predict_injury_and_loss(state_params, control_trainable, pulse_norm)