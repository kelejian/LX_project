import torch
import torch.nn as nn
import logging
from typing import Dict, Tuple

# 引入全局损伤风险计算模块与特征顺序常量
from common.metrics import injury_risk
from common.settings import FEATURE_ORDER
from src.core.param_manager import ParamManager

# [新增导入] 引入刚刚编写的物理约束管理器
from src.core.constraints import PhysicalConstraintManager

class SurrogateAdapter(nn.Module):
    """
    代理模型级联适配器 (Cascaded Surrogate Adapter)
    将预训练的 PulsePredict 与 InjuryPredict 严格级联，封装为全流程可微的环境模拟器。
    """
    def __init__(self, pulse_model: nn.Module, injury_model: nn.Module, param_manager: ParamManager, config: dict, data_processor):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.pulse_model = pulse_model
        self.injury_model = injury_model
        self.param_manager = param_manager
        
        # [新增] 实例化物理约束管理器，用于后续计算耦合关系的软惩罚
        self.constraint_manager = PhysicalConstraintManager(param_manager)
        
        if data_processor is None:
            raise ValueError("[致命错误] SurrogateAdapter 必须传入有效的 data_processor 实例！")
        self.data_processor = data_processor
        
        # 严苛约束：代理模型仅作为提供梯度的物理环境，严禁在寻优中发生权重更新
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

    def forward(self, state_params: torch.Tensor, control_trainable: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        前向级联传播与全导向计算图 (Fully Differentiable Cascaded Graph) 构建。
        """
        batch_size = state_params.shape[0]
        device = state_params.device
        total_dim = self.param_manager.get_total_feature_dim()
        
        # ==========================================
        # 1. 组装输入张量 (物理尺度)
        # ==========================================
        combined_phys = torch.zeros((batch_size, total_dim), device=device, dtype=torch.float32)
        
        combined_phys[:, self.param_manager.get_state_indices()] = state_params
        combined_phys[:, self.param_manager.get_control_trainable_indices()] = control_trainable
        
        fixed_indices, fixed_defaults = self.param_manager.get_control_fixed_defaults(device=device)
        if len(fixed_indices) > 0:
            combined_phys[:, fixed_indices] = fixed_defaults.unsqueeze(0).expand(batch_size, -1)
            
        # ==========================================
        # 2. 归一化特征 (保持梯度)
        # ==========================================
        model_input_norm = self.data_processor.process_by_name(
            values=combined_phys, 
            feature_names=FEATURE_ORDER, 
            inverse=False
        )
            
        # ==========================================
        # 3. 代理模型级联推理 (PulsePredict -> InjuryPredict)
        # ==========================================
        pulse_input = model_input_norm[:, 0:3] 
        pulse_output_raw = self.pulse_model(pulse_input)
        
        if hasattr(self.pulse_model, 'get_metrics_output'):
            waveform_norm = self.pulse_model.get_metrics_output(pulse_output_raw)
        else:
            waveform_norm = pulse_output_raw[-1][0] if isinstance(pulse_output_raw, (list, tuple)) else pulse_output_raw

        x_acc_xy = waveform_norm[:, 0:2, :] 
        x_att_continuous = model_input_norm[:, 0:11]
        x_att_discrete = model_input_norm[:, 11:13].to(torch.long)
        
        predictions_phys, _, _ = self.injury_model(x_acc_xy, x_att_continuous, x_att_discrete)

        # ==========================================
        # 4. 摊销寻优计算：非线性联合风险项 L_risk
        # ==========================================
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
        
        # ==========================================
        # 5. 摊销寻优计算：物理边界双重惩罚项 L_constraint
        # ==========================================
        # 5.1 绝对极值违规惩罚 (Min-Max)
        min_bounds, max_bounds = self.param_manager.get_trainable_bounds(device=device)
        exceed_max = torch.relu(control_trainable - max_bounds.unsqueeze(0))
        exceed_min = torch.relu(min_bounds.unsqueeze(0) - control_trainable)
        abs_penalty = (exceed_max + exceed_min).sum(dim=1)
        
        # 5.2 相对耦合违规惩罚 (如 AFT >= BTF + 25)
        rel_penalty = self.constraint_manager.compute_soft_penalty(control_trainable)
        
        # 5.3 汇总惩罚
        loss_constraint = abs_penalty + rel_penalty # [Batch]
        
        # ==========================================
        # 6. 总损失融合
        # ==========================================
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