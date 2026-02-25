import torch
import torch.nn as nn
import logging
from typing import Dict, Tuple

# 引入全局损伤风险计算模块与特征顺序常量
from common.metrics import injury_risk
from common.settings import FEATURE_ORDER
from src.core.param_manager import ParamManager

class SurrogateAdapter(nn.Module):
    """
    代理模型级联适配器 (Cascaded Surrogate Adapter)
    将预训练的 PulsePredict 与 InjuryPredict 严格级联，封装为全流程可微的环境模拟器。
    
    核心机制 (计算流):
    1. 组装物理尺度的状态参数(s)与决策参数(a) -> combined_phys [Batch, 13]
    2. 特征归一化 -> combined_norm [Batch, 13] (保持 Tensor 梯度穿透)
    3. PulsePredict 推理: f(combined_norm) -> waveform_norm (归一化波形)
    4. InjuryPredict 推理: g(combined_norm, waveform_norm) -> predictions_phys (物理损伤)
    5. 计算基于指数级权重的复合风险概率 (L_risk) 与越界软惩罚 (L_constraint)。
    """
    def __init__(self, pulse_model: nn.Module, injury_model: nn.Module, param_manager: ParamManager, config: dict, data_processor):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.pulse_model = pulse_model
        self.injury_model = injury_model
        self.param_manager = param_manager
        
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
        
        参数:
            state_params: [Batch, D_state] 物理尺度的状态参数
            control_trainable: [Batch, D_trainable] 物理尺度的可调约束系统参数 (a) 
                # WARNING: [核心约束] 此张量必须具有 requires_grad=True 属性。
            
        返回:
            total_loss: [Batch] 每个样本的融合标量损失
            predictions_phys: [Batch, 3] 预测的物理损伤绝对值 (HIC15, Dmax, Nij)
            info: 记录中间计算过程的监控字典
        """
        batch_size = state_params.shape[0]
        device = state_params.device
        total_dim = self.param_manager.get_total_feature_dim()
        
        # ==========================================
        # 1. 组装输入张量 (物理尺度)
        # ==========================================
        # [Batch, 13]
        combined_phys = torch.zeros((batch_size, total_dim), device=device, dtype=torch.float32)
        
        combined_phys[:, self.param_manager.get_state_indices()] = state_params
        combined_phys[:, self.param_manager.get_control_trainable_indices()] = control_trainable
        
        fixed_indices, fixed_defaults = self.param_manager.get_control_fixed_defaults(device=device)
        if len(fixed_indices) > 0:
            # [Dim] -> [1, Dim] -> [Batch, Dim]
            combined_phys[:, fixed_indices] = fixed_defaults.unsqueeze(0).expand(batch_size, -1)
            
        # ==========================================
        # 2. 归一化特征 (保持梯度)
        # ==========================================
        # [Batch, 13] -> [Batch, 13]
        model_input_norm = self.data_processor.process_by_name(
            values=combined_phys, 
            feature_names=FEATURE_ORDER, 
            inverse=False
        )
            
        # ==========================================
        # 3. 代理模型级联推理 (PulsePredict -> InjuryPredict)
        # ==========================================
        # 3.1 碰撞波形预测
        # 输入: 归一化 13维特征 -> 输出: 归一化波形 [Batch, Channels, SeqLen]
        # TODO: [需确认] 此处调用方法严格依赖于 PulsePredict 的 forward 定义，默认其仅需特征张量输入。
        waveform_norm = self.pulse_model(model_input_norm)
        
        # 3.2 损伤预测
        # 输入: 归一化特征 + 归一化波形 -> 输出: 物理损伤 [Batch, 3]
        # TODO: [需确认] 此处调用方法严格依赖于 InjuryPredict 的 forward 定义，假设其接收 (属性特征, 波形特征) 两个参数。
        predictions_phys = self.injury_model(model_input_norm, waveform_norm)

        # ==========================================
        # 4. 摊销寻优计算：非线性联合风险项 L_risk
        # ==========================================
        hic15 = predictions_phys[:, 0]
        dmax = predictions_phys[:, 1]
        nij = predictions_phys[:, 2]
        
        # TODO: [需确认] 依赖 common/metrics/injury_risk.py 中的具体函数名获取 P(AIS3+) 概率
        p_head = injury_risk.prob_hic15_ais3(hic15)
        p_chest = injury_risk.prob_dmax_ais3(dmax)
        p_neck = injury_risk.prob_nij_ais3(nij)
        
        # 极值截断: 防止后续指数运算发生 NaN 梯度崩溃
        p_head = torch.clamp(p_head, 1e-6, 1.0 - 1e-6)
        p_chest = torch.clamp(p_chest, 1e-6, 1.0 - 1e-6)
        p_neck = torch.clamp(p_neck, 1e-6, 1.0 - 1e-6)
        
        # L_risk = 1 - (1-P_head)^w_head * (1-P_chest)^w_chest * (1-P_neck)^w_neck
        term_head = torch.pow(1.0 - p_head, self.w_head)
        term_chest = torch.pow(1.0 - p_chest, self.w_chest)
        term_neck = torch.pow(1.0 - p_neck, self.w_neck)
        
        loss_risk = 1.0 - (term_head * term_chest * term_neck) # [Batch]
        
        # ==========================================
        # 5. 摊销寻优计算：物理边界惩罚项 L_constraint
        # ==========================================
        min_bounds, max_bounds = self.param_manager.get_trainable_bounds(device=device)
        
        # [Batch, D_trainable]
        exceed_max = torch.relu(control_trainable - max_bounds.unsqueeze(0))
        exceed_min = torch.relu(min_bounds.unsqueeze(0) - control_trainable)
        
        loss_constraint = (exceed_max + exceed_min).sum(dim=1) # [Batch]
        
        # ==========================================
        # 6. 总损失融合
        # ==========================================
        total_loss = loss_risk + self.weight_penalty * loss_constraint # [Batch]
        
        info = {
            "loss_risk": loss_risk.detach(),
            "loss_constraint": loss_constraint.detach(),
            "p_head": p_head.detach(),
            "p_chest": p_chest.detach(),
            "p_neck": p_neck.detach(),
            "hic15": hic15.detach(),
            "dmax": dmax.detach(),
            "nij": nij.detach()
        }
        
        return total_loss, predictions_phys, info