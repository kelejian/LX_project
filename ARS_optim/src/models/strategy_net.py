import torch
import torch.nn as nn
import logging
from typing import List

from src.core.param_manager import ParamManager
from src.core.constraints import PhysicalConstraintManager

class StrategyNet(nn.Module):
    """
    自适应寻优策略网络 (Adaptive Optimization Strategy Network)
    
    架构特征:
    1. 接收不可控的状态特征 (State Params) 作为输入。
    2. 使用多层 MLP 提取高阶非线性环境表征。
    3. 输出层执行强有界激活 (Sigmoid)，逆映射到各可调参数的真实物理区间。
    4. 执行硬投影截断 (Hard Projection)，确保相对耦合物理法则被严格遵守。
    """
    def __init__(
        self, 
        param_manager: ParamManager, 
        constraint_manager: PhysicalConstraintManager,
        hidden_dims: List[int] = [128, 256, 128],
        activation: str = "LeakyReLU",
        dropout: float = 0.1
    ):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.param_manager = param_manager
        self.constraint_manager = constraint_manager
        
        # 确定输入与输出维度
        self.input_dim = self.param_manager.get_state_dim()
        self.output_dim = self.param_manager.get_trainable_dim()
        
        # 严苛校验：若无任何参数需要寻优，则报错
        if self.output_dim == 0:
            raise ValueError("[致命错误] 策略网络初始化失败：没有设置任何 trainable=True 的可调参数！")

        # ---------------------------------------------------------
        # 1. 构建 MLP 骨干网络 (Backbone)
        # ---------------------------------------------------------
        layers = []
        in_features = self.input_dim
        
        # 动态解析激活函数
        act_layer = getattr(nn, activation)(inplace=True) if hasattr(nn, activation) else nn.ReLU(inplace=True)
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_features, hidden_dim, bias=False)) # BN前无需Bias
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(act_layer)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_features = hidden_dim
            
        # 最后一层映射到可调参数维度 (Plain Linear)
        layers.append(nn.Linear(in_features, self.output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # ---------------------------------------------------------
        # 2. 注册物理边界缓冲 (Buffers)
        # ---------------------------------------------------------
        # 从 param_manager 获取绝对上下限。
        # 使用 register_buffer 能够让这些张量随着模型 .to(device) 自动转移到 GPU/CPU，
        # 且不会被 Optimizer 错误当成网络权重去更新。
        min_b, max_b = self.param_manager.get_trainable_bounds()
        self.register_buffer("min_bounds", min_b)
        self.register_buffer("max_bounds", max_b)
        
        # 权重初始化
        self._initialize_weights()

    def _initialize_weights(self):
        """Kaiming 正态初始化，适配 ReLU/LeakyReLU 体系"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, state_features: torch.Tensor) -> torch.Tensor:
        """
        前向推理过程，融合了深度网络特征提取与严格的物理边界约束。
        
        参数:
            state_features: [Batch, D_state] 物理尺度的环境工况参数
            
        返回:
            actions: [Batch, D_trainable] 绝对合法的物理决策参数，并保留完整的梯度传播链
        """
        # 1. 骨干网络特征提取
        # [Batch, D_state] -> [Batch, D_trainable]
        raw_output = self.mlp(state_features)
        
        # 2. 绝对极值范围逆映射 (Sigmoid + Affine Transform)
        # 利用 Sigmoid 将输出强制压入 (0, 1) 区间，随后按对应参数的 Min-Max 拉伸
        # [Batch, D_trainable] -> [Batch, D_trainable]
        norm_actions = torch.sigmoid(raw_output)
        range_span = self.max_bounds - self.min_bounds
        
        abs_actions = norm_actions * range_span.unsqueeze(0) + self.min_bounds.unsqueeze(0)
        
        # 3. 相对耦合物理法则硬投影 (Hard Constraint Projection)
        # 依赖于 PhysicalConstraintManager，调用诸如 torch.min 等可微算子
        # 确保诸如 AFT < BTF + 25 等物理依赖在网络输出端被 100% 严格执行
        # [Batch, D_trainable] -> [Batch, D_trainable]
        final_actions = self.constraint_manager.project_forward(abs_actions)
        
        return final_actions