import torch
import torch.nn as nn
import logging
from typing import List

# 严格执行绝对路径引用规范
from ARS_optim.src.core.param_manager import ParamManager
from ARS_optim.src.core.constraints import PhysicalConstraintManager

class StrategyNet(nn.Module):
    """
    自适应寻优策略网络 (Adaptive Optimization Strategy Network)
    
    架构设计:
    本网络充当了寻优系统中的“智能体(Agent)”。
    1. 多模态物理感知：不仅接收结构化的标量状态参数（State），还通过内嵌的 1D-CNN 
       提取连续的车身碰撞加速度波形（Pulse）的高维动态特征。
    2. 鲁棒特征融合：利用自适应池化层（AdaptiveAvgPool1d）抹平波形序列长度（Seq_Len）
       的潜在波动，并与标量特征级联（Concat）送入多层感知机（MLP）。
    3. 强有界连续输出：输出层强制使用 Sigmoid 激活并结合逆向仿射变换，将张量严格
       约束至各执行参数的理论上下限内。
    4. 混合状态物理投影：最终调用 PhysicalConstraintManager，融合当前的 State 上下文，
       执行双向物理极值与耦合关系的硬投影截断。
    """
    def __init__(
        self, 
        param_manager: ParamManager, 
        constraint_manager: PhysicalConstraintManager,
        hidden_dims: List[int] = [128, 256, 128],
        activation: str = "LeakyReLU",
        dropout: float = 0.1,
        pulse_channels: int = 2,        # X轴和Y轴两向加速度波形
        pulse_embed_dim: int = 32       # 波形降维后的嵌入特征向量维度
    ):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.param_manager = param_manager
        self.constraint_manager = constraint_manager
        
        # 确定输入与输出基础维度
        self.state_dim = self.param_manager.get_state_dim()
        self.output_dim = self.param_manager.get_trainable_dim()
        # 记录波形编码器输出维度，方便在 forward 中做一致性检查
        self.pulse_embed_dim = pulse_embed_dim
        
        if self.output_dim == 0:
            raise ValueError("[致命错误] 策略网络初始化失败：没有设置任何 trainable=True 的可调参数！")

        # ---------------------------------------------------------
        # 1. 构建波形特征编码器 (1D-CNN Waveform Encoder)
        # ---------------------------------------------------------
        act_layer_cls = getattr(nn, activation) if hasattr(nn, activation) else nn.ReLU
        
        self.pulse_encoder = nn.Sequential(
            # Layer 1: [Batch, 2, Seq_Len] -> [Batch, 16, Seq_Len / 2]
            nn.Conv1d(in_channels=pulse_channels, out_channels=16, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(16),
            act_layer_cls(inplace=True),
            
            # Layer 2: [Batch, 16, Seq_Len / 2] -> [Batch, pulse_embed_dim, Seq_Len / 4]
            nn.Conv1d(in_channels=16, out_channels=pulse_embed_dim, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(pulse_embed_dim),
            act_layer_cls(inplace=True),
            
            # 引入自适应平均池化，彻底解耦序列长度的硬性依赖
            # [Batch, pulse_embed_dim, Seq_Len / 4] -> [Batch, pulse_embed_dim, 1]
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten() # -> [Batch, pulse_embed_dim]
        )
        
        # ---------------------------------------------------------
        # 2. 构建 MLP 决策骨干网络 (Decision Backbone)
        # ---------------------------------------------------------
        layers = []
        # MLP 的输入维度 = 标量状态特征维度 + 波形嵌入特征维度
        in_features = self.state_dim + pulse_embed_dim
        act_layer_instance = act_layer_cls(inplace=True)
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_features, hidden_dim, bias=False)) # BN前无需Bias
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(act_layer_instance)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_features = hidden_dim
            
        # 最后一层映射到可调参数维度 (Plain Linear)，不加激活与 BN
        layers.append(nn.Linear(in_features, self.output_dim))
        self.mlp = nn.Sequential(*layers)
        
        # ---------------------------------------------------------
        # 3. 注册物理边界缓冲 (Buffers)
        # ---------------------------------------------------------
        # 获取物理参数的绝对上下限，并驻留目标设备内存
        min_b, max_b = self.param_manager.get_trainable_bounds()
        self.register_buffer("min_bounds", min_b)
        self.register_buffer("max_bounds", max_b)
        
        # 权重初始化
        self._initialize_weights()

    # ------------------------------------------------------------------
    def to_normalized(self, actions_phys: torch.Tensor) -> torch.Tensor:
        """
        将策略网络的物理尺度动作转换到 [0,1] 归一化空间。
        这个工具主要供外部评估或日志使用，例如将输出与训练时的 `norm_actions` 对齐。
        由于归一化仅依赖于 min_bounds/max_bounds，该函数无需其它上下文。

        输入:
            actions_phys: [Batch, D_trainable] 物理尺度动作
        返回:
            norm: [Batch, D_trainable] 归一化动作 (0 到 1)
        """
        # 避免除零
        span = (self.max_bounds - self.min_bounds)
        safe_span = torch.where(span > 0, span, torch.ones_like(span))
        return (actions_phys - self.min_bounds.unsqueeze(0)) / safe_span.unsqueeze(0)

    def _initialize_weights(self):
        """深度网络权重初始化 (Kaiming Normal for LeakyReLU/ReLU)"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, state_features: torch.Tensor, pulse_features: torch.Tensor) -> torch.Tensor:
        """
        前向推理过程，融合多模态特征并输出受严格物理法则约束的寻优动作。
        
        参数:
            state_features: [Batch, D_state] 物理尺度的环境与乘员标量参数
            pulse_features: [Batch, 2, Seq_Len] 归一化后的二维（XY轴）碰撞加速度波形
            
        返回:
            final_actions: [Batch, D_trainable] 绝对合法的物理决策参数，并保留完整的梯度传播链
        """
        # 1. 提取波形动态特征
        # [Batch, 2, Seq_Len] -> [Batch, pulse_embed_dim]
        # 校验输入波形的通道数和序列长度
        assert pulse_features.dim() == 3 and pulse_features.size(1) == self.pulse_encoder[0].in_channels, \
            f"pulse_features should be [B, {self.pulse_encoder[0].in_channels}, Seq]"
        pulse_embed = self.pulse_encoder(pulse_features)
        assert pulse_embed.size(1) == self.pulse_embed_dim, \
            "pulse_encoder output dimension mismatch"
        
        # 2. 多模态特征级联
        # [Batch, D_state] ⊕ [Batch, pulse_embed_dim] -> [Batch, D_state + pulse_embed_dim]
        combined_features = torch.cat([state_features, pulse_embed], dim=1)
        
        # 3. 骨干网络非线性决策映射
        # [Batch, D_state + pulse_embed_dim] -> [Batch, D_trainable]
        raw_output = self.mlp(combined_features)
        
        # 4. 绝对极值范围逆映射 (Sigmoid + Affine Transform)
        # 利用 Sigmoid 将输出强制压入 (0, 1) 区间，随后按对应参数的 Min-Max 拉伸至真实物理标度
        norm_actions = torch.sigmoid(raw_output)
        range_span = self.max_bounds - self.min_bounds
        # 防止某些维度 min==max 导致 0 乘法
        safe_span = torch.where(range_span > 0, range_span, torch.ones_like(range_span))
        abs_actions = norm_actions * safe_span.unsqueeze(0) + self.min_bounds.unsqueeze(0)
        
        # 输出数值检查，某些激活和归一化组合在训练初期可能产生 NaN/Inf
        if torch.isnan(abs_actions).any() or torch.isinf(abs_actions).any():
            self.logger.warning("策略网络输出包含 NaN/Inf，可能发生梯度爆炸或数据异常。")
        
        # 5. 混合状态物理法则硬投影 (Hard Constraint Projection)
        # 依赖于 PhysicalConstraintManager，融入 state_features 上下文以解析混合参数耦合限制
        final_actions = self.constraint_manager.project_forward(abs_actions, state_features)
        
        return final_actions