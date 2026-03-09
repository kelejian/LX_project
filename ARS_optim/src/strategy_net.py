from typing import List

import torch
import torch.nn as nn

from ARS_optim.src.constraints import ConstraintEngine
from ARS_optim.src.param_manager import ParamManager
from common.data_utils.processor import UnifiedDataProcessor


class StrategyNet(nn.Module):
    """策略网络：context + pulse -> trainable controls。"""

    def __init__(
        self,
        param_manager: ParamManager,
        constraint_engine: ConstraintEngine,
        data_processor: UnifiedDataProcessor,
        hidden_dims: List[int] = None,
        activation: str = "LeakyReLU",
        dropout: float = 0.1,
        pulse_channels: int = 2,
        pulse_embed_dim: int = 32,
    ):
        super().__init__()
        self.param_manager = param_manager
        self.constraint_engine = constraint_engine
        self.data_processor = data_processor
        self.context_dim = self.param_manager.get_context_dim()
        self.output_dim = self.param_manager.get_trainable_dim()
        self.context_names = self.param_manager.get_context_names()
        self.pulse_embed_dim = int(pulse_embed_dim)

        if hidden_dims is None:
            hidden_dims = [128, 256, 128]
        if self.output_dim == 0:
            raise ValueError("当前 param_space.yaml 中没有 trainable=True 的控制参数")

        act_layer = getattr(nn, activation) if hasattr(nn, activation) else nn.ReLU
        self.pulse_encoder = nn.Sequential(
            nn.Conv1d(pulse_channels, 16, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(16),
            act_layer(inplace=True),
            nn.Conv1d(16, self.pulse_embed_dim, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(self.pulse_embed_dim),
            act_layer(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        layers = []
        in_features = self.context_dim + self.pulse_embed_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_features, hidden_dim, bias=False))
            layers.append(nn.BatchNorm1d(hidden_dim))
            if act_layer is nn.PReLU:
                layers.append(act_layer())
            else:
                layers.append(act_layer(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
            in_features = hidden_dim
        layers.append(nn.Linear(in_features, self.output_dim))
        self.mlp = nn.Sequential(*layers)

        min_bounds, max_bounds = self.param_manager.get_trainable_bounds()
        self.register_buffer("min_bounds", min_bounds)
        self.register_buffer("max_bounds", max_bounds)
        defaults = [param["default"] for param in self.param_manager.control_trainable_params]
        self.register_buffer("default_actions", torch.tensor(defaults, dtype=torch.float32))
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if getattr(module, "bias", None) is not None:
                    nn.init.constant_(module.bias, 0.0)

        out_layer = self.mlp[-1]
        if not isinstance(out_layer, nn.Linear) or out_layer.bias is None:
            raise ValueError("策略网络最后一层必须是带 bias 的 Linear")
        with torch.no_grad():
            nn.init.constant_(out_layer.weight, 0.0)
            # 令 sigmoid(bias) 恰好等于默认值在线性边界盒中的归一化位置：
            # ratio = (default - min) / (max - min), bias = log(ratio / (1 - ratio))。
            # 这样在最后一层权重清零时，网络未训练前就会稳定输出 default 动作。
            ratio = (self.default_actions - self.min_bounds) / torch.clamp(self.max_bounds - self.min_bounds, min=1e-12)
            ratio = torch.clamp(ratio, min=1e-6, max=1.0 - 1e-6)
            out_layer.bias.copy_(torch.log(ratio / (1.0 - ratio)))

    def _normalize_context(self, context_features: torch.Tensor) -> torch.Tensor:
        return self.data_processor.process_by_name(context_features, self.context_names, inverse=False)

    def forward(self, context_features: torch.Tensor, pulse_features: torch.Tensor) -> torch.Tensor:
        pulse_embed = self.pulse_encoder(pulse_features)
        context_norm = self._normalize_context(context_features)
        combined = torch.cat([context_norm, pulse_embed], dim=1)
        raw_output = self.mlp(combined)
        # 先用 sigmoid 把网络输出压到 [0, 1]，再线性映射回物理参数边界盒。最后再通过约束引擎修正 AFT/BTF、LL1/LL2、LLATTF 等耦合关系，避免网络直接回归物理量时频繁输出越界或违反硬约束的解。
        norm_actions = torch.sigmoid(raw_output)
        span = torch.where(self.max_bounds > self.min_bounds, self.max_bounds - self.min_bounds, torch.ones_like(self.max_bounds))
        actions = norm_actions * span.unsqueeze(0) + self.min_bounds.unsqueeze(0)
        return self.constraint_engine.project_forward(actions, context_features)
