from typing import List

import torch
import torch.nn as nn

from ARS_optim.src.constraints import ConstraintEngine
from ARS_optim.src.param_manager import ParamManager
from common.data_utils.processor import UnifiedDataProcessor


def build_strategy_net_from_config(
    param_manager: ParamManager,
    constraint_engine: ConstraintEngine,
    data_processor: UnifiedDataProcessor,
    config: dict,
) -> "StrategyNet":
    """根据配置构造策略网络。

    训练与评估必须读取同一份策略网络超参数，否则同名权重可能在两端对应到不同结构。
    这里保留一个很薄的构造函数，只负责把 config 中的结构参数解包出来，避免入口脚本各自重复拼装。
    """
    strat_cfg = config.get("strategy_net", {})
    return StrategyNet(
        param_manager=param_manager,
        constraint_engine=constraint_engine,
        data_processor=data_processor,
        hidden_dims=strat_cfg.get("hidden_dims", [128, 256, 128]),
        activation=strat_cfg.get("activation", "LeakyReLU"),
        dropout=float(strat_cfg.get("dropout", 0.1)),
        pulse_channels=int(strat_cfg.get("pulse_channels", 2)),
        pulse_embed_dim=int(strat_cfg.get("pulse_embed_dim", 32)),
    )


class StrategyNet(nn.Module):
    """策略网络：context + pulse -> trainable controls。

    这个模块只负责学习“给定工况时应该往哪个控制量方向调”。
    约束处理仍交给 ConstraintEngine，原因是同一套合法化规则还要被训练采样、
    评估 CSV、局部精调共同复用；若把规则散落到网络内部，训练端和评估端很容易逐渐偏离。
    """

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
        self.trainable_params = self.param_manager.get_trainable_params()
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
        defaults = self.param_manager.get_default_values(self.trainable_params)
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
        # context_features 的列顺序来自 ParamManager，而不是 data_processor 内部的固定顺序。
        # 这里按名称归一化，目的是把“特征顺序由谁定义”收敛到 ParamManager 一处，
        # 避免后续参数角色调整时因为列顺序假设失效而产生隐蔽错误。
        # 其中 is_driver_side / OT 会被 process_by_name 映射为整数类别编码，并直接作为标量输入 MLP。
        # 这与 InjuryPredictModel 为离散变量单独做 Embedding 的设计不同：策略网络这里只需要一个轻量决策器，
        # 而 OT 本身又带有明确的序数语义（5th < 50th < 95th），因此标量编码已经足够让 MLP 学到体型趋势。
        return self.data_processor.process_by_name(context_features, self.context_names, inverse=False)

    def _decode_actions_from_logits(self, raw_output: torch.Tensor, context_features: torch.Tensor) -> torch.Tensor:
        """将网络输出的无界 logits 还原为物理参数，并做连续可微投影。

        这里把“网络回归”和“动作约束”拆成两个连续步骤：
        1. 先用 sigmoid 和边界盒把输出限制在 trainable 参数的基础物理范围内；
        2. 再把动作拼回完整特征张量，由约束引擎统一做连续耦合投影。

        这样做的目的不是增加层次，而是避免让网络直接学习一整套硬规则；
        策略网络只负责学习从状态到动作的映射，参数合法化仍然由统一的约束层定义。
        """
        norm_actions = torch.sigmoid(raw_output)
        span = torch.where(
            self.max_bounds > self.min_bounds,
            self.max_bounds - self.min_bounds,
            torch.ones_like(self.max_bounds),
        )
        actions = norm_actions * span.unsqueeze(0) + self.min_bounds.unsqueeze(0)
        full_features = self.constraint_engine.compose_full_features(context_features, actions)
        projected_full = self.constraint_engine.project_forward(full_features, strict=False)
        _, projected_actions = self.constraint_engine.split_from_full(projected_full)
        return projected_actions

    def forward(self, context_features: torch.Tensor, pulse_features: torch.Tensor) -> torch.Tensor:
        # pulse 只作为工况补充信息编码进策略，不在这里重复生成，
        # 这样训练、验证和评估都可以复用同一份 pulse，避免同批样本重复跑代理前端。
        pulse_embed = self.pulse_encoder(pulse_features)
        context_norm = self._normalize_context(context_features)
        combined = torch.cat([context_norm, pulse_embed], dim=1)
        raw_output = self.mlp(combined)
        return self._decode_actions_from_logits(raw_output, context_features)
