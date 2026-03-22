from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

import torch
import torch.nn as nn

from InjuryPredict.utils.models import BaseMLP, DiscreteFeatureEmbedding, FeatureSELayer, TemporalBlock

from ARS_optim.src.constraints import ConstraintEngine
from ARS_optim.src.param_manager import ParamManager
from common.data_utils.processor import UnifiedDataProcessor


def build_strategy_net_from_config(
    param_manager: ParamManager,
    constraint_engine: ConstraintEngine,
    data_processor: UnifiedDataProcessor,
    config: dict,
) -> "StrategyNet":
    """根据配置构造策略网络。"""
    strat_cfg = config.get("strategy_net", {}) or {}
    return StrategyNet(
        param_manager=param_manager,
        constraint_engine=constraint_engine,
        data_processor=data_processor,
        pulse_channels=int(strat_cfg.get("pulse_channels", 2)),
        tcn_channels_list=[int(value) for value in strat_cfg.get("tcn_channels_list", [64, 128, 256])],
        tcn_output_dim=int(strat_cfg.get("tcn_output_dim", 128)),
        tcn_kernel_init=int(strat_cfg.get("tcn_kernel_init", 8)),
        tcn_kernel_mid=int(strat_cfg.get("tcn_kernel_mid", 3)),
        mlp_encoder_hidden=int(strat_cfg.get("mlp_encoder_hidden", 192)),
        mlp_encoder_layers=int(strat_cfg.get("mlp_encoder_layers", 3)),
        mlp_encoder_output_dim=int(strat_cfg.get("mlp_encoder_output_dim", 128)),
        mlp_decoder_hidden=int(strat_cfg.get("mlp_decoder_hidden", 160)),
        mlp_decoder_layers=int(strat_cfg.get("mlp_decoder_layers", 3)),
        mlp_decoder_output_dim=int(strat_cfg.get("mlp_decoder_output_dim", 96)),
        dropout_mlp=float(strat_cfg.get("dropout_mlp", 0.1)),
        dropout_tcn=float(strat_cfg.get("dropout_tcn", 0.0)),
    )


def resolve_strategy_run_artifacts(strategy_ckpt_path: Path) -> Dict[str, Path]:
    """根据策略权重路径定位同一次训练保存的配置快照。"""
    strategy_ckpt_path = Path(strategy_ckpt_path).resolve()
    if not strategy_ckpt_path.is_file():
        raise FileNotFoundError(f"策略权重不存在: {strategy_ckpt_path}")

    run_dir = strategy_ckpt_path.parent.parent
    configs_dir = run_dir / "configs_used"
    config_path = configs_dir / "config.yaml"
    param_space_path = configs_dir / "param_space.yaml"
    normalization_path = configs_dir / "normalization_config.json"

    for path, label in (
        (config_path, "config.yaml"),
        (param_space_path, "param_space.yaml"),
        (normalization_path, "normalization_config.json"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"策略权重目录缺少 {label}: {path}")

    return {
        "run_dir": run_dir,
        "configs_dir": configs_dir,
        "config": config_path,
        "param_space": param_space_path,
        "normalization_config": normalization_path,
    }


def load_strategy_run_config(strategy_ckpt_path: Path) -> Tuple[Dict[str, Any], Dict[str, Path]]:
    """读取策略权重所属运行目录中的配置快照。"""
    artifacts = resolve_strategy_run_artifacts(strategy_ckpt_path)
    with open(artifacts["config"], "r", encoding="utf-8") as file:
        saved_config = yaml.safe_load(file)
    return saved_config, artifacts


class PulseTCNEncoder(nn.Module):
    """轻量 TCN 波形编码器。

    这里复用 InjuryPredict 中的 `TemporalBlock` 残差块，但不再引入 ChannelAttention。
    策略网络只需要稳定提取 pulse 的阶段性模式作为决策增强特征，
    没必要把 InjuryPredict 的多任务输出与注意力分支整套搬进来。
    """

    def __init__(
        self,
        in_channels: int,
        channels_list: List[int],
        output_dim: int,
        kernel_size_init: int,
        kernel_size_mid: int,
        dropout: float,
    ):
        super().__init__()
        if not channels_list:
            raise ValueError("tcn_channels_list 不能为空")
        if kernel_size_init < 2 or kernel_size_mid < 1:
            raise ValueError("TCN 卷积核大小必须为正整数")

        padding_init = max(0, (kernel_size_init - 2) // 2)
        self.initial_conv = nn.Sequential(
            nn.Conv1d(in_channels, channels_list[0], kernel_size=kernel_size_init, stride=2, padding=padding_init, bias=False),
            nn.BatchNorm1d(channels_list[0]),
            nn.SiLU(inplace=True),
        )

        blocks = []
        in_dim = channels_list[0]
        for out_dim in channels_list[1:]:
            blocks.append(
                TemporalBlock(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    kernel_size=kernel_size_mid,
                    dropout=dropout,
                )
            )
            in_dim = out_dim
        self.temporal_blocks = nn.Sequential(*blocks)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.output_proj = nn.Linear(in_dim, output_dim)

    def forward(self, pulse_features: torch.Tensor) -> torch.Tensor:
        x = self.initial_conv(pulse_features)
        x = self.temporal_blocks(x)
        x = self.global_avg_pool(x).squeeze(-1)
        return self.output_proj(x)


class StrategyNet(nn.Module):
    """策略网络：context + pulse -> trainable controls。

    结构借鉴 InjuryPredictModel：
    - TCN 波形编码器负责提取碰撞 pulse 的时序表征；
    - MLP 编码器负责处理 context 中的连续量与离散嵌入；
    - 融合层把 [pulse 编码, context 编码, 原始标量特征] 拼接后再做规范化与重标定；
    - MLP 解码器输出共享决策表示；
    - 最终输出头只保留一个共享 Linear，把表示映射到所有 trainable 控制参数的 logits。

    约束投影仍由 ConstraintEngine 统一执行，避免策略网络、局部精调和评估 CSV
    各自复制一套物理规则。
    """

    def __init__(
        self,
        param_manager: ParamManager,
        constraint_engine: ConstraintEngine,
        data_processor: UnifiedDataProcessor,
        pulse_channels: int = 2,
        tcn_channels_list: List[int] = None,
        tcn_output_dim: int = 128,
        tcn_kernel_init: int = 8,
        tcn_kernel_mid: int = 3,
        mlp_encoder_hidden: int = 192,
        mlp_encoder_layers: int = 3,
        mlp_encoder_output_dim: int = 128,
        mlp_decoder_hidden: int = 160,
        mlp_decoder_layers: int = 3,
        mlp_decoder_output_dim: int = 96,
        dropout_mlp: float = 0.1,
        dropout_tcn: float = 0.0,
    ):
        super().__init__()
        self.param_manager = param_manager
        self.constraint_engine = constraint_engine
        self.data_processor = data_processor

        self.context_names = self.param_manager.get_context_names()
        self.trainable_params = self.param_manager.get_trainable_params()
        self.trainable_params_names = self.param_manager.get_trainable_names()
        self.context_dim = self.param_manager.get_context_dim()
        self.output_dim = self.param_manager.get_trainable_dim()
        if self.output_dim == 0:
            raise ValueError("当前 param_space.yaml 中没有 trainable=True 的控制参数")

        discrete_num_classes = self.data_processor.get_discrete_num_classes()
        required_discrete = ["is_driver_side", "OT"]
        for name in required_discrete:
            if name not in self.context_names:
                raise ValueError(f"策略网络的 context 输入中必须包含离散特征 {name}")
            if name not in discrete_num_classes:
                raise ValueError(f"normalization_config 中缺少离散特征 {name} 的类别配置")

        self.continuous_context_names = [name for name in self.context_names if name not in required_discrete]
        self.discrete_context_names = required_discrete
        self.continuous_context_indices = [self.context_names.index(name) for name in self.continuous_context_names]
        self.discrete_context_indices = [self.context_names.index(name) for name in self.discrete_context_names]

        discrete_classes = [int(discrete_num_classes[name]) for name in self.discrete_context_names]
        self.discrete_embedding = DiscreteFeatureEmbedding(discrete_classes)
        self.discrete_embedding_dim = int(self.discrete_embedding.output_dim)
        self.scalar_feature_dim = len(self.continuous_context_names) + self.discrete_embedding_dim

        if tcn_channels_list is None:
            tcn_channels_list = [64, 128, 256]
        self.pulse_encoder = PulseTCNEncoder(
            in_channels=pulse_channels,
            channels_list=tcn_channels_list,
            output_dim=tcn_output_dim,
            kernel_size_init=tcn_kernel_init,
            kernel_size_mid=tcn_kernel_mid,
            dropout=dropout_tcn,
        )

        self.context_encoder = BaseMLP(
            in_features=self.scalar_feature_dim,
            hidden_features=mlp_encoder_hidden,
            out_features=mlp_encoder_output_dim,
            num_layers=mlp_encoder_layers,
            dropout=dropout_mlp,
        )

        fusion_dim = tcn_output_dim + mlp_encoder_output_dim + self.scalar_feature_dim
        self.fusion_norm = nn.LayerNorm(fusion_dim)
        self.fusion_se = FeatureSELayer(fusion_dim, reduction=8)
        self.decoder = BaseMLP(
            in_features=fusion_dim,
            hidden_features=mlp_decoder_hidden,
            out_features=mlp_decoder_output_dim,
            num_layers=mlp_decoder_layers,
            dropout=dropout_mlp,
        )
        self.post_decoder_block = nn.Sequential(
            nn.BatchNorm1d(mlp_decoder_output_dim),
            nn.SiLU(inplace=True),
        )
        self.output_head = nn.Linear(mlp_decoder_output_dim, self.output_dim)

        min_bounds, max_bounds = self.param_manager.get_trainable_opt_bounds()
        self.register_buffer("min_bounds", min_bounds)
        self.register_buffer("max_bounds", max_bounds)
        defaults = self.param_manager.get_default_values(self.trainable_params)
        self.register_buffer("default_actions", torch.tensor(defaults, dtype=torch.float32))

        self.arch_info = {
            "context_dim": self.context_dim,
            "continuous_context_names": list(self.continuous_context_names),
            "discrete_context_names": list(self.discrete_context_names),
            "scalar_feature_dim": self.scalar_feature_dim,
            "pulse_channels": int(pulse_channels),
            "tcn_channels_list": [int(value) for value in tcn_channels_list],
            "tcn_output_dim": int(tcn_output_dim),
            "mlp_encoder_output_dim": int(mlp_encoder_output_dim),
            "mlp_decoder_output_dim": int(mlp_decoder_output_dim),
            "output_dim": self.output_dim,
            "output_trainable_params": list(self.trainable_params_names),
        }

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

        with torch.no_grad():
            self.output_head.weight.zero_()
            # 输出头权重清零后，网络对任意输入都会退化为一个固定 bias。
            # 这里把 bias 设为 default 动作在 [opt_min, opt_max] 内的 logit，
            # 从而保证训练开始前任何合法输入都会输出 param_space.yaml 中的 default。
            ratio = (self.default_actions - self.min_bounds) / torch.clamp(self.max_bounds - self.min_bounds, min=1e-12)
            ratio = torch.clamp(ratio, min=1e-6, max=1.0 - 1e-6)
            self.output_head.bias.copy_(torch.log(ratio / (1.0 - ratio)))

    def _prepare_scalar_context_features(self, context_features: torch.Tensor) -> torch.Tensor:
        parts = []
        if self.continuous_context_indices:
            continuous_raw = context_features[:, self.continuous_context_indices]
            continuous_norm = self.data_processor.process_by_name(
                continuous_raw,
                self.continuous_context_names,
                inverse=False,
            )
            parts.append(continuous_norm.to(dtype=context_features.dtype))

        discrete_raw = context_features[:, self.discrete_context_indices]
        discrete_encoded = self.data_processor.process_by_name(
            discrete_raw,
            self.discrete_context_names,
            inverse=False,
        ).to(dtype=torch.long)
        discrete_embed = self.discrete_embedding(discrete_encoded).to(dtype=context_features.dtype)
        parts.append(discrete_embed)

        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)

    def _decode_actions_from_logits(
        self,
        raw_output: torch.Tensor,
        context_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """把网络输出的 logits 映射回物理动作，并做前向投影。"""
        norm_actions = torch.sigmoid(raw_output)
        spans = torch.clamp(self.max_bounds - self.min_bounds, min=1e-12)
        actions = norm_actions * spans.unsqueeze(0) + self.min_bounds.unsqueeze(0)
        raw_full_features = self.constraint_engine.compose_full_features(context_features, actions)
        projected_full = self.constraint_engine.project_forward(raw_full_features, strict=False)
        _, projected_actions = self.constraint_engine.split_from_full(projected_full)
        return projected_actions, raw_full_features

    def forward(
        self,
        context_features: torch.Tensor,
        pulse_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pulse_encoded = self.pulse_encoder(pulse_features)
        scalar_context = self._prepare_scalar_context_features(context_features)
        context_encoded = self.context_encoder(scalar_context)
        fusion_vec = torch.cat([pulse_encoded, context_encoded, scalar_context], dim=1)
        fusion_vec = self.fusion_norm(fusion_vec)
        fusion_vec = self.fusion_se(fusion_vec)
        decoded = self.decoder(fusion_vec)
        decoded = self.post_decoder_block(decoded)
        raw_output = self.output_head(decoded)
        return self._decode_actions_from_logits(raw_output, context_features)
