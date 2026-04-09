import logging
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from common.metrics import injury_risk
from common.settings import (
    CONTINUOUS_INDICES,
    DISCRETE_INDICES,
    FEATURE_ORDER,
    INJURY_PREDICT_DIR,
    PULSE_PREDICT_DIR,
    WAVEFORM_LENGTH,
)
from InjuryPredict.utils.models import InjuryPredictModel
from PulsePredict.model.model import HybridPulseCNN

from ARS_optim.src.constraints import ConstraintEngine
from ARS_optim.src.distribution_penalty import DistributionPenalty
from ARS_optim.src.param_manager import ParamManager

LOGGER = logging.getLogger(__name__)

def _resolve_checkpoint_path(base_dir: Path, cfg_value: str) -> Path:
    """把相对或绝对权重路径统一解析为绝对路径。"""
    raw = str(cfg_value or "").strip()
    candidate = Path(raw).expanduser()
    return candidate if candidate.is_absolute() else (base_dir / candidate).resolve()


def load_surrogate_models(
    config: dict,
    device: torch.device,
) -> Tuple[HybridPulseCNN, InjuryPredictModel]:
    """按子项目原有保存格式重建 PulsePredict 与 InjuryPredict。

    作用：
    - 读取 `surrogate` 配置中声明的权重路径；
    - 严格按照各子项目保存产物中已有的结构信息重建模型实例；
    - 加载权重并返回可直接推理的模型对象。

    注意：
    - 这里只接受子项目正式保存产物，不猜测缺失配置；
    - 若 checkpoint 同目录缺少必需的 `config.json` 或 `TrainingRecord.json`，直接报错。
    """
    surrogate_cfg = config.get("surrogate", {})

    pulse_ckpt_path = _resolve_checkpoint_path(
        Path(PULSE_PREDICT_DIR),
        surrogate_cfg.get("pulse_ckpt_rel_path", ""),
    )
    if not pulse_ckpt_path.is_file():
        raise FileNotFoundError(f"pulse model checkpoint not found: {pulse_ckpt_path}")
    pulse_config_path = pulse_ckpt_path.parent / "config.json"
    if not pulse_config_path.is_file():
        raise FileNotFoundError(
            f"Pulse checkpoint 同目录缺少 config.json: {pulse_config_path}"
        )
    with open(pulse_config_path, "r", encoding="utf-8") as file:
        pulse_saved_cfg = json.load(file)
    pulse_arch = pulse_saved_cfg.get("arch", {})
    if pulse_arch.get("type") != "HybridPulseCNN":
        raise ValueError(f"Pulse arch type 非 HybridPulseCNN: {pulse_arch.get('type')}")
    pulse_model = HybridPulseCNN(**pulse_arch.get("args", {})).to(device)
    pulse_ckpt = torch.load(str(pulse_ckpt_path), map_location=device, weights_only=False)
    pulse_model.load_state_dict(pulse_ckpt["state_dict"])
    LOGGER.info("Loaded PulsePredict checkpoint from %s", pulse_ckpt_path)

    injury_ckpt_path = _resolve_checkpoint_path(
        Path(INJURY_PREDICT_DIR),
        surrogate_cfg.get("injury_ckpt_rel_path", ""),
    )
    if not injury_ckpt_path.is_file():
        raise FileNotFoundError(f"injury model checkpoint not found: {injury_ckpt_path}")
    injury_training_record_path = injury_ckpt_path.parent / "TrainingRecord.json"
    if not injury_training_record_path.is_file():
        raise FileNotFoundError(
            f"Injury checkpoint 同目录缺少 TrainingRecord.json: {injury_training_record_path}"
        )
    with open(injury_training_record_path, "r", encoding="utf-8") as file:
        training_record = json.load(file)
    model_args = training_record.get("hyperparameters", {}).get("model")
    if model_args is None:
        raise ValueError(
            f"TrainingRecord.json 缺少 hyperparameters.model: {injury_training_record_path}"
        )
    injury_model = InjuryPredictModel(**model_args).to(device)
    injury_model.load_state_dict(
        torch.load(str(injury_ckpt_path), map_location=device, weights_only=False)
    )
    LOGGER.info("Loaded InjuryPredict checkpoint from %s", injury_ckpt_path)
    return pulse_model, injury_model


class SurrogateAdapter(nn.Module):
    """统一封装波形生成、损伤预测与目标函数计算。

    作用：
    - 负责把 `context + trainable control` 映射到代理模型需要的输入形式；
    - 调用 PulsePredict 生成归一化波形；
    - 调用 InjuryPredict 输出损伤标量，并进一步拆出风险项、约束惩罚和分布偏离惩罚。

    注意：
    - 这里不训练代理模型，只做冻结推理与目标值计算；
    - 输入归一化始终以根目录共享的 `normalization_config.json` 为准。
    """

    def __init__(
        self,
        pulse_model: nn.Module,
        injury_model: nn.Module,
        param_manager: ParamManager,
        constraint_engine: ConstraintEngine,
        config: dict,
        data_processor,
    ):
        super().__init__()
        self.pulse_model = pulse_model
        self.injury_model = injury_model
        self.param_manager = param_manager
        self.constraint_engine = constraint_engine
        self.data_processor = data_processor

        if not self.data_processor.load_config():
            raise RuntimeError("无法加载 normalization_config.json")
        self.data_processor.validate_config(raise_on_error=True)

        self.pulse_model.eval()
        self.injury_model.eval()
        for param in self.pulse_model.parameters():
            param.requires_grad = False
        for param in self.injury_model.parameters():
            param.requires_grad = False

        obj_cfg = config.get("optimization", {}).get("objectives", {})
        self.w_head = float(obj_cfg.get("weight_hic", 1.0))
        self.w_chest = float(obj_cfg.get("weight_dmax", 1.0))
        self.w_neck = float(obj_cfg.get("weight_nij", 1.0))
        self.weight_penalty = float(obj_cfg.get("weight_penalty", 10.0))
        self.distribution_penalty = DistributionPenalty(config)
        self.weight_distribution = float(
            config.get("optimization", {})
            .get("distribution_penalty", {})
            .get("weight", 0.0)
        )

        # 波形代理只依赖三项碰撞工况。把索引缓存下来，避免高频前向时重复查表。
        self._impact_names = ["impact_velocity", "impact_angle", "overlap"]
        self._impact_indices = [FEATURE_ORDER.index(name) for name in self._impact_names]
        self._ot_index = self.param_manager.get_param("OT")["index"]

    def fit_distribution_reference(self, reference_features: torch.Tensor) -> None:
        """拟合分布偏离惩罚所需的参考统计量。

        输入：
        - `reference_features`: 训练经验池中的完整特征张量。

        注意：
        - 仅当分布偏离惩罚启用时才真正拟合；
        - 该统计量对应经验分布约束，不负责显式可行域约束。
        """
        if self.distribution_penalty.enabled:
            self.distribution_penalty.fit(reference_features)

    def _prepare_normalized_inputs(
        self,
        context_params: torch.Tensor,
        control_trainable: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """拼回完整特征，并映射到代理模型使用的归一化空间。

        作用：
        - 将 `context_params` 与 `control_trainable` 拼成完整 `FEATURE_ORDER`；
        - 基于共享归一化配置生成代理模型前向所需的归一化输入。

        输出：
        - `combined_phys`: 完整物理尺度特征；
        - `model_input_norm`: 与 `combined_phys` 列顺序一致的归一化特征。
        """
        combined_phys = self.constraint_engine.compose_full_features(
            context_params,
            control_trainable,
        )
        model_input_norm = self.data_processor.process_by_name(
            values=combined_phys,
            feature_names=FEATURE_ORDER,
            inverse=False,
        )
        return combined_phys, model_input_norm

    def predict_injury_and_loss(
        self,
        context_params: torch.Tensor,
        control_trainable: torch.Tensor,
        pulse_norm: torch.Tensor,
        include_opt_bounds: bool = False,
        detach_info: bool = True,
        penalty_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """基于代理链路计算逐样本损伤预测与目标函数分项。

        作用：
        - 将当前候选解送入 InjuryPredict，得到物理尺度下的 `HIC/Dmax/Nij`；
        - 将损伤标量映射为部位风险概率与联合风险；
        - 进一步拆出 `loss_risk / loss_constraint / loss_distribution`。

        输入：
        - `context_params`: 当前批次的 `context` 特征，列顺序与 `ParamManager.get_context_names()` 一致。
        - `control_trainable`: 当前候选 trainable control，列顺序与 `ParamManager.get_trainable_names()` 一致。
        - `pulse_norm`: PulsePredict 生成的归一化二维波形，直接作为 InjuryPredict 的波形输入。
        - `include_opt_bounds`: 是否把 `opt_min/opt_max` 也纳入反向约束软惩罚。
        - `detach_info`: 是否对返回的分项指标做 `detach()`。
        - `penalty_features`: 若显式提供，则约束软惩罚作用在这份完整特征上；通常用于惩罚投影前原始游离解。

        输出：
        - `total_loss`: 逐样本总目标值。
        - `predictions_phys`: 物理尺度下的 `[HIC15, Dmax, Nij]`。
        - `info`: 逐样本分项指标字典，供训练日志、评估导出和 trace 使用。

        注意：
        - `loss_risk` 是训练/精调使用的加权联合风险目标；
        - `joint_risk` 是评估导出使用的无权重联合风险口径。
        """
        # `combined_phys` 保持完整 `FEATURE_ORDER` 的物理尺度；
        # `model_input_norm` 与其列顺序一致，只是数值已归一化。
        combined_phys, model_input_norm = self._prepare_normalized_inputs(
            context_params,
            control_trainable,
        )
        # InjuryPredict 的标量输入拆分统一依照 `common.settings` 中基于 `FEATURE_ORDER` 的索引定义。
        x_att_continuous = model_input_norm[:, CONTINUOUS_INDICES]
        x_att_discrete = model_input_norm[:, DISCRETE_INDICES].to(torch.long)

        predictions_phys, _, _ = self.injury_model(
            pulse_norm,
            x_att_continuous,
            x_att_discrete,
        )
        hic15 = predictions_phys[:, 0]
        dmax = predictions_phys[:, 1]
        nij = predictions_phys[:, 2]
        ot_tensor = combined_phys[:, self._ot_index]

        # 先将三个部位的代理输出映射为风险概率，再构造优化目标和评估指标。
        p_head = torch.clamp(injury_risk.Injury_prob_cal_head(hic15), 1e-6, 1.0 - 1e-6)
        p_chest = torch.clamp(
            injury_risk.Injury_prob_cal_chest(dmax, OT=ot_tensor),
            1e-6,
            1.0 - 1e-6,
        )
        p_neck = torch.clamp(injury_risk.Injury_prob_cal_neck(nij), 1e-6, 1.0 - 1e-6)

        # `loss_risk` 用于训练和精调；`joint_risk` 用于评估导出。
        loss_risk = 1.0 - (
            torch.pow(1.0 - p_head, self.w_head)
            * torch.pow(1.0 - p_chest, self.w_chest)
            * torch.pow(1.0 - p_neck, self.w_neck)
        )
        joint_risk = 1.0 - ((1.0 - p_head) * (1.0 - p_chest) * (1.0 - p_neck))

        # 默认惩罚当前完整特征；若上游传入 `penalty_features`，则改为惩罚投影前原始游离解。
        penalty_source = combined_phys if penalty_features is None else penalty_features
        loss_constraint = self.constraint_engine.compute_soft_penalty(
            penalty_source,
            include_opt_bounds=include_opt_bounds,
        )

        # 分布偏离惩罚只回答“当前合法解是否偏离经验分布”，不负责处理可行域越界。
        loss_distribution = self.distribution_penalty.compute(
            context_params,
            control_trainable,
        )
        total_loss = (
            loss_risk
            + self.weight_penalty * loss_constraint
            + self.weight_distribution * loss_distribution
        )

        if detach_info:
            info = {
                "loss_risk": loss_risk.detach(),
                "loss_constraint": loss_constraint.detach(),
                "loss_distribution": loss_distribution.detach(),
                "p_head": p_head.detach(),
                "p_chest": p_chest.detach(),
                "p_neck": p_neck.detach(),
                "joint_risk": joint_risk.detach(),
                "hic15": hic15.detach(),
                "dmax": dmax.detach(),
                "nij": nij.detach(),
            }
        else:
            info = {
                "loss_risk": loss_risk,
                "loss_constraint": loss_constraint,
                "loss_distribution": loss_distribution,
                "p_head": p_head,
                "p_chest": p_chest,
                "p_neck": p_neck,
                "joint_risk": joint_risk,
                "hic15": hic15,
                "dmax": dmax,
                "nij": nij,
            }
        return total_loss, predictions_phys, info

    def generate_pulse(self, context_params: torch.Tensor) -> torch.Tensor:
        """基于 `context` 生成归一化二维碰撞波形。

        作用：
        - 从完整特征中抽取 PulsePredict 真正依赖的三项碰撞工况；
        - 经过共享归一化后送入 PulsePredict；
        - 输出可直接供 InjuryPredict 使用的二维归一化波形。

        注意：
        - PulsePredict 只依赖 `impact_velocity / impact_angle / overlap`；
        - 返回值形状固定为 `[batch, 2, WAVEFORM_LENGTH]`。
        """
        # `context` 不是完整 `FEATURE_ORDER`，需要先补齐再抽取三项碰撞工况。
        full = self.constraint_engine.compose_full_features(context_params=context_params)
        impact_phys = full[:, self._impact_indices]
        impact_norm = self.data_processor.process_by_name(
            impact_phys,
            self._impact_names,
            inverse=False,
        )
        pulse_output_raw = self.pulse_model(impact_norm)
        waveform_norm = self.pulse_model.get_metrics_output(pulse_output_raw)
        pulse_xy = waveform_norm[:, 0:2, :]
        if (
            pulse_xy.dim() != 3
            or pulse_xy.size(1) != 2
            or pulse_xy.size(2) != WAVEFORM_LENGTH
        ):
            raise ValueError(f"generate_pulse 输出形状异常: {tuple(pulse_xy.shape)}")
        return pulse_xy
