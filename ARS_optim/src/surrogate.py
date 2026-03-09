import json
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn

from common.metrics import injury_risk
from common.settings import CONTINUOUS_INDICES, DISCRETE_INDICES, FEATURE_ORDER, INJURY_PREDICT_DIR, PULSE_PREDICT_DIR, WAVEFORM_LENGTH
from InjuryPredict.utils.models import InjuryPredictModel
from PulsePredict.model.model import HybridPulseCNN

from ARS_optim.src.constraints import ConstraintEngine
from ARS_optim.src.distribution_penalty import DistributionPenalty
from ARS_optim.src.param_manager import ParamManager


def _resolve_checkpoint_path(base_dir: Path, cfg_value: str) -> Path:
    raw = str(cfg_value or "").strip()
    candidate = Path(raw).expanduser()
    return candidate if candidate.is_absolute() else (base_dir / candidate).resolve()


def load_surrogate_models(config: dict, device: torch.device) -> Tuple[HybridPulseCNN, InjuryPredictModel]:
    surrogate_cfg = config.get("surrogate", {})

    pulse_ckpt_path = _resolve_checkpoint_path(Path(PULSE_PREDICT_DIR), surrogate_cfg.get("pulse_checkpoint", ""))
    if not pulse_ckpt_path.is_file():
        raise FileNotFoundError(f"pulse model checkpoint not found: {pulse_ckpt_path}")
    pulse_config_path = pulse_ckpt_path.parent / "config.json"
    if not pulse_config_path.is_file():
        raise FileNotFoundError(f"Pulse checkpoint 同目录缺少 config.json: {pulse_config_path}")
    with open(pulse_config_path, "r", encoding="utf-8") as file:
        pulse_saved_cfg = json.load(file)
    pulse_arch = pulse_saved_cfg.get("arch", {})
    if pulse_arch.get("type") != "HybridPulseCNN":
        raise ValueError(f"Pulse arch type 非 HybridPulseCNN: {pulse_arch.get('type')}")
    pulse_model = HybridPulseCNN(**pulse_arch.get("args", {})).to(device)
    pulse_ckpt = torch.load(str(pulse_ckpt_path), map_location=device, weights_only=False)
    pulse_model.load_state_dict(pulse_ckpt["state_dict"])

    injury_ckpt_path = _resolve_checkpoint_path(Path(INJURY_PREDICT_DIR), surrogate_cfg.get("checkpoint_rel_path", ""))
    if not injury_ckpt_path.is_file():
        raise FileNotFoundError(f"injury model checkpoint not found: {injury_ckpt_path}")
    training_record_path = injury_ckpt_path.parent / "TrainingRecord.json"
    if not training_record_path.is_file():
        raise FileNotFoundError(f"Injury checkpoint 同目录缺少 TrainingRecord.json: {training_record_path}")
    with open(training_record_path, "r", encoding="utf-8") as file:
        training_record = json.load(file)
    model_args = training_record.get("hyperparameters", {}).get("model")
    if model_args is None:
        raise ValueError(f"TrainingRecord.json 缺少 hyperparameters.model: {training_record_path}")
    injury_model = InjuryPredictModel(**model_args).to(device)
    injury_model.load_state_dict(torch.load(str(injury_ckpt_path), map_location=device, weights_only=False))
    return pulse_model, injury_model


class SurrogateAdapter(nn.Module):
    """封装波形预测、损伤预测与寻优目标计算。"""

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
        self.weight_distribution = float(config.get("optimization", {}).get("distribution_penalty", {}).get("weight", 0.0))
        # impact 参数在完整 FEATURE_ORDER 中的索引固定不变，初始化时缓存可避免高频推理阶段重复查找。
        self._impact_names = ["impact_velocity", "impact_angle", "overlap"]
        self._impact_indices = [FEATURE_ORDER.index(name) for name in self._impact_names]

    def fit_distribution_reference(self, reference_features: torch.Tensor) -> None:
        if self.distribution_penalty.enabled:
            self.distribution_penalty.fit(reference_features)

    def _prepare_normalized_inputs(self, context_params: torch.Tensor, control_trainable: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        combined_phys = self.constraint_engine.compose_full_features(context_params, control_trainable)
        model_input_norm = self.data_processor.process_by_name(
            values=combined_phys,
            feature_names=FEATURE_ORDER,
            inverse=False,
        )
        return combined_phys, model_input_norm

    def predict_injury_and_loss(self, context_params: torch.Tensor, control_trainable: torch.Tensor, pulse_norm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        combined_phys, model_input_norm = self._prepare_normalized_inputs(context_params, control_trainable)
        x_att_continuous = model_input_norm[:, CONTINUOUS_INDICES]
        x_att_discrete = model_input_norm[:, DISCRETE_INDICES].to(torch.long)

        predictions_phys, _, _ = self.injury_model(pulse_norm, x_att_continuous, x_att_discrete)
        hic15 = predictions_phys[:, 0]
        dmax = predictions_phys[:, 1]
        nij = predictions_phys[:, 2]
        ot_tensor = combined_phys[:, self.param_manager.params_by_name["OT"]["index"]]

        p_head = torch.clamp(injury_risk.Injury_prob_cal_head(hic15), 1e-6, 1.0 - 1e-6)
        p_chest = torch.clamp(injury_risk.Injury_prob_cal_chest(dmax, OT=ot_tensor), 1e-6, 1.0 - 1e-6)
        p_neck = torch.clamp(injury_risk.Injury_prob_cal_neck(nij), 1e-6, 1.0 - 1e-6)

        # 这里对应 ARS.md 步骤三中的联合损伤风险项。训练时允许对头/胸/颈三部分别加权，以便在不改变整体乘法结构的前提下调节不同部位在优化目标中的相对重要性。
        loss_risk = 1.0 - (
            torch.pow(1.0 - p_head, self.w_head)
            * torch.pow(1.0 - p_chest, self.w_chest)
            * torch.pow(1.0 - p_neck, self.w_neck)
        )
        joint_risk = 1.0 - ((1.0 - p_head) * (1.0 - p_chest) * (1.0 - p_neck))

        loss_constraint = self.constraint_engine.compute_soft_penalty(control_trainable, context_params)
        loss_distribution = self.distribution_penalty.compute(context_params, control_trainable)
        total_loss = loss_risk + self.weight_penalty * loss_constraint + self.weight_distribution * loss_distribution

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
        return total_loss, predictions_phys, info

    def generate_pulse(self, context_params: torch.Tensor) -> torch.Tensor:
        # context_params 只包含 context 子集，列顺序并不等于 FEATURE_ORDER。先补全到完整特征向量，再按完整索引抽取 impact 参数，才能保证送入 PulsePredict 的速度/角度/重叠率与全项目统一数据接口严格对齐。
        full = self.constraint_engine.compose_full_features(context_params=context_params)
        impact_phys = full[:, self._impact_indices]
        impact_norm = self.data_processor.process_by_name(impact_phys, self._impact_names, inverse=False)
        pulse_output_raw = self.pulse_model(impact_norm)
        if hasattr(self.pulse_model, "get_metrics_output"):
            waveform_norm = self.pulse_model.get_metrics_output(pulse_output_raw)
        else:
            waveform_norm = pulse_output_raw[-1][0] if isinstance(pulse_output_raw, (list, tuple)) else pulse_output_raw
        pulse_xy = waveform_norm[:, 0:2, :]
        if pulse_xy.dim() != 3 or pulse_xy.size(1) != 2 or pulse_xy.size(2) != WAVEFORM_LENGTH:
            raise ValueError(f"generate_pulse 输出形状异常: {tuple(pulse_xy.shape)}")
        return pulse_xy

    def forward(self, context_params: torch.Tensor, control_trainable: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        pulse_norm = self.generate_pulse(context_params)
        return self.predict_injury_and_loss(context_params, control_trainable, pulse_norm)
