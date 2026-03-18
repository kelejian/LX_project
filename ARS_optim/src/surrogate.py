import json
from pathlib import Path
from typing import Dict, Optional, Tuple

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
    """把配置中的相对/绝对权重路径统一解析成绝对路径。"""
    raw = str(cfg_value or "").strip()
    candidate = Path(raw).expanduser()
    return candidate if candidate.is_absolute() else (base_dir / candidate).resolve()


def load_surrogate_models(config: dict, device: torch.device) -> Tuple[HybridPulseCNN, InjuryPredictModel]:
    """按各自子项目保存格式重建 PulsePredict 与 InjuryPredict。

    这里不在 ARS_optim 内另造一套模型配置协议，而是直接复用：
    - PulsePredict checkpoint 同目录下的 config.json；
    - InjuryPredict checkpoint 同目录下的 TrainingRecord.json。

    这样模型结构的真源仍然留在各自子项目中，ARS_optim 只负责装配与调用。
    """
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
    """封装波形生成、损伤预测与优化目标计算。

    这里的职责边界是：
    - 负责把 context/control 组织成代理模型真正需要的输入；
    - 负责计算逐样本损伤风险和优化目标；
    - 不负责动作合法化。动作是否合法由 ConstraintEngine 或 LocalRefiner 决定。
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
        self.weight_distribution = float(config.get("optimization", {}).get("distribution_penalty", {}).get("weight", 0.0))
        # impact 参数在完整 FEATURE_ORDER 中的索引固定不变，初始化时缓存可避免高频推理阶段重复查找。
        self._impact_names = ["impact_velocity", "impact_angle", "overlap"]
        self._impact_indices = [FEATURE_ORDER.index(name) for name in self._impact_names]
        self._ot_index = self.param_manager.get_param("OT")["index"]

    def fit_distribution_reference(self, reference_features: torch.Tensor) -> None:
        """拟合分布偏离惩罚所需的训练参考统计量。"""
        if self.distribution_penalty.enabled:
            self.distribution_penalty.fit(reference_features)

    def _prepare_normalized_inputs(self, context_params: torch.Tensor, control_trainable: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """把 context/trainable 重新拼成完整特征，并映射到代理模型需要的归一化空间。

        PulsePredict 和 InjuryPredict 依赖的是全量 FEATURE_ORDER 输入；
        ARS_optim 内部为了拆清职责，常常把 context 与 trainable 分开传递。
        因此这里集中做一次“物理空间拼接 -> 统一归一化”，避免每个调用点重复维护同样的列对齐逻辑。
        """
        combined_phys = self.constraint_engine.compose_full_features(context_params, control_trainable)
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
        # 这里返回逐样本 loss 而不是 batch mean，原因是局部精调和评估表都需要保留样本粒度；
        # batch 聚合只在更外层训练循环里完成，避免优化器和评估脚本各自再拆一次总损失。
        combined_phys, model_input_norm = self._prepare_normalized_inputs(context_params, control_trainable)
        x_att_continuous = model_input_norm[:, CONTINUOUS_INDICES]
        x_att_discrete = model_input_norm[:, DISCRETE_INDICES].to(torch.long)

        predictions_phys, _, _ = self.injury_model(pulse_norm, x_att_continuous, x_att_discrete)
        hic15 = predictions_phys[:, 0]
        dmax = predictions_phys[:, 1]
        nij = predictions_phys[:, 2]
        ot_tensor = combined_phys[:, self._ot_index]

        # 代理模型输出已经是物理尺度的损伤值；这里再依据 common.metrics.injury_risk
        # 中的风险曲线把各部位损伤值映射为 AIS3+ 概率，供优化目标与评估汇总共同使用。
        p_head = torch.clamp(injury_risk.Injury_prob_cal_head(hic15), 1e-6, 1.0 - 1e-6)
        p_chest = torch.clamp(injury_risk.Injury_prob_cal_chest(dmax, OT=ot_tensor), 1e-6, 1.0 - 1e-6)
        p_neck = torch.clamp(injury_risk.Injury_prob_cal_neck(nij), 1e-6, 1.0 - 1e-6)

        # loss_risk 是训练/精调时真正回传梯度的目标函数：
        #   loss_risk = 1 - Π_k (1 - p_k)^{w_k}
        # 其中 w_k 用来调节头/胸/颈三部分在优化时的相对重要性。
        # joint_risk 则是无权重版本：
        #   joint_risk = 1 - Π_k (1 - p_k)
        # 它只作为评估报告与结果表中的统一指标，不参与额外加权。
        # 当所有权重均为 1.0 时，loss_risk 与 joint_risk 在数值上完全相同。
        loss_risk = 1.0 - (
            torch.pow(1.0 - p_head, self.w_head)
            * torch.pow(1.0 - p_chest, self.w_chest)
            * torch.pow(1.0 - p_neck, self.w_neck)
        )
        joint_risk = 1.0 - ((1.0 - p_head) * (1.0 - p_chest) * (1.0 - p_neck))

        # 默认在当前送入代理模型的物理特征上计算软惩罚。
        # 训练策略网络时会额外传入 penalty_features=投影前特征，用于在前向投影截断违约量之后，仍保留把动作拉回合法域的梯度补偿。
        penalty_source = combined_phys if penalty_features is None else penalty_features
        loss_constraint = self.constraint_engine.compute_soft_penalty(
            penalty_source,
            include_opt_bounds=include_opt_bounds,
        )
        # 分布偏离惩罚独立于显式物理约束：
        # 即使当前动作已经合法，也可能落在训练经验池稀疏甚至未覆盖的区域。
        # 因此它单独作为一个软项存在，而不是合并进 ConstraintEngine。
        loss_distribution = self.distribution_penalty.compute(context_params, control_trainable)
        total_loss = loss_risk + self.weight_penalty * loss_constraint + self.weight_distribution * loss_distribution

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
        """基于 context 里的碰撞工况生成归一化后的 XY 波形。

        这里显式只取 impact_velocity / impact_angle / overlap 三个工况参数，
        再走与 PulsePredict 推理一致的归一化与 `get_metrics_output` 提取流程。
        输出保持归一化后的 XY 两通道，直接供 InjuryPredict 与策略网络复用。
        """
        # context_params 只包含 context 子集，列顺序并不等于 FEATURE_ORDER。
        # 这里先补全为完整特征向量，再按 FEATURE_ORDER 中的固定索引抽取 impact 参数，
        # 目的是确保送入 PulsePredict 的速度/角度/重叠率与全项目统一特征接口保持一致。
        full = self.constraint_engine.compose_full_features(context_params=context_params)
        impact_phys = full[:, self._impact_indices]
        impact_norm = self.data_processor.process_by_name(impact_phys, self._impact_names, inverse=False)
        pulse_output_raw = self.pulse_model(impact_norm)
        waveform_norm = self.pulse_model.get_metrics_output(pulse_output_raw)
        pulse_xy = waveform_norm[:, 0:2, :]
        if pulse_xy.dim() != 3 or pulse_xy.size(1) != 2 or pulse_xy.size(2) != WAVEFORM_LENGTH:
            raise ValueError(f"generate_pulse 输出形状异常: {tuple(pulse_xy.shape)}")
        return pulse_xy

