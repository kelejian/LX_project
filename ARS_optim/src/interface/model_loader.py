import json
from pathlib import Path
from typing import Tuple

import torch

from common.settings import PULSE_PREDICT_DIR, INJURY_PREDICT_DIR
from PulsePredict.model.model import HybridPulseCNN
from InjuryPredict.utils.models import InjuryPredictModel
from ARS_optim.src.utils.path_utils import resolve_checkpoint_path


def load_surrogate_models(config: dict, device: torch.device) -> Tuple[HybridPulseCNN, InjuryPredictModel]:
    """严格加载 PulsePredict 与 InjuryPredict 的已训练模型。

    设计原则：
    - Pulse 模型结构仅从 checkpoint 同目录的 config.json 读取；
    - Injury 模型结构仅从 checkpoint 同目录的 TrainingRecord.json 读取；
    """
    surrogate_cfg = config.get('surrogate', {})

    # ---------------- PulsePredict ----------------
    pulse_ckpt = resolve_checkpoint_path(Path(PULSE_PREDICT_DIR), surrogate_cfg.get('pulse_checkpoint', ''))
    pulse_ckpt_path = Path(pulse_ckpt)
    if not pulse_ckpt_path.is_file():
        raise FileNotFoundError(f"pulse model checkpoint not found: {pulse_ckpt_path}")

    pulse_config_path = pulse_ckpt_path.parent / 'config.json'
    if not pulse_config_path.is_file():
        raise FileNotFoundError(
            f"Pulse checkpoint 同目录缺少 config.json: {pulse_config_path}"
        )

    with open(pulse_config_path, 'r', encoding='utf-8') as f:
        pulse_saved_cfg = json.load(f)

    pulse_arch = pulse_saved_cfg.get('arch', {})
    pulse_arch_type = pulse_arch.get('type')
    pulse_arch_args = pulse_arch.get('args', {})
    if pulse_arch_type != 'HybridPulseCNN':
        raise ValueError(f"Pulse arch type 非 HybridPulseCNN: {pulse_arch_type}")

    pulse_model = HybridPulseCNN(**pulse_arch_args).to(device)
    pulse_ckpt_obj = torch.load(str(pulse_ckpt_path), map_location=device, weights_only=False)
    if not isinstance(pulse_ckpt_obj, dict) or 'state_dict' not in pulse_ckpt_obj:
        raise ValueError(f"Pulse checkpoint 格式错误，期望包含 state_dict: {pulse_ckpt_path}")
    pulse_model.load_state_dict(pulse_ckpt_obj['state_dict'])

    # ---------------- InjuryPredict ----------------
    inj_ckpt = resolve_checkpoint_path(Path(INJURY_PREDICT_DIR), surrogate_cfg.get('checkpoint_rel_path', ''))
    inj_ckpt_path = Path(inj_ckpt)
    if not inj_ckpt_path.is_file():
        raise FileNotFoundError(f"injury model checkpoint not found: {inj_ckpt_path}")

    injury_record_path = inj_ckpt_path.parent / 'TrainingRecord.json'
    if not injury_record_path.is_file():
        raise FileNotFoundError(
            f"Injury checkpoint 同目录缺少 TrainingRecord.json: {injury_record_path}"
        )

    with open(injury_record_path, 'r', encoding='utf-8') as f:
        injury_record = json.load(f)

    injury_model_params = injury_record.get('hyperparameters', {}).get('model', None)
    if injury_model_params is None:
        raise ValueError(f"TrainingRecord.json 缺少 hyperparameters.model: {injury_record_path}")

    injury_model = InjuryPredictModel(**injury_model_params).to(device)
    injury_state_dict = torch.load(str(inj_ckpt_path), map_location=device, weights_only=False)
    injury_model.load_state_dict(injury_state_dict)

    return pulse_model, injury_model
