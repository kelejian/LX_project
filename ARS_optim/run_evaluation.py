import argparse
import hashlib
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

from common.data_utils.processor import UnifiedDataProcessor
from common.metrics.injury_risk import (
    AIS_cal_chest,
    AIS_cal_head,
    AIS_cal_neck,
)
from common.settings import FEATURE_ORDER, NORMALIZATION_CONFIG_PATH, RAW_DATA_DIR, SPLIT_INDICES_DIR

from ARS_optim.src.core.constraints import PhysicalConstraintManager
from ARS_optim.src.core.optimizer import ARSLocalOptimizer
from ARS_optim.src.core.param_manager import ParamManager
from ARS_optim.src.core.rule_engine import RuleEngine
from ARS_optim.src.interface.data_loader import StateDataLoaderManager
from ARS_optim.src.interface.model_loader import load_surrogate_models
from ARS_optim.src.interface.surrogate_adapter import SurrogateAdapter
from ARS_optim.src.models.strategy_net import StrategyNet
from ARS_optim.src.utils.logger import setup_logger as setup_ars_logger
from ARS_optim.src.utils.metrics import MetricsTracker


def parse_args():
    parser = argparse.ArgumentParser(description="ARS Local Refinement Evaluator")
    parser.add_argument(
        "--input_csv",
        type=str,
        default=None,
        help="可选：输入工况参数CSV。若不提供，则自动使用 injury_test_indices 对应的测试集工况。",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="evaluation_results.csv",
        help="输出 CSV 文件名或路径。默认在本次 saved_eval 子目录下保存。",
    )
    parser.add_argument(
        "--strategy_ckpt",
        type=str,
        default=None,
        help="可选：策略网络权重文件路径。",
    )
    parser.add_argument(
        "--direct_inference",
        action="store_true",
        help="启用策略网络直推（覆盖配置中的 direct_inference）。",
    )
    return parser.parse_args()


def _safe_stem(text: str) -> str:
    stem = Path(text).stem
    valid = []
    for ch in stem:
        if ch.isalnum() or ch in ["_", "-"]:
            valid.append(ch)
        else:
            valid.append("_")
    name = "".join(valid).strip("_")
    return name if name else "evaluation"


def _build_output_dir(base_dir: Path, input_csv: Optional[str]) -> Path:
    ts = datetime.now().strftime("%m%d_%H%M%S")
    if input_csv:
        prefix = f"eval_{_safe_stem(input_csv)}"
    else:
        prefix = "eval_injury_test_split"
    out_dir = base_dir / "saved_eval" / f"{prefix}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir


def _resolve_output_csv_path(output_arg: str, output_dir: Path) -> Path:
    arg_path = Path(output_arg)
    filename = arg_path.name if arg_path.name else "evaluation_results.csv"
    return output_dir / filename


def _load_default_val_best_ckpt(base_dir: Path) -> Optional[Path]:
    saved_models_dir = base_dir / "saved_models"
    if not saved_models_dir.exists():
        return None

    runs = [d for d in saved_models_dir.iterdir() if d.is_dir() and d.name.startswith("strategy_net_")]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    ckpt_priority = ["val_best_model.pth"]
    for run_dir in runs:
        for ckpt_name in ckpt_priority:
            ckpt = run_dir / ckpt_name
            if ckpt.is_file():
                return ckpt
    return None


def _load_default_ckpt_with_fallback(base_dir: Path) -> Optional[Path]:
    saved_models_dir = base_dir / "saved_models"
    if not saved_models_dir.exists():
        return None

    runs = [d for d in saved_models_dir.iterdir() if d.is_dir() and d.name.startswith("strategy_net_")]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    # 非严格模式下：优先 val_best；若缺失则回退 train_best/final。
    ckpt_priority = ["val_best_model.pth", "train_best_model.pth", "final_model.pth"]
    for run_dir in runs:
        for ckpt_name in ckpt_priority:
            ckpt = run_dir / ckpt_name
            if ckpt.is_file():
                return ckpt
    return None


def _resolve_default_strategy_ckpt(base_dir: Path, config: Dict) -> Optional[Path]:
    """解析默认策略权重路径。

    优先级：
    1. evaluation.default_strategy_ckpt（支持绝对/相对路径）；
    2. strict_default_ckpt=true 时：仅自动查找 val_best_model.pth；
    3. strict_default_ckpt=false 时：自动查找 val/train/final 回退。
    """
    eval_cfg = config.get("evaluation", {}) or {}
    cfg_path = eval_cfg.get("default_strategy_ckpt", None)
    strict_default_ckpt = bool(eval_cfg.get("strict_default_ckpt", False))
    if cfg_path:
        p = Path(str(cfg_path))
        if not p.is_absolute():
            p = (base_dir / p).resolve()
        return p
    if strict_default_ckpt:
        return _load_default_val_best_ckpt(base_dir)
    return _load_default_ckpt_with_fallback(base_dir)


def _snapshot_configs(cfg_path: Path, param_space_path: Path, output_dir: Path) -> Dict[str, str]:
    cfg_snapshot = output_dir / "config_used.yaml"
    param_snapshot = output_dir / "param_space_used.yaml"
    shutil.copy2(str(cfg_path), str(cfg_snapshot))
    shutil.copy2(str(param_space_path), str(param_snapshot))
    return {
        "saved_default_config_snapshot": str(cfg_snapshot),
        "saved_param_space_snapshot": str(param_snapshot),
    }


def _sha1(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()


def _build_context_dataframe(df_input: pd.DataFrame, param_manager: ParamManager, logger) -> Tuple[pd.DataFrame, List[str]]:
    context_params = param_manager.get_context_params()
    context_df = pd.DataFrame(index=df_input.index)
    missing = []

    for p in context_params:
        name = p["name"]
        if name in df_input.columns:
            context_df[name] = pd.to_numeric(df_input[name], errors="raise")
        else:
            if "default" not in p:
                raise ValueError(f"输入缺失 context 参数 '{name}'，且未配置 default。")
            context_df[name] = float(p["default"])
            missing.append(name)

    if missing:
        logger.warning(f"输入缺失部分 context 参数，已回退 default: {missing}")
    return context_df, missing


def _build_baseline_trainable_dataframe(df_input: pd.DataFrame, param_manager: ParamManager, logger) -> Tuple[pd.DataFrame, List[str]]:
    trainable_params = param_manager.control_trainable_params
    trainable_df = pd.DataFrame(index=df_input.index)
    missing = []

    for p in trainable_params:
        name = p["name"]
        if name in df_input.columns:
            trainable_df[name] = pd.to_numeric(df_input[name], errors="raise")
        else:
            trainable_df[name] = float(p["default"])
            missing.append(name)

    if missing:
        logger.warning(f"输入未提供部分可调参数，baseline 已回退 default: {missing}")
    return trainable_df, missing


def _compute_joint_risk(p_head: np.ndarray, p_chest: np.ndarray, p_neck: np.ndarray) -> np.ndarray:
    return 1.0 - (1.0 - p_head) * (1.0 - p_chest) * (1.0 - p_neck)


def _write_yaml(path: Path, data: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def _compute_predictions_batch(
    context_tensor: torch.Tensor,
    baseline_trainable: torch.Tensor,
    surrogate: SurrogateAdapter,
    optimizer: ARSLocalOptimizer,
    rule_engine: RuleEngine,
    eval_batch_size: int,
) -> Dict[str, object]:
    device = context_tensor.device
    n_samples = context_tensor.shape[0]

    baseline_loss_parts: List[torch.Tensor] = []
    baseline_pred_parts: List[torch.Tensor] = []
    baseline_info_parts: Dict[str, List[torch.Tensor]] = {k: [] for k in ["p_head", "p_chest", "p_neck", "joint_risk"]}

    optimized_action_parts: List[torch.Tensor] = []
    optimized_pred_parts: List[torch.Tensor] = []
    opt_final_loss_parts: List[torch.Tensor] = []
    opt_info_parts: Dict[str, List[torch.Tensor]] = {k: [] for k in ["p_head", "p_chest", "p_neck", "joint_risk"]}

    init_action_parts: List[torch.Tensor] = []
    init_pred_parts: List[torch.Tensor] = []
    init_loss_parts: List[torch.Tensor] = []
    init_detail_parts: Dict[str, List[torch.Tensor]] = {k: [] for k in ["p_head", "p_chest", "p_neck", "joint_risk"]}

    total_time_cost = 0.0
    trajectory_all: List[float] = []

    def _cat_or_nan(parts: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat(parts, dim=0) if parts else torch.full((n_samples,), float("nan"), device=device)

    for start in range(0, n_samples, eval_batch_size):
        end = min(start + eval_batch_size, n_samples)
        ctx_b = context_tensor[start:end]
        base_b = baseline_trainable[start:end]

        with torch.no_grad():
            pulse_b = surrogate.generate_pulse(ctx_b)
            base_loss_b, base_pred_b, base_info_b = surrogate.predict_injury_and_loss(ctx_b, base_b, pulse_b)

        opt_act_b, opt_pred_b, opt_info_b = optimizer.optimize(ctx_b, pulse_norm=pulse_b)
        _, opt_act_b = rule_engine.sanitize_context_and_trainable(ctx_b, opt_act_b)

        baseline_loss_parts.append(base_loss_b.detach())
        baseline_pred_parts.append(base_pred_b.detach())
        for key in baseline_info_parts:
            if key in base_info_b:
                baseline_info_parts[key].append(base_info_b[key].detach())

        optimized_action_parts.append(opt_act_b.detach())
        optimized_pred_parts.append(opt_pred_b.detach())
        if opt_info_b.get("final_loss_batch") is not None:
            opt_final_loss_parts.append(opt_info_b["final_loss_batch"].detach())
        for key in opt_info_parts:
            if key in opt_info_b:
                opt_info_parts[key].append(opt_info_b[key].detach())

        init_obj = opt_info_b.get("initial", {})
        if init_obj.get("actions") is not None:
            init_actions_b = init_obj["actions"].detach()
            _, init_actions_b = rule_engine.sanitize_context_and_trainable(ctx_b, init_actions_b)
            init_action_parts.append(init_actions_b)
        if init_obj.get("preds") is not None:
            init_pred_parts.append(init_obj["preds"].detach())
        if init_obj.get("loss_batch") is not None:
            init_loss_parts.append(init_obj["loss_batch"].detach())
        init_detail = init_obj.get("detail", {})
        for key in init_detail_parts:
            if key in init_detail:
                init_detail_parts[key].append(init_detail[key].detach())

        total_time_cost += float(opt_info_b.get("time_cost", 0.0))
        trajectory_all.extend(opt_info_b.get("trajectory", []))

    return {
        "baseline_loss_batch": torch.cat(baseline_loss_parts, dim=0),
        "baseline_preds": torch.cat(baseline_pred_parts, dim=0),
        "baseline_info": {key: _cat_or_nan(parts) for key, parts in baseline_info_parts.items()},
        "optimized_actions": torch.cat(optimized_action_parts, dim=0),
        "optimized_preds": torch.cat(optimized_pred_parts, dim=0),
        "opt_final_loss": _cat_or_nan(opt_final_loss_parts),
        "opt_info": {key: _cat_or_nan(parts) for key, parts in opt_info_parts.items()},
        "init_actions": torch.cat(init_action_parts, dim=0) if init_action_parts else None,
        "init_preds": torch.cat(init_pred_parts, dim=0) if init_pred_parts else None,
        "init_loss_batch": torch.cat(init_loss_parts, dim=0) if init_loss_parts else None,
        "init_detail": {key: _cat_or_nan(parts) for key, parts in init_detail_parts.items()},
        "total_time_cost": total_time_cost,
        "trajectory_all": trajectory_all,
    }


def _build_all_result_dfs(
    df_input: pd.DataFrame,
    context_df: pd.DataFrame,
    baseline_trainable: torch.Tensor,
    optimized_actions: torch.Tensor,
    init_actions: Optional[torch.Tensor],
    baseline_preds: torch.Tensor,
    optimized_preds: torch.Tensor,
    init_preds: Optional[torch.Tensor],
    baseline_info: Dict[str, torch.Tensor],
    opt_info: Dict[str, torch.Tensor],
    init_detail: Dict[str, torch.Tensor],
    baseline_loss_batch: torch.Tensor,
    init_loss_batch: Optional[torch.Tensor],
    opt_final_loss: torch.Tensor,
    truth_arrays: Dict[str, np.ndarray],
    trainable_names: List[str],
    fixed_names: List[str],
) -> Dict[str, pd.DataFrame]:
    n_samples = len(df_input)
    device = baseline_trainable.device

    def _build_control_df(tensor: Optional[torch.Tensor], prefix: str, names: List[str]) -> pd.DataFrame:
        if tensor is None:
            return pd.DataFrame({f"{prefix}_{name}": [np.nan] * n_samples for name in names})
        return pd.DataFrame(tensor.detach().cpu().numpy(), columns=[f"{prefix}_{name}" for name in names])

    def _build_injury_df(tensor: Optional[torch.Tensor], prefix: str) -> pd.DataFrame:
        columns = [f"{prefix}_HIC", f"{prefix}_Dmax", f"{prefix}_Nij"]
        if tensor is None:
            return pd.DataFrame({col: [np.nan] * n_samples for col in columns})
        return pd.DataFrame(tensor.detach().cpu().numpy(), columns=columns)

    def _tensor_to_numpy(tensor: Optional[torch.Tensor]) -> np.ndarray:
        if tensor is None:
            return np.full((n_samples,), np.nan, dtype=np.float32)
        return tensor.detach().cpu().numpy()

    if fixed_names:
        fixed_context_tensor = torch.tensor(context_df[fixed_names].values, dtype=torch.float32, device=device)
        full_baseline = torch.cat([baseline_trainable, fixed_context_tensor], dim=1)
        full_optimized = torch.cat([optimized_actions, fixed_context_tensor], dim=1)
        full_init = torch.cat([init_actions, fixed_context_tensor], dim=1) if init_actions is not None else None
    else:
        full_baseline = baseline_trainable
        full_optimized = optimized_actions
        full_init = init_actions

    all_control_names = trainable_names + fixed_names
    df_base_ctrl = _build_control_df(full_baseline, "Base", all_control_names)
    df_init_ctrl = _build_control_df(full_init, "Init", all_control_names)
    df_opt_ctrl = _build_control_df(full_optimized, "Opt", all_control_names)

    df_base_inj = _build_injury_df(baseline_preds, "Base")
    df_init_inj = _build_injury_df(init_preds, "Init")
    df_opt_inj = _build_injury_df(optimized_preds, "Opt")

    ot_array = context_df["OT"].to_numpy(dtype=np.float32)

    base_hic = df_base_inj["Base_HIC"].to_numpy(dtype=np.float32)
    base_dmax = df_base_inj["Base_Dmax"].to_numpy(dtype=np.float32)
    base_nij = df_base_inj["Base_Nij"].to_numpy(dtype=np.float32)
    opt_hic = df_opt_inj["Opt_HIC"].to_numpy(dtype=np.float32)
    opt_dmax = df_opt_inj["Opt_Dmax"].to_numpy(dtype=np.float32)
    opt_nij = df_opt_inj["Opt_Nij"].to_numpy(dtype=np.float32)
    init_hic = df_init_inj["Init_HIC"].to_numpy(dtype=np.float32)
    init_dmax = df_init_inj["Init_Dmax"].to_numpy(dtype=np.float32)
    init_nij = df_init_inj["Init_Nij"].to_numpy(dtype=np.float32)

    base_ais_head = AIS_cal_head(base_hic)
    base_ais_chest = AIS_cal_chest(base_dmax, ot_array)
    base_ais_neck = AIS_cal_neck(base_nij)
    opt_ais_head = AIS_cal_head(opt_hic)
    opt_ais_chest = AIS_cal_chest(opt_dmax, ot_array)
    opt_ais_neck = AIS_cal_neck(opt_nij)
    init_ais_head = AIS_cal_head(init_hic) if init_preds is not None else np.full((n_samples,), np.nan)
    init_ais_chest = AIS_cal_chest(init_dmax, ot_array) if init_preds is not None else np.full((n_samples,), np.nan)
    init_ais_neck = AIS_cal_neck(init_nij) if init_preds is not None else np.full((n_samples,), np.nan)

    base_p_head = _tensor_to_numpy(baseline_info["p_head"])
    base_p_chest = _tensor_to_numpy(baseline_info["p_chest"])
    base_p_neck = _tensor_to_numpy(baseline_info["p_neck"])
    base_joint = _tensor_to_numpy(baseline_info["joint_risk"])
    opt_p_head = _tensor_to_numpy(opt_info["p_head"])
    opt_p_chest = _tensor_to_numpy(opt_info["p_chest"])
    opt_p_neck = _tensor_to_numpy(opt_info["p_neck"])
    opt_joint = _tensor_to_numpy(opt_info["joint_risk"])
    if np.isnan(opt_joint).all():
        opt_joint = _compute_joint_risk(opt_p_head, opt_p_chest, opt_p_neck)

    init_p_head = _tensor_to_numpy(init_detail["p_head"]) if init_preds is not None else np.full((n_samples,), np.nan)
    init_p_chest = _tensor_to_numpy(init_detail["p_chest"]) if init_preds is not None else np.full((n_samples,), np.nan)
    init_p_neck = _tensor_to_numpy(init_detail["p_neck"]) if init_preds is not None else np.full((n_samples,), np.nan)
    init_joint = _tensor_to_numpy(init_detail["joint_risk"]) if init_preds is not None else np.full((n_samples,), np.nan)

    df_prob = pd.DataFrame(
        {
            "Base_Phead": base_p_head,
            "Base_Pchest": base_p_chest,
            "Base_Pneck": base_p_neck,
            "Base_JointRisk": base_joint,
            "Init_Phead": init_p_head,
            "Init_Pchest": init_p_chest,
            "Init_Pneck": init_p_neck,
            "Init_JointRisk": init_joint,
            "Opt_Phead": opt_p_head,
            "Opt_Pchest": opt_p_chest,
            "Opt_Pneck": opt_p_neck,
            "Opt_JointRisk": opt_joint,
        }
    )

    df_ais = pd.DataFrame(
        {
            "Base_AIS_head": base_ais_head,
            "Base_AIS_chest": base_ais_chest,
            "Base_AIS_neck": base_ais_neck,
            "Base_AIS_max": np.maximum.reduce([base_ais_head, base_ais_chest, base_ais_neck]),
            "Init_AIS_head": init_ais_head,
            "Init_AIS_chest": init_ais_chest,
            "Init_AIS_neck": init_ais_neck,
            "Init_AIS_max": np.nanmax(np.vstack([init_ais_head, init_ais_chest, init_ais_neck]), axis=0),
            "Opt_AIS_head": opt_ais_head,
            "Opt_AIS_chest": opt_ais_chest,
            "Opt_AIS_neck": opt_ais_neck,
            "Opt_AIS_max": np.maximum.reduce([opt_ais_head, opt_ais_chest, opt_ais_neck]),
        }
    )

    df_loss = pd.DataFrame(
        {
            "Base_Loss": baseline_loss_batch.detach().cpu().numpy(),
            "Init_Loss": _tensor_to_numpy(init_loss_batch),
            "Opt_Loss": opt_final_loss.detach().cpu().numpy(),
        }
    )

    eps = 1e-8
    reduction_df = pd.DataFrame(
        {
            "Reduction_HIC_abs": base_hic - opt_hic,
            "Reduction_Dmax_abs": base_dmax - opt_dmax,
            "Reduction_Nij_abs": base_nij - opt_nij,
            "Reduction_HIC_pct": (base_hic - opt_hic) / np.maximum(np.abs(base_hic), eps),
            "Reduction_Dmax_pct": (base_dmax - opt_dmax) / np.maximum(np.abs(base_dmax), eps),
            "Reduction_Nij_pct": (base_nij - opt_nij) / np.maximum(np.abs(base_nij), eps),
            "Reduction_Phead_abs": base_p_head - opt_p_head,
            "Reduction_Pchest_abs": base_p_chest - opt_p_chest,
            "Reduction_Pneck_abs": base_p_neck - opt_p_neck,
            "Reduction_JointRisk_abs": base_joint - opt_joint,
            "Reduction_AIS_head_abs": base_ais_head - opt_ais_head,
            "Reduction_AIS_chest_abs": base_ais_chest - opt_ais_chest,
            "Reduction_AIS_neck_abs": base_ais_neck - opt_ais_neck,
            "Reduction_AIS_max_abs": df_ais["Base_AIS_max"].to_numpy() - df_ais["Opt_AIS_max"].to_numpy(),
            "Reduction_Loss_abs": df_loss["Base_Loss"].to_numpy() - df_loss["Opt_Loss"].to_numpy(),
            "Reduction_Loss_pct": (df_loss["Base_Loss"].to_numpy() - df_loss["Opt_Loss"].to_numpy()) / np.maximum(np.abs(df_loss["Base_Loss"].to_numpy()), eps),
        }
    )

    df_truth = pd.DataFrame(index=df_input.index)
    if all(key in truth_arrays for key in ["y_HIC", "y_Dmax", "y_Nij"]):
        true_hic = np.asarray(truth_arrays["y_HIC"], dtype=np.float32)
        true_dmax = np.asarray(truth_arrays["y_Dmax"], dtype=np.float32)
        true_nij = np.asarray(truth_arrays["y_Nij"], dtype=np.float32)
        true_ais_head = np.asarray(truth_arrays["ais_head"]) if "ais_head" in truth_arrays else AIS_cal_head(true_hic)
        true_ais_chest = np.asarray(truth_arrays["ais_chest"]) if "ais_chest" in truth_arrays else AIS_cal_chest(true_dmax, ot_array)
        true_ais_neck = np.asarray(truth_arrays["ais_neck"]) if "ais_neck" in truth_arrays else AIS_cal_neck(true_nij)
        df_truth = pd.DataFrame(
            {
                "True_HIC": true_hic,
                "True_Dmax": true_dmax,
                "True_Nij": true_nij,
                "True_AIS_head": true_ais_head,
                "True_AIS_chest": true_ais_chest,
                "True_AIS_neck": true_ais_neck,
                "True_AIS_max": np.maximum.reduce([true_ais_head, true_ais_chest, true_ais_neck]),
                "True_vs_Base_HIC": true_hic - base_hic,
                "True_vs_Base_Dmax": true_dmax - base_dmax,
                "True_vs_Base_Nij": true_nij - base_nij,
                "True_vs_Opt_HIC": true_hic - opt_hic,
                "True_vs_Opt_Dmax": true_dmax - opt_dmax,
                "True_vs_Opt_Nij": true_nij - opt_nij,
            }
        )

    df_final = pd.concat(
        [
            df_input,
            context_df,
            df_truth,
            df_base_ctrl,
            df_init_ctrl,
            df_opt_ctrl,
            df_base_inj,
            df_init_inj,
            df_opt_inj,
            df_prob,
            df_ais,
            df_loss,
            reduction_df,
        ],
        axis=1,
    )
    df_final = df_final.loc[:, ~df_final.columns.duplicated()]

    return {
        "df_final": df_final,
        "df_truth": df_truth,
        "df_base_ctrl": df_base_ctrl,
        "df_init_ctrl": df_init_ctrl,
        "df_opt_ctrl": df_opt_ctrl,
        "df_base_inj": df_base_inj,
        "df_init_inj": df_init_inj,
        "df_opt_inj": df_opt_inj,
        "df_prob": df_prob,
        "df_ais": df_ais,
        "df_loss": df_loss,
        "reduction_df": reduction_df,
    }


def _compute_macro_summary(reduction_df: pd.DataFrame, df_prob: pd.DataFrame, n_samples: int) -> Dict[str, float]:
    return {
        "mean_reduction_HIC": float(np.nanmean(reduction_df["Reduction_HIC_abs"])),
        "mean_reduction_Dmax": float(np.nanmean(reduction_df["Reduction_Dmax_abs"])),
        "mean_reduction_Nij": float(np.nanmean(reduction_df["Reduction_Nij_abs"])),
        "mean_reduction_Phead": float(np.nanmean(reduction_df["Reduction_Phead_abs"])),
        "mean_reduction_Pchest": float(np.nanmean(reduction_df["Reduction_Pchest_abs"])),
        "mean_reduction_Pneck": float(np.nanmean(reduction_df["Reduction_Pneck_abs"])),
        "mean_reduction_joint_risk": float(np.nanmean(reduction_df["Reduction_JointRisk_abs"])),
        "mean_base_joint_risk": float(np.nanmean(df_prob["Base_JointRisk"])),
        "mean_opt_joint_risk": float(np.nanmean(df_prob["Opt_JointRisk"])),
        "n_samples": int(n_samples),
    }


def main():
    args = parse_args()
    logger = setup_ars_logger(name="Evaluator")

    base_dir = Path(__file__).resolve().parent
    cfg_path = base_dir / "configs" / "default_config.yaml"
    param_space_path = base_dir / "configs" / "param_space.yaml"

    if not cfg_path.is_file():
        raise FileNotFoundError(f"config file not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if "optimization" not in config:
        config["optimization"] = {}

    if args.direct_inference:
        config["optimization"]["direct_inference"] = True

    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"评估设备: {device}")

    output_dir = _build_output_dir(base_dir, args.input_csv)
    output_csv_path = _resolve_output_csv_path(args.output_csv, output_dir)

    # 初始化核心组件
    param_manager = ParamManager(str(param_space_path))
    constraint_manager = PhysicalConstraintManager(param_manager)
    rule_engine = RuleEngine(param_manager)
    data_processor = UnifiedDataProcessor(str(NORMALIZATION_CONFIG_PATH))

    pulse_model, injury_model = load_surrogate_models(config=config, device=device)
    surrogate = SurrogateAdapter(
        pulse_model=pulse_model,
        injury_model=injury_model,
        param_manager=param_manager,
        constraint_manager=constraint_manager,
        config=config,
        data_processor=data_processor,
    ).to(device)

    # 评估阶段若启用了分布惩罚，先拟合一次训练分布参考，避免逐批告警并保持目标语义一致。
    if surrogate.distribution_penalty.enabled:
        try:
            train_idx_path = SPLIT_INDICES_DIR / "injury_train_indices.npy"
            ref_loader = StateDataLoaderManager(
                param_manager=param_manager,
                batch_size=1024,
                device=device,
                seed=int(config.get("seed", 42)),
                split_indices_path=str(train_idx_path),
                jitter_ratio=0.0,
                jitter_prob=0.0,
            )
            max_ref_samples = int(config.get("optimization", {}).get("distribution_penalty", {}).get("max_ref_samples", 0))
            ref_context = ref_loader.get_distribution_reference(
                max_samples=max_ref_samples,
                shuffle=False,
                feature_space=surrogate.distribution_penalty.feature_space,
                trainable_indices=param_manager.get_control_trainable_indices(),
            )
            surrogate.fit_distribution_reference(ref_context)
            logger.info(f"评估阶段已拟合分布惩罚参考集: n_ref={ref_context.shape[0]}")
        except Exception as ex:
            logger.warning(f"评估阶段拟合分布参考失败，将回退为无分布惩罚: {ex}")
            surrogate.distribution_penalty.enabled = False

    strat_cfg = config.get("strategy_net", {})
    strategy_net = StrategyNet(
        param_manager=param_manager,
        constraint_manager=constraint_manager,
        data_processor=data_processor,
        hidden_dims=strat_cfg.get("hidden_dims", [128, 256, 128]),
        activation=strat_cfg.get("activation", "LeakyReLU"),
        dropout=float(strat_cfg.get("dropout", 0.0)),
        pulse_channels=int(strat_cfg.get("pulse_channels", 2)),
        pulse_embed_dim=int(strat_cfg.get("pulse_embed_dim", 32)),
    ).to(device)

    # 策略权重加载逻辑
    strategy_ckpt_path = None
    if args.strategy_ckpt:
        strategy_ckpt_path = Path(args.strategy_ckpt)
    else:
        strategy_ckpt_path = _resolve_default_strategy_ckpt(base_dir, config)

    strict_default_ckpt = bool(config.get("evaluation", {}).get("strict_default_ckpt", False))

    if strategy_ckpt_path is not None:
        if not strategy_ckpt_path.is_file():
            raise FileNotFoundError(f"策略权重不存在: {strategy_ckpt_path}")
        strategy_net.load_state_dict(torch.load(str(strategy_ckpt_path), map_location=device, weights_only=True))
        config["optimization"]["direct_inference"] = True
        logger.info(f"已加载策略权重: {strategy_ckpt_path}")
    else:
        if strict_default_ckpt:
            raise FileNotFoundError(
                "strict_default_ckpt=true 且未找到可用默认策略权重（val_best_model.pth）。"
                "请先训练生成 val_best_model.pth，或显式设置 evaluation.default_strategy_ckpt。"
            )
        logger.warning("未找到默认策略权重，将仅使用 default 初值 + 局部精调。")

    optimizer = ARSLocalOptimizer(
        config=config,
        param_manager=param_manager,
        constraint_manager=constraint_manager,
        surrogate=surrogate,
        strategy_net=strategy_net,
    )

    # 读取输入
    truth_arrays: Dict[str, np.ndarray] = {}
    input_source: Dict[str, str] = {}

    if args.input_csv:
        input_csv_path = Path(args.input_csv)
        if not input_csv_path.is_file():
            raise FileNotFoundError(f"input_csv 不存在: {input_csv_path}")
        df_input = pd.read_csv(str(input_csv_path))
        input_source = {"type": "input_csv", "path": str(input_csv_path.resolve())}

        if all(col in df_input.columns for col in ["y_HIC", "y_Dmax", "y_Nij"]):
            truth_arrays["y_HIC"] = df_input["y_HIC"].to_numpy(dtype=np.float32)
            truth_arrays["y_Dmax"] = df_input["y_Dmax"].to_numpy(dtype=np.float32)
            truth_arrays["y_Nij"] = df_input["y_Nij"].to_numpy(dtype=np.float32)
    else:
        pool_npz_path = RAW_DATA_DIR / "raw_data_packed.npz"
        test_idx_path = SPLIT_INDICES_DIR / "injury_test_indices.npy"
        if not pool_npz_path.exists() or not test_idx_path.exists():
            raise FileNotFoundError("自动测试集模式需要 raw_data_packed.npz 与 injury_test_indices.npy")

        test_indices = np.load(str(test_idx_path)).astype(np.int64)
        with np.load(str(pool_npz_path), allow_pickle=True) as data:
            if "x_att_raw" not in data:
                raise KeyError("raw_data_packed.npz 缺失 x_att_raw")
            x_att_raw = data["x_att_raw"][test_indices]
            if x_att_raw.ndim != 2 or x_att_raw.shape[1] != len(FEATURE_ORDER):
                raise ValueError(f"x_att_raw 形状异常: {x_att_raw.shape}")

            case_ids = data["case_ids"][test_indices] if "case_ids" in data else np.arange(len(test_indices))
            df_input = pd.DataFrame(x_att_raw, columns=FEATURE_ORDER)
            df_input.insert(0, "case_id", case_ids)

            for key in ["y_HIC", "y_Dmax", "y_Nij", "ais_head", "ais_chest", "ais_neck"]:
                if key in data:
                    truth_arrays[key] = np.asarray(data[key][test_indices])

        input_source = {
            "type": "test_split",
            "path": str(test_idx_path.resolve()),
            "test_indices_path": str(test_idx_path.resolve()),
            "raw_data_npz_path": str(pool_npz_path.resolve()),
        }
        logger.info(f"未指定 input_csv，自动加载测试集: {len(df_input)} 条")

    if "case_id" not in df_input.columns:
        df_input.insert(0, "case_id", np.arange(len(df_input), dtype=np.int64))

    # 组装 context 与 baseline trainable，并统一规则化
    context_df, missing_context_cols = _build_context_dataframe(df_input, param_manager, logger)
    baseline_trainable_df, missing_trainable_cols = _build_baseline_trainable_dataframe(df_input, param_manager, logger)

    context_names = [p["name"] for p in param_manager.get_context_params()]
    trainable_names = [p["name"] for p in param_manager.control_trainable_params]
    fixed_names = [p["name"] for p in param_manager.control_fixed_params]

    context_tensor_raw = torch.tensor(context_df[context_names].values, dtype=torch.float32, device=device)
    baseline_trainable_raw = torch.tensor(
        baseline_trainable_df[trainable_names].values,
        dtype=torch.float32,
        device=device,
    )

    context_tensor, baseline_trainable = rule_engine.sanitize_context_and_trainable(
        context_params=context_tensor_raw,
        control_trainable=baseline_trainable_raw,
    )

    # 以规则化后的 context 反写，确保导出和真实评估一致
    context_df = pd.DataFrame(context_tensor.detach().cpu().numpy(), columns=context_names, index=df_input.index)

    eval_cfg = config.get("evaluation", {})
    eval_batch_size = int(eval_cfg.get("eval_batch_size", 512))
    if eval_batch_size <= 0:
        raise ValueError("eval_batch_size 必须为正整数")

    n_samples = context_tensor.shape[0]
    prediction_outputs = _compute_predictions_batch(
        context_tensor=context_tensor,
        baseline_trainable=baseline_trainable,
        surrogate=surrogate,
        optimizer=optimizer,
        rule_engine=rule_engine,
        eval_batch_size=eval_batch_size,
    )

    baseline_loss_batch = prediction_outputs["baseline_loss_batch"]
    baseline_preds = prediction_outputs["baseline_preds"]
    baseline_info = prediction_outputs["baseline_info"]
    optimized_actions = prediction_outputs["optimized_actions"]
    optimized_preds = prediction_outputs["optimized_preds"]
    opt_final_loss = prediction_outputs["opt_final_loss"]
    opt_info = prediction_outputs["opt_info"]
    init_actions = prediction_outputs["init_actions"]
    init_preds = prediction_outputs["init_preds"]
    init_loss_batch = prediction_outputs["init_loss_batch"]
    init_detail = prediction_outputs["init_detail"]
    total_time_cost = float(prediction_outputs["total_time_cost"])
    trajectory_all = prediction_outputs["trajectory_all"]

    result_frames = _build_all_result_dfs(
        df_input=df_input,
        context_df=context_df,
        baseline_trainable=baseline_trainable,
        optimized_actions=optimized_actions,
        init_actions=init_actions,
        baseline_preds=baseline_preds,
        optimized_preds=optimized_preds,
        init_preds=init_preds,
        baseline_info=baseline_info,
        opt_info=opt_info,
        init_detail=init_detail,
        baseline_loss_batch=baseline_loss_batch,
        init_loss_batch=init_loss_batch,
        opt_final_loss=opt_final_loss,
        truth_arrays=truth_arrays,
        trainable_names=trainable_names,
        fixed_names=fixed_names,
    )

    df_final = result_frames["df_final"]
    df_prob = result_frames["df_prob"]
    df_loss = result_frames["df_loss"]
    reduction_df = result_frames["reduction_df"]
    df_final.to_csv(str(output_csv_path), index=False)

    tracker = MetricsTracker()
    case_ids_series = df_input["case_id"] if "case_id" in df_input.columns else pd.Series(range(len(df_input)))
    avg_time = total_time_cost / max(1, len(df_input))

    for i in range(len(df_input)):
        case_result = {
            "initial": {
                "loss_mean": float(df_loss.iloc[i]["Init_Loss"]) if not np.isnan(df_loss.iloc[i]["Init_Loss"]) else float(df_loss.iloc[i]["Base_Loss"])
            },
            "final_loss_batch": torch.tensor([float(df_loss.iloc[i]["Opt_Loss"])], dtype=torch.float32),
            "time_cost": avg_time,
            "trajectory": trajectory_all,
        }
        init_action_i = None
        if init_actions is not None:
            init_action_i = init_actions[i].detach().cpu().numpy()
        opt_action_i = optimized_actions[i].detach().cpu().numpy()
        tracker.update(case_result, case_id=int(case_ids_series.iloc[i]), initial_action=init_action_i, optimized_action=opt_action_i)

    tracker.log_summary()

    summary = {
        "summary_metrics": _compute_macro_summary(reduction_df=reduction_df, df_prob=df_prob, n_samples=len(df_final)),
        "formulas": {
            "joint_risk": "L_risk = 1 - Π_k (1 - P_k)",
            "reported_reduction": "mean(Baseline - Optimized)",
            "true_vs_base_error": "mean(True - BaselineModel), only when truth exists",
            "true_vs_opt_error": "mean(True - OptimizedModel), only when truth exists",
        },
    }
    _write_yaml(output_dir / "evaluation_summary.yaml", summary)

    snapshots = _snapshot_configs(cfg_path=cfg_path, param_space_path=param_space_path, output_dir=output_dir)

    eval_record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(output_dir),
        "output_csv_path": str(output_csv_path),
        "input_source": input_source,
        "strategy_checkpoint_path": str(strategy_ckpt_path) if strategy_ckpt_path is not None else None,
        "direct_inference": bool(config.get("optimization", {}).get("direct_inference", False)),
        "config_files": {
            "default_config_path": str(cfg_path),
            "param_space_path": str(param_space_path),
            **snapshots,
            "normalization_config_path": str(NORMALIZATION_CONFIG_PATH),
            "normalization_config_sha1": _sha1(Path(NORMALIZATION_CONFIG_PATH)),
        },
        "evaluation_config": config.get("evaluation", {}),
        "optimization_config": config.get("optimization", {}),
        "missing_context_filled_by_default": missing_context_cols,
        "missing_trainable_filled_by_default": missing_trainable_cols,
    }
    _write_yaml(output_dir / "evaluation_record.yaml", eval_record)

    logger.info(f"评估完成，结果目录: {output_dir}")
    logger.info(f"结果 CSV: {output_csv_path}")


if __name__ == "__main__":
    main()
