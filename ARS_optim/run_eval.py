import argparse
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import yaml

from common.data_utils.split_io import load_int_vector_csv
from common.data_utils.processor import UnifiedDataProcessor
from common.metrics.injury_risk import AIS_cal_chest, AIS_cal_head, AIS_cal_neck
from common.settings import FEATURE_ORDER, NORMALIZATION_CONFIG_PATH, RAW_DATA_DIR, SPLIT_INDICES_DIR
from common.tools.logger import setup_logger

from ARS_optim.src.constraints import ConstraintEngine
from ARS_optim.src.data_sampler import StateDataSampler
from ARS_optim.src.optimizer import LocalRefiner
from ARS_optim.src.param_manager import ParamManager
from ARS_optim.src.strategy_net import build_strategy_net_from_config
from ARS_optim.src.surrogate import SurrogateAdapter, load_surrogate_models


def parse_args():
    parser = argparse.ArgumentParser(description="ARS Local Refinement Evaluator")
    parser.add_argument("--input_csv", type=str, default=None, help="输入工况参数 CSV；若不提供则自动使用 injury test split")
    parser.add_argument("--output_csv", type=str, default="evaluation_results.csv", help="输出 CSV 文件名")
    parser.add_argument("--strategy_ckpt", type=str, default=None, help="策略网络权重路径")
    parser.add_argument("--direct_inference", action="store_true", help="强制启用策略网络直推")
    return parser.parse_args()


def _safe_stem(text: str) -> str:
    stem = Path(text).stem
    normalized = [char if char.isalnum() or char in {"_", "-"} else "_" for char in stem]
    name = "".join(normalized).strip("_")
    return name or "evaluation"


def _build_output_dir(base_dir: Path, input_csv: Optional[str]) -> Path:
    suffix = _safe_stem(input_csv) if input_csv else "injury_test_split"
    output_dir = base_dir / "saved_eval" / f"eval_{suffix}_{datetime.now().strftime('%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def _resolve_strategy_ckpt(args, base_dir: Path, config: Dict) -> Optional[Path]:
    if args.strategy_ckpt:
        return Path(args.strategy_ckpt).resolve()

    eval_cfg = config.get("evaluation", {}) or {}
    configured = eval_cfg.get("strategy_checkpoint")
    if configured:
        candidate = Path(str(configured))
        return candidate if candidate.is_absolute() else (base_dir / candidate).resolve()
    return None


def _copy_config_snapshots(cfg_path: Path, param_space_path: Path, output_dir: Path) -> Dict[str, str]:
    cfg_snapshot = output_dir / "config_used.yaml"
    param_snapshot = output_dir / "param_space.yaml"
    norm_snapshot = output_dir / "normalization_config.json"
    shutil.copy2(str(cfg_path), str(cfg_snapshot))
    shutil.copy2(str(param_space_path), str(param_snapshot))
    shutil.copy2(str(NORMALIZATION_CONFIG_PATH), str(norm_snapshot))
    return {
        "config_used": str(cfg_snapshot),
        "param_space_used": str(param_snapshot),
        "normalization_config": str(norm_snapshot),
    }


def _write_yaml(path: Path, content: Dict) -> None:
    with open(path, "w", encoding="utf-8") as file:
        yaml.safe_dump(content, file, allow_unicode=True, sort_keys=False)


def _safe_nanmean(series: pd.Series) -> float:
    values = np.asarray(series, dtype=np.float32)
    return float(np.nan) if np.isnan(values).all() else float(np.nanmean(values))


def _build_param_dataframe(
    df_input: pd.DataFrame,
    params: List[dict],
    fill_missing_with_default: bool,
) -> tuple[pd.DataFrame, List[str], List[str]]:
    output_df = pd.DataFrame(index=df_input.index)
    missing = []
    provided = []
    for param in params:
        name = param["name"]
        if name in df_input.columns:
            output_df[name] = pd.to_numeric(df_input[name], errors="coerce")
            provided.append(name)
        else:
            fill_value = float(param["default"]) if fill_missing_with_default else np.nan
            output_df[name] = fill_value
            missing.append(name)
    return output_df, missing, provided


def _format_row_list(row_indices: np.ndarray, max_items: int = 12) -> str:
    if row_indices.size == 0:
        return "[]"
    preview = row_indices[:max_items].tolist()
    suffix = " ..." if row_indices.size > max_items else ""
    return f"{preview}{suffix}"


def _prepare_eval_inputs(
    df_input: pd.DataFrame,
    param_manager: ParamManager,
    constraint_engine: ConstraintEngine,
    device: torch.device,
    logger,
    strict_provided_validation: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], List[str], np.ndarray, List[int], List[int]]:
    context_params = param_manager.get_context_params()
    trainable_params = param_manager.get_trainable_params()
    context_names = param_manager.get_context_names()
    trainable_names = param_manager.get_trainable_names()
    default_trainable = np.asarray(param_manager.get_default_values(trainable_params), dtype=np.float32)

    context_df_raw, missing_context, provided_context = _build_param_dataframe(
        df_input,
        context_params,
        fill_missing_with_default=not strict_provided_validation,
    )
    baseline_df_raw, missing_trainable, provided_trainable = _build_param_dataframe(
        df_input,
        trainable_params,
        fill_missing_with_default=True,
    )

    if missing_context:
        if strict_provided_validation:
            logger.warning("input_csv 缺失 context 列，这些列对应的 case 将被逐行跳过: %s", missing_context)
        else:
            logger.warning("输入缺失部分 context 参数，已回退 default: %s", missing_context)
    if missing_trainable:
        logger.warning("输入未提供部分可调参数，baseline 已回退 default: %s", missing_trainable)

    if not strict_provided_validation:
        context_eval_df = context_df_raw.astype(np.float32)
        baseline_eval_df = baseline_df_raw.astype(np.float32)
        context_tensor = torch.tensor(context_eval_df[context_names].values, dtype=torch.float32, device=device)
        baseline_tensor = torch.tensor(baseline_eval_df[trainable_names].values, dtype=torch.float32, device=device)
        full_features = constraint_engine.compose_full_features(context_tensor, baseline_tensor)
        valid_mask = constraint_engine.is_valid_physics(full_features).detach().cpu().numpy().astype(bool)
        invalid_rows = np.flatnonzero(~valid_mask)
        if invalid_rows.size > 0:
            logger.warning(
                "当前评估输入来自内部测试集，其中 %d 个 case 不满足物理约束，已跳过。行号: %s",
                invalid_rows.size,
                _format_row_list(invalid_rows),
            )
        return (
            context_eval_df,
            context_eval_df,
            baseline_eval_df,
            missing_context,
            missing_trainable,
            valid_mask,
            invalid_rows.astype(int).tolist(),
            [],
        )

    context_output_df = context_df_raw.copy()
    row_has_context_nan = context_df_raw[context_names].isna().any(axis=1).to_numpy()
    valid_context_candidates = ~row_has_context_nan
    skipped_rows: List[int] = np.flatnonzero(row_has_context_nan).astype(int).tolist()

    valid_mask = np.zeros(len(df_input), dtype=bool)
    context_eval_df = pd.DataFrame(np.nan, index=df_input.index, columns=context_names, dtype=np.float32)
    baseline_eval_df = pd.DataFrame(np.tile(default_trainable, (len(df_input), 1)), index=df_input.index, columns=trainable_names, dtype=np.float32)
    reverted_baseline_rows: List[int] = []

    if np.any(valid_context_candidates):
        candidate_indices = np.flatnonzero(valid_context_candidates)
        candidate_context_raw = context_df_raw.iloc[candidate_indices]
        candidate_context_tensor = torch.tensor(candidate_context_raw[context_names].values, dtype=torch.float32, device=device)
        context_full = constraint_engine.compose_full_features(candidate_context_tensor)
        context_valid_local = constraint_engine.is_valid_physics(context_full).detach().cpu().numpy().astype(bool)
        if (~context_valid_local).any():
            logger.warning(
                "以下 input_csv 行的 context 参数不合法，将跳过该 case。行号: %s",
                _format_row_list(candidate_indices[~context_valid_local]),
            )
        invalid_context_indices = candidate_indices[~context_valid_local]
        if invalid_context_indices.size > 0:
            skipped_rows.extend(invalid_context_indices.astype(int).tolist())

        legal_context_indices = candidate_indices[context_valid_local]
        if legal_context_indices.size > 0:
            valid_mask[legal_context_indices] = True
            context_eval_df.loc[legal_context_indices, context_names] = context_df_raw.loc[legal_context_indices, context_names]

            baseline_candidate = baseline_df_raw.loc[legal_context_indices, trainable_names].copy()
            provided_trainable_nan_rows = baseline_candidate[provided_trainable].isna().any(axis=1).to_numpy() if provided_trainable else np.zeros(len(legal_context_indices), dtype=bool)
            baseline_candidate.loc[:, trainable_names] = baseline_candidate.loc[:, trainable_names].fillna(pd.Series(default_trainable, index=trainable_names))

            legal_context_tensor = torch.tensor(
                context_eval_df.loc[legal_context_indices, context_names].values,
                dtype=torch.float32,
                device=device,
            )
            baseline_candidate_tensor = torch.tensor(
                baseline_candidate[trainable_names].values,
                dtype=torch.float32,
                device=device,
            )
            baseline_full = constraint_engine.compose_full_features(legal_context_tensor, baseline_candidate_tensor)
            baseline_valid_local = constraint_engine.is_valid_physics(baseline_full).detach().cpu().numpy().astype(bool)
            baseline_invalid_mask_local = (~baseline_valid_local) | provided_trainable_nan_rows
            if baseline_invalid_mask_local.any():
                logger.warning(
                    "以下 input_csv 行的 baseline 可调控制参数不合法，已回退为 param_space.yaml 的 default。行号: %s",
                    _format_row_list(legal_context_indices[baseline_invalid_mask_local]),
                )
            reverted_indices = legal_context_indices[baseline_invalid_mask_local]
            if reverted_indices.size > 0:
                reverted_baseline_rows.extend(reverted_indices.astype(int).tolist())

            baseline_eval_df.loc[legal_context_indices, trainable_names] = baseline_candidate.loc[legal_context_indices, trainable_names]
            if reverted_indices.size > 0:
                baseline_eval_df.loc[reverted_indices, trainable_names] = default_trainable

    if skipped_rows:
        skipped_array = np.asarray(sorted(set(skipped_rows)), dtype=np.int64)
        logger.warning("input_csv 中共有 %d 个 case 因 context 缺失或非法被跳过，行号: %s", skipped_array.size, _format_row_list(skipped_array))
    if reverted_baseline_rows:
        reverted_array = np.asarray(sorted(set(reverted_baseline_rows)), dtype=np.int64)
        logger.warning("input_csv 中共有 %d 个 case 的 baseline 可调控制参数被回退为 default，行号: %s", reverted_array.size, _format_row_list(reverted_array))

    return (
        context_output_df,
        context_eval_df,
        baseline_eval_df,
        missing_context,
        missing_trainable,
        valid_mask,
        sorted(set(skipped_rows)),
        sorted(set(reverted_baseline_rows)),
    )


def _fit_distribution_reference_if_needed(surrogate: SurrogateAdapter, sampler: StateDataSampler, param_manager: ParamManager, config: dict) -> None:
    if not surrogate.distribution_penalty.enabled:
        return
    max_ref_samples = int(config.get("optimization", {}).get("distribution_penalty", {}).get("max_ref_samples", 0))
    reference = sampler.get_distribution_reference(
        max_samples=max_ref_samples,
        shuffle=False,
        feature_space=surrogate.distribution_penalty.feature_space,
        trainable_indices=param_manager.get_control_trainable_indices(),
    )
    surrogate.fit_distribution_reference(reference)


def _load_eval_input(args, logger) -> tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, str]]:
    """装载评估输入与可选真值列。

    该函数只负责确定评估样本和可选真值从哪里读取，
    不参与参数修复、baseline 回填或模型推理，避免 main 里混杂两类职责。
    """
    truth_arrays: Dict[str, np.ndarray] = {}
    if args.input_csv:
        input_csv_path = Path(args.input_csv)
        if not input_csv_path.is_file():
            raise FileNotFoundError(f"input_csv 不存在: {input_csv_path}")
        df_input = pd.read_csv(str(input_csv_path))
        input_source = {"type": "input_csv", "path": str(input_csv_path.resolve())}
        for key in ["y_HIC", "y_Dmax", "y_Nij"]:
            if key in df_input.columns:
                truth_arrays[key] = df_input[key].to_numpy(dtype=np.float32)
        return df_input, truth_arrays, input_source

    pool_path = RAW_DATA_DIR / "raw_data_packed.npz"
    test_idx_path = SPLIT_INDICES_DIR / "injury_test_indices.csv"
    if not pool_path.exists() or not test_idx_path.exists():
        raise FileNotFoundError("自动测试集模式需要 raw_data_packed.npz 和 injury_test_indices.csv")

    test_indices = load_int_vector_csv(test_idx_path)
    with np.load(str(pool_path), allow_pickle=True) as data:
        x_att_raw = data["x_att_raw"][test_indices]
        df_input = pd.DataFrame(x_att_raw, columns=FEATURE_ORDER)
        case_ids = data["case_ids"][test_indices] if "case_ids" in data else np.arange(len(test_indices))
        df_input.insert(0, "case_id", case_ids)
        for key in ["y_HIC", "y_Dmax", "y_Nij", "ais_head", "ais_chest", "ais_neck"]:
            if key in data:
                truth_arrays[key] = np.asarray(data[key][test_indices])
    logger.info(f"未指定 input_csv，自动加载测试集: {len(df_input)} 条")
    input_source = {
        "type": "test_split",
        "path": str(test_idx_path.resolve()),
        "raw_data_npz_path": str(pool_path.resolve()),
    }
    return df_input, truth_arrays, input_source


def _compute_predictions_batch(
    context_tensor: torch.Tensor,
    baseline_trainable: torch.Tensor,
    surrogate: SurrogateAdapter,
    optimizer: LocalRefiner,
    eval_batch_size: int,
) -> Dict[str, object]:
    sample_count = context_tensor.shape[0]
    device = context_tensor.device
    stage_parts = {
        key: {
            "preds": [],
            "actions": [],
            "loss": [],
            "info": {name: [] for name in ["p_head", "p_chest", "p_neck", "joint_risk"]},
        }
        for key in ["Base", "Opt1", "Opt2"]
    }
    total_time_cost = 0.0
    trajectory_all: List[float] = []

    def cat_or_nan(parts: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat(parts, dim=0) if parts else torch.full((sample_count,), float("nan"), device=device)

    for start in range(0, sample_count, eval_batch_size):
        end = min(start + eval_batch_size, sample_count)
        context_batch = context_tensor[start:end]
        baseline_batch = baseline_trainable[start:end]

        with torch.no_grad():
            pulse_batch = surrogate.generate_pulse(context_batch)
            base_loss, base_preds, base_info = surrogate.predict_injury_and_loss(context_batch, baseline_batch, pulse_batch)
        stage_parts["Base"]["preds"].append(base_preds.detach())
        stage_parts["Base"]["actions"].append(baseline_batch.detach())
        stage_parts["Base"]["loss"].append(base_loss.detach())
        for key in stage_parts["Base"]["info"]:
            stage_parts["Base"]["info"][key].append(base_info[key].detach())

        if optimizer.direct_inference or optimizer.refine_steps > 0:
            opt_actions, opt_preds, opt_info = optimizer.optimize(context_batch, pulse_norm=pulse_batch)
            direct_stage = opt_info.get("direct_stage", {})
            if direct_stage.get("enabled") and direct_stage.get("actions") is not None:
                stage_parts["Opt1"]["actions"].append(direct_stage["actions"].detach())
                stage_parts["Opt1"]["preds"].append(direct_stage["preds"].detach())
                stage_parts["Opt1"]["loss"].append(direct_stage["loss_batch"].detach())
                direct_detail = direct_stage.get("detail", {})
                for key in stage_parts["Opt1"]["info"]:
                    stage_parts["Opt1"]["info"][key].append(direct_detail[key].detach())

            if opt_info.get("refine_stage_enabled", False):
                stage_parts["Opt2"]["preds"].append(opt_preds.detach())
                stage_parts["Opt2"]["actions"].append(opt_actions.detach())
                stage_parts["Opt2"]["loss"].append(opt_info["final_loss_batch"].detach())
                for key in stage_parts["Opt2"]["info"]:
                    stage_parts["Opt2"]["info"][key].append(opt_info[key].detach())

            total_time_cost += float(opt_info.get("time_cost", 0.0))
            trajectory_all.extend(opt_info.get("trajectory", []))

    output = {"total_time_cost": total_time_cost, "trajectory_all": trajectory_all}
    for prefix, content in stage_parts.items():
        output[prefix] = {
            "preds": torch.cat(content["preds"], dim=0) if content["preds"] else None,
            "actions": torch.cat(content["actions"], dim=0) if content["actions"] else None,
            "loss": cat_or_nan(content["loss"]),
            "info": {key: cat_or_nan(parts) for key, parts in content["info"].items()},
        }
    return output


def _expand_stage_outputs_to_full(
    stage_outputs: Dict[str, object],
    valid_indices: np.ndarray,
    total_count: int,
    trainable_dim: int,
) -> Dict[str, object]:
    valid_indices_tensor = torch.as_tensor(valid_indices, dtype=torch.long)
    expanded = {
        "total_time_cost": stage_outputs["total_time_cost"],
        "trajectory_all": stage_outputs["trajectory_all"],
    }
    for stage_name in ["Base", "Opt1", "Opt2"]:
        stage = stage_outputs[stage_name]
        preds_src = stage["preds"]
        actions_src = stage["actions"]
        preds_full = None
        actions_full = None
        if preds_src is not None:
            preds_src_cpu = preds_src.detach().cpu()
            preds_full = torch.full((total_count, preds_src_cpu.shape[1]), float("nan"), dtype=preds_src_cpu.dtype)
            preds_full[valid_indices_tensor] = preds_src_cpu
        if actions_src is not None:
            actions_src_cpu = actions_src.detach().cpu()
            actions_full = torch.full((total_count, trainable_dim), float("nan"), dtype=actions_src_cpu.dtype)
            actions_full[valid_indices_tensor] = actions_src_cpu
        loss_full = torch.full((total_count,), float("nan"), dtype=torch.float32)
        loss_src = stage["loss"]
        if loss_src.numel() > 0:
            loss_full[valid_indices_tensor] = loss_src.detach().cpu().to(torch.float32)
        info_full = {}
        for key, values in stage["info"].items():
            buffer = torch.full((total_count,), float("nan"), dtype=torch.float32)
            if values.numel() > 0:
                buffer[valid_indices_tensor] = values.detach().cpu().to(torch.float32)
            info_full[key] = buffer
        expanded[stage_name] = {
            "preds": preds_full,
            "actions": actions_full,
            "loss": loss_full,
            "info": info_full,
        }
    return expanded


def _build_stage_metric_dataframe(
    stage_label: str,
    stage: Dict[str, object],
    row_count: int,
    ot_array: np.ndarray,
) -> pd.DataFrame:
    preds = stage["preds"]
    info = stage["info"]

    if preds is None:
        nan_array = np.full(row_count, np.nan, dtype=np.float32)
        return pd.DataFrame(
            {
                f"{stage_label}_HIC": nan_array.copy(),
                f"{stage_label}_Dmax": nan_array.copy(),
                f"{stage_label}_Nij": nan_array.copy(),
                f"{stage_label}_Phead": nan_array.copy(),
                f"{stage_label}_Pchest": nan_array.copy(),
                f"{stage_label}_Pneck": nan_array.copy(),
                f"{stage_label}_JointRisk": nan_array.copy(),
                f"{stage_label}_AIS_head": nan_array.copy(),
                f"{stage_label}_AIS_chest": nan_array.copy(),
                f"{stage_label}_AIS_neck": nan_array.copy(),
                f"{stage_label}_AIS_max": nan_array.copy(),
            }
        )

    pred_array = preds.detach().cpu().numpy()
    info_arrays = {name: info[name].detach().cpu().numpy() for name in ["p_head", "p_chest", "p_neck", "joint_risk"]}
    invalid_rows = np.isnan(pred_array).any(axis=1)

    stage_df = pd.DataFrame(
        {
            f"{stage_label}_HIC": pred_array[:, 0],
            f"{stage_label}_Dmax": pred_array[:, 1],
            f"{stage_label}_Nij": pred_array[:, 2],
            f"{stage_label}_Phead": info_arrays["p_head"],
            f"{stage_label}_Pchest": info_arrays["p_chest"],
            f"{stage_label}_Pneck": info_arrays["p_neck"],
            f"{stage_label}_JointRisk": info_arrays["joint_risk"],
        }
    )
    stage_df[f"{stage_label}_AIS_head"] = AIS_cal_head(stage_df[f"{stage_label}_HIC"].to_numpy(dtype=np.float32))
    stage_df[f"{stage_label}_AIS_chest"] = AIS_cal_chest(stage_df[f"{stage_label}_Dmax"].to_numpy(dtype=np.float32), ot_array)
    stage_df[f"{stage_label}_AIS_neck"] = AIS_cal_neck(stage_df[f"{stage_label}_Nij"].to_numpy(dtype=np.float32))
    stage_df[f"{stage_label}_AIS_max"] = np.maximum.reduce(
        [
            stage_df[f"{stage_label}_AIS_head"].to_numpy(dtype=np.float32),
            stage_df[f"{stage_label}_AIS_chest"].to_numpy(dtype=np.float32),
            stage_df[f"{stage_label}_AIS_neck"].to_numpy(dtype=np.float32),
        ]
    )
    if np.any(invalid_rows):
        stage_df.loc[invalid_rows, [
            f"{stage_label}_AIS_head",
            f"{stage_label}_AIS_chest",
            f"{stage_label}_AIS_neck",
            f"{stage_label}_AIS_max",
        ]] = np.nan
    return stage_df


def _build_result_dataframe(
    df_input: pd.DataFrame,
    context_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    stage_outputs: Dict[str, object],
    truth_arrays: Dict[str, np.ndarray],
    trainable_names: List[str],
    optimized_stage_name: str,
) -> tuple[pd.DataFrame, Dict[str, float], str]:
    excluded_input_cols = set(FEATURE_ORDER) | {"y_HIC", "y_Dmax", "y_Nij", "ais_head", "ais_chest", "ais_neck"}
    metadata_cols = [col for col in df_input.columns if col not in excluded_input_cols]
    metadata_df = df_input[metadata_cols].reset_index(drop=True)
    frame_parts = [metadata_df, context_df.reset_index(drop=True)]

    ot_array = pd.to_numeric(context_df["OT"], errors="coerce").to_numpy(dtype=np.float32)
    optimized_stage = stage_outputs[optimized_stage_name]

    baseline_df = baseline_df.reset_index(drop=True).rename(columns={name: f"Base_{name}" for name in trainable_names})
    frame_parts.append(baseline_df)

    opt_actions = optimized_stage["actions"]
    if opt_actions is None:
        opt_action_df = pd.DataFrame({f"Opt_{name}": np.full(len(df_input), np.nan, dtype=np.float32) for name in trainable_names})
    else:
        opt_array = opt_actions.detach().cpu().numpy()
        opt_action_df = pd.DataFrame({f"Opt_{name}": opt_array[:, idx] for idx, name in enumerate(trainable_names)})
    frame_parts.append(opt_action_df.reset_index(drop=True))

    frame_parts.append(_build_stage_metric_dataframe("Base", stage_outputs["Base"], len(df_input), ot_array).reset_index(drop=True))
    frame_parts.append(_build_stage_metric_dataframe("Opt", optimized_stage, len(df_input), ot_array).reset_index(drop=True))

    result_df = pd.concat(frame_parts, axis=1)

    if all(key in truth_arrays for key in ["y_HIC", "y_Dmax", "y_Nij"]):
        truth_df = pd.DataFrame(
            [
                np.asarray(truth_arrays["y_HIC"], dtype=np.float32),
                np.asarray(truth_arrays["y_Dmax"], dtype=np.float32),
                np.asarray(truth_arrays["y_Nij"], dtype=np.float32),
            ],
            index=["True_HIC", "True_Dmax", "True_Nij"],
        ).T.reset_index(drop=True)
        truth_df["True_AIS_head"] = np.asarray(
            truth_arrays.get("ais_head", AIS_cal_head(truth_df["True_HIC"].to_numpy(dtype=np.float32)))
        )
        truth_df["True_AIS_chest"] = np.asarray(
            truth_arrays.get("ais_chest", AIS_cal_chest(truth_df["True_Dmax"].to_numpy(dtype=np.float32), ot_array))
        )
        truth_df["True_AIS_neck"] = np.asarray(
            truth_arrays.get("ais_neck", AIS_cal_neck(truth_df["True_Nij"].to_numpy(dtype=np.float32)))
        )
        truth_df["True_AIS_max"] = np.maximum.reduce(
            [
                truth_df["True_AIS_head"].to_numpy(dtype=np.float32),
                truth_df["True_AIS_chest"].to_numpy(dtype=np.float32),
                truth_df["True_AIS_neck"].to_numpy(dtype=np.float32),
            ]
        )
        for prefix in ["Base", "Opt"]:
            for name in ["HIC", "Dmax", "Nij"]:
                truth_df[f"True_vs_{prefix}_{name}"] = truth_df[f"True_{name}"] - result_df[f"{prefix}_{name}"]
        result_df = pd.concat([result_df, truth_df], axis=1)

    reduction_specs = [
        ("HIC", "Base_HIC", "Opt_HIC"),
        ("Dmax", "Base_Dmax", "Opt_Dmax"),
        ("Nij", "Base_Nij", "Opt_Nij"),
        ("Phead", "Base_Phead", "Opt_Phead"),
        ("Pchest", "Base_Pchest", "Opt_Pchest"),
        ("Pneck", "Base_Pneck", "Opt_Pneck"),
        ("JointRisk", "Base_JointRisk", "Opt_JointRisk"),
        ("AIS_head", "Base_AIS_head", "Opt_AIS_head"),
        ("AIS_chest", "Base_AIS_chest", "Opt_AIS_chest"),
        ("AIS_neck", "Base_AIS_neck", "Opt_AIS_neck"),
        ("AIS_max", "Base_AIS_max", "Opt_AIS_max"),
    ]
    reduction_data = {}
    for alias, base_col, opt_col in reduction_specs:
        reduction_abs = result_df[base_col] - result_df[opt_col]
        reduction_data[f"Reduction_{alias}"] = reduction_abs

    if reduction_data:
        result_df = pd.concat([result_df, pd.DataFrame(reduction_data, index=df_input.index).reset_index(drop=True)], axis=1)

    summary = {
        "optimized_stage_source": optimized_stage_name,
        "mean_reduction_HIC": _safe_nanmean(result_df["Reduction_HIC"]),
        "mean_reduction_Dmax": _safe_nanmean(result_df["Reduction_Dmax"]),
        "mean_reduction_Nij": _safe_nanmean(result_df["Reduction_Nij"]),
        "mean_reduction_Phead": _safe_nanmean(result_df["Reduction_Phead"]),
        "mean_reduction_Pchest": _safe_nanmean(result_df["Reduction_Pchest"]),
        "mean_reduction_Pneck": _safe_nanmean(result_df["Reduction_Pneck"]),
        "mean_reduction_joint_risk": _safe_nanmean(result_df["Reduction_JointRisk"]),
        "mean_base_joint_risk": _safe_nanmean(result_df["Base_JointRisk"]),
        "mean_opt_joint_risk": _safe_nanmean(result_df["Opt_JointRisk"]),
        "mean_base_ais_max": _safe_nanmean(result_df["Base_AIS_max"]),
        "mean_opt_ais_max": _safe_nanmean(result_df["Opt_AIS_max"]),
        "n_samples": int(len(result_df)),
    }
    return result_df, summary, optimized_stage_name


def _build_eval_info(
    args,
    output_dir: Path,
    output_csv_path: Path,
    config_snapshots: Dict[str, str],
    input_source: Dict[str, str],
    strategy_ckpt_path: Optional[Path],
    config: dict,
    param_manager: ParamManager,
    context_names: List[str],
    trainable_names: List[str],
    missing_context: List[str],
    missing_trainable: List[str],
    skipped_case_rows: List[int],
    reverted_baseline_rows: List[int],
    opt1_generated: bool,
    opt2_generated: bool,
    optimized_stage_name: str,
    surrogate: SurrogateAdapter,
    summary_metrics: Dict[str, float],
    stage_outputs: Dict[str, object],
    result_row_count: int,
) -> Dict[str, object]:
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(output_dir),
        "output_csv": str(output_csv_path),
        "input_source": input_source,
        "strategy_checkpoint_path": str(strategy_ckpt_path) if strategy_ckpt_path is not None else None,
        "direct_inference": bool(config.get("optimization", {}).get("direct_inference", False)),
        "config_files": config_snapshots,
        "evaluation_config": config.get("evaluation", {}),
        "optimization_config": config.get("optimization", {}),
        "parameter_roles": {
            "context": context_names,
            "trainable_control": trainable_names,
            "fixed_control": param_manager.get_fixed_control_names(),
        },
        "missing_context_filled_by_default": missing_context,
        "missing_trainable_filled_by_default": missing_trainable,
        "skipped_case_rows": skipped_case_rows,
        "skipped_cases_count": len(skipped_case_rows),
        "reverted_baseline_rows": reverted_baseline_rows,
        "input_validation_policy": {
            "input_csv": "rowwise_skip_invalid_context_rowwise_revert_invalid_baseline" if args.input_csv else None,
            "test_split": "rowwise_skip_invalid_internal_cases" if not args.input_csv else None,
        },
        "stage_definition": {
            "Base": "输入 CSV 提供的 baseline control；缺失时回填 default",
            "Opt1": "策略网络直推结果；仅在 direct_inference=True 且权重可用时存在",
            "Opt2": "局部精调结果；仅在 refine_steps>0 时存在",
        },
        "stage_status": {
            "base_generated": True,
            "opt1_generated": opt1_generated,
            "opt2_generated": opt2_generated,
            "reported_optimized_stage": optimized_stage_name,
        },
        "distribution_penalty": {
            "enabled_after_fit": bool(surrogate.distribution_penalty.enabled),
            "feature_space": surrogate.distribution_penalty.feature_space,
        },
        "summary_metrics": summary_metrics,
        "formulas": {
            "joint_risk": "L_risk = 1 - Π_k (1 - P_k)",
            "reported_reduction": "mean(Base - ReportedOpt)",
        },
        "runtime": {
            "total_time_cost_sec": float(stage_outputs["total_time_cost"]),
            "avg_time_cost_sec": float(stage_outputs["total_time_cost"] / max(1, result_row_count)),
            "trajectory_steps_logged": len(stage_outputs["trajectory_all"]),
        },
    }


def main():
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    logger = setup_logger(name="ARS_optim.eval")
    cfg_path = base_dir / "configs" / "default_config.yaml"
    param_space_path = base_dir / "configs" / "param_space.yaml"

    if not cfg_path.is_file():
        raise FileNotFoundError(f"config file not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    config.setdefault("optimization", {})
    if args.direct_inference:
        config["optimization"]["direct_inference"] = True

    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    param_manager = ParamManager(param_space_path)
    constraint_engine = ConstraintEngine(param_manager)
    data_processor = UnifiedDataProcessor(str(NORMALIZATION_CONFIG_PATH))
    pulse_model, injury_model = load_surrogate_models(config=config, device=device)
    surrogate = SurrogateAdapter(
        pulse_model=pulse_model,
        injury_model=injury_model,
        param_manager=param_manager,
        constraint_engine=constraint_engine,
        config=config,
        data_processor=data_processor,
    ).to(device)

    ref_sampler = StateDataSampler(
        param_manager=param_manager,
        constraint_engine=constraint_engine,
        batch_size=1024,
        device=device,
        seed=int(config.get("seed", 42)),
        split_indices_path=str(SPLIT_INDICES_DIR / "injury_train_indices.csv"),
        jitter_ratio=0.0,
        jitter_prob=0.0,
    )
    _fit_distribution_reference_if_needed(surrogate, ref_sampler, param_manager, config)

    strategy_ckpt_path = _resolve_strategy_ckpt(args, base_dir, config)
    strategy_net = None
    if bool(config.get("optimization", {}).get("direct_inference", False)):
        if strategy_ckpt_path is None:
            raise ValueError("direct_inference=True 时必须显式提供 strategy_checkpoint 或 --strategy_ckpt")
        if not strategy_ckpt_path.is_file():
            raise FileNotFoundError(f"策略权重不存在: {strategy_ckpt_path}")
        strategy_net = build_strategy_net_from_config(
            param_manager=param_manager,
            constraint_engine=constraint_engine,
            data_processor=data_processor,
            config=config,
        ).to(device)
        try:
            strategy_net.load_state_dict(torch.load(str(strategy_ckpt_path), map_location=device, weights_only=True))
        except Exception as exc:
            raise RuntimeError(f"策略权重与当前参数空间不兼容: {strategy_ckpt_path}") from exc

    optimizer = LocalRefiner(
        config=config,
        param_manager=param_manager,
        constraint_engine=constraint_engine,
        surrogate=surrogate,
        strategy_net=strategy_net,
    )

    df_input, truth_arrays, input_source = _load_eval_input(args, logger)

    if "case_id" not in df_input.columns:
        df_input.insert(0, "case_id", np.arange(len(df_input), dtype=np.int64))

    context_output_df, context_eval_df, baseline_df, missing_context, missing_trainable, valid_mask, skipped_case_rows, reverted_baseline_rows = _prepare_eval_inputs(
        df_input=df_input,
        param_manager=param_manager,
        constraint_engine=constraint_engine,
        device=device,
        logger=logger,
        strict_provided_validation=bool(args.input_csv),
    )
    context_names = param_manager.get_context_names()
    trainable_names = param_manager.get_trainable_names()
    valid_indices = np.flatnonzero(valid_mask)
    context_tensor = torch.tensor(context_eval_df.loc[valid_indices, context_names].values, dtype=torch.float32, device=device)
    baseline_tensor = torch.tensor(baseline_df.loc[valid_indices, trainable_names].values, dtype=torch.float32, device=device)

    eval_batch_size = int(config.get("evaluation", {}).get("eval_batch_size", 512))
    if eval_batch_size <= 0:
        raise ValueError("eval_batch_size 必须为正整数")
    stage_outputs_valid = _compute_predictions_batch(
        context_tensor=context_tensor,
        baseline_trainable=baseline_tensor,
        surrogate=surrogate,
        optimizer=optimizer,
        eval_batch_size=eval_batch_size,
    )
    stage_outputs = _expand_stage_outputs_to_full(
        stage_outputs=stage_outputs_valid,
        valid_indices=valid_indices,
        total_count=len(df_input),
        trainable_dim=len(trainable_names),
    )

    if optimizer.refine_steps > 0:
        optimized_stage_name = "Opt2"
    elif optimizer.direct_inference:
        optimized_stage_name = "Opt1"
    else:
        raise ValueError("当前配置未启用任何优化阶段，无法组织评估输出")

    result_df, summary_metrics, optimized_stage_name = _build_result_dataframe(
        df_input=df_input,
        context_df=context_output_df,
        baseline_df=baseline_df,
        stage_outputs=stage_outputs,
        truth_arrays=truth_arrays,
        trainable_names=trainable_names,
        optimized_stage_name=optimized_stage_name,
    )
    # 在所有前置校验和模型推理都完成后再创建输出目录，避免早退时留下空目录。
    output_dir = _build_output_dir(base_dir, args.input_csv)
    output_csv_path = output_dir / Path(args.output_csv).name
    config_snapshots = _copy_config_snapshots(cfg_path, param_space_path, output_dir)
    result_df.to_csv(str(output_csv_path), index=False)
    opt1_generated = stage_outputs["Opt1"]["preds"] is not None
    opt2_generated = stage_outputs["Opt2"]["preds"] is not None

    eval_info = _build_eval_info(
        args=args,
        output_dir=output_dir,
        output_csv_path=output_csv_path,
        config_snapshots=config_snapshots,
        input_source=input_source,
        strategy_ckpt_path=strategy_ckpt_path,
        config=config,
        param_manager=param_manager,
        context_names=context_names,
        trainable_names=trainable_names,
        missing_context=missing_context,
        missing_trainable=missing_trainable,
        skipped_case_rows=skipped_case_rows,
        reverted_baseline_rows=reverted_baseline_rows,
        opt1_generated=opt1_generated,
        opt2_generated=opt2_generated,
        optimized_stage_name=optimized_stage_name,
        surrogate=surrogate,
        summary_metrics=summary_metrics,
        stage_outputs=stage_outputs,
        result_row_count=len(result_df),
    )
    _write_yaml(output_dir / "eval_info.yaml", eval_info)
    logger.info(f"评估完成，结果目录: {output_dir}")
    logger.info(f"结果 CSV: {output_csv_path}")


if __name__ == "__main__":
    main()
