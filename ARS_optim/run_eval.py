import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = 'T'
import warnings
warnings.filterwarnings('ignore')
import argparse
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from common.data_utils.split_io import load_int_vector_csv
from common.data_utils.processor import UnifiedDataProcessor
from common.metrics.injury_risk import (
    AIS_cal_chest,
    AIS_cal_head,
    AIS_cal_neck,
    Injury_prob_cal_chest,
    Injury_prob_cal_head,
    Injury_prob_cal_neck,
)
from common.settings import FEATURE_ORDER, NORMALIZATION_CONFIG_PATH, RAW_DATA_DIR, SPLIT_INDICES_DIR
from common.tools.logger import setup_logger
from common.tools.seeding import set_random_seed

from InjuryPredict.utils.tools import convert_numpy_types

from ARS_optim.src.constraints import ConstraintEngine
from ARS_optim.src.data_sampler import StateDataSampler
from ARS_optim.src.optimizer import LocalRefiner
from ARS_optim.src.param_manager import ParamManager
from ARS_optim.src.strategy_net import build_strategy_net_from_config, load_strategy_run_config
from ARS_optim.src.surrogate import SurrogateAdapter, load_surrogate_models


STAGE_NAMES = ("Base", "Opt1", "Opt2")
STAGE_INFO_KEYS = ("p_head", "p_chest", "p_neck", "joint_risk")
OPT_STAGE_NAMES = STAGE_NAMES[1:]
RESULT_METRIC_NAMES = (
    "HIC",
    "Dmax",
    "Nij",
    "Phead",
    "Pchest",
    "Pneck",
    "JointRisk",
    "AIS_head",
    "AIS_chest",
    "AIS_neck",
    "MAIS",
)
TRUE_VS_METRIC_NAMES = ("HIC", "Dmax", "Nij")


def parse_args():
    parser = argparse.ArgumentParser(description="ARS Local Refinement Evaluator")
    parser.add_argument("--input_csv", type=str, default=None, help="输入工况参数 CSV 的绝对路径；若不提供则自动使用 injury test split")
    parser.add_argument("--output_csv", type=str, default="evaluation_results.csv", help="输出 CSV 文件名")
    parser.add_argument("--strategy_ckpt", type=str, default=None, help="策略网络权重绝对路径")
    parser.add_argument("--direct_inference", action="store_true", help="强制启用策略网络直推")
    return parser.parse_args()


def _build_output_dir(base_dir: Path, input_csv: Optional[str]) -> Path:
    if input_csv:
        stem = Path(input_csv).stem
        normalized = [char if char.isalnum() or char in {"_", "-"} else "_" for char in stem]
        suffix = "".join(normalized).strip("_") or "evaluation"
    else:
        suffix = "injury_test_split"
    output_dir = base_dir / "saved_eval" / f"eval_{suffix}_{datetime.now().strftime('%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def _resolve_strategy_ckpt(args, base_dir: Path, config: Dict) -> Optional[Path]:
    """解析评估阶段允许使用的策略权重路径。

    优先级固定为：CLI 显式指定 > config.evaluation.strategy_checkpoint > 关闭策略直推。
    这里不做自动搜目录或按时间戳兜底，避免评估在用户不知情时悄悄换了一份权重。
    """
    if args.strategy_ckpt:
        return Path(args.strategy_ckpt).resolve()

    eval_cfg = config.get("evaluation", {}) or {}
    configured = eval_cfg.get("strategy_checkpoint")
    if configured:
        candidate = Path(str(configured))
        return candidate if candidate.is_absolute() else (base_dir / candidate).resolve()
    return None


def _assert_strategy_runtime_compatibility(
    current_param_space_path: Path,
    current_normalization_path: Path,
    strategy_artifacts: Dict[str, Path],
) -> None:
    """确保当前评估运行环境与策略权重保存时的结构性配置一致。

    策略网络不仅依赖 `config.yaml` 里的层参数，还隐式依赖：
    - `param_space.yaml` 中的 context/trainable 角色划分与输出维度；
    - `normalization_config.json` 中的离散特征类别定义。

    因此评估阶段若继续使用当前工作区里的不同版本配置，会出现
    “权重来自 A 目录，但网络结构按 B 配置重建”的隐性错配。
    这里直接做严格比对，不做静默 fallback。
    """
    comparisons = (
        ("param_space", current_param_space_path, strategy_artifacts["param_space"]),
        ("normalization_config", current_normalization_path, strategy_artifacts["normalization_config"]),
    )
    for label, current_path, saved_path in comparisons:
        if current_path.read_bytes() != saved_path.read_bytes():
            raise ValueError(
                f"当前 {label} 与策略权重目录中的快照不一致。\n"
                f"current: {current_path}\n"
                f"saved: {saved_path}\n"
                "请切换到与该策略权重配套的配置后再评估。"
            )


def _copy_config_snapshots(cfg_path: Path, param_space_path: Path, output_dir: Path) -> Dict[str, str]:
    config_dir = output_dir / "configs_used"
    config_dir.mkdir(parents=True, exist_ok=False)
    cfg_snapshot = config_dir / "config.yaml"
    param_snapshot = config_dir / "param_space.yaml"
    norm_snapshot = config_dir / "normalization_config.json"
    shutil.copy2(str(cfg_path), str(cfg_snapshot))
    shutil.copy2(str(param_space_path), str(param_snapshot))
    shutil.copy2(str(NORMALIZATION_CONFIG_PATH), str(norm_snapshot))
    return {
        "config_used": str(cfg_snapshot),
        "param_space_used": str(param_snapshot),
        "normalization_config": str(norm_snapshot),
    }


def _build_stage_summary_metrics(result_df: pd.DataFrame, available_opt_stages: List[str]) -> Dict[str, object]:
    def safe_nanmean(series: pd.Series) -> float:
        values = np.asarray(series, dtype=np.float32)
        return float(np.nan) if np.isnan(values).all() else float(np.nanmean(values))

    metrics = {
        "stage_mean": {},
        "reduction_vs_base": {},
    }
    stage_names = ["Base"] + available_opt_stages
    for stage_name in stage_names:
        metrics["stage_mean"][stage_name] = {
            metric_name: safe_nanmean(result_df[f"{stage_name}_{metric_name}"])
            for metric_name in RESULT_METRIC_NAMES
        }
    for stage_name in available_opt_stages:
        metrics["reduction_vs_base"][stage_name] = {
            metric_name: safe_nanmean(result_df[f"Reduction_{stage_name}_{metric_name}"])
            for metric_name in RESULT_METRIC_NAMES
        }
    return metrics


def _build_top_cases_summary(
    result_df: pd.DataFrame,
    optimized_stage_name: Optional[str],
    top_n: int = 5,
) -> List[Dict[str, object]]:
    if optimized_stage_name is None:
        return []
    joint_col = f"Reduction_{optimized_stage_name}_JointRisk"
    mais_col = f"Reduction_{optimized_stage_name}_MAIS"
    if joint_col not in result_df.columns or mais_col not in result_df.columns:
        return []
    top_df = (
        result_df.loc[result_df[joint_col].notna(), ["case_id", joint_col, mais_col]]
        .sort_values(by=joint_col, ascending=False)
        .head(top_n)
    )
    output = []
    for _, row in top_df.iterrows():
        output.append(
            {
                "case_id": row["case_id"],
                "reduction_joint_risk": float(row[joint_col]),
                "reduction_mais": float(row[mais_col]),
            }
        )
    return output


def _build_param_dataframe(
    df_input: pd.DataFrame,
    params: List[dict],
) -> tuple[pd.DataFrame, List[str]]:
    """按参数清单抽取输入列，并把缺失列显式保留为 NaN。

    这里故意不提供“缺列自动回填 default”的分支。
    当前 `run_eval` 的正式语义已经固定为：
    - context 缺列/缺值 -> 逐行跳过；
    - baseline trainable 缺列/缺值 -> 整行回退 default。

    因此辅助函数只负责列抽取，后续如何处理 NaN 由调用方按阶段语义显式决定，
    避免低层工具再次长出隐性 fallback。
    """
    output_df = pd.DataFrame(index=df_input.index)
    missing = []
    for param in params:
        name = param["name"]
        if name in df_input.columns:
            output_df[name] = pd.to_numeric(df_input[name], errors="coerce")
        else:
            output_df[name] = np.nan
            missing.append(name)
    return output_df, missing


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
    """整理评估输入。

    strict_provided_validation=True 只用于外部 input_csv：
    - 已提供的 context 值必须逐行合法，否则跳过该 case；
    - baseline 只有在“整组 trainable control 列齐全且整组合法”时才采用用户值；
    - 只要 baseline 缺列、缺值或整组不合法，就整行回退到 default；
    - 输入端校验只按完整物理定义域执行，不额外要求落在优化子范围内。

    strict_provided_validation=False 只用于内部 test split：
    - baseline 直接使用测试集已有完整参数；
    - 只检查缺失值，不再按优化子范围做过滤。
    """
    context_params = param_manager.get_context_params()
    trainable_params = param_manager.get_trainable_params()
    context_names = param_manager.get_context_names()
    trainable_names = param_manager.get_trainable_names()
    default_trainable = np.asarray(param_manager.get_default_values(trainable_params), dtype=np.float32)

    context_df_raw, missing_context = _build_param_dataframe(
        df_input,
        context_params,
    )
    baseline_df_raw, missing_trainable = _build_param_dataframe(
        df_input,
        trainable_params,
    )

    if missing_context:
        if strict_provided_validation:
            logger.warning("input_csv 缺失 context 列，这些列对应的 case 将被逐行跳过: %s", missing_context)
        else:
            raise ValueError(f"测试集缺失 context 列: {missing_context}")
    if missing_trainable:
        if strict_provided_validation:
            logger.warning("input_csv 缺失部分 trainable control 列；这些 case 的 baseline 将整组回退为 default: %s", missing_trainable)
        else:
            raise ValueError(f"测试集缺失 trainable control 列: {missing_trainable}")

    if not strict_provided_validation:
        context_eval_df = context_df_raw.astype(np.float32)
        baseline_eval_df = baseline_df_raw.astype(np.float32)
        missing_context_rows = context_eval_df[context_names].isna().any(axis=1).to_numpy()
        missing_trainable_rows = baseline_eval_df[trainable_names].isna().any(axis=1).to_numpy()
        if missing_context_rows.any() or missing_trainable_rows.any():
            bad_rows = np.flatnonzero(missing_context_rows | missing_trainable_rows)
            raise ValueError(
                "测试集样本存在缺失值，不符合评估前提。行号: "
                f"{_format_row_list(bad_rows)}"
            )
        valid_mask = np.ones(len(df_input), dtype=bool)
        return (
            context_eval_df,
            context_eval_df,
            baseline_eval_df,
            missing_context,
            missing_trainable,
            valid_mask,
            [],
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
        # 外部 input_csv 的 context 校验只针对 context 本身，不再补默认 trainable control。
        # 这样 trainable 属性未来变化时，输入端语义仍然稳定：只看当前真正属于 context 的列。
        context_valid_local = constraint_engine.validate_context_input(candidate_context_tensor).detach().cpu().numpy().astype(bool)
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
            context_eval_df.loc[legal_context_indices, context_names] = context_df_raw.loc[
                legal_context_indices,
                context_names,
            ].to_numpy(dtype=np.float32)

            baseline_candidate = baseline_df_raw.loc[legal_context_indices, trainable_names].copy()
            # baseline 采用严格整组策略：只要 trainable 列缺失、该行存在 NaN，
            # 或整组不满足输入端完整物理定义域，就整行回退为 default。
            # 这里故意不做“用户部分列 + default 其余列”的混搭，避免结果表出现隐性语义。
            baseline_has_missing_local = baseline_candidate[trainable_names].isna().any(axis=1).to_numpy()

            legal_context_tensor = torch.tensor(
                context_eval_df.loc[legal_context_indices, context_names].values,
                dtype=torch.float32,
                device=device,
            )
            baseline_valid_local = np.zeros(len(legal_context_indices), dtype=bool)
            baseline_complete_local = ~baseline_has_missing_local
            if baseline_complete_local.any():
                complete_row_indices = np.flatnonzero(baseline_complete_local)
                baseline_complete_tensor = torch.tensor(
                    baseline_candidate.iloc[complete_row_indices][trainable_names].values,
                    dtype=torch.float32,
                    device=device,
                )
                baseline_full = constraint_engine.compose_full_features(
                    legal_context_tensor[complete_row_indices],
                    baseline_complete_tensor,
                )
                baseline_valid_local[complete_row_indices] = (
                    constraint_engine.validate_full_input(baseline_full)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(bool)
                )

            baseline_invalid_mask_local = ~baseline_valid_local
            if baseline_invalid_mask_local.any():
                logger.warning(
                    "以下 input_csv 行的 baseline trainable control 缺失或整组不合法，已整行回退为 param_space.yaml 的 default。行号: %s",
                    _format_row_list(legal_context_indices[baseline_invalid_mask_local]),
                )
            reverted_indices = legal_context_indices[baseline_invalid_mask_local]
            if reverted_indices.size > 0:
                reverted_baseline_rows.extend(reverted_indices.astype(int).tolist())

            accepted_indices = legal_context_indices[baseline_valid_local]
            if accepted_indices.size > 0:
                baseline_eval_df.loc[accepted_indices, trainable_names] = baseline_candidate.loc[
                    accepted_indices,
                    trainable_names,
                ].to_numpy(dtype=np.float32)

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
    if test_indices.size == 0:
        raise ValueError("自动测试集模式对应的 injury_test_indices.csv 为空；请提供 --input_csv 或重新生成划分。")
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
    """按批次计算 Base/Opt1/Opt2 三个阶段的预测结果。

    这里显式把 Base、Opt1、Opt2 分开缓存。
    结果 CSV 会直接按真实生成阶段展开，而不是再压缩成单一 `Opt_` 口径。

    局部精调虽然在实现上按 batch 调用一次 `optimizer.optimize`，但优化器内部维护的是
    `[batch_size, trainable_dim]` 形状的潜变量，并按样本逐行计算损失与 Adam 状态。
    也就是说，这里只是把多个逐点优化并行向量化，而不是把一个 batch 当成单个分布目标来优化。
    """
    sample_count = context_tensor.shape[0]
    device = context_tensor.device
    stage_parts = {
        key: {
            "preds": [],
            "actions": [],
            "loss": [],
            "info": {name: [] for name in STAGE_INFO_KEYS},
        }
        for key in STAGE_NAMES
    }
    total_time_cost = 0.0
    trajectory_all: List[float] = []

    def cat_or_nan(parts: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat(parts, dim=0) if parts else torch.full((sample_count,), float("nan"), device=device)

    batch_starts = range(0, sample_count, eval_batch_size)
    for start in tqdm(batch_starts, total=(sample_count + eval_batch_size - 1) // eval_batch_size, desc="Evaluating batches"):
        end = min(start + eval_batch_size, sample_count)
        context_batch = context_tensor[start:end]
        baseline_batch = baseline_trainable[start:end]

        with torch.no_grad():
            pulse_batch = surrogate.generate_pulse(context_batch)
            base_loss, base_preds, base_info = surrogate.predict_injury_and_loss(context_batch, baseline_batch, pulse_batch)
        stage_parts["Base"]["preds"].append(base_preds.detach())
        stage_parts["Base"]["actions"].append(baseline_batch.detach())
        stage_parts["Base"]["loss"].append(base_loss.detach())
        for key in STAGE_INFO_KEYS:
            stage_parts["Base"]["info"][key].append(base_info[key].detach())

        if optimizer.direct_inference or optimizer.refine_steps > 0:
            opt_actions, opt_preds, opt_info = optimizer.optimize(context_batch, pulse_norm=pulse_batch)
            direct_stage = opt_info.get("direct_stage")
            if direct_stage is not None:
                stage_parts["Opt1"]["preds"].append(direct_stage["preds"].detach())
                stage_parts["Opt1"]["actions"].append(direct_stage["actions"].detach())
                stage_parts["Opt1"]["loss"].append(direct_stage["loss_batch"].detach())
                for key in STAGE_INFO_KEYS:
                    stage_parts["Opt1"]["info"][key].append(direct_stage["detail"][key].detach())

            if opt_info.get("refine_stage_enabled", False):
                stage_parts["Opt2"]["preds"].append(opt_preds.detach())
                stage_parts["Opt2"]["actions"].append(opt_actions.detach())
                stage_parts["Opt2"]["loss"].append(opt_info["final_loss_batch"].detach())
                for key in STAGE_INFO_KEYS:
                    stage_parts["Opt2"]["info"][key].append(opt_info[key].detach())

            total_time_cost += float(opt_info.get("time_cost", 0.0))
            trajectory_all.extend(opt_info.get("trajectory", []))

    output = {"total_time_cost": total_time_cost, "trajectory_all": trajectory_all}
    for prefix in STAGE_NAMES:
        content = stage_parts[prefix]
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
    """把“仅对有效 case 计算出来的阶段结果”扩回完整行数。

    前面 input_csv 模式可能按行跳过非法 context，因此模型真正推理的张量只覆盖 valid_indices。
    结果导出时仍需保留原 CSV 的完整行顺序，所以这里把有效结果写回对应位置，
    其余 skipped 行统一填 NaN，保证结果表、summary 和 warning 行号能一一对应。
    """
    valid_indices_tensor = torch.as_tensor(valid_indices, dtype=torch.long)
    expanded = {
        "total_time_cost": stage_outputs["total_time_cost"],
        "trajectory_all": stage_outputs["trajectory_all"],
    }
    for stage_name in STAGE_NAMES:
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
    """把单阶段输出整理成结果表所需的损伤、风险和 AIS 列。"""
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
                f"{stage_label}_MAIS": nan_array.copy(),
            }
        )

    pred_array = preds.detach().cpu().numpy()
    info_arrays = {name: info[name].detach().cpu().numpy() for name in STAGE_INFO_KEYS}
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
    stage_df[f"{stage_label}_MAIS"] = np.maximum.reduce(
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
            f"{stage_label}_MAIS",
        ]] = np.nan
    return stage_df


def _build_truth_metric_dataframe(
    truth_arrays: Dict[str, np.ndarray],
    ot_array: np.ndarray,
) -> pd.DataFrame:
    """把测试集真值整理成与各阶段同口径的损伤、风险和 AIS 列。

    run_eval 的结果表需要同时支持两类用途：
    1. 比较 Base、Opt1、Opt2 各阶段的降损效果；
    2. 在测试集模式下，把代理预测与仿真真值并排查看。

    因此这里把 True_ 列也整理成与各阶段平行的结构。
    这样下游绘图脚本和结果分析代码只需按统一列命名规则取值，
    不必为测试集真值再额外维护一套特例分支。
    """
    truth_df = pd.DataFrame(
        {
            "True_HIC": np.asarray(truth_arrays["y_HIC"], dtype=np.float32),
            "True_Dmax": np.asarray(truth_arrays["y_Dmax"], dtype=np.float32),
            "True_Nij": np.asarray(truth_arrays["y_Nij"], dtype=np.float32),
        }
    )
    truth_df["True_Phead"] = np.asarray(
        Injury_prob_cal_head(truth_df["True_HIC"].to_numpy(dtype=np.float32)),
        dtype=np.float32,
    )
    truth_df["True_Pchest"] = np.asarray(
        Injury_prob_cal_chest(truth_df["True_Dmax"].to_numpy(dtype=np.float32), OT=ot_array),
        dtype=np.float32,
    )
    truth_df["True_Pneck"] = np.asarray(
        Injury_prob_cal_neck(truth_df["True_Nij"].to_numpy(dtype=np.float32)),
        dtype=np.float32,
    )
    truth_df["True_JointRisk"] = 1.0 - (
        (1.0 - truth_df["True_Phead"])
        * (1.0 - truth_df["True_Pchest"])
        * (1.0 - truth_df["True_Pneck"])
    )
    truth_df["True_AIS_head"] = np.asarray(
        truth_arrays.get("ais_head", AIS_cal_head(truth_df["True_HIC"].to_numpy(dtype=np.float32)))
    )
    truth_df["True_AIS_chest"] = np.asarray(
        truth_arrays.get("ais_chest", AIS_cal_chest(truth_df["True_Dmax"].to_numpy(dtype=np.float32), ot_array))
    )
    truth_df["True_AIS_neck"] = np.asarray(
        truth_arrays.get("ais_neck", AIS_cal_neck(truth_df["True_Nij"].to_numpy(dtype=np.float32)))
    )
    truth_df["True_MAIS"] = np.maximum.reduce(
        [
            truth_df["True_AIS_head"].to_numpy(dtype=np.float32),
            truth_df["True_AIS_chest"].to_numpy(dtype=np.float32),
            truth_df["True_AIS_neck"].to_numpy(dtype=np.float32),
        ]
    )
    return truth_df


def _build_result_dataframe(
    df_input: pd.DataFrame,
    context_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    stage_outputs: Dict[str, object],
    truth_arrays: Dict[str, np.ndarray],
    trainable_names: List[str],
) -> tuple[pd.DataFrame, Dict[str, object]]:
    """组装最终导出 CSV 和宏观汇总指标。

    对外结果表只保留：
    - 一份原始 metadata/context；
    - 一份 baseline trainable control；
    - 每个真实生成阶段各自的可调参数、损伤、风险、AIS；
    - 每个优化阶段各自相对 baseline 的 reduction 列。

    这里不再把多个优化阶段压缩成一组含糊的 `Opt_` 列。
    若同时存在策略直推与局部精调，结果表必须显式保留 `Opt1_` 与 `Opt2_`，
    否则下游绘图和结果分析无法区分“直推收益”和“精调新增收益”。
    """
    excluded_input_cols = set(FEATURE_ORDER) | {"y_HIC", "y_Dmax", "y_Nij", "ais_head", "ais_chest", "ais_neck"}
    metadata_cols = [col for col in df_input.columns if col not in excluded_input_cols]
    metadata_df = df_input[metadata_cols].reset_index(drop=True)
    frame_parts = [metadata_df, context_df.reset_index(drop=True)]

    ot_array = pd.to_numeric(context_df["OT"], errors="coerce").to_numpy(dtype=np.float32)
    available_opt_stages = [stage_name for stage_name in OPT_STAGE_NAMES if stage_outputs[stage_name]["preds"] is not None]

    baseline_df = baseline_df.reset_index(drop=True).rename(columns={name: f"Base_{name}" for name in trainable_names})
    frame_parts.append(baseline_df)
    frame_parts.append(_build_stage_metric_dataframe("Base", stage_outputs["Base"], len(df_input), ot_array).reset_index(drop=True))

    for stage_name in available_opt_stages:
        opt_actions = stage_outputs[stage_name]["actions"]
        if opt_actions is None:
            action_df = pd.DataFrame({f"{stage_name}_{name}": np.full(len(df_input), np.nan, dtype=np.float32) for name in trainable_names})
        else:
            opt_array = opt_actions.detach().cpu().numpy()
            action_df = pd.DataFrame({f"{stage_name}_{name}": opt_array[:, idx] for idx, name in enumerate(trainable_names)})
        frame_parts.append(action_df.reset_index(drop=True))
        frame_parts.append(_build_stage_metric_dataframe(stage_name, stage_outputs[stage_name], len(df_input), ot_array).reset_index(drop=True))

    result_df = pd.concat(frame_parts, axis=1)
    truth_vs_df = None

    if all(key in truth_arrays for key in ["y_HIC", "y_Dmax", "y_Nij"]):
        # 测试集模式下把真值也整理成与各阶段同口径的列，便于直接并排比较。
        truth_df = _build_truth_metric_dataframe(truth_arrays=truth_arrays, ot_array=ot_array).reset_index(drop=True)
        for stage_name in STAGE_NAMES:
            if stage_name != "Base" and stage_name not in available_opt_stages:
                continue
            for metric_name in TRUE_VS_METRIC_NAMES:
                if truth_vs_df is None:
                    truth_vs_df = pd.DataFrame(index=df_input.index)
                truth_vs_df[f"True_vs_{stage_name}_{metric_name}"] = truth_df[f"True_{metric_name}"] - result_df[f"{stage_name}_{metric_name}"]
        result_df = pd.concat([result_df, truth_df], axis=1)

    reduction_data = {}
    for stage_name in available_opt_stages:
        for metric_name in RESULT_METRIC_NAMES:
            reduction_data[f"Reduction_{stage_name}_{metric_name}"] = result_df[f"Base_{metric_name}"] - result_df[f"{stage_name}_{metric_name}"]

    if reduction_data:
        result_df = pd.concat([result_df, pd.DataFrame(reduction_data, index=df_input.index).reset_index(drop=True)], axis=1)

    if truth_vs_df is not None:
        # True_vs_ 放在所有 Reduction_ 列之后，便于先看各阶段本体，再看降幅，再看与真值的偏差。
        result_df = pd.concat([result_df, truth_vs_df.reset_index(drop=True)], axis=1)

    summary_metrics = _build_stage_summary_metrics(
        result_df=result_df,
        available_opt_stages=available_opt_stages,
    )
    return result_df, summary_metrics


def _build_evaluation_record(
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
    optimized_stage_name: Optional[str],
    surrogate: SurrogateAdapter,
    dist_ref_info: Optional[Dict[str, object]],
    summary_metrics: Dict[str, object],
    top_cases: List[Dict[str, object]],
    stage_outputs: Dict[str, object],
    evaluated_case_count: int,
) -> Dict[str, object]:
    record = {
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
        "missing_context_columns": missing_context,
        "missing_trainable_columns_triggering_default_baseline": missing_trainable,
        "evaluated_case_count": evaluated_case_count,
        "skipped_case_rows": skipped_case_rows,
        "skipped_cases_count": len(skipped_case_rows),
        "reverted_baseline_rows": reverted_baseline_rows,
        "reverted_baseline_count": len(reverted_baseline_rows),
        "stage_status": {
            "base_generated": True,
            "opt1_generated": opt1_generated,
            "opt2_generated": opt2_generated,
            "reported_optimized_stage": optimized_stage_name,
        },
        "distribution_penalty": {
            "enabled_after_fit": bool(surrogate.distribution_penalty.enabled),
            "feature_space": surrogate.distribution_penalty.feature_space,
            "reference_sampling": dist_ref_info,
        },
        "summary_metrics": summary_metrics,
        "top_cases_by_reported_joint_risk_reduction": top_cases,
        "runtime": {
            "total_time_cost_sec": float(stage_outputs["total_time_cost"]),
            "avg_time_cost_sec": float(stage_outputs["total_time_cost"] / max(1, evaluated_case_count)),
            "trajectory_steps_logged": len(stage_outputs["trajectory_all"]),
        },
    }
    # summary_metrics / top_cases / distribution_penalty 里可能混入 np.float32、np.int64 等标量；
    # 这里统一在导出前递归转成原生 Python 类型，避免 YAML 序列化阶段再因单个字段失败。
    return convert_numpy_types(record)

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
    seed = int(config.get("seed", 42))
    set_random_seed(seed)

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

    dist_ref_info = None
    if surrogate.distribution_penalty.enabled:
        logger.info("Distribution Penalty 已启用：正在读取 injury_train split，用于拟合评估阶段的训练流形参考分布。")
        ref_sampler = StateDataSampler(
            param_manager=param_manager,
            constraint_engine=constraint_engine,
            batch_size=1024,
            device=device,
            seed=seed,
            split_indices_path=str(SPLIT_INDICES_DIR / "injury_train_indices.csv"),
            jitter_ratio=0.0,
            jitter_prob=0.0,
        )
        max_ref_samples = int(config.get("optimization", {}).get("distribution_penalty", {}).get("max_ref_samples", 0))
        # 评估端必须与训练端复用同一条参考集抽样语义：
        # 未截断时取完整训练池，截断时用固定 seed 的一次性随机子样本。
        # 不能在这里退回“前 N 条样本”，否则 penalty 的参考分布会和训练端脱节。
        reference, dist_ref_info = ref_sampler.get_distribution_reference(
            max_samples=max_ref_samples,
            feature_space=surrogate.distribution_penalty.feature_space,
            trainable_indices=param_manager.get_control_trainable_indices(),
            sample_seed=seed,
        )
        surrogate.fit_distribution_reference(reference)
        logger.info("Distribution Penalty 参考分布拟合完成：mode=%s, sample_count=%s, pool_size=%s", dist_ref_info["sampling_mode"], dist_ref_info["sample_count"], dist_ref_info["pool_size"])
    else:
        logger.info("Distribution Penalty 未启用：跳过训练流形参考分布加载与拟合。")

    strategy_ckpt_path = _resolve_strategy_ckpt(args, base_dir, config)
    strategy_net = None
    if bool(config.get("optimization", {}).get("direct_inference", False)):
        if strategy_ckpt_path is None:
            raise ValueError("direct_inference=True 时必须显式提供 strategy_checkpoint 或 --strategy_ckpt")
        if not strategy_ckpt_path.is_file():
            raise FileNotFoundError(f"策略权重不存在: {strategy_ckpt_path}")
        # 策略网络的层参数不能再从当前工作区配置里重建，
        # 必须回到该权重所属训练目录的 configs_used/config.yaml，
        # 否则会出现“权重来自旧实验，结构却按新配置重建”的隐性错配。
        strategy_config, strategy_artifacts = load_strategy_run_config(strategy_ckpt_path)
        _assert_strategy_runtime_compatibility(
            current_param_space_path=param_space_path,
            current_normalization_path=NORMALIZATION_CONFIG_PATH,
            strategy_artifacts=strategy_artifacts,
        )
        logger.info("策略网络结构配置来自: %s", strategy_artifacts["config"])
        strategy_net = build_strategy_net_from_config(
            param_manager=param_manager,
            constraint_engine=constraint_engine,
            data_processor=data_processor,
            config=strategy_config,
        ).to(device)
        try:
            strategy_net.load_state_dict(torch.load(str(strategy_ckpt_path), map_location=device, weights_only=True))
            logger.info("策略网络权重加载成功: %s", str(strategy_ckpt_path))
        except Exception as exc:
            raise RuntimeError(
                "策略权重与其权重目录中的结构配置不兼容，无法完成加载: "
                f"{strategy_ckpt_path}"
            ) from exc

    optimizer = LocalRefiner(
        config=config,
        param_manager=param_manager,
        constraint_engine=constraint_engine,
        surrogate=surrogate,
        strategy_net=strategy_net,
    )
    logger.info(
        "评估配置: direct_inference=%s, refine_steps=%d, eval_batch_size=%d",
        optimizer.direct_inference,
        optimizer.refine_steps,
        int(config.get("evaluation", {}).get("eval_batch_size", 512)),
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
    logger.info("评估样本统计: total=%d, valid=%d, skipped=%d, reverted_baseline=%d", len(df_input), int(valid_mask.sum()), len(skipped_case_rows), len(reverted_baseline_rows))
    if valid_indices.size == 0:
        raise ValueError("所有 case 都因输入端校验失败被跳过，未生成任何可评估样本。")
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

    optimized_stage_name = "Opt2" if stage_outputs["Opt2"]["preds"] is not None else ("Opt1" if stage_outputs["Opt1"]["preds"] is not None else None)
    logger.info("阶段结果: Base=always, Opt1=%s, Opt2=%s, reported_stage=%s", stage_outputs["Opt1"]["preds"] is not None, stage_outputs["Opt2"]["preds"] is not None, optimized_stage_name)

    result_df, summary_metrics = _build_result_dataframe(
        df_input=df_input,
        context_df=context_output_df,
        baseline_df=baseline_df,
        stage_outputs=stage_outputs,
        truth_arrays=truth_arrays,
        trainable_names=trainable_names,
    )
    evaluated_case_count = int(valid_mask.sum())
    top_cases = _build_top_cases_summary(result_df, optimized_stage_name=optimized_stage_name, top_n=5)
    # 在所有前置校验和模型推理都完成后再创建输出目录，避免早退时留下空目录。
    output_dir = _build_output_dir(base_dir, args.input_csv)
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=False)
    output_csv_path = results_dir / Path(args.output_csv).name
    config_snapshots = _copy_config_snapshots(cfg_path, param_space_path, output_dir)
    result_df.to_csv(str(output_csv_path), index=False)
    opt1_generated = stage_outputs["Opt1"]["preds"] is not None
    opt2_generated = stage_outputs["Opt2"]["preds"] is not None

    evaluation_record = _build_evaluation_record(
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
        dist_ref_info=dist_ref_info,
        summary_metrics=summary_metrics,
        top_cases=top_cases,
        stage_outputs=stage_outputs,
        evaluated_case_count=evaluated_case_count,
    )
    with open(results_dir / "evaluation_record.yaml", "w", encoding="utf-8") as file:
        yaml.safe_dump(evaluation_record, file, allow_unicode=True, sort_keys=False)
    logger.info(f"评估完成，结果目录: {output_dir}")
    logger.info(f"结果 CSV: {output_csv_path}")


if __name__ == "__main__":
    main()

