import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = 'T'
import warnings
warnings.filterwarnings('ignore')
import argparse
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

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
from common.settings import FEATURE_ORDER, NORMALIZATION_CONFIG_PATH, RAW_DATA, get_split_indices_path
from common.tools.logger import setup_logger
from common.tools.seeding import set_random_seed

from InjuryPredict.utils.tools import convert_numpy_types

from ARS_optim.src.constraints import ConstraintEngine
from ARS_optim.src.data_sampler import StateDataSampler
from ARS_optim.src.optimizer import Optimizer
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
    parser.add_argument(
        "--input_csv",
        type=str,
        default=None,
        help="输入工况参数 CSV 路径；绝对路径或相对路径均可。若不提供则优先使用 injury test split，不可用时自动回退到 injury val split",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="evaluation_results.csv",
        help="结果 CSV 文件名；若传入路径，仅使用其中的文件名部分",
    )
    parser.add_argument(
        "--strategy_ckpt",
        type=str,
        default=None,
        help="策略网络权重路径；绝对路径或相对路径均可",
    )
    parser.add_argument(
        "--trace_case_ids",
        nargs="+",
        default=None,
        help="需要导出局部精调逐步轨迹的 case_id 列表，例如 --trace_case_ids 101 205 999",
    )
    parser.add_argument("--direct_inference", action="store_true", help="强制启用策略网络直推")

    return parser.parse_args()


def _build_output_dir(
    base_dir: Path,
    input_csv: Optional[str],
    input_source: Optional[Dict[str, object]] = None,
) -> Path:
    if input_csv:
        stem = Path(input_csv).stem
        normalized = [char if char.isalnum() or char in {"_", "-"} else "_" for char in stem]
        suffix = "".join(normalized).strip("_") or "evaluation"
    else:
        split_name = "default"
        if input_source and input_source.get("type") == "default_split":
            split_name = str(input_source.get("selected_split", "default"))
        suffix = f"injury_{split_name}_split"
    output_dir = base_dir / "saved_eval" / f"eval_{suffix}_{datetime.now().strftime('%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir
def _save_case_trace_tables(
    results_dir: Path,
    trace_records: Dict[int, Dict[str, object]],
    requested_case_ids: Set[str],
    logger,
) -> Dict[str, object]:
    """把逐 case 优化轨迹导出为独立 CSV。"""
    if not requested_case_ids:
        return {
            "enabled": False,
            "requested_case_ids": [],
            "written_case_count": 0,
            "trace_dir": None,
            "files": [],
            "missing_case_ids": [],
        }

    if not trace_records:
        missing_case_ids = sorted(requested_case_ids)
        logger.warning("未找到任何需要导出轨迹的 case_id: %s", missing_case_ids)
        return {
            "enabled": True,
            "requested_case_ids": sorted(requested_case_ids),
            "written_case_count": 0,
            "trace_dir": None,
            "files": [],
            "missing_case_ids": missing_case_ids,
        }

    trace_dir = results_dir / "optimization_traces"
    trace_dir.mkdir(parents=True, exist_ok=False)
    file_records = []
    matched_case_ids = set()

    for row_index in sorted(trace_records):
        trace_record = trace_records[row_index]
        case_id = str(trace_record["case_id"])
        rows = trace_record.get("rows", [])
        if not rows:
            continue
        matched_case_ids.add(case_id)
        trace_df = pd.DataFrame(rows)
        safe_case_id = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in case_id).strip("_") or "case"
        trace_path = trace_dir / f"case_{safe_case_id}_row_{int(row_index)}.csv"
        trace_df.to_csv(str(trace_path), index=False)
        file_records.append(
            {
                "case_id": case_id,
                "row_index": int(row_index),
                "path": str(trace_path),
            }
        )

    missing_case_ids = sorted(requested_case_ids - matched_case_ids)
    if missing_case_ids:
        logger.warning("以下 trace_case_ids 未命中有效评估 case: %s", missing_case_ids)

    return {
        "enabled": True,
        "requested_case_ids": sorted(requested_case_ids),
        "written_case_count": len(file_records),
        "trace_dir": str(trace_dir),
        "files": file_records,
        "missing_case_ids": missing_case_ids,
    }


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


def _copy_config_snapshots(
    cfg_path: Path,
    param_space_path: Path,
    output_dir: Path,
) -> Dict[str, str]:
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
    """基于逐 case 结果表计算宏观汇总指标。

    作用：
    - 按阶段统计 `HIC/Dmax/Nij/Phead/Pchest/Pneck/JointRisk/AIS/MAIS` 的均值；
    - 按优化阶段统计相对 `Base` 的平均降幅。

    输入：
    - `result_df`: `_build_result_dataframe()` 产出的逐 case 结果表。
    - `available_opt_stages`: 本次评估实际生成的优化阶段名称列表，只能取 `Opt1/Opt2` 的子集。

    输出：
    - 返回一个字典，供 `evaluation_record.yaml` 写入宏观摘要。

    注意：
    - 这里使用 `nanmean`，因此会自动跳过被 `NaN` 占位的 skipped case。
    - 该函数只读取现成列，不参与任何额外推理或结果修正。
    """
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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], List[str], np.ndarray, List[int], List[int], List[int]]:
    """整理评估输入。

    `input_csv` 模式下，按输入端规则逐行处理：
    - `context` 缺失或不合法时跳过该行。
    - `baseline` trainable control 只有整组合法才采用，否则整行回退为 default。

    默认 split 评估使用数据集已有参数，但仍按当前约束配置过滤不合法样本。

    返回值依次为：
    - `context_output_df`: 结果表中保留的 context 视图, 包含原始输入里的非法行
    - `context_eval_df`: 真正送入模型的 context 数值表；被跳过的行（如缺失或非法）保持 NaN
    - `baseline_eval_df`: baseline 损伤评估时使用的 trainable control 参数表；必要时已整行回退为 default
    - `missing_context` / `missing_trainable`: 输入表缺失的参数列名
    - `valid_mask`: 最终进入模型评估的行掩码
    - `skipped_rows`: 因 context 缺失或不合法而跳过的原始行号
    - `reverted_baseline_rows`: baseline 被整行回退为 default 的原始行号
    - `invalid_default_baseline_rows`: baseline 回退到 default 后仍与当前 context 不兼容、因此被跳过的原始行号
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
            raise ValueError(f"默认 split 评估样本缺失 context 列: {missing_context}")
    if missing_trainable:
        if strict_provided_validation:
            logger.warning("input_csv 缺失部分 trainable control 列；这些 case 的 baseline 将整组回退为 default: %s", missing_trainable)
        else:
            raise ValueError(f"默认 split 评估样本缺失 trainable control 列: {missing_trainable}")

    if not strict_provided_validation:
        # 默认 split 不做数值修补；若历史数据与当前约束配置不一致，则跳过不合法行，避免优化器在不可修复的 context 上报错。
        context_output_df = context_df_raw.astype(np.float32)
        context_eval_df = context_output_df.copy()
        baseline_eval_df = baseline_df_raw.astype(np.float32)
        missing_context_rows = context_eval_df[context_names].isna().any(axis=1).to_numpy()
        missing_trainable_rows = baseline_eval_df[trainable_names].isna().any(axis=1).to_numpy()
        if missing_context_rows.any() or missing_trainable_rows.any():
            bad_rows = np.flatnonzero(missing_context_rows | missing_trainable_rows)
            raise ValueError(
                "默认 split 评估样本存在缺失值，不符合评估前提。行号: "
                f"{_format_row_list(bad_rows)}"
            )

        context_tensor = torch.tensor(context_eval_df[context_names].values, dtype=torch.float32, device=device)
        baseline_tensor = torch.tensor(baseline_eval_df[trainable_names].values, dtype=torch.float32, device=device)
        context_valid = (
            constraint_engine.validate_context_input(context_tensor)
            .detach()
            .cpu()
            .numpy()
            .astype(bool)
        )
        baseline_full = constraint_engine.compose_full_features(context_tensor, baseline_tensor)
        baseline_valid = (
            constraint_engine.validate_full_input(baseline_full)
            .detach()
            .cpu()
            .numpy()
            .astype(bool)
        )
        valid_mask = context_valid & baseline_valid
        skipped_rows = np.flatnonzero(~valid_mask).astype(int).tolist()
        invalid_context_rows = np.flatnonzero(~context_valid)
        invalid_baseline_rows = np.flatnonzero(context_valid & ~baseline_valid)
        if invalid_context_rows.size > 0:
            logger.warning(
                "默认 split 中共有 %d 个 case 的 context 不满足当前约束，将跳过。行号: %s",
                invalid_context_rows.size,
                _format_row_list(invalid_context_rows),
            )
        if invalid_baseline_rows.size > 0:
            logger.warning(
                "默认 split 中共有 %d 个 case 的 baseline trainable control 与当前 context 不兼容，将跳过。行号: %s",
                invalid_baseline_rows.size,
                _format_row_list(invalid_baseline_rows),
            )
        if skipped_rows:
            context_eval_df.loc[skipped_rows, context_names] = np.nan
            baseline_eval_df.loc[skipped_rows, trainable_names] = np.nan

        return (
            context_output_df,
            context_eval_df,
            baseline_eval_df,
            missing_context,
            missing_trainable,
            valid_mask,
            sorted(set(skipped_rows)),
            [],
            [],
        )

    context_output_df = context_df_raw.copy()
    row_has_context_nan = context_df_raw[context_names].isna().any(axis=1).to_numpy()
    valid_context_candidates = ~row_has_context_nan
    skipped_rows: List[int] = np.flatnonzero(row_has_context_nan).astype(int).tolist()

    # 严格 `input_csv` 模式下，导出视图与模型输入分开维护；
    # 后者只保留真正参与评估的合法数值。
    valid_mask = np.zeros(len(df_input), dtype=bool)
    context_eval_df = pd.DataFrame(np.nan, index=df_input.index, columns=context_names, dtype=np.float32)
    baseline_eval_df = pd.DataFrame(np.tile(default_trainable, (len(df_input), 1)), index=df_input.index, columns=trainable_names, dtype=np.float32)
    reverted_baseline_rows: List[int] = []
    invalid_default_baseline_rows: List[int] = []

    if np.any(valid_context_candidates):
        candidate_indices = np.flatnonzero(valid_context_candidates)
        candidate_context_raw = context_df_raw.iloc[candidate_indices]
        candidate_context_tensor = torch.tensor(candidate_context_raw[context_names].values, dtype=torch.float32, device=device)
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
            baseline_candidate = baseline_df_raw.loc[legal_context_indices, trainable_names].copy()
            # baseline 采用严格整组策略，不做“用户部分列 + default”的混搭。
            baseline_has_missing_local = baseline_candidate[trainable_names].isna().any(axis=1).to_numpy()

            legal_context_tensor = torch.tensor(
                context_df_raw.loc[legal_context_indices, context_names].values,
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
                # baseline 的合法性按完整样本校验：必须与当前 context 联合后仍满足输入端规则。
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
            # default 只能作为整组 fallback 候选，仍需与当前 context 联合校验；
            # 否则一旦未来 trainable / context 角色切换，结果表里会混入物理上不合法的 baseline。
            fallback_valid_local = np.zeros(len(legal_context_indices), dtype=bool)
            if baseline_invalid_mask_local.any():
                fallback_row_indices = np.flatnonzero(baseline_invalid_mask_local)
                fallback_default_tensor = torch.tensor(
                    np.tile(default_trainable, (fallback_row_indices.size, 1)),
                    dtype=torch.float32,
                    device=device,
                )
                fallback_full = constraint_engine.compose_full_features(
                    legal_context_tensor[fallback_row_indices],
                    fallback_default_tensor,
                )
                fallback_valid_local[fallback_row_indices] = (
                    constraint_engine.validate_full_input(fallback_full)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(bool)
                )
                invalid_default_indices = legal_context_indices[
                    fallback_row_indices[~fallback_valid_local[fallback_row_indices]]
                ]
                if invalid_default_indices.size > 0:
                    logger.warning(
                        "以下 input_csv 行在 baseline 回退为 default 后，仍与当前 context 不兼容，将跳过该 case。行号: %s",
                        _format_row_list(invalid_default_indices),
                    )
                    invalid_default_baseline_rows.extend(
                        invalid_default_indices.astype(int).tolist()
                    )
                    skipped_rows.extend(invalid_default_indices.astype(int).tolist())
                    baseline_eval_df.loc[invalid_default_indices, trainable_names] = np.nan

                reverted_indices = legal_context_indices[
                    fallback_row_indices[fallback_valid_local[fallback_row_indices]]
                ]
                if reverted_indices.size > 0:
                    reverted_baseline_rows.extend(reverted_indices.astype(int).tolist())

            accepted_indices = legal_context_indices[baseline_valid_local]
            if accepted_indices.size > 0:
                baseline_eval_df.loc[accepted_indices, trainable_names] = baseline_candidate.loc[
                    accepted_indices,
                    trainable_names,
                ].to_numpy(dtype=np.float32)

            final_valid_local = baseline_valid_local | fallback_valid_local
            final_valid_indices = legal_context_indices[final_valid_local]
            if final_valid_indices.size > 0:
                valid_mask[final_valid_indices] = True
                context_eval_df.loc[final_valid_indices, context_names] = context_df_raw.loc[
                    final_valid_indices,
                    context_names,
                ].to_numpy(dtype=np.float32)

    if skipped_rows:
        skipped_array = np.asarray(sorted(set(skipped_rows)), dtype=np.int64)
        logger.warning("input_csv 中共有 %d 个 case 因 context 缺失、context 非法或 default baseline 与 context 不兼容而被跳过，行号: %s", skipped_array.size, _format_row_list(skipped_array))
    if reverted_baseline_rows:
        reverted_array = np.asarray(sorted(set(reverted_baseline_rows)), dtype=np.int64)
        logger.warning("input_csv 中共有 %d 个 case 的 baseline 可调控制参数被回退为 default，行号: %s", reverted_array.size, _format_row_list(reverted_array))
    if invalid_default_baseline_rows:
        invalid_default_array = np.asarray(sorted(set(invalid_default_baseline_rows)), dtype=np.int64)
        logger.warning("input_csv 中共有 %d 个 case 在 baseline 回退 default 后仍不合法，因此被跳过，行号: %s", invalid_default_array.size, _format_row_list(invalid_default_array))

    return (
        context_output_df,
        context_eval_df,
        baseline_eval_df,
        missing_context,
        missing_trainable,
        valid_mask,
        sorted(set(skipped_rows)),
        sorted(set(reverted_baseline_rows)),
        sorted(set(invalid_default_baseline_rows)),
    )

def _load_default_split_eval_input(logger) -> tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, object]]:
    """装载未指定 input_csv 时的默认评估输入。

    默认优先使用 injury test split；
    若 test split 文件不存在或为空，则自动回退到 injury val split。
    两者都不可用时直接报错，不引入额外隐式 fallback。
    """
    pool_path = RAW_DATA
    if not pool_path.exists():
        raise FileNotFoundError(f"默认 split 评估模式需要原始打包数据文件: {pool_path}")

    split_status: Dict[str, str] = {}
    selected_split = None
    selected_idx_path = None
    selected_indices = None
    for split_name in ("test", "val"):
        idx_path = get_split_indices_path("injury", split_name)
        if not idx_path.exists():
            split_status[split_name] = "missing"
            continue
        indices = load_int_vector_csv(idx_path)
        if indices.size == 0:
            split_status[split_name] = "empty"
            continue
        split_status[split_name] = "selected"
        selected_split = split_name
        selected_idx_path = idx_path
        selected_indices = indices
        break

    if selected_split is None or selected_idx_path is None or selected_indices is None:
        status_text = ", ".join(f"{name}={split_status.get(name, 'missing')}" for name in ("test", "val"))
        raise ValueError(
            "未指定 input_csv 时，默认评估需要可用的 injury_test 或 injury_val split。"
            f"当前状态: {status_text}"
        )

    truth_arrays: Dict[str, np.ndarray] = {}
    with np.load(str(pool_path), allow_pickle=True) as data:
        x_att_raw = data["x_att_raw"][selected_indices]
        df_input = pd.DataFrame(x_att_raw, columns=FEATURE_ORDER)
        case_ids = data["case_ids"][selected_indices] if "case_ids" in data else np.arange(len(selected_indices))
        df_input.insert(0, "case_id", case_ids)
        for key in ["y_HIC", "y_Dmax", "y_Nij", "ais_head", "ais_chest", "ais_neck"]:
            if key in data:
                truth_arrays[key] = np.asarray(data[key][selected_indices])

    if selected_split == "test":
        logger.info("未指定 input_csv，自动加载默认评估 split=test: %d 条", len(df_input))
    else:
        logger.warning(
            "未指定 input_csv，injury_test split %s，已自动回退到默认评估 split=val: %d 条",
            split_status.get("test", "missing"),
            len(df_input),
        )
    input_source = {
        "type": "default_split",
        "selected_split": selected_split,
        "candidate_priority": ["test", "val"],
        "fallback_from_test": selected_split != "test",
        "split_status": split_status,
        "path": str(selected_idx_path.resolve()),
        "raw_data_npz_path": str(pool_path.resolve()),
    }
    return df_input, truth_arrays, input_source


def _load_eval_input(args, logger) -> tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, object]]:
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

    return _load_default_split_eval_input(logger)


def _compute_predictions_batch(
    context_tensor: torch.Tensor,
    baseline_trainable: torch.Tensor,
    case_ids: List[str],
    row_indices: List[int],
    surrogate: SurrogateAdapter,
    optimizer: Optimizer,
    eval_batch_size: int,
    trace_case_ids: Set[str],
) -> Dict[str, object]:
    """按评估批次生成 `Base / Opt1 / Opt2` 三阶段结果。

    作用：
    - 对已经通过输入端校验的有效 case 执行代理评估；
    - 在配置允许时，继续执行策略直推和局部精调；
    - 把各阶段的预测值、动作、损失分项、收敛步数和 trace 统一收集起来。

    输入：
    - `context_tensor`: 仅包含有效 case 的 `context` 张量。
    - `baseline_trainable`: 与 `context_tensor` 严格对齐的 baseline 可调控制参数。
    - `case_ids` / `row_indices`: 有效 case 在原输入表中的标识，用于 trace 和后续结果回填。
    - `surrogate`: 统一封装波形生成、损伤预测和目标函数计算的代理接口。
    - `optimizer`: 负责 `Opt1/Opt2` 的二阶段优化器。
    - `eval_batch_size`: 仅控制并行切分粒度，不改变逐 case 优化语义。
    - `trace_case_ids`: 需要额外导出局部精调过程表的 case 标识集合。

    输出：
    - 返回一个按阶段组织的字典，其中 `Base` 总是存在；
    - `Opt1` 仅在启用策略直推时存在；
    - `Opt2` 仅在启用局部精调时存在；
    - 额外字段包括总耗时、逐 case 收敛步数和指定 case 的轨迹记录。

    注意：
    - 虽然实现上按 batch 调用 `optimizer.optimize()`，但局部精调仍是逐 case 优化；
      这里只是把多个样本的逐点优化做成并行向量化。
    - 当前函数只输出“有效 case 的紧凑结果”；skipped 行的 `NaN` 回填在
      `_expand_stage_outputs_to_full()` 中统一处理。
    """
    sample_count = context_tensor.shape[0]
    device = context_tensor.device
    # 每个阶段都按同一结构暂存：预测值、动作、逐样本 loss 和风险细项。
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
    opt2_convergence_parts: List[torch.Tensor] = []  # 各 batch 的逐 case Opt2 收敛步数
    trace_records: Dict[int, Dict[str, object]] = {}  # 指定 case 的逐步优化明细

    def cat_or_nan(parts: List[torch.Tensor]) -> torch.Tensor:
        # 某个阶段未启用时，用 NaN 向量占位，保证下游结果组装逻辑不必再分支判断长度。
        return torch.cat(parts, dim=0) if parts else torch.full((sample_count,), float("nan"), device=device)

    batch_starts = range(0, sample_count, eval_batch_size)
    # 评估按有效样本的既有行序连续切 batch，不做随机打乱。
    for start in tqdm(batch_starts, total=(sample_count + eval_batch_size - 1) // eval_batch_size, desc="Evaluating batches"):
        end = min(start + eval_batch_size, sample_count)
        context_batch = context_tensor[start:end]
        baseline_batch = baseline_trainable[start:end]

        with torch.no_grad():
            # Base 阶段直接评估“输入 context + baseline control”，不做任何优化。
            pulse_batch = surrogate.generate_pulse(context_batch)
            base_loss, base_preds, base_info = surrogate.predict_injury_and_loss(context_batch, baseline_batch, pulse_batch)
        stage_parts["Base"]["preds"].append(base_preds.detach())
        stage_parts["Base"]["actions"].append(baseline_batch.detach())
        stage_parts["Base"]["loss"].append(base_loss.detach())
        # 各损伤风险分项与损失分量按阶段逐 batch 累积，后面统一拼接。
        for key in STAGE_INFO_KEYS:
            stage_parts["Base"]["info"][key].append(base_info[key].detach())
        # 按配置决定是否继续执行 Opt1 / Opt2。
        if optimizer.direct_inference or optimizer.refine_steps > 0:
            opt_actions, opt_preds, opt_info = optimizer.optimize(
                context_batch,
                pulse_norm=pulse_batch,
                case_ids=case_ids[start:end],
                row_indices=row_indices[start:end],
                trace_case_ids=trace_case_ids,
            )
            direct_stage = opt_info.get("direct_stage")
            if direct_stage is not None:
                # Opt1 是策略网络直推阶段；这里记录的是“直推后、局部精调前”的阶段结果。
                stage_parts["Opt1"]["preds"].append(direct_stage["preds"].detach())
                stage_parts["Opt1"]["actions"].append(direct_stage["actions"].detach())
                stage_parts["Opt1"]["loss"].append(direct_stage["loss_batch"].detach())
                for key in STAGE_INFO_KEYS:
                    stage_parts["Opt1"]["info"][key].append(direct_stage["detail"][key].detach())

            if opt_info.get("refine_stage_enabled", False):
                # Opt2 是局部精调后的最终结果；若未启用 refine，则这一整段保持为空。
                stage_parts["Opt2"]["preds"].append(opt_preds.detach())
                stage_parts["Opt2"]["actions"].append(opt_actions.detach())
                stage_parts["Opt2"]["loss"].append(opt_info["final_loss_batch"].detach())
                for key in STAGE_INFO_KEYS:
                    stage_parts["Opt2"]["info"][key].append(opt_info[key].detach())
                convergence_steps = opt_info.get("convergence_steps")  # 当前 batch 的逐 case 收敛步数
                if convergence_steps is not None:
                    opt2_convergence_parts.append(convergence_steps.detach())

            # 这些附加信息按 batch 汇总，最终统一写入结果记录。
            total_time_cost += float(opt_info.get("time_cost", 0.0))
            for row_index, trace_record in opt_info.get("trace_records", {}).items():
                if row_index in trace_records:
                    raise RuntimeError(f"重复写入同一 row_index 的优化轨迹: {row_index}")
                trace_records[int(row_index)] = trace_record

    # 这里先保持“只覆盖有效 case”的紧凑结构；跳过行的 NaN 回填在 _expand_stage_outputs_to_full() 里统一处理。
    output = {
        "total_time_cost": total_time_cost,
        "opt2_convergence_step": torch.cat(opt2_convergence_parts, dim=0) if opt2_convergence_parts else None,
        "trace_records": trace_records,
    }
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
    """把“只覆盖有效 case”的阶段结果扩回完整行序。

    作用：
    - 将代理推理和优化阶段的紧凑输出写回原始输入表的完整行数；
    - 为 skipped case 统一补 `NaN` 占位，保持导出结果与原输入逐行对齐。

    输入：
    - `stage_outputs`: `_compute_predictions_batch()` 产出的紧凑阶段结果。
    - `valid_indices`: 真正参与评估的原始行号，升序且无重复。
    - `total_count`: 原始输入总行数。
    - `trainable_dim`: 当前 trainable control 维数，用于构造动作列的占位张量。

    输出：
    - 返回与原始输入等长的阶段结果字典，可直接供结果表组装与汇总统计使用。

    注意：
    - 这里不重新排序，也不重算任何指标，只做“按行号回填 + skipped 行补 NaN”。
    - `valid_indices` 是这一步的唯一对齐依据；只要切片和回填都使用同一份索引，
      最终结果就不会发生 case 错配。
    """
    valid_indices_tensor = torch.as_tensor(valid_indices, dtype=torch.long)
    expanded = {
        "total_time_cost": stage_outputs["total_time_cost"],
        "trace_records": stage_outputs.get("trace_records", {}),
    }
    convergence_src = stage_outputs.get("opt2_convergence_step")
    convergence_full = torch.full((total_count,), float("nan"), dtype=torch.float32)
    if convergence_src is not None and convergence_src.numel() > 0:
        convergence_full[valid_indices_tensor] = convergence_src.detach().cpu().to(torch.float32)
    expanded["opt2_convergence_step"] = convergence_full
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
    """把单阶段输出整理成统一列结构的结果子表。

    作用：
    - 将单个阶段的 `preds/info` 张量转换成 `DataFrame`；
    - 统一生成 `HIC/Dmax/Nij/Phead/Pchest/Pneck/JointRisk/AIS/MAIS` 列。

    输入：
    - `stage_label`: 阶段名前缀，如 `Base`、`Opt1`、`Opt2`。
    - `stage`: 某一阶段在 `stage_outputs` 中对应的字典。
    - `row_count`: 结果表总行数。
    - `ot_array`: 与结果表行序对齐的 `OT` 数组，用于胸部 AIS 与风险计算。

    输出：
    - 返回一个与结果表等长的 `DataFrame`。

    注意：
    - 若该阶段未生成结果，则整张子表返回 `NaN`；
    - 若某些行的预测值本身为 `NaN`，对应的 AIS/MAIS 也会显式置为 `NaN`，
      避免把无效预测继续映射成表面上“有值”的等级。
    """
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
    # `MAIS` 统一取头/胸/颈三个 AIS 等级中的最大值。
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
    """把真值数组整理成与阶段结果平行的 `True_*` 列。

    作用：
    - 将默认 split 或外部 CSV 自带的真值转换成统一列命名；
    - 让 `True_*` 与 `Base/Opt1/Opt2_*` 共享同一套下游绘图和分析逻辑。

    输入：
    - `truth_arrays`: 至少包含 `y_HIC/y_Dmax/y_Nij`，可选包含 `ais_*` 真值。
    - `ot_array`: 与结果表行序对齐的 `OT` 数组，用于胸部风险和 AIS 计算。

    输出：
    - 返回一个 `DataFrame`，列名统一以 `True_` 为前缀。

    注意：
    - 若输入已经提供 `ais_head/ais_chest/ais_neck` 真值，则优先直接使用；
    - 若未提供，则根据 `True_HIC/True_Dmax/True_Nij` 现场计算。
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
    """组装最终导出的逐 case 结果表，并生成宏观汇总指标。

    作用：
    - 把 `context`、baseline、`Base/Opt1/Opt2` 各阶段输出以及可选真值拼成一张总表；
    - 补齐每个阶段对应的动作列、损伤列、风险列、AIS/MAIS 列和 reduction 列；
    - 基于结果表进一步生成 `evaluation_record.yaml` 需要的宏观摘要。

    输入：
    - `df_input`: 原始输入表，仅用于保留稳定标识列和完整行数。
    - `context_df`: 当前评估实际使用的 `context` 表，行序已与原输入对齐。
    - `baseline_df`: 当前评估实际使用的 baseline trainable control 表，行序已与原输入对齐。
    - `stage_outputs`: 已扩回完整行数的阶段结果字典。
    - `truth_arrays`: 可选真值数组；若缺失，则结果表不生成 `True_*` 与 `True_vs_*` 列。
    - `trainable_names`: 当前 trainable control 名称列表，用于组装动作列。

    输出：
    - `result_df`: 最终导出的逐 case 结果表。
    - `summary_metrics`: 基于 `result_df` 计算得到的宏观汇总指标字典。

    关键注意点：
    - 结果表必须严格保持与原始输入一致的行序；所有子表在拼接前都要 `reset_index(drop=True)`。
    - `available_opt_stages` 以“该阶段是否真的产出了预测结果”为准，而不是仅看配置是否启用。
    - `Reduction_*` 统一采用 `Base - 当前阶段` 的口径，便于汇总和绘图复用。
    - `Opt2_ConvergenceStep` 只属于局部精调阶段；`Opt1` 是一次性直推，不存在收敛步数。
    """
    # 自定义 input_csv 可能带有备注等无关列；结果表只保留稳定标识列 `case_id`，
    # 其余参数一律以当前评估实际使用的 context/Base/Opt 列为准。
    metadata_cols = ["case_id"] if "case_id" in df_input.columns else []
    metadata_df = df_input[metadata_cols].reset_index(drop=True)
    frame_parts = [metadata_df, context_df.reset_index(drop=True)]

    ot_array = pd.to_numeric(context_df["OT"], errors="coerce").to_numpy(dtype=np.float32)
    # 是否存在 `Opt1 / Opt2` 由该阶段是否真的产出预测结果决定，而不是只看配置开关。
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
        if stage_name == "Opt2":
            # 收敛步数只对应局部精调阶段；Opt1 是一次性直推，不存在“迭代收敛步数”的语义。
            convergence_steps = stage_outputs.get("opt2_convergence_step")
            if convergence_steps is not None:
                convergence_array = convergence_steps.detach().cpu().numpy().astype(np.float32, copy=False)
                frame_parts.append(
                    pd.DataFrame(
                        {
                            "Opt2_ConvergenceStep": convergence_array,
                        }
                    ).reset_index(drop=True)
                )

    result_df = pd.concat(frame_parts, axis=1)
    truth_vs_df = None

    if all(key in truth_arrays for key in ["y_HIC", "y_Dmax", "y_Nij"]):
        # 只要输入侧带有真值，就把 `True_*` 列与各阶段结果并排展开。
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
        # `Reduction_*` 始终表示“相对 baseline 下降了多少”。
        for metric_name in RESULT_METRIC_NAMES:
            reduction_data[f"Reduction_{stage_name}_{metric_name}"] = result_df[f"Base_{metric_name}"] - result_df[f"{stage_name}_{metric_name}"]

    if reduction_data:
        result_df = pd.concat([result_df, pd.DataFrame(reduction_data, index=df_input.index).reset_index(drop=True)], axis=1)

    if truth_vs_df is not None:
        # True_vs_ 放在所有 Reduction_ 列之后，便于先看各阶段本体，再看降幅，再看与真值的偏差。
        result_df = pd.concat([result_df, truth_vs_df.reset_index(drop=True)], axis=1)
    
    # 这份摘要不写回 CSV，而是写入 evaluation_record 供整体统计查看。
    summary_metrics = _build_stage_summary_metrics(
        result_df=result_df,
        available_opt_stages=available_opt_stages,
    )

    return result_df, summary_metrics


def _build_evaluation_record(
    output_dir: Path,
    output_csv_path: Path,
    config_snapshots: Dict[str, str],
    input_source: Dict[str, object],
    strategy_ckpt_path: Optional[Path],
    config: dict,
    param_manager: ParamManager,
    context_names: List[str],
    trainable_names: List[str],
    missing_context: List[str],
    missing_trainable: List[str],
    skipped_case_rows: List[int],
    reverted_baseline_rows: List[int],
    invalid_default_baseline_rows: List[int],
    opt1_generated: bool,
    opt2_generated: bool,
    optimized_stage_name: Optional[str],
    surrogate: SurrogateAdapter,
    dist_ref_info: Optional[Dict[str, object]],
    summary_metrics: Dict[str, object],
    top_cases: List[Dict[str, object]],
    stage_outputs: Dict[str, object],
    evaluated_case_count: int,
    trace_outputs: Dict[str, object],
) -> Dict[str, object]:
    convergence_array = stage_outputs.get("opt2_convergence_step")
    convergence_summary = None
    if convergence_array is not None and stage_outputs["Opt2"]["preds"] is not None:
        convergence_np = convergence_array.detach().cpu().numpy().astype(np.float32, copy=False)
        converged_steps = convergence_np[np.isfinite(convergence_np)]
        convergence_summary = {
            "converged_case_count": int(converged_steps.size),
            "mean_convergence_step": None if converged_steps.size == 0 else float(np.mean(converged_steps)),
            "median_convergence_step": None if converged_steps.size == 0 else float(np.median(converged_steps)),
            "max_convergence_step": None if converged_steps.size == 0 else int(np.max(converged_steps)),
        }

    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(output_dir),
        "output_csv": str(output_csv_path),
        "input_source": input_source,
        "strategy_checkpoint_path": str(strategy_ckpt_path) if strategy_ckpt_path is not None else None,
        "direct_inference": bool(config.get("optimization", {}).get("direct_inference", False)),
        "config_files": config_snapshots,
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
        "skipped_due_to_invalid_default_baseline_rows": invalid_default_baseline_rows,
        "skipped_due_to_invalid_default_baseline_count": len(invalid_default_baseline_rows),
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
        "trace_outputs": trace_outputs,
        "runtime": {
            "total_time_cost_sec": float(stage_outputs["total_time_cost"]), # 单位: 秒
            "avg_time_cost_sec": float(stage_outputs["total_time_cost"] / max(1, evaluated_case_count)),
            "opt2_convergence": convergence_summary,
        },
    }
    # 统一把 numpy 标量递归转成原生 Python 类型，避免 YAML 序列化被单个字段卡住。
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
            split_indices_path=str(get_split_indices_path("injury", "train")),
            jitter_ratio=0.0,
            jitter_prob=0.0,
        )
        max_ref_samples = int(config.get("optimization", {}).get("distribution_penalty", {}).get("max_ref_samples", 0))

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
    trace_case_ids = {str(case_id) for case_id in (args.trace_case_ids or [])}
    strategy_net = None
    if bool(config.get("optimization", {}).get("direct_inference", False)):
        if strategy_ckpt_path is None:
            raise ValueError("direct_inference=True 时必须显式提供 strategy_checkpoint 或 --strategy_ckpt")
        if not strategy_ckpt_path.is_file():
            raise FileNotFoundError(f"策略权重不存在: {strategy_ckpt_path}")
        # 策略网络的层参数由该权重所属训练目录的 configs_used/config.yaml 重建，而非当前评估的 cfg_path
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


    optimizer = Optimizer(
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
    if trace_case_ids:
        logger.info("已启用局部精调轨迹导出: trace_case_ids=%s", sorted(trace_case_ids))

    df_input, truth_arrays, input_source = _load_eval_input(args, logger)

    # 如果输入 CSV 没有 case_id 列，则自动添加一个连续整数的 case_id 列，确保后续处理和结果表中都有稳定的行标识。
    # 主要服务于trace_case_ids，与其它需要case_id标识的逻辑。但实际逐点优化评估时不依赖 case_id 来规定顺序，是由行顺序决定。
    if "case_id" not in df_input.columns:
        df_input.insert(0, "case_id", np.arange(len(df_input)) + 1, dtype=np.int32)

    context_output_df, context_eval_df, baseline_df, missing_context, missing_trainable, valid_mask, skipped_case_rows, reverted_baseline_rows, invalid_default_baseline_rows = _prepare_eval_inputs(
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
    logger.info(
        "评估样本统计: total=%d, valid=%d, skipped=%d, reverted_baseline=%d, invalid_default_baseline=%d",
        len(df_input),
        int(valid_mask.sum()),
        len(skipped_case_rows),
        len(reverted_baseline_rows),
        len(invalid_default_baseline_rows),
    )
    if valid_indices.size == 0:
        raise ValueError("所有 case 都因输入端校验失败被跳过，未生成任何可评估样本。")
    context_tensor = torch.tensor(context_eval_df.loc[valid_indices, context_names].values, dtype=torch.float32, device=device)
    baseline_tensor = torch.tensor(baseline_df.loc[valid_indices, trainable_names].values, dtype=torch.float32, device=device)

    eval_batch_size = int(config.get("evaluation", {}).get("eval_batch_size", 512))
    if eval_batch_size <= 0:
        raise ValueError("eval_batch_size 必须为正整数")
    # 核心评估逻辑：按批次计算 Base/Opt1/Opt2 阶段结果。内部会启用具体的策略直推和局部精调逻辑。
    stage_outputs_valid = _compute_predictions_batch(
        context_tensor=context_tensor,
        baseline_trainable=baseline_tensor,
        case_ids=[str(case_id) for case_id in df_input.loc[valid_indices, "case_id"].tolist()],
        row_indices=valid_indices.tolist(),
        surrogate=surrogate,
        optimizer=optimizer,
        eval_batch_size=eval_batch_size,
        trace_case_ids=trace_case_ids,
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
    output_dir = _build_output_dir(base_dir, args.input_csv, input_source=input_source)
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=False)
    output_csv_path = results_dir / Path(args.output_csv).name
    config_snapshots = _copy_config_snapshots(
        cfg_path,
        param_space_path,
        output_dir,
    )
    result_df.to_csv(str(output_csv_path), index=False)
    trace_outputs = _save_case_trace_tables(
        results_dir=results_dir,
        trace_records=stage_outputs.get("trace_records", {}),
        requested_case_ids=trace_case_ids,
        logger=logger,
    )
    if trace_outputs["written_case_count"] > 0:
        logger.info("已导出 %d 个 case 的优化轨迹: %s", trace_outputs["written_case_count"], trace_outputs["trace_dir"])
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
        invalid_default_baseline_rows=invalid_default_baseline_rows,
        opt1_generated=opt1_generated,
        opt2_generated=opt2_generated,
        optimized_stage_name=optimized_stage_name,
        surrogate=surrogate,
        dist_ref_info=dist_ref_info,
        summary_metrics=summary_metrics,
        top_cases=top_cases,
        stage_outputs=stage_outputs,
        evaluated_case_count=evaluated_case_count,
        trace_outputs=trace_outputs,
    )
    with open(results_dir / "evaluation_record.yaml", "w", encoding="utf-8") as file:
        yaml.safe_dump(evaluation_record, file, allow_unicode=True, sort_keys=False)
    logger.info(f"评估完成，结果目录: {output_dir}")
    logger.info(f"结果 CSV: {output_csv_path}")


if __name__ == "__main__":
    main()

