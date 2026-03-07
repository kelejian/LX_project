import argparse
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import yaml

from common.data_utils.processor import UnifiedDataProcessor
from common.metrics.injury_risk import AIS_cal_chest, AIS_cal_head, AIS_cal_neck
from common.settings import FEATURE_ORDER, NORMALIZATION_CONFIG_PATH, RAW_DATA_DIR, SPLIT_INDICES_DIR
from common.tools.logger import setup_logger

from ARS_optim.src.constraints import ConstraintEngine
from ARS_optim.src.data_sampler import StateDataSampler
from ARS_optim.src.optimizer import LocalRefiner
from ARS_optim.src.param_manager import ParamManager
from ARS_optim.src.strategy_net import StrategyNet
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
    configured = eval_cfg.get("default_strategy_ckpt")
    if configured:
        candidate = Path(str(configured))
        return candidate if candidate.is_absolute() else (base_dir / candidate).resolve()

    strict = bool(eval_cfg.get("strict_default_ckpt", False))
    run_dirs = sorted(
        [path for path in (base_dir / "saved_models").glob("strategy_net_*") if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    priorities = ["val_best_model.pth"] if strict else ["val_best_model.pth", "train_best_model.pth", "final_model.pth"]
    for run_dir in run_dirs:
        for filename in priorities:
            candidate = run_dir / filename
            if candidate.is_file():
                return candidate
    return None


def _copy_config_snapshots(cfg_path: Path, param_space_path: Path, output_dir: Path) -> Dict[str, str]:
    cfg_snapshot = output_dir / "config_used.yaml"
    param_snapshot = output_dir / "param_space.yaml"
    shutil.copy2(str(cfg_path), str(cfg_snapshot))
    shutil.copy2(str(param_space_path), str(param_snapshot))
    return {
        "config_used": str(cfg_snapshot),
        "param_space_used": str(param_snapshot),
        "normalization_config": str(NORMALIZATION_CONFIG_PATH),
    }


def _write_yaml(path: Path, content: Dict) -> None:
    with open(path, "w", encoding="utf-8") as file:
        yaml.safe_dump(content, file, allow_unicode=True, sort_keys=False)


def _compute_joint_risk(prob_head: np.ndarray, prob_chest: np.ndarray, prob_neck: np.ndarray) -> np.ndarray:
    return 1.0 - (1.0 - prob_head) * (1.0 - prob_chest) * (1.0 - prob_neck)


def _build_param_dataframe(df_input: pd.DataFrame, params: List[dict], logger, missing_message: str) -> tuple[pd.DataFrame, List[str]]:
    output_df = pd.DataFrame(index=df_input.index)
    missing = []
    for param in params:
        name = param["name"]
        if name in df_input.columns:
            output_df[name] = pd.to_numeric(df_input[name], errors="raise")
        else:
            output_df[name] = float(param["default"])
            missing.append(name)
    if missing:
        logger.warning(missing_message.format(missing=missing))
    return output_df, missing


def _fit_distribution_reference_if_needed(surrogate: SurrogateAdapter, sampler: StateDataSampler, param_manager: ParamManager, config: dict, logger) -> None:
    if not surrogate.distribution_penalty.enabled:
        return
    try:
        max_ref_samples = int(config.get("optimization", {}).get("distribution_penalty", {}).get("max_ref_samples", 0))
        reference = sampler.get_distribution_reference(
            max_samples=max_ref_samples,
            shuffle=False,
            feature_space=surrogate.distribution_penalty.feature_space,
            trainable_indices=param_manager.get_control_trainable_indices(),
        )
        surrogate.fit_distribution_reference(reference)
    except Exception as exc:
        logger.warning(f"评估阶段拟合分布参考失败，将关闭分布惩罚: {exc}")
        surrogate.distribution_penalty.enabled = False


def _compute_predictions_batch(
    context_tensor: torch.Tensor,
    baseline_trainable: torch.Tensor,
    surrogate: SurrogateAdapter,
    optimizer: LocalRefiner,
    constraint_engine: ConstraintEngine,
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
        for key in ["Base", "Init", "Opt"]
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

        opt_actions, opt_preds, opt_info = optimizer.optimize(context_batch, pulse_norm=pulse_batch)
        _, opt_actions = constraint_engine.sanitize_context_and_trainable(context_batch, opt_actions)
        stage_parts["Opt"]["preds"].append(opt_preds.detach())
        stage_parts["Opt"]["actions"].append(opt_actions.detach())
        stage_parts["Opt"]["loss"].append(opt_info["final_loss_batch"].detach())
        for key in stage_parts["Opt"]["info"]:
            stage_parts["Opt"]["info"][key].append(opt_info[key].detach())

        init_data = opt_info.get("initial", {})
        if init_data.get("actions") is not None:
            init_actions = init_data["actions"].detach()
            _, init_actions = constraint_engine.sanitize_context_and_trainable(context_batch, init_actions)
            stage_parts["Init"]["actions"].append(init_actions)
        if init_data.get("preds") is not None:
            stage_parts["Init"]["preds"].append(init_data["preds"].detach())
        if init_data.get("loss_batch") is not None:
            stage_parts["Init"]["loss"].append(init_data["loss_batch"].detach())
        init_detail = init_data.get("detail", {})
        for key in stage_parts["Init"]["info"]:
            if key in init_detail:
                stage_parts["Init"]["info"][key].append(init_detail[key].detach())

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


def _build_result_dataframe(
    df_input: pd.DataFrame,
    context_df: pd.DataFrame,
    stage_outputs: Dict[str, object],
    truth_arrays: Dict[str, np.ndarray],
    trainable_names: List[str],
    fixed_names: List[str],
) -> tuple[pd.DataFrame, Dict[str, float]]:
    frame_parts = [df_input.reset_index(drop=True), context_df.reset_index(drop=True)]

    ot_array = context_df["OT"].to_numpy(dtype=np.float32)
    fixed_array = context_df[fixed_names].to_numpy(dtype=np.float32) if fixed_names else None

    for prefix in ["Base", "Init", "Opt"]:
        stage = stage_outputs[prefix]
        preds = stage["preds"]
        actions = stage["actions"]
        loss = stage["loss"]
        info = stage["info"]

        pred_array = np.full((len(df_input), 3), np.nan, dtype=np.float32) if preds is None else preds.detach().cpu().numpy()

        info_arrays = {}
        for name in ["p_head", "p_chest", "p_neck", "joint_risk"]:
            info_arrays[name] = info[name].detach().cpu().numpy()
        joint_risk = info_arrays["joint_risk"]
        if np.isnan(joint_risk).all():
            joint_risk = _compute_joint_risk(info_arrays["p_head"], info_arrays["p_chest"], info_arrays["p_neck"])

        stage_data = {
            f"{prefix}_HIC": pred_array[:, 0],
            f"{prefix}_Dmax": pred_array[:, 1],
            f"{prefix}_Nij": pred_array[:, 2],
            f"{prefix}_Loss": loss.detach().cpu().numpy(),
            f"{prefix}_Phead": info_arrays["p_head"],
            f"{prefix}_Pchest": info_arrays["p_chest"],
            f"{prefix}_Pneck": info_arrays["p_neck"],
            f"{prefix}_JointRisk": joint_risk,
        }
        stage_data[f"{prefix}_AIS_head"] = AIS_cal_head(stage_data[f"{prefix}_HIC"].astype(np.float32))
        stage_data[f"{prefix}_AIS_chest"] = AIS_cal_chest(stage_data[f"{prefix}_Dmax"].astype(np.float32), ot_array)
        stage_data[f"{prefix}_AIS_neck"] = AIS_cal_neck(stage_data[f"{prefix}_Nij"].astype(np.float32))
        stage_data[f"{prefix}_AIS_max"] = np.nanmax(
            np.vstack(
                [
                    stage_data[f"{prefix}_AIS_head"].astype(np.float32),
                    stage_data[f"{prefix}_AIS_chest"].astype(np.float32),
                    stage_data[f"{prefix}_AIS_neck"].astype(np.float32),
                ]
            ),
            axis=0,
        )

        if actions is None:
            for name in trainable_names + fixed_names:
                stage_data[f"{prefix}_{name}"] = np.full(len(df_input), np.nan, dtype=np.float32)
        else:
            action_array = actions.detach().cpu().numpy()
            for idx, name in enumerate(trainable_names):
                stage_data[f"{prefix}_{name}"] = action_array[:, idx]
            for idx, name in enumerate(fixed_names):
                stage_data[f"{prefix}_{name}"] = fixed_array[:, idx]

        frame_parts.append(pd.DataFrame(stage_data, index=df_input.index).reset_index(drop=True))

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

    eps = 1e-8
    reduction_specs = [
        ("HIC", "Base_HIC", "Opt_HIC", True),
        ("Dmax", "Base_Dmax", "Opt_Dmax", True),
        ("Nij", "Base_Nij", "Opt_Nij", True),
        ("Phead", "Base_Phead", "Opt_Phead", False),
        ("Pchest", "Base_Pchest", "Opt_Pchest", False),
        ("Pneck", "Base_Pneck", "Opt_Pneck", False),
        ("JointRisk", "Base_JointRisk", "Opt_JointRisk", False),
        ("AIS_head", "Base_AIS_head", "Opt_AIS_head", False),
        ("AIS_chest", "Base_AIS_chest", "Opt_AIS_chest", False),
        ("AIS_neck", "Base_AIS_neck", "Opt_AIS_neck", False),
        ("AIS_max", "Base_AIS_max", "Opt_AIS_max", False),
        ("Loss", "Base_Loss", "Opt_Loss", True),
    ]
    reduction_data = {}
    for alias, base_col, opt_col, with_pct in reduction_specs:
        reduction_abs = result_df[base_col] - result_df[opt_col]
        reduction_data[f"Reduction_{alias}_abs"] = reduction_abs
        if with_pct:
            reduction_data[f"Reduction_{alias}_pct"] = reduction_abs / np.maximum(np.abs(result_df[base_col]), eps)

    if reduction_data:
        result_df = pd.concat([result_df, pd.DataFrame(reduction_data, index=df_input.index).reset_index(drop=True)], axis=1)

    summary = {
        "mean_reduction_HIC": float(np.nanmean(result_df["Reduction_HIC_abs"])),
        "mean_reduction_Dmax": float(np.nanmean(result_df["Reduction_Dmax_abs"])),
        "mean_reduction_Nij": float(np.nanmean(result_df["Reduction_Nij_abs"])),
        "mean_reduction_Phead": float(np.nanmean(result_df["Reduction_Phead_abs"])),
        "mean_reduction_Pchest": float(np.nanmean(result_df["Reduction_Pchest_abs"])),
        "mean_reduction_Pneck": float(np.nanmean(result_df["Reduction_Pneck_abs"])),
        "mean_reduction_joint_risk": float(np.nanmean(result_df["Reduction_JointRisk_abs"])),
        "mean_base_joint_risk": float(np.nanmean(result_df["Base_JointRisk"])),
        "mean_opt_joint_risk": float(np.nanmean(result_df["Opt_JointRisk"])),
        "n_samples": int(len(result_df)),
    }
    return result_df, summary


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
    output_dir = _build_output_dir(base_dir, args.input_csv)
    output_csv_path = output_dir / Path(args.output_csv).name

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
        split_indices_path=str(SPLIT_INDICES_DIR / "injury_train_indices.npy"),
        jitter_ratio=0.0,
        jitter_prob=0.0,
    )
    _fit_distribution_reference_if_needed(surrogate, ref_sampler, param_manager, config, logger)

    strat_cfg = config.get("strategy_net", {})
    strategy_net = StrategyNet(
        param_manager=param_manager,
        constraint_engine=constraint_engine,
        data_processor=data_processor,
        hidden_dims=strat_cfg.get("hidden_dims", [128, 256, 128]),
        activation=strat_cfg.get("activation", "LeakyReLU"),
        dropout=float(strat_cfg.get("dropout", 0.1)),
        pulse_channels=int(strat_cfg.get("pulse_channels", 2)),
        pulse_embed_dim=int(strat_cfg.get("pulse_embed_dim", 32)),
    ).to(device)

    strategy_ckpt_path = _resolve_strategy_ckpt(args, base_dir, config)
    strict_default_ckpt = bool(config.get("evaluation", {}).get("strict_default_ckpt", False))
    if strategy_ckpt_path is not None:
        if not strategy_ckpt_path.is_file():
            raise FileNotFoundError(f"策略权重不存在: {strategy_ckpt_path}")
        strategy_net.load_state_dict(torch.load(str(strategy_ckpt_path), map_location=device, weights_only=True))
        config["optimization"]["direct_inference"] = True
    elif strict_default_ckpt:
        raise FileNotFoundError("strict_default_ckpt=true 且未找到默认策略权重")

    optimizer = LocalRefiner(
        config=config,
        param_manager=param_manager,
        constraint_engine=constraint_engine,
        surrogate=surrogate,
        strategy_net=strategy_net,
    )

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
    else:
        pool_path = RAW_DATA_DIR / "raw_data_packed.npz"
        test_idx_path = SPLIT_INDICES_DIR / "injury_test_indices.npy"
        if not pool_path.exists() or not test_idx_path.exists():
            raise FileNotFoundError("自动测试集模式需要 raw_data_packed.npz 和 injury_test_indices.npy")
        test_indices = np.load(str(test_idx_path)).astype(np.int64)
        with np.load(str(pool_path), allow_pickle=True) as data:
            x_att_raw = data["x_att_raw"][test_indices]
            df_input = pd.DataFrame(x_att_raw, columns=FEATURE_ORDER)
            case_ids = data["case_ids"][test_indices] if "case_ids" in data else np.arange(len(test_indices))
            df_input.insert(0, "case_id", case_ids)
            for key in ["y_HIC", "y_Dmax", "y_Nij", "ais_head", "ais_chest", "ais_neck"]:
                if key in data:
                    truth_arrays[key] = np.asarray(data[key][test_indices])
        input_source = {
            "type": "test_split",
            "path": str(test_idx_path.resolve()),
            "raw_data_npz_path": str(pool_path.resolve()),
        }
        logger.info(f"未指定 input_csv，自动加载测试集: {len(df_input)} 条")

    if "case_id" not in df_input.columns:
        df_input.insert(0, "case_id", np.arange(len(df_input), dtype=np.int64))

    context_df, missing_context = _build_param_dataframe(
        df_input,
        param_manager.get_context_params(),
        logger,
        "输入缺失部分 context 参数，已回退 default: {missing}",
    )
    baseline_df, missing_trainable = _build_param_dataframe(
        df_input,
        param_manager.control_trainable_params,
        logger,
        "输入未提供部分可调参数，baseline 已回退 default: {missing}",
    )
    context_names = param_manager.get_context_names()
    trainable_names = [param["name"] for param in param_manager.control_trainable_params]
    fixed_names = [param["name"] for param in param_manager.control_fixed_params]

    context_tensor_raw = torch.tensor(context_df[context_names].values, dtype=torch.float32, device=device)
    baseline_tensor_raw = torch.tensor(baseline_df[trainable_names].values, dtype=torch.float32, device=device)
    context_tensor, baseline_tensor = constraint_engine.sanitize_context_and_trainable(context_tensor_raw, baseline_tensor_raw)
    context_df = pd.DataFrame(context_tensor.detach().cpu().numpy(), columns=context_names, index=df_input.index)

    eval_batch_size = int(config.get("evaluation", {}).get("eval_batch_size", 512))
    if eval_batch_size <= 0:
        raise ValueError("eval_batch_size 必须为正整数")
    stage_outputs = _compute_predictions_batch(
        context_tensor=context_tensor,
        baseline_trainable=baseline_tensor,
        surrogate=surrogate,
        optimizer=optimizer,
        constraint_engine=constraint_engine,
        eval_batch_size=eval_batch_size,
    )

    result_df, summary_metrics = _build_result_dataframe(
        df_input=df_input,
        context_df=context_df,
        stage_outputs=stage_outputs,
        truth_arrays=truth_arrays,
        trainable_names=trainable_names,
        fixed_names=fixed_names,
    )
    result_df.to_csv(str(output_csv_path), index=False)

    eval_info = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(output_dir),
        "output_csv": str(output_csv_path),
        "input_source": input_source,
        "strategy_checkpoint_path": str(strategy_ckpt_path) if strategy_ckpt_path is not None else None,
        "direct_inference": bool(config.get("optimization", {}).get("direct_inference", False)),
        "config_files": _copy_config_snapshots(cfg_path, param_space_path, output_dir),
        "evaluation_config": config.get("evaluation", {}),
        "optimization_config": config.get("optimization", {}),
        "missing_context_filled_by_default": missing_context,
        "missing_trainable_filled_by_default": missing_trainable,
        "summary_metrics": summary_metrics,
        "formulas": {
            "joint_risk": "L_risk = 1 - Π_k (1 - P_k)",
            "reported_reduction": "mean(Baseline - Optimized)",
        },
        "runtime": {
            "total_time_cost_sec": float(stage_outputs["total_time_cost"]),
            "avg_time_cost_sec": float(stage_outputs["total_time_cost"] / max(1, len(result_df))),
            "trajectory_steps_logged": len(stage_outputs["trajectory_all"]),
        },
    }
    _write_yaml(output_dir / "eval_info.yaml", eval_info)
    logger.info(f"评估完成，结果目录: {output_dir}")
    logger.info(f"结果 CSV: {output_csv_path}")


if __name__ == "__main__":
    main()
