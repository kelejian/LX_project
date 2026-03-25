import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = 'T'
import warnings
warnings.filterwarnings('ignore')
import argparse
import csv
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.optim as optim
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from common.data_utils.processor import UnifiedDataProcessor
from common.settings import NORMALIZATION_CONFIG_PATH, get_split_indices_path
from common.tools.logger import setup_logger
from common.tools.seeding import set_random_seed

from ARS_optim.src.constraints import ConstraintEngine
from ARS_optim.src.data_sampler import StateDataSampler
from ARS_optim.src.param_manager import ParamManager
from ARS_optim.src.strategy_net import build_strategy_net_from_config
from ARS_optim.src.surrogate import SurrogateAdapter, load_surrogate_models


def parse_args():
    parser = argparse.ArgumentParser(description="Train strategy network for ARS")
    parser.add_argument("--config", type=str, default=None, help="override default config file path")
    parser.add_argument("--batch_size", type=int, help="override batch size")
    parser.add_argument("--lr", type=float, help="override learning rate")
    parser.add_argument("--weight_decay", type=float, help="override weight decay")
    parser.add_argument("--max_iterations", type=int, help="override max training iterations")
    parser.add_argument("--device", type=str, help="override device (cpu/cuda)")
    return parser.parse_args()


def _build_scheduler(optimizer, train_cfg: dict, max_iters: int):
    scheduler_cfg = train_cfg.get("scheduler", {}) or {}
    scheduler_type = str(scheduler_cfg.get("type", "none")).lower()
    if scheduler_type in {"cosine", "cosineannealinglr"}:
        t_max_cfg = scheduler_cfg.get("T_max", max_iters)
        if isinstance(t_max_cfg, str):
            t_max_text = t_max_cfg.strip().lower()
            if t_max_text == "max_iters":
                t_max_value = max_iters
            else:
                try:
                    t_max_value = int(t_max_cfg)
                except ValueError as exc:
                    raise ValueError("训练阶段 scheduler.T_max 只允许写整数或 'max_iters'") from exc
        else:
            t_max_value = max_iters if t_max_cfg is None else int(t_max_cfg)
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, t_max_value),
            eta_min=float(scheduler_cfg.get("eta_min", 1e-5)),
        )
    if scheduler_type in {"step", "steplr"}:
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(1, int(scheduler_cfg.get("step_size", max_iters // 5 or 1))),
            gamma=float(scheduler_cfg.get("gamma", 0.5)),
        )
    return None



def _collect_strategy_net_diagnostics(
    strategy_net,
    surrogate: SurrogateAdapter,
    sample_context: torch.Tensor,
    logger,
) -> dict:
    """记录网络结构、参数量和 default 初始化检查结果。"""
    sample_context = sample_context[: min(3, sample_context.shape[0])].detach()
    total_params = sum(param.numel() for param in strategy_net.parameters())
    trainable_params = sum(param.numel() for param in strategy_net.parameters() if param.requires_grad)
    param_stats = {
        "total": int(total_params),
        "trainable": int(trainable_params),
        "non_trainable": int(total_params - trainable_params),
    }

    strategy_net.eval()
    with torch.no_grad():
        pulse_norm = surrogate.generate_pulse(sample_context)
        projected_actions, _ = strategy_net(sample_context, pulse_norm)

    default_actions = strategy_net.default_actions.detach().cpu()
    output_cpu = projected_actions.detach().cpu()
    max_abs_diff = float((output_cpu - default_actions.unsqueeze(0)).abs().max().item())
    diagnostics = {
        "parameter_count": param_stats,
        "arch_info": dict(strategy_net.arch_info),
        "default_actions": default_actions.tolist(),
        "init_output_samples": output_cpu.tolist(),
        "init_output_max_abs_diff_vs_default": max_abs_diff,
    }

    logger.info("StrategyNet 参数量: total=%d, trainable=%d", param_stats["total"], param_stats["trainable"])
    logger.info("StrategyNet 结构摘要: %s", strategy_net.arch_info)
    logger.info("StrategyNet 网络结构:\n%s", strategy_net)
    logger.info("初始化 default 动作: %s", diagnostics["default_actions"])
    logger.info("初始化后随机合法输入输出: %s", diagnostics["init_output_samples"])
    logger.info("初始化输出与 default 的最大绝对误差: %.6e", max_abs_diff)
    return diagnostics

def _build_training_summary(
    train_cfg: dict,
    val_interval: int,
    ema_enabled: bool,
    ema_alpha: float,
    ema_warmup_iters: int,
    tensorboard_enabled: bool,
    surrogate: SurrogateAdapter,
    param_manager: ParamManager,
    train_sampler: StateDataSampler,
    val_sampler: StateDataSampler,
    dist_ref_info: Optional[dict],
    config_snapshots: Dict[str, str],
    network_info: dict,
    train_best_iter: int,
    train_best_loss: float,
    train_metric_name: str,
    train_best_path: Path,
    val_best_iter: int,
    val_best_loss: float,
    val_best_path: Path,
    final_path: Path,
    save_best: bool,
    save_last: bool,
) -> dict:
    def checkpoint_record(
        saved: bool,
        path: Path,
        *,
        status: str,
        iter_idx: Optional[int] = None,
        loss: Optional[float] = None,
        metric: Optional[str] = None,
    ) -> dict:
        """把 checkpoint 摘要统一成同一结构。

        训练摘要里必须区分三种状态：
        - `saved`：该类权重真实落盘，ckpt 指向真实文件名；
        - `disabled` / `validation_disabled`：本次运行压根没有启用这类产物；
        - `not_triggered`：逻辑允许生成，但本次运行中没有触发保存。

        这样摘要文件才能和磁盘事实一一对应，避免出现“summary 里写着有 final_model，
        但实际因为 save_last=False 根本没有生成”的误导记录。
        """
        return {
            "saved": bool(saved),
            "status": status,
            "iter": None if iter_idx is None or iter_idx < 0 else int(iter_idx),
            "loss": None if loss is None else float(loss),
            "metric": metric,
            "ckpt": path.name if saved else None,
        }

    if not save_best:
        train_best_record = checkpoint_record(
            saved=False,
            path=train_best_path,
            status="disabled",
            metric=train_metric_name,
        )
    elif train_best_iter < 0:
        train_best_record = checkpoint_record(
            saved=False,
            path=train_best_path,
            status="not_triggered",
            metric=train_metric_name,
        )
    else:
        train_best_record = checkpoint_record(
            saved=True,
            path=train_best_path,
            status="saved",
            iter_idx=train_best_iter,
            loss=train_best_loss,
            metric=train_metric_name,
        )

    if val_interval <= 0:
        val_best_record = checkpoint_record(
            saved=False,
            path=val_best_path,
            status="validation_disabled",
            metric="val_loss",
        )
    elif val_best_iter < 0:
        val_best_record = checkpoint_record(
            saved=False,
            path=val_best_path,
            status="not_triggered",
            metric="val_loss",
        )
    else:
        val_best_record = checkpoint_record(
            saved=True,
            path=val_best_path,
            status="saved",
            iter_idx=val_best_iter,
            loss=val_best_loss,
            metric="val_loss",
        )

    if save_last:
        final_model_record = checkpoint_record(
            saved=True,
            path=final_path,
            status="saved",
            iter_idx=train_cfg["max_iterations"],
        )
    else:
        final_model_record = checkpoint_record(
            saved=False,
            path=final_path,
            status="disabled",
        )

    return {
        "max_iterations": train_cfg["max_iterations"],
        "val_interval": val_interval,
        "ema": {
            "enabled": ema_enabled,
            "alpha": ema_alpha,
            "warmup_iters": ema_warmup_iters,
            "tensorboard": tensorboard_enabled,
        },
        "data_flow": {
            "train_sampler": train_sampler.get_source_info(),
            "val_sampler": val_sampler.get_source_info(),
        },
        "distribution_penalty": {
            "enabled": bool(surrogate.distribution_penalty.enabled),
            "method": surrogate.distribution_penalty.method,
            "feature_space": surrogate.distribution_penalty.feature_space,
            "weight": surrogate.weight_distribution,
            "k": surrogate.distribution_penalty.k,
            "eps": surrogate.distribution_penalty.eps,
            "clip_max": surrogate.distribution_penalty.clip_max,
            "normalize_by_train_stats": surrogate.distribution_penalty.normalize_by_train_stats,
            "reference_sampling": dist_ref_info,
        },
        "parameter_roles": {
            "context": param_manager.get_context_names(),
            "trainable_control": param_manager.get_trainable_names(),
            "fixed_control": param_manager.get_fixed_control_names(),
        },
        # 运行目录下的configs_used中已经直接保存了 config.yaml / param_space.yaml / normalization_config.json；
        # 这里不再把同一批外部工件路径重复展开到 summary，避免摘要和快照文件各维护一份信息。
        "config_files": config_snapshots,
        "network": network_info,
        "train_best": train_best_record,
        "val_best": val_best_record,
        "final_model": final_model_record,
    }


def _evaluate_strategy_batch(
    strategy_net,
    surrogate: SurrogateAdapter,
    context_params: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    """统一一批 context 上的策略前向与代理评估。

    训练批和验证批都走同一条链路：先基于 context 生成 pulse，再由策略网络产出动作，
    最后交给代理模型计算逐样本 loss 与风险拆解。

    这样做的目的不是增加封装层，而是避免训练循环和验证循环各自维护一套相同的前向逻辑。
    同时也把 pulse 的来源固定为“当前 batch 显式生成一次并沿链路传递”，
    不允许策略网、代理层或局部优化器在内部再各跑一次隐式 pulse fallback。
    """
    pulse_norm = surrogate.generate_pulse(context_params)
    actions, raw_full_features = strategy_net(context_params, pulse_norm)
    loss_batch, _, info = surrogate.predict_injury_and_loss(
        context_params=context_params,
        control_trainable=actions,
        pulse_norm=pulse_norm,
        include_opt_bounds=False,
        penalty_features=raw_full_features,
    )
    return loss_batch, info


def _run_gradient_sanity_check(
    strategy_net,
    surrogate: SurrogateAdapter,
    context_params: torch.Tensor,
) -> None:
    """在正式训练前做一次轻量梯度流检查。"""
    strategy_net.train()
    strategy_net.zero_grad(set_to_none=True)
    loss_batch, _ = _evaluate_strategy_batch(strategy_net, surrogate, context_params)
    loss_mean = loss_batch.mean()
    if torch.isnan(loss_mean) or torch.isinf(loss_mean):
        raise RuntimeError("梯度流自检失败：sanity batch 的 loss 出现 NaN/Inf")
    loss_mean.backward()

    out_layer = strategy_net.output_head
    if not isinstance(out_layer, torch.nn.Linear):
        raise TypeError("梯度流自检失败：策略网络最后一层不是 Linear")

    for name, param in {"weight": out_layer.weight, "bias": out_layer.bias}.items():
        if param is None or param.grad is None:
            raise RuntimeError(f"梯度流自检失败：输出层 {name} 梯度缺失")
        if not torch.isfinite(param.grad).all():
            raise RuntimeError(f"梯度流自检失败：输出层 {name} 梯度存在 NaN/Inf")
        if float(param.grad.abs().max().item()) <= 0.0:
            raise RuntimeError(f"梯度流自检失败：输出层 {name} 梯度全为 0")

    strategy_net.zero_grad(set_to_none=True)


def main():
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    logger = setup_logger(name="ARS_optim.train")

    cfg_path = Path(args.config) if args.config else base_dir / "configs" / "default_config.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"config file not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if "strategy_net" not in config or "train" not in config["strategy_net"]:
        raise KeyError("配置文件中缺失 strategy_net.train")
    config.setdefault("optimization", {})
    train_cfg = config["strategy_net"]["train"]

    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size
    if args.lr is not None:
        train_cfg["lr"] = args.lr
    if args.weight_decay is not None:
        train_cfg["weight_decay"] = args.weight_decay
    if args.max_iterations is not None:
        train_cfg["max_iterations"] = args.max_iterations
    if args.device is not None:
        config["device"] = args.device

    train_cfg["batch_size"] = int(train_cfg.get("batch_size", 0))
    train_cfg["max_iterations"] = int(train_cfg.get("max_iterations", 0))
    train_cfg["lr"] = float(train_cfg.get("lr", 0.0))
    train_cfg["weight_decay"] = float(train_cfg.get("weight_decay", 0.0))
    if train_cfg["batch_size"] <= 0 or train_cfg["max_iterations"] <= 0:
        raise ValueError("batch_size 和 max_iterations 必须为正整数")
    if train_cfg["lr"] <= 0.0 or train_cfg["weight_decay"] < 0.0:
        raise ValueError("lr 必须为正数，weight_decay 必须非负")

    ema_cfg = train_cfg.get("ema", {}) or {}
    ema_enabled = bool(ema_cfg.get("enabled", True))
    ema_alpha = float(ema_cfg.get("alpha", 0.98))
    ema_warmup_iters = int(ema_cfg.get("warmup_iters", 100))
    ema_log_to_tb = bool(ema_cfg.get("log_to_tensorboard", True))
    if not (0.0 <= ema_alpha < 1.0):
        raise ValueError("EMA alpha 必须满足 0<=alpha<1")
    if ema_warmup_iters < 0:
        raise ValueError("EMA warmup_iters 必须非负")

    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    seed = int(config.get("seed", 42))
    set_random_seed(seed)
    logger.info(f"训练设备: {device}")
    logger.info(f"随机种子: {seed}")

    param_space_path = base_dir / "configs" / "param_space.yaml"
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

    strategy_net = build_strategy_net_from_config(
        param_manager=param_manager,
        constraint_engine=constraint_engine,
        data_processor=data_processor,
        config=config,
    ).to(device)

    optimizer = optim.AdamW(strategy_net.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    scheduler = _build_scheduler(optimizer, train_cfg, train_cfg["max_iterations"])
    val_batch_size = int(train_cfg.get("val_batch_size", 1024))

    train_sampler = StateDataSampler(
        param_manager=param_manager,
        constraint_engine=constraint_engine,
        batch_size=train_cfg["batch_size"],
        device=device,
        seed=seed,
        split_indices_path=str(get_split_indices_path("injury", "train")),
        jitter_ratio=float(train_cfg.get("jitter_ratio", 0.01)),
        jitter_prob=float(train_cfg.get("jitter_prob", 1.0)),
        jitter_max_attempts=int(train_cfg.get("jitter_max_attempts", 1)),
        jitter_min_active_dims=int(train_cfg.get("jitter_min_active_dims", 1)),
    )
    val_sampler = StateDataSampler(
        param_manager=param_manager,
        constraint_engine=constraint_engine,
        batch_size=val_batch_size,
        device=device,
        seed=seed,
        split_indices_path=str(get_split_indices_path("injury", "val")),
        jitter_ratio=0.0,
        jitter_prob=0.0,
    )
    train_generator = train_sampler.get_infinite_generator()
    dist_ref_info = None

    if surrogate.distribution_penalty.enabled:
        dist_cfg = config.get("optimization", {}).get("distribution_penalty", {})
        # 分布惩罚的参考集必须尽量代表整个训练经验池，而不是池中前缀切片。
        # 这里统一使用“完整训练池”或“基于全局 seed 的一次性随机子样本”，让训练和评估两端拟合出的参考分布保持同源、可复现。
        ref_context, dist_ref_info = train_sampler.get_distribution_reference(
            max_samples=int(dist_cfg.get("max_ref_samples", 0)),
            feature_space=surrogate.distribution_penalty.feature_space,
            trainable_indices=param_manager.get_control_trainable_indices(),
            sample_seed=seed,
        )
        surrogate.fit_distribution_reference(ref_context)

    sanity_context = next(train_generator)
    network_info = _collect_strategy_net_diagnostics(strategy_net, surrogate, sanity_context, logger)
    _run_gradient_sanity_check(strategy_net, surrogate, sanity_context)
    logger.info("梯度流自检通过：策略网络输出层梯度存在且有限")

    tensorboard_enabled = bool(train_cfg.get("tensorboard", True))

    save_root = base_dir / "saved_models"
    save_root.mkdir(parents=True, exist_ok=True)
    run_dir = save_root / f"strategy_net_{datetime.now().strftime('%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    config_dir = run_dir / "configs_used"
    checkpoint_dir = run_dir / "checkpoints"
    record_dir = run_dir / "records"
    tensorboard_dir = run_dir / "tensorboard"
    config_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    record_dir.mkdir(parents=True, exist_ok=False)
    if tensorboard_enabled:
        tensorboard_dir.mkdir(parents=True, exist_ok=False)

    config_used_path = config_dir / "config.yaml"
    param_snapshot_path = config_dir / "param_space.yaml"
    run_norm_snapshot_path = config_dir / "normalization_config.json"
    with open(config_used_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)
    shutil.copy2(str(param_space_path), str(param_snapshot_path))
    shutil.copy2(str(NORMALIZATION_CONFIG_PATH), str(run_norm_snapshot_path))
    config_snapshots = {
        "config_used": str(config_used_path),
        "param_space_used": str(param_snapshot_path),
        "normalization_config": str(run_norm_snapshot_path),
    }

    writer = SummaryWriter(log_dir=str(tensorboard_dir)) if tensorboard_enabled else None

    train_best_path = checkpoint_dir / "train_best_model.pth"
    val_best_path = checkpoint_dir / "val_best_model.pth"
    final_path = checkpoint_dir / "final_model.pth"
    history_path = record_dir / "training_history.csv"
    summary_path = record_dir / "training_summary.yaml"
    history_rows = []
    train_best_loss = float("inf")
    val_best_loss = float("inf")
    train_best_iter = -1
    val_best_iter = -1
    ema_train_loss: Optional[float] = None

    def evaluate_full_val() -> tuple[float, float, float, float]:
        """在 injury_val 全量样本上做一次无扰动评估。

        训练阶段使用无限采样流，不存在自然 epoch；
        因此验证要显式遍历固定的 injury_val 切片，才能让不同迭代点的验证指标具有可比性。
        """
        strategy_net.eval()
        total_loss = 0.0
        total_risk = 0.0
        total_constraint = 0.0
        total_distribution = 0.0
        sample_count = 0
        with torch.no_grad():
            for batch_context in val_sampler.iter_dataset_batches(batch_size=val_batch_size, shuffle=False):
                loss_batch, info = _evaluate_strategy_batch(strategy_net, surrogate, batch_context)
                total_loss += float(loss_batch.sum().item())
                total_risk += float(info["loss_risk"].sum().item())
                total_constraint += float(info["loss_constraint"].sum().item())
                total_distribution += float(info["loss_distribution"].sum().item())
                sample_count += int(loss_batch.numel())
        strategy_net.train()
        if sample_count == 0:
            raise ValueError("验证集为空")
        return (
            total_loss / sample_count,
            total_risk / sample_count,
            total_constraint / sample_count,
            total_distribution / sample_count,
        )

    log_interval = int(train_cfg.get("log_interval", 10))
    val_interval = int(train_cfg.get("val_interval", 500))
    if val_sampler.pool_size == 0:
        if val_interval > 0:
            logger.warning("injury_val 切片为空：本次训练将跳过固定验证与 val_best 权重保存。")
        val_interval = 0
    grad_clip = float(train_cfg.get("gradient_clip_max_norm", 1.0))
    save_best = bool(train_cfg.get("save_best", True))
    save_last = bool(train_cfg.get("save_last", True))
    train_metric_name = "ema_train_loss" if ema_enabled else "train_loss"

    strategy_net.train()
    try:
        for iter_idx in tqdm(range(train_cfg["max_iterations"]), desc="Training StrategyNet"):
            optimizer.zero_grad()
            context_params = next(train_generator)
            total_loss, info = _evaluate_strategy_batch(strategy_net, surrogate, context_params)
            loss_mean = total_loss.mean()
            if torch.isnan(loss_mean) or torch.isinf(loss_mean):
                raise RuntimeError(f"iter={iter_idx + 1}: loss 出现 NaN/Inf，请检查当前配置、数据流或代理模型输出")

            loss_mean.backward()
            torch.nn.utils.clip_grad_norm_(strategy_net.parameters(), max_norm=grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            loss_value = float(loss_mean.item())
            loss_risk = float(info["loss_risk"].mean().item())
            loss_constraint = float(info["loss_constraint"].mean().item())
            loss_distribution = float(info["loss_distribution"].mean().item())
            sampling_stats = train_sampler.get_last_sampling_stats()
            jitter_attempt_rejection_rate = float(sampling_stats["attempt_rejection_rate"])
            jitter_final_fallback_rate = float(sampling_stats["final_fallback_rate"])
            jitter_changed_param_ratio = float(sampling_stats["mean_changed_param_ratio"])
            current_lr = float(optimizer.param_groups[0]["lr"])

            ema_train_loss = loss_value if ema_train_loss is None else ema_alpha * ema_train_loss + (1.0 - ema_alpha) * loss_value
            train_select_metric = ema_train_loss if ema_enabled else loss_value
            # 训练最优权重默认按 EMA 指标挑选，是为了弱化随机采样流带来的单步抖动；
            # 但在 warmup 结束前 EMA 还没有稳定含义，因此这段区间不更新 train_best。
            # val_best 则始终来自固定 injury_val 全量评估，表示“泛化到固定验证切片的最好点”。
            # 两者故意分开记录，避免把流式训练噪声下的最优点和固定验证集最优点混成一个概念。
            ema_ready = (iter_idx + 1) >= max(1, ema_warmup_iters) if ema_enabled else True

            if writer is not None:
                writer.add_scalar("Train/Loss", loss_value, iter_idx)
                writer.add_scalar("Train/LossRisk", loss_risk, iter_idx)
                writer.add_scalar("Train/LossConstraint", loss_constraint, iter_idx)
                writer.add_scalar("Train/LossDistribution", loss_distribution, iter_idx)
                writer.add_scalar("Train/JitterAttemptRejectionRate", jitter_attempt_rejection_rate, iter_idx)
                writer.add_scalar("Train/JitterFinalFallbackRate", jitter_final_fallback_rate, iter_idx)
                writer.add_scalar("Train/JitterChangedParamRatio", jitter_changed_param_ratio, iter_idx)
                writer.add_scalar("Train/LR", current_lr, iter_idx)
                if ema_log_to_tb:
                    writer.add_scalar("Train/EMA_Loss", float(ema_train_loss), iter_idx)

            val_loss = None
            val_loss_risk = None
            val_loss_constraint = None
            val_loss_distribution = None
            if val_interval > 0 and (iter_idx + 1) % val_interval == 0:
                val_loss, val_loss_risk, val_loss_constraint, val_loss_distribution = evaluate_full_val()
                if writer is not None:
                    writer.add_scalar("Val/Loss", val_loss, iter_idx)
                    writer.add_scalar("Val/LossRisk", val_loss_risk, iter_idx)
                    writer.add_scalar("Val/LossConstraint", val_loss_constraint, iter_idx)
                    writer.add_scalar("Val/LossDistribution", val_loss_distribution, iter_idx)
                if val_loss < val_best_loss:
                    val_best_loss = val_loss
                    val_best_iter = iter_idx + 1
                    torch.save(strategy_net.state_dict(), val_best_path)

            if save_best and ema_ready and train_select_metric < train_best_loss:
                train_best_loss = float(train_select_metric)
                train_best_iter = iter_idx + 1
                torch.save(strategy_net.state_dict(), train_best_path)

            history_rows.append(
                {
                    "iteration": iter_idx + 1,
                    "train_loss": loss_value,
                    "train_ema_loss": float(ema_train_loss),
                    "train_loss_risk": loss_risk,
                    "train_loss_constraint": loss_constraint,
                    "train_loss_distribution": loss_distribution,
                    "jitter_attempt_rejection_rate": jitter_attempt_rejection_rate,
                    "jitter_final_fallback_rate": jitter_final_fallback_rate,
                    "jitter_changed_param_ratio": jitter_changed_param_ratio,
                    "val_loss": val_loss,
                    "val_loss_risk": val_loss_risk,
                    "val_loss_constraint": val_loss_constraint,
                    "val_loss_distribution": val_loss_distribution,
                    "lr": current_lr,
                }
            )
            if iter_idx % max(1, log_interval) == 0:
                tqdm.write(
                    f"iter={iter_idx + 1} loss={loss_value:.4f} ema={ema_train_loss:.4f} "
                    f"risk={loss_risk:.4f} penalty={loss_constraint:.4f} dist={loss_distribution:.4f} "
                    f"att_rej={jitter_attempt_rejection_rate:.2%} "
                    f"fallback={jitter_final_fallback_rate:.2%} "
                    f"changed={jitter_changed_param_ratio:.2%} "
                    f"lr={current_lr:.2e}"
                )

        if save_last:
            torch.save(strategy_net.state_dict(), final_path)

        with open(history_path, "w", newline="", encoding="utf-8") as file:
            writer_csv = csv.DictWriter(file, fieldnames=list(history_rows[0].keys()) if history_rows else ["iteration"])
            writer_csv.writeheader()
            writer_csv.writerows(history_rows)

        with open(summary_path, "w", encoding="utf-8") as file:
            yaml.safe_dump(
                _build_training_summary(
                    train_cfg=train_cfg,
                    val_interval=val_interval,
                    ema_enabled=ema_enabled,
                    ema_alpha=ema_alpha,
                    ema_warmup_iters=ema_warmup_iters,
                    tensorboard_enabled=tensorboard_enabled,
                    surrogate=surrogate,
                    param_manager=param_manager,
                    train_sampler=train_sampler,
                    val_sampler=val_sampler,
                    dist_ref_info=dist_ref_info,
                    config_snapshots=config_snapshots,
                    network_info=network_info,
                    train_best_iter=train_best_iter,
                    train_best_loss=train_best_loss,
                    train_metric_name=train_metric_name,
                    train_best_path=train_best_path,
                    val_best_iter=val_best_iter,
                    val_best_loss=val_best_loss,
                    val_best_path=val_best_path,
                    final_path=final_path,
                    save_best=save_best,
                    save_last=save_last,
                ),
                file,
                allow_unicode=True,
                sort_keys=False,
            )

        if train_best_iter > 0:
            logger.info(f"训练最优权重: {train_best_path}")
        if val_best_iter > 0:
            logger.info(f"验证最优权重: {val_best_path}")
        if save_last:
            logger.info(f"最终权重: {final_path}")
        logger.info(f"训练产物目录: {run_dir}")
    finally:
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    main()

