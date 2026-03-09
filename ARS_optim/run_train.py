import argparse
import csv
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.optim as optim
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from common.data_utils.processor import UnifiedDataProcessor
from common.settings import NORMALIZATION_CONFIG_PATH, SPLIT_INDICES_DIR
from common.tools.logger import setup_logger
from common.tools.seeding import set_random_seed

from ARS_optim.src.constraints import ConstraintEngine
from ARS_optim.src.data_sampler import StateDataSampler
from ARS_optim.src.param_manager import ParamManager
from ARS_optim.src.strategy_net import StrategyNet
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
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(scheduler_cfg.get("T_max", max_iters))),
            eta_min=float(scheduler_cfg.get("eta_min", 1e-5)),
        )
    if scheduler_type in {"step", "steplr"}:
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(1, int(scheduler_cfg.get("step_size", max_iters // 5 or 1))),
            gamma=float(scheduler_cfg.get("gamma", 0.5)),
        )
    return None


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

    optimizer = optim.Adam(strategy_net.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    scheduler = _build_scheduler(optimizer, train_cfg, train_cfg["max_iterations"])

    train_sampler = StateDataSampler(
        param_manager=param_manager,
        constraint_engine=constraint_engine,
        batch_size=train_cfg["batch_size"],
        device=device,
        seed=seed,
        jitter_ratio=float(train_cfg.get("jitter_ratio", 0.01)),
        jitter_prob=float(train_cfg.get("jitter_prob", 1.0)),
    )
    val_sampler = StateDataSampler(
        param_manager=param_manager,
        constraint_engine=constraint_engine,
        batch_size=int(train_cfg.get("val_batch_size", 1024)),
        device=device,
        seed=seed,
        split_indices_path=str(SPLIT_INDICES_DIR / "injury_val_indices.npy"),
        jitter_ratio=0.0,
        jitter_prob=0.0,
    )
    train_generator = train_sampler.get_infinite_generator()

    if surrogate.distribution_penalty.enabled:
        dist_cfg = config.get("optimization", {}).get("distribution_penalty", {})
        ref_context = train_sampler.get_distribution_reference(
            max_samples=int(dist_cfg.get("max_ref_samples", 0)),
            shuffle=False,
            feature_space=surrogate.distribution_penalty.feature_space,
            trainable_indices=param_manager.get_control_trainable_indices(),
        )
        surrogate.fit_distribution_reference(ref_context)

    tensorboard_enabled = bool(train_cfg.get("tensorboard", True))

    save_root = base_dir / "saved_models"
    save_root.mkdir(parents=True, exist_ok=True)
    run_dir = save_root / f"strategy_net_{datetime.now().strftime('%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    run_norm_snapshot_path = run_dir / "normalization_config.json"
    shutil.copy2(str(NORMALIZATION_CONFIG_PATH), str(run_norm_snapshot_path))
    writer = SummaryWriter(log_dir=str(run_dir)) if tensorboard_enabled else None

    train_best_path = run_dir / "train_best_model.pth"
    val_best_path = run_dir / "val_best_model.pth"
    final_path = run_dir / "final_model.pth"
    history_rows = []
    train_best_loss = float("inf")
    val_best_loss = float("inf")
    train_best_iter = -1
    val_best_iter = -1
    ema_train_loss: Optional[float] = None

    def evaluate_full_val() -> tuple[float, float, float, float]:
        strategy_net.eval()
        total_loss = 0.0
        total_risk = 0.0
        total_constraint = 0.0
        total_distribution = 0.0
        sample_count = 0
        with torch.no_grad():
            for batch_context in val_sampler.iter_dataset_batches(batch_size=int(train_cfg.get("val_batch_size", 1024)), shuffle=False):
                pulse_norm = surrogate.generate_pulse(batch_context)
                actions = strategy_net(batch_context, pulse_norm)
                loss_batch, _, info = surrogate.predict_injury_and_loss(batch_context, actions, pulse_norm)
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
    val_batch_size = int(train_cfg.get("val_batch_size", 1024))
    grad_clip = float(train_cfg.get("gradient_clip_max_norm", 1.0))
    save_best = bool(train_cfg.get("save_best", True))
    save_last = bool(train_cfg.get("save_last", True))
    train_metric_name = "ema_train_loss" if ema_enabled else "train_loss"

    strategy_net.train()
    try:
        for iter_idx in tqdm(range(train_cfg["max_iterations"]), desc="Training StrategyNet"):
            optimizer.zero_grad()
            context_params = next(train_generator)
            with torch.no_grad():
                pulse_norm = surrogate.generate_pulse(context_params)
            actions = strategy_net(context_params, pulse_norm)
            total_loss, _, info = surrogate.predict_injury_and_loss(context_params, actions, pulse_norm)
            loss_mean = total_loss.mean()
            if torch.isnan(loss_mean) or torch.isinf(loss_mean):
                logger.warning(f"iter={iter_idx}: loss 出现 NaN/Inf，跳过本次更新")
                optimizer.zero_grad(set_to_none=True)
                continue

            loss_mean.backward()
            torch.nn.utils.clip_grad_norm_(strategy_net.parameters(), max_norm=grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            loss_value = float(loss_mean.item())
            loss_risk = float(info["loss_risk"].mean().item())
            loss_constraint = float(info["loss_constraint"].mean().item())
            loss_distribution = float(info["loss_distribution"].mean().item())
            current_lr = float(optimizer.param_groups[0]["lr"])

            ema_train_loss = loss_value if ema_train_loss is None else ema_alpha * ema_train_loss + (1.0 - ema_alpha) * loss_value
            train_select_metric = ema_train_loss if ema_enabled else loss_value
            ema_ready = (iter_idx + 1) >= max(1, ema_warmup_iters) if ema_enabled else True

            if writer is not None:
                writer.add_scalar("Train/Loss", loss_value, iter_idx)
                writer.add_scalar("Train/LossRisk", loss_risk, iter_idx)
                writer.add_scalar("Train/LossConstraint", loss_constraint, iter_idx)
                writer.add_scalar("Train/LossDistribution", loss_distribution, iter_idx)
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
                    f"risk={loss_risk:.4f} penalty={loss_constraint:.4f} dist={loss_distribution:.4f} lr={current_lr:.2e}"
                )

        if save_last:
            torch.save(strategy_net.state_dict(), final_path)

        with open(run_dir / "training_history.csv", "w", newline="", encoding="utf-8") as file:
            writer_csv = csv.DictWriter(file, fieldnames=list(history_rows[0].keys()) if history_rows else ["iteration"])
            writer_csv.writeheader()
            writer_csv.writerows(history_rows)

        with open(run_dir / "training_summary.yaml", "w", encoding="utf-8") as file:
            yaml.safe_dump(
                {
                    "max_iterations": train_cfg["max_iterations"],
                    "val_interval": val_interval,
                    "val_batch_size": val_batch_size,
                    "ema": {
                        "enabled": ema_enabled,
                        "alpha": ema_alpha,
                        "warmup_iters": ema_warmup_iters,
                        "tensorboard": tensorboard_enabled,
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
                    },
                    "parameter_roles": {
                        "context": param_manager.get_context_names(),
                        "trainable_control": [param["name"] for param in param_manager.control_trainable_params],
                        "fixed_control": [param["name"] for param in param_manager.control_fixed_params],
                    },
                    "external_artifacts": {
                        "normalization_config": run_norm_snapshot_path.name,
                        "pulse_checkpoint": config.get("surrogate", {}).get("pulse_checkpoint"),
                        "injury_checkpoint": config.get("surrogate", {}).get("checkpoint_rel_path"),
                    },
                    "train_best": {
                        "iter": train_best_iter,
                        "loss": None if train_best_iter < 0 else float(train_best_loss),
                        "metric": train_metric_name,
                        "ckpt": train_best_path.name,
                    },
                    "val_best": {
                        "iter": val_best_iter,
                        "loss": None if val_best_iter < 0 else float(val_best_loss),
                        "ckpt": val_best_path.name,
                    },
                    "final_model": {
                        "iter": train_cfg["max_iterations"],
                        "ckpt": final_path.name,
                    },
                },
                file,
                allow_unicode=True,
                sort_keys=False,
            )

        with open(run_dir / "config_used.yaml", "w", encoding="utf-8") as file:
            yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)
        shutil.copy2(str(param_space_path), str(run_dir / "param_space.yaml"))

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
