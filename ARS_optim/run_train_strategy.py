import os
import yaml
import torch
import torch.optim as optim
import argparse
import shutil
import hashlib
from datetime import datetime
import csv
from typing import Optional, Tuple
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from common.data_utils.processor import UnifiedDataProcessor
from common.settings import NORMALIZATION_CONFIG_PATH, SPLIT_INDICES_DIR
from common.utils.seeding import set_random_seed

from ARS_optim.src.core.param_manager import ParamManager
from ARS_optim.src.core.constraints import PhysicalConstraintManager
from ARS_optim.src.interface.data_loader import StateDataLoaderManager
from ARS_optim.src.interface.model_loader import load_surrogate_models
from ARS_optim.src.interface.surrogate_adapter import SurrogateAdapter
from ARS_optim.src.models.strategy_net import StrategyNet
from ARS_optim.src.utils.logger import setup_logger as setup_ars_logger

"""
训练入口脚本（python -m ARS_optim.run_train_strategy）。

本脚本职责：
1) 读取配置并实例化模型/数据流；
2) 执行“状态采样 -> 波形生成 -> 策略输出 -> 损伤评估 -> 反向更新”的自监督训练；
3) 保存权重与关键配置指纹（用于后续评估一致性检查）。

输入样例（配置）：
- strategy_net.train.batch_size: 256
- strategy_net.train.max_iterations: 50000

训练主循环中的典型张量形状：
- context_params: [Batch, D_context]
- pulse_norm: [Batch, 2, Seq_Len]
- actions: [Batch, D_trainable]
- total_loss: [Batch]
"""

def parse_args():
    parser = argparse.ArgumentParser(description="Train strategy network for ARS")
    parser.add_argument('--config', type=str, default=None,
                        help="override default config file path")
    parser.add_argument('--batch_size', type=int, help="override batch size")
    parser.add_argument('--lr', type=float, help="override learning rate")
    parser.add_argument('--weight_decay', type=float, help="override weight decay")
    parser.add_argument('--max_iterations', type=int, help="override max training iterations")
    parser.add_argument('--device', type=str, help="override device (cpu/cuda)")
    return parser.parse_args()


def _normalize_and_validate_train_config(train_cfg: dict) -> Tuple[dict, dict]:
    train_cfg['batch_size'] = int(train_cfg.get('batch_size', 0))
    train_cfg['max_iterations'] = int(train_cfg.get('max_iterations', 0))
    train_cfg['lr'] = float(train_cfg.get('lr', 0.0))
    train_cfg['weight_decay'] = float(train_cfg.get('weight_decay', 0.0))

    ema_cfg = train_cfg.get('ema', {}) or {}
    ema_enabled = bool(ema_cfg.get('enabled', True))
    ema_alpha = float(ema_cfg.get('alpha', 0.98))
    ema_warmup_iters = int(ema_cfg.get('warmup_iters', 0))
    ema_log_to_tb = bool(ema_cfg.get('log_to_tensorboard', True))

    validations = [
        (train_cfg['batch_size'] > 0, "batch_size 必须为正整数"),
        (train_cfg['max_iterations'] > 0, "max_iterations 必须为正整数"),
        (train_cfg['lr'] > 0.0, "lr 必须为正数"),
        (train_cfg['weight_decay'] >= 0.0, "weight_decay 必须为非负数"),
        (0.0 <= ema_alpha < 1.0, "EMA 配置错误：alpha 必须满足 0<=alpha<1"),
        (ema_warmup_iters >= 0, "EMA 配置错误：warmup_iters 必须为非负整数"),
    ]
    for ok, message in validations:
        if not ok:
            raise ValueError(message)

    return train_cfg, {
        'enabled': ema_enabled,
        'alpha': ema_alpha,
        'warmup_iters': ema_warmup_iters,
        'log_to_tensorboard': ema_log_to_tb,
    }


def main():
    logger = setup_ars_logger(name="TrainStrategy")
    logger.info("初始化自监督摊销优化管线 (Amortized Optimization Pipeline)...")

    args = parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path = args.config if args.config is not None else os.path.join(base_dir, 'configs', 'default_config.yaml')
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"config file not found: {cfg_path}")
    with open(cfg_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 基本配置校验：避免运行到中途才发现关键字段缺失
    if 'strategy_net' not in config or 'train' not in config['strategy_net']:
        raise KeyError("配置文件中缺失 'strategy_net.train' 部分，请检查配置。")
    if 'surrogate' not in config:
        raise KeyError("配置文件中缺失 'surrogate' 部分，请检查配置。")
    if 'optimization' not in config:
        config['optimization'] = {}

    # 统一配置来源：CLI 覆盖 YAML
    train_cfg = config['strategy_net']['train']
    # allow CLI overrides
    if args.batch_size is not None:
        train_cfg['batch_size'] = args.batch_size
    if args.lr is not None:
        train_cfg['lr'] = args.lr
    if args.weight_decay is not None:
        train_cfg['weight_decay'] = args.weight_decay
    if args.max_iterations is not None:
        train_cfg['max_iterations'] = args.max_iterations
    if args.device is not None:
        config['device'] = args.device

    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    logger.info(f"计算设备挂载: {device}")

    seed = int(config.get('seed', 42))
    set_random_seed(seed)
    logger.info(f"全局随机种子已设置: {seed}")

    train_cfg, ema_runtime = _normalize_and_validate_train_config(train_cfg)
    ema_enabled = ema_runtime['enabled']
    ema_alpha = ema_runtime['alpha']
    ema_warmup_iters = ema_runtime['warmup_iters']
    ema_log_to_tb = ema_runtime['log_to_tensorboard']

    param_space_path = os.path.join(base_dir, 'configs', 'param_space.yaml')
    param_manager = ParamManager(param_space_path)
    constraint_manager = PhysicalConstraintManager(param_manager)
    data_processor = UnifiedDataProcessor(str(NORMALIZATION_CONFIG_PATH))

    pulse_model, injury_model = load_surrogate_models(config=config, device=device)

    surrogate = SurrogateAdapter(
        pulse_model=pulse_model, 
        injury_model=injury_model, 
        param_manager=param_manager, 
        constraint_manager=constraint_manager,
        config=config, 
        data_processor=data_processor
    ).to(device)

    strat_cfg = config.get('strategy_net', {})
    strategy_net = StrategyNet(
        param_manager=param_manager,
        constraint_manager=constraint_manager,
        data_processor=data_processor,
        hidden_dims=strat_cfg.get('hidden_dims', [128, 256, 128]),
        activation=strat_cfg.get('activation', 'LeakyReLU'),
        dropout=float(strat_cfg.get('dropout', 0.0)),
        pulse_channels=int(strat_cfg.get('pulse_channels', 2)),
        pulse_embed_dim=int(strat_cfg.get('pulse_embed_dim', 32)),
    ).to(device)
    
    optimizer = optim.Adam(
        strategy_net.parameters(),
        lr=float(train_cfg.get('lr')),
        weight_decay=float(train_cfg.get('weight_decay', 0.0))
    )

    max_iters = int(train_cfg.get('max_iterations'))
    scheduler_cfg = train_cfg.get('scheduler', {})
    scheduler_type = str(scheduler_cfg.get('type', 'none')).lower()
    scheduler = None
    if scheduler_type in {'cosine', 'cosineannealinglr'}:
        t_max = int(scheduler_cfg.get('T_max', max_iters))
        eta_min = float(scheduler_cfg.get('eta_min', 1e-5))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, t_max),
            eta_min=eta_min
        )
    elif scheduler_type in {'step', 'steplr'}:
        step_size = int(scheduler_cfg.get('step_size', max(1, max_iters // 5)))
        gamma = float(scheduler_cfg.get('gamma', 0.5))
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(1, step_size),
            gamma=gamma
        )

    data_loader_manager = StateDataLoaderManager(
        param_manager=param_manager, 
        batch_size=int(train_cfg.get('batch_size')), 
        device=device,
        seed=int(config.get('seed', 42)),
        jitter_ratio=float(train_cfg.get('jitter_ratio', 0.01)),
        jitter_prob=float(train_cfg.get('jitter_prob', 1.0))
    )
    context_generator = data_loader_manager.get_infinite_generator()

    dist_cfg = config.get('optimization', {}).get('distribution_penalty', {})
    if surrogate.distribution_penalty.enabled:
        max_ref_samples = int(dist_cfg.get('max_ref_samples', 0))
        ref_context = data_loader_manager.get_distribution_reference(
            max_samples=max_ref_samples,
            shuffle=False,
            feature_space=surrogate.distribution_penalty.feature_space,
            trainable_indices=param_manager.get_control_trainable_indices(),
        )
        surrogate.fit_distribution_reference(ref_context)
        logger.info(
            f"已拟合训练分布参考统计: method={surrogate.distribution_penalty.method}, "
            f"feature_space={surrogate.distribution_penalty.feature_space}, "
            f"weight={surrogate.weight_distribution}, n_ref={ref_context.shape[0]}"
        )

    val_indices_path = SPLIT_INDICES_DIR / 'injury_val_indices.npy'
    val_loader_manager = StateDataLoaderManager(
        param_manager=param_manager,
        batch_size=int(train_cfg.get('val_batch_size', 1024)),
        device=device,
        seed=int(config.get('seed', 42)),
        split_indices_path=str(val_indices_path),
        jitter_ratio=0.0,
        jitter_prob=0.0,
    )

    logger.info(f"开始自监督训练，最大迭代次数: {max_iters}, batch_size={train_cfg.get('batch_size')}, lr={train_cfg.get('lr')}, weight_decay={train_cfg.get('weight_decay')}")
    logger.info(
        f"EMA 选优配置: enabled={ema_enabled}, alpha={ema_alpha}, "
        f"warmup_iters={ema_warmup_iters}, log_to_tensorboard={ema_log_to_tb}"
    )
    if scheduler is not None:
        logger.info(f"启用学习率调度器: {scheduler.__class__.__name__}")

    log_interval = int(train_cfg.get('log_interval', 10))
    save_best = bool(train_cfg.get('save_best', True))
    save_last = bool(train_cfg.get('save_last', True))
    grad_clip_max_norm = float(train_cfg.get('gradient_clip_max_norm', 1.0))
    val_interval = int(train_cfg.get('val_interval', 500))
    val_batch_size = int(train_cfg.get('val_batch_size', 1024))

    runs_root = os.path.join(base_dir, 'saved_models')
    os.makedirs(runs_root, exist_ok=True)
    run_id = datetime.now().strftime("%m%d_%H%M%S")
    save_dir = os.path.join(runs_root, f'strategy_net_{run_id}')
    os.makedirs(save_dir, exist_ok=False)
    writer = SummaryWriter(log_dir=save_dir)
    train_best_path = os.path.join(save_dir, 'train_best_model.pth')
    val_best_path = os.path.join(save_dir, 'val_best_model.pth')
    final_path = os.path.join(save_dir, 'final_model.pth')
    train_best_loss = float('inf')
    train_best_iter = -1
    train_best_metric_name = 'ema_train_loss' if ema_enabled else 'train_loss'
    val_best_loss = float('inf')
    val_best_iter = -1
    ema_train_loss: Optional[float] = None

    history_rows = []

    def evaluate_full_val() -> Tuple[float, float, float]:
        """在完整验证集上评估 val 指标（不更新参数）。"""
        strategy_net.eval()
        total_loss = 0.0
        total_risk = 0.0
        total_constraint = 0.0
        count = 0
        with torch.no_grad():
            for val_context in val_loader_manager.iter_dataset_batches(batch_size=val_batch_size, shuffle=False):
                pulse_norm_val = surrogate.generate_pulse(val_context)
                val_actions = strategy_net(val_context, pulse_norm_val)
                val_loss_batch, _, val_info = surrogate.predict_injury_and_loss(val_context, val_actions, pulse_norm_val)
                total_loss += float(val_loss_batch.sum().item())
                total_risk += float(val_info['loss_risk'].sum().item())
                total_constraint += float(val_info['loss_constraint'].sum().item())
                count += int(val_loss_batch.numel())
        strategy_net.train()
        if count <= 0:
            raise ValueError("验证集样本数为0，无法计算 val_loss。")
        return total_loss / count, total_risk / count, total_constraint / count

    strategy_net.train()
    
    pbar = tqdm(range(max_iters), desc="Training StrategyNet")
    try:
        for iter_idx in pbar:
            optimizer.zero_grad()

            context_params = next(context_generator)
            with torch.no_grad():
                pulse_norm = surrogate.generate_pulse(context_params)

            actions = strategy_net(context_params, pulse_norm)
            total_loss, _, info = surrogate.predict_injury_and_loss(context_params, actions, pulse_norm)

            loss_mean = total_loss.mean()

            if torch.isnan(loss_mean) or torch.isinf(loss_mean):
                logger.warning(f"iter={iter_idx}: loss 出现 NaN/Inf，已跳过本次参数更新。")
                optimizer.zero_grad(set_to_none=True)
                continue

            loss_mean.backward()
        
            # 梯度裁剪：防止个别 batch 导致梯度过大
            torch.nn.utils.clip_grad_norm_(strategy_net.parameters(), max_norm=grad_clip_max_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            loss_value = float(loss_mean.item())
            loss_risk_value = float(info['loss_risk'].mean().item())
            loss_penalty_value = float(info['loss_constraint'].mean().item())
            loss_distribution_value = float(info.get('loss_distribution', torch.zeros_like(total_loss)).mean().item())
            current_lr = float(optimizer.param_groups[0]['lr'])

            if ema_train_loss is None:
                ema_train_loss = loss_value
            else:
                ema_train_loss = ema_alpha * ema_train_loss + (1.0 - ema_alpha) * loss_value

            ema_is_warmed = (iter_idx + 1) >= max(1, ema_warmup_iters) if ema_enabled else True
            train_select_metric = ema_train_loss if ema_enabled else loss_value

            writer.add_scalar("Train/Loss", loss_value, iter_idx)
            writer.add_scalar("Train/LossRisk", loss_risk_value, iter_idx)
            writer.add_scalar("Train/LossPenalty", loss_penalty_value, iter_idx)
            writer.add_scalar("Train/LossDistribution", loss_distribution_value, iter_idx)
            writer.add_scalar("Train/LR", current_lr, iter_idx)
            if ema_log_to_tb and ema_train_loss is not None:
                writer.add_scalar("Train/EMA_Loss", float(ema_train_loss), iter_idx)

            current_val_loss = None
            current_val_loss_risk = None
            current_val_loss_constraint = None
            if val_interval > 0 and ((iter_idx + 1) % val_interval == 0):
                current_val_loss, current_val_loss_risk, current_val_loss_constraint = evaluate_full_val()
                writer.add_scalar("Val/Loss", current_val_loss, iter_idx)
                writer.add_scalar("Val/LossRisk", current_val_loss_risk, iter_idx)
                writer.add_scalar("Val/LossConstraint", current_val_loss_constraint, iter_idx)

                if current_val_loss < val_best_loss:
                    val_best_loss = current_val_loss
                    val_best_iter = iter_idx + 1
                    torch.save(strategy_net.state_dict(), val_best_path)

            if save_best and ema_is_warmed and train_select_metric < train_best_loss:
                train_best_loss = float(train_select_metric)
                train_best_iter = iter_idx + 1
                torch.save(strategy_net.state_dict(), train_best_path)

            history_rows.append({
                'iteration': int(iter_idx + 1),
                'train_loss': float(loss_value),
                'train_ema_loss': float(ema_train_loss) if ema_train_loss is not None else None,
                'train_loss_risk': float(loss_risk_value),
                'train_loss_constraint': float(loss_penalty_value),
                'train_loss_distribution': float(loss_distribution_value),
                'val_loss': float(current_val_loss) if current_val_loss is not None else None,
                'val_loss_risk': float(current_val_loss_risk) if current_val_loss_risk is not None else None,
                'val_loss_constraint': float(current_val_loss_constraint) if current_val_loss_constraint is not None else None,
                'lr': float(current_lr),
            })

            if iter_idx % max(1, log_interval) == 0:
                pbar.set_postfix({
                    "Loss": f"{loss_value:.4f}", 
                    "EMA": f"{ema_train_loss:.4f}" if ema_train_loss is not None else "nan",
                    "Risk": f"{loss_risk_value:.4f}",
                    "Penalty": f"{loss_penalty_value:.4f}",
                    "Dist": f"{loss_distribution_value:.4f}",
                    "LR": f"{current_lr:.2e}"
                })

        if save_last:
            torch.save(strategy_net.state_dict(), final_path)

        history_csv_path = os.path.join(save_dir, 'training_history.csv')
        with open(history_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer_csv = csv.DictWriter(
                f,
                fieldnames=[
                    'iteration',
                    'train_loss',
                    'train_ema_loss',
                    'train_loss_risk',
                    'train_loss_constraint',
                    'train_loss_distribution',
                    'val_loss',
                    'val_loss_risk',
                    'val_loss_constraint',
                    'lr'
                ]
            )
            writer_csv.writeheader()
            writer_csv.writerows(history_rows)

        summary_path = os.path.join(save_dir, 'training_summary.yaml')
        with open(summary_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(
                {
                    'max_iterations': max_iters,
                    'val_interval': val_interval,
                    'val_batch_size': val_batch_size,
                    'ema': {
                        'enabled': bool(ema_enabled),
                        'alpha': float(ema_alpha),
                        'warmup_iters': int(ema_warmup_iters),
                    },
                    'distribution_penalty': {
                        'enabled': bool(surrogate.distribution_penalty.enabled),
                        'method': surrogate.distribution_penalty.method,
                        'feature_space': surrogate.distribution_penalty.feature_space,
                        'weight': float(surrogate.weight_distribution),
                        'k': int(surrogate.distribution_penalty.k),
                        'eps': float(surrogate.distribution_penalty.eps),
                        'clip_max': float(surrogate.distribution_penalty.clip_max),
                        'normalize_by_train_stats': bool(surrogate.distribution_penalty.normalize_by_train_stats),
                    },
                    'train_best': {
                        'iter': int(train_best_iter),
                        'loss': float(train_best_loss),
                        'metric': train_best_metric_name,
                        'ckpt': os.path.basename(train_best_path),
                    },
                    'val_best': {
                        'iter': int(val_best_iter),
                        'loss': float(val_best_loss) if val_best_iter > 0 else None,
                        'ckpt': os.path.basename(val_best_path),
                    },
                    'final_model': {
                        'iter': int(max_iters),
                        'ckpt': os.path.basename(final_path),
                    }
                },
                f,
                allow_unicode=True,
                sort_keys=False,
            )

        try:
            cfg_path = os.path.join(save_dir, 'config_used.yaml')
            with open(cfg_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(config, f)
            shutil.copy(param_space_path, os.path.join(save_dir, 'param_space.yaml'))
            with open(param_space_path, 'rb') as f:
                param_space_sha = hashlib.sha1(f.read()).hexdigest()
            with open(str(NORMALIZATION_CONFIG_PATH), 'rb') as f:
                normalization_sha = hashlib.sha1(f.read()).hexdigest()

            sha_info_path = os.path.join(save_dir, 'sha_checksums.yaml')
            with open(sha_info_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(
                    {
                        'param_space_sha1': param_space_sha,
                        'normalization_config_sha1': normalization_sha,
                    },
                    f,
                    sort_keys=False,
                    allow_unicode=True,
                )

            if save_best and train_best_iter > 0:
                logger.info(
                    f"训练集最优权重: {train_best_path} "
                    f"(iter={train_best_iter}, {train_best_metric_name}={train_best_loss:.6f})"
                )
            if val_best_iter > 0:
                logger.info(f"验证集最优权重: {val_best_path} (iter={val_best_iter}, val_loss={val_best_loss:.6f})")
            if save_last:
                logger.info(f"最终权重: {final_path}")
            logger.info(f"训练产物目录: {save_dir}")
            logger.info(f"TensorBoard 日志目录: {save_dir}")
            logger.info(f"param_space sha1: {param_space_sha}")
            logger.info(f"normalization_config sha1: {normalization_sha}")
        except Exception as e:
            logger.warning(f"保存配置信息失败: {e}")

    finally:
        writer.close()

if __name__ == "__main__":
    main()