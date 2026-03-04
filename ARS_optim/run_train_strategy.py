import os
import yaml
import torch
import torch.optim as optim
import argparse
import shutil
import hashlib
from datetime import datetime
import csv
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from common.data_utils.processor import UnifiedDataProcessor
from common.settings import NORMALIZATION_CONFIG_PATH
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


def main():
    logger = setup_ars_logger(name="TrainStrategy")
    logger.info("初始化自监督摊销优化管线 (Amortized Optimization Pipeline)...")

    # 1) 命令行解析与配置加载
    # 说明：命令行参数优先级高于 YAML，用于快速做实验对比（例如临时调整 batch_size、lr）。
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
        logger.warning("配置中未包含 'optimization' 部分，使用默认值。")

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

    # 兼容旧配置：历史版本可能把 device 写在 strategy_net.train 下
    legacy_train_device = train_cfg.get('device', None)
    if 'device' not in config and legacy_train_device is not None:
        logger.warning("检测到旧配置 strategy_net.train.device；已迁移为顶层 device 真源。")
        config['device'] = legacy_train_device
    elif 'device' in config and legacy_train_device is not None and str(config['device']) != str(legacy_train_device):
        logger.warning("检测到 strategy_net.train.device 与顶层 device 不一致；将使用顶层 device 作为唯一真源。")

    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    logger.info(f"计算设备挂载: {device}")

    # 统一设置随机种子，保证模型初始化、Dropout、数据扰动过程可复现
    seed = int(config.get('seed', 42))
    set_random_seed(seed)
    logger.info(f"全局随机种子已设置: {seed}")

    # 数值型超参转成明确类型，减少 YAML 字面量差异带来的隐式类型问题
    train_cfg['batch_size'] = int(train_cfg.get('batch_size', 0))
    train_cfg['max_iterations'] = int(train_cfg.get('max_iterations', 0))
    train_cfg['lr'] = float(train_cfg.get('lr', 0.0))
    train_cfg['weight_decay'] = float(train_cfg.get('weight_decay', 0.0))

    if train_cfg.get('batch_size', 0) <= 0:
        raise ValueError("batch_size 必须为正整数")
    if train_cfg.get('max_iterations', 0) <= 0:
        raise ValueError("max_iterations 必须为正整数")
    if train_cfg.get('lr', 0.0) <= 0.0:
        raise ValueError("lr 必须为正数")
    if train_cfg.get('weight_decay', 0.0) < 0.0:
        raise ValueError("weight_decay 必须为非负数")

    # 2) 核心组件实例化
    # ParamManager 管理“参数顺序/边界/可调属性”，是后续张量列切片的一致性来源。
    param_space_path = os.path.join(base_dir, 'configs', 'param_space.yaml')
    param_manager = ParamManager(param_space_path)
    constraint_manager = PhysicalConstraintManager(param_manager)
    
    # 实例化数据归一化处理器 (依赖根目录全局配置)
    data_processor = UnifiedDataProcessor(str(NORMALIZATION_CONFIG_PATH))

    # 实例化并加载波形与损伤代理模型权重
    pulse_model, injury_model = load_surrogate_models(config=config, device=device)

    # 构建物理环境代理器
    surrogate = SurrogateAdapter(
        pulse_model=pulse_model, 
        injury_model=injury_model, 
        param_manager=param_manager, 
        config=config, 
        data_processor=data_processor
    ).to(device)

    # 构建策略网络与优化器
    # 约定：策略网络输入为 context + pulse 两路特征，输出仅包含 trainable control 参数。
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

    # 学习率调度器（可选）
    # 例：cosine 调度下，学习率会从初值平滑下降到 eta_min。
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

    # 构建训练数据流（损伤预测训练集 + 扰动）
    # 每次 next(context_generator) 返回一批 context 参数，形状 [Batch, D_context]。
    data_loader_manager = StateDataLoaderManager(
        param_manager=param_manager, 
        batch_size=int(train_cfg.get('batch_size')), 
        device=device,
        seed=int(config.get('seed', 42)),
        jitter_ratio=float(train_cfg.get('jitter_ratio', 0.01)),
        jitter_prob=float(train_cfg.get('jitter_prob', 1.0))
    )
    context_generator = data_loader_manager.get_infinite_generator()

    # 构建验证集迭代器（损伤预测验证集，不加扰动）
    from common.settings import SPLIT_INDICES_DIR
    val_indices_path = SPLIT_INDICES_DIR / 'injury_val_indices.npy'
    val_loader_manager = StateDataLoaderManager(
        param_manager=param_manager,
        batch_size=int(train_cfg.get('val_batch_size', 1024)),
        device=device,
        seed=int(config.get('seed', 42)),
        train_indices_path=str(val_indices_path),
        jitter_ratio=0.0,
        jitter_prob=0.0,
    )

    # 3) 摊销训练主循环
    # 训练目标：最小化代理模型给出的总损失（风险项 + 约束惩罚项）。
    logger.info(f"开始自监督训练，最大迭代次数: {max_iters}, batch_size={train_cfg.get('batch_size')}, lr={train_cfg.get('lr')}, weight_decay={train_cfg.get('weight_decay')}")
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
    val_best_loss = float('inf')
    val_best_iter = -1

    # 训练日志记录：逐迭代写入内存，训练结束写 CSV
    history_rows = []

    def evaluate_full_val() -> float:
        """在完整验证集上评估 val_loss（不更新参数）。"""
        strategy_net.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for val_context in val_loader_manager.iter_dataset_batches(batch_size=val_batch_size, shuffle=False):
                pulse_norm_val = surrogate.generate_pulse(val_context)          # [Bv, 2, Seq_Len]
                val_actions = strategy_net(val_context, pulse_norm_val)         # [Bv, D_trainable]
                val_loss_batch, _, _ = surrogate.predict_injury_and_loss(val_context, val_actions, pulse_norm_val)  # [Bv]
                total += float(val_loss_batch.sum().item())
                count += int(val_loss_batch.numel())
        strategy_net.train()
        if count <= 0:
            raise ValueError("验证集样本数为0，无法计算 val_loss。")
        return total / count

    strategy_net.train()
    
    pbar = tqdm(range(max_iters), desc="Training StrategyNet")
    try:
        for iter_idx in pbar:
            optimizer.zero_grad()

            # Step A: 从经验池采样上下文参数
            # [Batch, D_context]
            context_params = next(context_generator) # [Batch, D_context]
        
            # Step B: 根据 context 生成碰撞波形特征
            # 这里用 no_grad 是因为 pulse_model 在策略训练中作为冻结环境，不参与更新。
            # [Batch, D_context] -> [Batch, 2, Seq_Len]
            with torch.no_grad():
                pulse_norm = surrogate.generate_pulse(context_params) # [Batch, 2, Seq_Len]
            
            # Step C: 策略网络前向，输出可调控制参数
            # [Batch, D_context] + [Batch, 2, Seq_Len] -> [Batch, D_trainable]
            actions = strategy_net(context_params, pulse_norm) # [Batch, D_trainable]
        
            # Step D: 计算总损失
            # total_loss 是逐样本向量，例如 Batch=3 时可能是 [0.31, 0.27, 0.29]。
            total_loss, _, info = surrogate.predict_injury_and_loss(context_params, actions, pulse_norm)
        
            loss_mean = total_loss.mean()

            # 防御性保护：若当前批次损失异常（NaN/Inf），跳过本次更新
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
            current_lr = float(optimizer.param_groups[0]['lr'])

            writer.add_scalar("Train/Loss", loss_value, iter_idx)
            writer.add_scalar("Train/LossRisk", loss_risk_value, iter_idx)
            writer.add_scalar("Train/LossPenalty", loss_penalty_value, iter_idx)
            writer.add_scalar("Train/LR", current_lr, iter_idx)

            current_val_loss = None
            # 周期性验证：每隔 val_interval 在完整验证集上计算一次均值损失
            if val_interval > 0 and ((iter_idx + 1) % val_interval == 0):
                current_val_loss = evaluate_full_val()
                writer.add_scalar("Val/Loss", current_val_loss, iter_idx)

                if current_val_loss < val_best_loss:
                    val_best_loss = current_val_loss
                    val_best_iter = iter_idx + 1
                    torch.save(strategy_net.state_dict(), val_best_path)

            if save_best and loss_value < train_best_loss:
                train_best_loss = loss_value
                train_best_iter = iter_idx + 1
                torch.save(strategy_net.state_dict(), train_best_path)

            history_rows.append({
                'iteration': int(iter_idx + 1),
                'train_loss': float(loss_value),
                'train_loss_risk': float(loss_risk_value),
                'train_loss_constraint': float(loss_penalty_value),
                'val_loss': float(current_val_loss) if current_val_loss is not None else None,
                'lr': float(current_lr),
            })

            if iter_idx % max(1, log_interval) == 0:
                pbar.set_postfix({
                    "Loss": f"{loss_value:.4f}", 
                    "Risk": f"{loss_risk_value:.4f}",
                    "Penalty": f"{loss_penalty_value:.4f}",
                    "LR": f"{current_lr:.2e}"
                })

        # 4) 保存模型与训练记录
        if save_last:
            torch.save(strategy_net.state_dict(), final_path)

        # 训练历史 CSV 记录
        history_csv_path = os.path.join(save_dir, 'training_history.csv')
        with open(history_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer_csv = csv.DictWriter(f, fieldnames=['iteration', 'train_loss', 'train_loss_risk', 'train_loss_constraint', 'val_loss', 'lr'])
            writer_csv.writeheader()
            writer_csv.writerows(history_rows)

        # 训练摘要 YAML
        summary_path = os.path.join(save_dir, 'training_summary.yaml')
        with open(summary_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(
                {
                    'max_iterations': max_iters,
                    'val_interval': val_interval,
                    'val_batch_size': val_batch_size,
                    'train_best': {
                        'iter': int(train_best_iter),
                        'loss': float(train_best_loss),
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

        # 额外保存配置与关键文件哈希
        # 用途：评估阶段可以检查“当前环境是否与训练时一致”（尤其是归一化配置）。
        try:
            cfg_path = os.path.join(save_dir, 'config_used.yaml')
            with open(cfg_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(config, f)
            shutil.copy(param_space_path, os.path.join(save_dir, 'param_space.yaml'))
            with open(param_space_path, 'rb') as f:
                param_space_sha = hashlib.sha1(f.read()).hexdigest()
            with open(str(NORMALIZATION_CONFIG_PATH), 'rb') as f:
                normalization_sha = hashlib.sha1(f.read()).hexdigest()

            # 保存关键配置指纹（示例文件：sha_checksums.yaml）
            # 内容示例：
            # param_space_sha1: <40位hex>
            # normalization_config_sha1: <40位hex>
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
                logger.info(f"训练集最优权重: {train_best_path} (iter={train_best_iter}, train_loss={train_best_loss:.6f})")
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