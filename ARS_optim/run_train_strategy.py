import os
import yaml
import torch
import torch.optim as optim
import logging
import argparse
from pathlib import Path
from tqdm import tqdm

# 全局禁用冗余引用，严格执行根目录模块化导入 (Absolute Import for CLI)
from common.data_utils.processor import UnifiedDataProcessor
from common.settings import PULSE_PREDICT_DIR, INJURY_PREDICT_DIR, NORMALIZATION_CONFIG_PATH
from PulsePredict.model.model import HybridPulseCNN
from InjuryPredict.utils.models import InjuryPredictModel

from ARS_optim.src.core.param_manager import ParamManager
from ARS_optim.src.core.constraints import PhysicalConstraintManager
from ARS_optim.src.interface.data_loader import StateDataLoaderManager
from ARS_optim.src.interface.surrogate_adapter import SurrogateAdapter
from ARS_optim.src.models.strategy_net import StrategyNet


def _resolve_checkpoint_path(base_dir: Path, cfg_value: str) -> str:
    """
    统一解析 checkpoint 路径：
    - 支持历史配置里误写的 r'...'/r"..." 文本
    - 去除前导斜杠，强制按“相对子路径”拼接到 base_dir
    """
    if cfg_value is None:
        return ""
    raw = str(cfg_value).strip()
    if (raw.startswith("r'") and raw.endswith("'")) or (raw.startswith('r"') and raw.endswith('"')):
        raw = raw[2:-1]
    raw = raw.strip().lstrip("/\\")
    return str(base_dir / Path(raw))

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger("TrainStrategy")

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
    logger = setup_logger()
    logger.info("初始化自监督摊销优化管线 (Amortized Optimization Pipeline)...")

    # 1. 命令行解析与配置文件加载
    args = parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path = args.config if args.config is not None else os.path.join(base_dir, 'configs', 'default_config.yaml')
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"config file not found: {cfg_path}")
    with open(cfg_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 校验配置中必备部分
    if 'strategy_net' not in config or 'train' not in config['strategy_net']:
        raise KeyError("配置文件中缺失 'strategy_net.train' 部分，请检查配置。")
    if 'surrogate' not in config:
        raise KeyError("配置文件中缺失 'surrogate' 部分，请检查配置。")
    if 'optimization' not in config:
        logger.warning("配置中未包含 'optimization' 部分，使用默认值。")

    # normalization of configuration structure
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
        train_cfg['device'] = args.device

    device = torch.device(train_cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    logger.info(f"计算设备挂载: {device}")

    # 参数值校验
    if train_cfg.get('batch_size', 0) <= 0:
        raise ValueError("batch_size 必须为正整数")
    if train_cfg.get('max_iterations', 0) <= 0:
        raise ValueError("max_iterations 必须为正整数")
    if train_cfg.get('lr', 0.0) <= 0.0:
        raise ValueError("lr 必须为正数")
    if train_cfg.get('weight_decay', 0.0) < 0.0:
        raise ValueError("weight_decay 必须为非负数")

    # 2. 核心组件实例化
    param_space_path = os.path.join(base_dir, 'configs', 'param_space.yaml')
    param_manager = ParamManager(param_space_path)
    constraint_manager = PhysicalConstraintManager(param_manager)
    
    # 实例化数据归一化处理器 (依赖根目录全局配置)
    data_processor = UnifiedDataProcessor(str(NORMALIZATION_CONFIG_PATH))

    # 实例化并加载波形与损伤代理模型权重
    pulse_model = HybridPulseCNN(GauNll_use=False).to(device)
    pulse_ckpt = _resolve_checkpoint_path(Path(PULSE_PREDICT_DIR), config.get('surrogate', {}).get('pulse_checkpoint', ''))
    if not os.path.isfile(pulse_ckpt):
        raise FileNotFoundError(f"pulse model checkpoint not found: {pulse_ckpt}")
    pulse_model.load_state_dict(torch.load(pulse_ckpt, map_location=device))
    
    injury_model = InjuryPredictModel(num_classes_of_discrete=[2, 3]).to(device)
    inj_ckpt = _resolve_checkpoint_path(Path(INJURY_PREDICT_DIR), config.get('surrogate', {}).get('checkpoint_rel_path', ''))
    if not os.path.isfile(inj_ckpt):
        raise FileNotFoundError(f"injury model checkpoint not found: {inj_ckpt}")
    injury_model.load_state_dict(torch.load(inj_ckpt, map_location=device))

    # 构建物理环境代理器
    surrogate = SurrogateAdapter(
        pulse_model=pulse_model, 
        injury_model=injury_model, 
        param_manager=param_manager, 
        config=config, 
        data_processor=data_processor
    ).to(device)

    # 构建策略网络与优化器
    strat_cfg = config.get('strategy_net', {})
    strategy_net = StrategyNet(
        param_manager=param_manager,
        constraint_manager=constraint_manager,
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

    # 构建高吞吐向量化数据流生成器
    data_loader_manager = StateDataLoaderManager(
        param_manager=param_manager, 
        batch_size=int(train_cfg.get('batch_size')), 
        device=device,
        seed=int(config.get('seed', 42))
    )
    state_generator = data_loader_manager.get_infinite_generator()

    # 3. 摊销训练主循环 (Amortized Training Loop)
    max_iters = int(train_cfg.get('max_iterations'))
    logger.info(f"开始自监督训练，最大迭代次数: {max_iters}, batch_size={train_cfg.get('batch_size')}, lr={train_cfg.get('lr')}, weight_decay={train_cfg.get('weight_decay')}")

    strategy_net.train()
    
    pbar = tqdm(range(max_iters), desc="Training StrategyNet")
    for iter_idx in pbar:
        optimizer.zero_grad()

        # Step A: 零拷贝在 Device 上生成物理工况态
        state_params = next(state_generator) # [Batch, D_State]
        
        # Step B: 提取环境波形特征 (前置特征，阻断梯度)
        with torch.no_grad():
            pulse_norm = surrogate.generate_pulse(state_params) # [Batch, 2, Seq_Len]
            
        # Step C: 策略网络多模态融合推断，输出合法动作
        actions = strategy_net(state_params, pulse_norm) # [Batch, D_trainable]
        
        # Step D: 计算物理损伤与惩罚损失
        total_loss, preds, info = surrogate.predict_injury_and_loss(state_params, actions, pulse_norm)
        
        loss_mean = total_loss.mean()
        loss_mean.backward()
        
        # 梯度裁剪防爆
        torch.nn.utils.clip_grad_norm_(strategy_net.parameters(), max_norm=1.0)
        optimizer.step()

        if iter_idx % 10 == 0:
            pbar.set_postfix({
                "Loss": f"{loss_mean.item():.4f}", 
                "Risk": f"{info['loss_risk'].mean().item():.4f}",
                "Penalty": f"{info['loss_constraint'].mean().item():.4f}"
            })

    # 4. 保存模型
    save_dir = os.path.join(base_dir, 'saved_models')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'strategy_net_best.pth')
    torch.save(strategy_net.state_dict(), save_path)
    # 另外保存配置和参数空间用于记录
    try:
        import hashlib, shutil
        cfg_path = os.path.join(save_dir, 'config_used.yaml')
        with open(cfg_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(config, f)
        shutil.copy(param_space_path, os.path.join(save_dir, 'param_space.yaml'))
        with open(param_space_path, 'rb') as f:
            sha = hashlib.sha1(f.read()).hexdigest()
        logger.info(f"策略网络训练完成，已固化权重至: {save_path} (param_space sha1: {sha})")
    except Exception as e:
        logger.warning(f"保存配置信息失败: {e}")

if __name__ == "__main__":
    main()