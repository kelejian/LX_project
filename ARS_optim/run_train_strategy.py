import os
import yaml
import torch
import logging
from tqdm import tqdm
from pathlib import Path

# 引入项目全局依赖
from common.settings import INJURY_PREDICT_DIR, NORMALIZATION_CONFIG_PATH
from common.data_utils.processor import UnifiedDataProcessor

# 引入子项目核心模块
from ARS_optim.src.core.param_manager import ParamManager
from ARS_optim.src.core.constraints import PhysicalConstraintManager
from ARS_optim.src.interface.data_loader import StateDataLoaderManager
from ARS_optim.src.interface.surrogate_adapter import SurrogateAdapter
from ARS_optim.src.models.strategy_net import StrategyNet

# TODO: [需确认] 根据您实际的项目路径引入模型架构定义，以下为假设的引用路径
from PulsePredict.model.model import HybridPulseCNN  # 假设波形模型类名
from InjuryPredict.utils.models import InjuryPredictModel  # 假设损伤模型类名

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger("TrainStrategy")

def load_surrogate_models(device: torch.device, logger: logging.Logger):
    """
    加载级联代理模型 (PulsePredict + InjuryPredict)。
    WARNING: [假设] 这里强假设了模型初始化的参数，实际工程中应从它们各自的 config.json 中读取。
    """
    logger.info("正在加载级联代理模型权重...")
    
    # 1. 加载波形模型
    # TODO: [需确认] 替换为您实际的波形模型初始化参数与权重路径
    pulse_model = HybridPulseCNN().to(device) 
    # pulse_checkpoint = torch.load("PulsePredict/saved/models/.../model_best.pth", map_location=device)
    # pulse_model.load_state_dict(pulse_checkpoint['state_dict'])
    pulse_model.eval()

    # 2. 加载损伤模型
    # TODO: [需确认] 替换为您实际的损伤模型初始化参数与权重路径
    injury_model = InjuryPredictModel(num_classes_of_discrete=[2, 3]).to(device)
    # injury_checkpoint = torch.load("InjuryPredict/runs/.../best_val_loss.pth", map_location=device)
    # injury_model.load_state_dict(injury_checkpoint['state_dict'])
    injury_model.eval()
    
    return pulse_model, injury_model

def main():
    logger = setup_logger()
    logger.info("启动策略网络摊销优化 (Amortized Optimization) 训练流...")

    # 1. 解析配置
    config_path = Path("ARS_optim/configs/default_config.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    device = torch.device(config.get("device", "cpu"))
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    
    # 2. 初始化底层基建
    param_manager = ParamManager(
        param_space_path="ARS_optim/configs/param_space.yaml",
        norm_config_path=NORMALIZATION_CONFIG_PATH
    )
    constraint_manager = PhysicalConstraintManager(param_manager)
    
    processor = UnifiedDataProcessor(config_path=NORMALIZATION_CONFIG_PATH)
    processor.load_config()

    # 3. 加载环境代理
    pulse_model, injury_model = load_surrogate_models(device, logger)
    surrogate_env = SurrogateAdapter(
        pulse_model=pulse_model,
        injury_model=injury_model,
        param_manager=param_manager,
        config=config,
        data_processor=processor
    ).to(device)

    # 4. 构建数据流与智能体
    train_cfg = config['strategy_net']['train']
    data_manager = StateDataLoaderManager(
        param_manager=param_manager,
        batch_size=train_cfg['batch_size'],
        device=device
    )
    state_generator = data_manager.get_infinite_generator()
    
    strategy_net = StrategyNet(
        param_manager=param_manager,
        constraint_manager=constraint_manager,
        hidden_dims=config['strategy_net']['hidden_dims'],
        activation=config['strategy_net']['activation'],
        dropout=config['strategy_net']['dropout']
    ).to(device)
    
    # 仅优化 StrategyNet 的权重
    optimizer = torch.optim.Adam(
        strategy_net.parameters(), 
        lr=train_cfg['lr'], 
        weight_decay=float(train_cfg['weight_decay'])
    )

    # 5. 执行无限流训练循环
    max_iters = train_cfg['max_iterations']
    logger.info(f"开始训练，最大迭代步数: {max_iters}")
    
    best_loss = float('inf')
    save_dir = Path("ARS_optim/saved_models")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    strategy_net.train()
    
    pbar = tqdm(range(max_iters), desc="Amortized Training")
    for step in pbar:
        optimizer.zero_grad()
        
        # 采样子监督环境状态
        # state_params: [Batch, D_state]
        state_params = next(state_generator)
        
        # 策略网络推断动作
        # actions: [Batch, D_trainable]
        actions = strategy_net(state_params)
        
        # 代理模型评估风险
        # loss_batch: [Batch]
        loss_batch, preds_phys, info = surrogate_env(state_params, actions)
        
        # 标量化并反传梯队
        loss = loss_batch.mean()
        loss.backward()
        
        # 梯度裁剪防止代理模型局部陡峭引发梯度爆炸
        torch.nn.utils.clip_grad_norm_(strategy_net.parameters(), max_norm=1.0)
        optimizer.step()
        
        # 记录与监控
        current_loss = loss.item()
        pbar.set_postfix({
            "Loss": f"{current_loss:.4f}", 
            "Risk": f"{info['loss_risk'].mean().item():.4f}",
            "Penalty": f"{info['loss_constraint'].mean().item():.4f}"
        })
        
        # 定期保存检查点
        if step % 1000 == 0 and step > 0:
            if current_loss < best_loss:
                best_loss = current_loss
                save_path = save_dir / "strategy_net_best.pth"
                torch.save({
                    'step': step,
                    'state_dict': strategy_net.state_dict(),
                    'loss': best_loss
                }, save_path)
                logger.info(f"Step {step}: 发现更优模型，已保存至 {save_path} (Loss: {best_loss:.4f})")

    # 最终保存
    torch.save(strategy_net.state_dict(), save_dir / "strategy_net_final.pth")
    logger.info("摊销优化训练流程结束。")

if __name__ == "__main__":
    main()