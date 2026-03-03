import os
import yaml
import torch
import pandas as pd
import logging
import argparse
from pathlib import Path

from common.data_utils.processor import UnifiedDataProcessor
from common.settings import FEATURE_ORDER, NORMALIZATION_CONFIG_PATH, PULSE_PREDICT_DIR, INJURY_PREDICT_DIR
from PulsePredict.model.model import HybridPulseCNN
from InjuryPredict.utils.models import InjuryPredictModel

from ARS_optim.src.core.param_manager import ParamManager
from ARS_optim.src.core.constraints import PhysicalConstraintManager
from ARS_optim.src.interface.surrogate_adapter import SurrogateAdapter
from ARS_optim.src.models.strategy_net import StrategyNet
from ARS_optim.src.core.optimizer import ARSLocalOptimizer


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
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    return logging.getLogger("Evaluator")

def parse_args():
    parser = argparse.ArgumentParser(description="ARS Local Refinement Evaluator")
    parser.add_argument('--input_csv', type=str, required=True, help="输入的本地工况参数CSV文件路径")
    parser.add_argument('--output_csv', type=str, default='evaluation_results.csv', help="输出的对比结果CSV文件路径")
    parser.add_argument('--strategy_ckpt', type=str, default=None,
                        help="可选：策略网络权重文件路径，若指定则加载并启用直推模式")
    parser.add_argument('--direct_inference', action='store_true',
                        help="启用策略网络直推（忽略 config 中的相应设置）")
    return parser.parse_args()

def main():
    args = parse_args()
    logger = setup_logger()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(base_dir, 'configs', 'default_config.yaml')
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"config file not found: {cfg_path}")
    with open(cfg_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # basic validation
    if 'surrogate' not in config:
        raise KeyError("配置文件中缺失 'surrogate' 部分，请检查配置。")
    if 'optimization' not in config:
        logger.warning("配置文件中缺失 'optimization' 部分，将使用默认设置。")
    if 'evaluation' not in config:
        logger.warning("配置文件中缺失 'evaluation' 部分，将使用默认设置。")

    eval_cfg = config.get('evaluation', {})
    device = torch.device(eval_cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

    param_space_path = os.path.join(base_dir, 'configs', 'param_space.yaml')
    param_manager = ParamManager(param_space_path)
    constraint_manager = PhysicalConstraintManager(param_manager)
    data_processor = UnifiedDataProcessor(str(NORMALIZATION_CONFIG_PATH))

    # 加载代理模型权重路径
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
    
    surrogate = SurrogateAdapter(
        pulse_model=pulse_model, injury_model=injury_model, 
        param_manager=param_manager, config=config, data_processor=data_processor
    ).to(device)

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

    # 允许通过命令行覆盖 config 里的设置
    if args.strategy_ckpt:
        if not os.path.isfile(args.strategy_ckpt):
            raise FileNotFoundError(f"指定的策略网络权重不存在: {args.strategy_ckpt}")
        strategy_net.load_state_dict(torch.load(args.strategy_ckpt, map_location=device))
        logger.info(f"已加载策略网络权重: {args.strategy_ckpt}")
        config['optimization']['direct_inference'] = True
    if args.direct_inference:
        logger.info("命令行指定启用直接推理模式 (direct_inference)")
        config['optimization']['direct_inference'] = True

    # 实例化核心寻优引擎
    optimizer = ARSLocalOptimizer(
        config=config, param_manager=param_manager, constraint_manager=constraint_manager,
        surrogate=surrogate, strategy_net=strategy_net
    )

    # 1. 解析输入 CSV
    logger.info(f"读取输入工况文件: {args.input_csv}")
    df_input = pd.read_csv(args.input_csv)
    
    # 组装上下文特征（state + fixed-control）并转换为物理张量
    state_names = [p['name'] for p in param_manager.state_params]
    fixed_params = param_manager.control_fixed_params
    fixed_names = [p['name'] for p in fixed_params]

    for col in state_names:
        if col not in df_input.columns:
            raise ValueError(f"输入 CSV 缺失必填状态列: {col}")

    context_df = df_input[state_names].copy()
    missing_fixed_cols = []
    for p in fixed_params:
        name = p['name']
        if name in df_input.columns:
            context_df[name] = df_input[name].values
        else:
            context_df[name] = float(p['default'])
            missing_fixed_cols.append(name)
    if missing_fixed_cols:
        logger.warning(f"输入 CSV 缺失固定控制参数列，已回退 default: {missing_fixed_cols}")

    context_names = param_manager.get_context_names()
    context_tensor = torch.tensor(context_df[context_names].values, dtype=torch.float32, device=device)

    # 2. 算力复用执行推理
    logger.info("开始执行基线推断与联合局部精调...")
    
    # 构造仅包含可训练参数的 baseline 动作向量（SurrogateAdapter 只看 trainable 部分）
    trainable_defaults = [p['default'] for p in param_manager.control_trainable_params]
    baseline_trainable = torch.tensor(trainable_defaults, dtype=torch.float32, device=device)
    baseline_trainable = baseline_trainable.unsqueeze(0).expand(context_tensor.shape[0], -1)

    # 同时准备完整的“基线控制参数”用于结果输出（可调 + 固定）
    fixed_idxs, fixed_defaults = param_manager.get_control_fixed_defaults(device=device)
    if fixed_defaults.numel() > 0:
        full_baseline = torch.cat([baseline_trainable, fixed_defaults.unsqueeze(0).expand(context_tensor.shape[0], -1)], dim=1)
    else:
        full_baseline = baseline_trainable

    with torch.no_grad():
        pulse_norm = surrogate.generate_pulse(context_tensor)
        # 评测 Baseline 损伤值，同时记录损失信息
        baseline_loss_batch, baseline_preds, baseline_info = surrogate.predict_injury_and_loss(context_tensor, baseline_trainable, pulse_norm)

    # 执行 StrategyNet 零次推断 + 局部反向微调
    optimized_actions, optimized_preds, opt_info = optimizer.optimize(context_tensor)

    # 日志一些简要指标
    logger.info(f"优化耗时: {opt_info.get('time_cost', float('nan')):.6f} s")
    if 'initial' in opt_info:
        logger.info(f"初始平均损伤: {opt_info['initial'].get('loss_mean', float('nan')):.4f}")
    if baseline_loss_batch is not None:
        mean_base = baseline_loss_batch.mean().item()
        logger.info(f"基线平均损伤: {mean_base:.4f}")

    # 3. 结果合并与输出
    # 基线/优化完整控制参数（含固定）
    control_names = [p['name'] for p in param_manager.control_trainable_params]
    all_control_names = control_names + fixed_names
    df_base_ctrl = pd.DataFrame(full_baseline.cpu().numpy(), columns=[f"Base_{n}" for n in all_control_names])
    if fixed_defaults.numel() > 0:
        full_optimized = torch.cat([optimized_actions, fixed_defaults.unsqueeze(0).expand(context_tensor.shape[0], -1)], dim=1)
    else:
        full_optimized = optimized_actions
    df_opt_actions = pd.DataFrame(full_optimized.cpu().numpy(), columns=[f"Opt_{n}" for n in all_control_names])

    # 拼装损伤对比
    df_baseline_inj = pd.DataFrame(baseline_preds.cpu().numpy(), columns=['Base_HIC', 'Base_Dmax', 'Base_Nij'])
    df_opt_inj = pd.DataFrame(optimized_preds.cpu().numpy(), columns=['Opt_HIC', 'Opt_Dmax', 'Opt_Nij'])
    # add loss columns if available
    df_base_loss = pd.DataFrame(baseline_loss_batch.cpu().numpy(), columns=['Base_Loss'])
    opt_loss_batch = opt_info.get('final_loss_batch')
    if opt_loss_batch is not None:
        df_opt_loss = pd.DataFrame(opt_loss_batch.cpu().numpy(), columns=['Opt_Loss'])
    else:
        df_opt_loss = pd.DataFrame([], columns=['Opt_Loss'])

    df_final = pd.concat([df_input, df_base_ctrl, df_opt_actions, df_baseline_inj, df_opt_inj, df_base_loss, df_opt_loss], axis=1)
    df_final.to_csv(args.output_csv, index=False)

    logger.info(f"评估完成！对比结果已保存至: {args.output_csv}")

if __name__ == "__main__":
    main()