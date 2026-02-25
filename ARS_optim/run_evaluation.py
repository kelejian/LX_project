import os
import yaml
import torch
import pandas as pd
import logging
from tqdm import tqdm
from pathlib import Path

# 引入项目全局依赖
from common.settings import NORMALIZATION_CONFIG_PATH
from common.data_utils.processor import UnifiedDataProcessor

# 引入子项目核心模块
from ARS_optim.src.core.param_manager import ParamManager
from ARS_optim.src.core.constraints import PhysicalConstraintManager
from ARS_optim.src.core.optimizer import ARSLocalOptimizer
from ARS_optim.src.interface.surrogate_adapter import SurrogateAdapter
from ARS_optim.src.models.strategy_net import StrategyNet

# TODO: [需确认] 替换为您实际的模型类
from PulsePredict.model.model import HybridPulseCNN 
from InjuryPredict.utils.models import InjuryPredictModel  

def setup_logger():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    return logging.getLogger("EvaluateARS")

def main():
    logger = setup_logger()
    logger.info("="*60)
    logger.info("启动 ARS_optim 在线评估流水线 (Evaluation Pipeline)")
    logger.info("="*60)

    # 1. 路径与输入文件配置
    # TODO: [假设] 用户提供需要被评估的工况数据 (仅含状态参数)
    input_csv_path = "ARS_optim/input_cases.csv" 
    output_csv_path = "ARS_optim/output_results.csv"
    
    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"[致命错误] 找不到输入数据文件: {input_csv_path}")

    config_path = Path("ARS_optim/configs/default_config.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    device = torch.device(config.get("device", "cpu"))
    
    # 2. 初始化底层基建
    param_manager = ParamManager(
        param_space_path="ARS_optim/configs/param_space.yaml",
        norm_config_path=NORMALIZATION_CONFIG_PATH
    )
    constraint_manager = PhysicalConstraintManager(param_manager)
    processor = UnifiedDataProcessor(config_path=NORMALIZATION_CONFIG_PATH)
    processor.load_config()

    # 3. 加载代理模型环境
    # TODO: [需确认] 请加入您真实的权重加载逻辑
    pulse_model = HybridPulseCNN().to(device) 
    injury_model = InjuryPredictModel(num_classes_of_discrete=[2, 3]).to(device)
    
    surrogate_env = SurrogateAdapter(
        pulse_model=pulse_model, injury_model=injury_model,
        param_manager=param_manager, config=config, data_processor=processor
    ).to(device)

    # 4. 根据 direct_inference 加载策略网络
    is_direct_infer = config['optimization'].get('direct_inference', False)
    strategy_net = None
    if is_direct_infer:
        logger.info("[模式] 开启 StrategyNet 直推 + 局部精调")
        strategy_net = StrategyNet(
            param_manager=param_manager, constraint_manager=constraint_manager,
            hidden_dims=config['strategy_net']['hidden_dims']
        ).to(device)
        # TODO: 加载训练好的权重
        # weight_path = "ARS_optim/saved_models/strategy_net_best.pth"
        # strategy_net.load_state_dict(torch.load(weight_path, map_location=device)['state_dict'])
        strategy_net.eval()
    else:
        logger.info("[模式] 关闭 StrategyNet，仅使用 Default 起点执行局部精调")

    # 初始化寻优引擎
    optimizer = ARSLocalOptimizer(
        config=config, param_manager=param_manager, 
        constraint_manager=constraint_manager, 
        surrogate=surrogate_env, strategy_net=strategy_net
    )

    # 5. 读取数据并校验列名
    df_input = pd.read_csv(input_csv_path)
    state_names = [p['name'] for p in param_manager.state_params]
    missing_cols = [col for col in state_names if col not in df_input.columns]
    if missing_cols:
        raise ValueError(f"输入 CSV 缺失必需的状态参数列: {missing_cols}")

    # 将工况参数转换为 Tensor
    state_tensor = torch.tensor(df_input[state_names].values, dtype=torch.float32, device=device)
    batch_size = state_tensor.shape[0]
    
    logger.info(f"成功读取待优化工况: {batch_size} 条")

    # ==========================================
    # 6. 计算 Baseline (优化前：输入Default动作)
    # ==========================================
    logger.info(">>> 正在计算 Baseline (未经优化的默认损伤)...")
    trainable_defaults = [p['default'] for p in param_manager.control_trainable_params]
    base_actions = torch.tensor(trainable_defaults, dtype=torch.float32, device=device).unsqueeze(0).expand(batch_size, -1)
    
    with torch.no_grad():
        base_loss, base_preds_phys, base_info = surrogate_env(state_tensor, base_actions)

    # ==========================================
    # 7. 计算 Optimized (执行在线寻优)
    # ==========================================
    logger.info(f">>> 正在执行 ARS 在线寻优 (精调步数={optimizer.refine_steps})...")
    opt_actions, opt_preds_phys, opt_info = optimizer.optimize(state_tensor)

    # ==========================================
    # 8. 组装结果并保存
    # ==========================================
    trainable_names = [p['name'] for p in param_manager.control_trainable_params]
    
    # 提取评估数据并转为 numpy
    base_act_np = base_actions.cpu().numpy()
    opt_act_np = opt_actions.cpu().numpy()
    base_inj_np = base_preds_phys.cpu().numpy()
    opt_inj_np = opt_preds_phys.cpu().numpy()
    
    # 构建对比结果字典
    results_dict = {}
    # 1. 填入环境状态
    for i, name in enumerate(state_names):
        results_dict[name] = df_input[name].values
        
    # 2. 填入动作对比
    for i, name in enumerate(trainable_names):
        results_dict[f"Baseline_{name}"] = base_act_np[:, i]
        results_dict[f"Optimized_{name}"] = opt_act_np[:, i]
        
    # 3. 填入损伤对比 (0:HIC, 1:Dmax, 2:Nij)
    results_dict["Baseline_HIC15"] = base_inj_np[:, 0]
    results_dict["Optimized_HIC15"] = opt_inj_np[:, 0]
    results_dict["Baseline_Dmax"] = base_inj_np[:, 1]
    results_dict["Optimized_Dmax"] = opt_inj_np[:, 1]
    results_dict["Baseline_Nij"] = base_inj_np[:, 2]
    results_dict["Optimized_Nij"] = opt_inj_np[:, 2]
    
    # 计算综合风险下降率
    base_risk = base_info['loss_risk'].cpu().numpy()
    opt_risk = opt_info['loss_risk'].cpu().numpy()
    results_dict["Baseline_TotalRisk"] = base_risk
    results_dict["Optimized_TotalRisk"] = opt_risk
    results_dict["Risk_Reduction(%)"] = (base_risk - opt_risk) / (base_risk + 1e-8) * 100.0

    df_output = pd.DataFrame(results_dict)
    df_output.to_csv(output_csv_path, index=False)
    
    logger.info("="*60)
    logger.info(f"评估完成！结果已保存至: {output_csv_path}")
    logger.info(f"平均综合风险: 优化前={base_risk.mean():.4f} -> 优化后={opt_risk.mean():.4f}")
    logger.info("="*60)

if __name__ == "__main__":
    main()