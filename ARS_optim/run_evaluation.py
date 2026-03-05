import os
import yaml
import torch
import pandas as pd
import argparse
import hashlib
import numpy as np
import re
from datetime import datetime

from common.data_utils.processor import UnifiedDataProcessor
from common.data_utils.split_io import load_int_vector_csv
from common.settings import NORMALIZATION_CONFIG_PATH, RAW_DATA_DIR, SPLIT_INDICES_DIR, FEATURE_ORDER
from common.metrics.injury_risk import AIS_cal_head, AIS_cal_chest, AIS_cal_neck

from ARS_optim.src.core.param_manager import ParamManager
from ARS_optim.src.core.constraints import PhysicalConstraintManager
from ARS_optim.src.interface.model_loader import load_surrogate_models
from ARS_optim.src.interface.surrogate_adapter import SurrogateAdapter
from ARS_optim.src.models.strategy_net import StrategyNet
from ARS_optim.src.core.optimizer import ARSLocalOptimizer
from ARS_optim.src.utils.logger import setup_logger as setup_ars_logger
from ARS_optim.src.utils.metrics import MetricsTracker

"""
评估入口脚本（python -m ARS_optim.run_evaluation）。

核心流程：
1) 读取输入 CSV，补齐缺失上下文/基线参数；
2) 计算 baseline 损伤；
3) 执行“策略直推 + 局部精调”；
4) 输出 baseline/init/optimized 三组对比结果到 CSV。

输入 CSV 最小样例：
case_id
1001
1002

若省略上下文列（如 impact_velocity、OT），会自动回退到 param_space.yaml 中的 default，
并在日志中提醒。
"""

def parse_args():
    parser = argparse.ArgumentParser(description="ARS Local Refinement Evaluator")
    parser.add_argument('--input_csv', type=str, default=None,
                        help="可选：输入工况参数CSV。若不提供，则自动使用 injury_test_indices.csv 对应的测试集工况。")
    parser.add_argument('--output_csv', type=str, default='evaluation_results.csv', help="输出的对比结果CSV文件路径")
    parser.add_argument('--strategy_ckpt', type=str, default=None,
                        help="可选：策略网络权重文件路径，若指定则加载并启用直推模式")
    parser.add_argument('--direct_inference', action='store_true',
                        help="启用策略网络直推（忽略 config 中的相应设置）")
    return parser.parse_args()


def _sanitize_name(name: str) -> str:
    """将任意字符串转换为适合目录名的安全片段。"""
    s = re.sub(r'[^0-9a-zA-Z\u4e00-\u9fa5._-]+', '_', str(name))
    s = s.strip('._-')
    return s if s else 'eval'

def main():
    args = parse_args()
    logger = setup_ars_logger(name="Evaluator")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(base_dir, 'configs', 'default_config.yaml')
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"config file not found: {cfg_path}")
    with open(cfg_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 配置完整性校验：确保 surrogate/optimization 存在
    if 'surrogate' not in config:
        raise KeyError("配置文件中缺失 'surrogate' 部分，请检查配置。")
    if 'optimization' not in config:
        logger.warning("配置文件中缺失 'optimization' 部分，将使用默认设置。")
        config['optimization'] = {}
    if 'evaluation' not in config:
        logger.warning("配置文件中缺失 'evaluation' 部分，将使用默认设置。")

    eval_cfg = config.get('evaluation', {})
    if 'device' in eval_cfg and 'device' in config and str(eval_cfg['device']) != str(config['device']):
        logger.warning("检测到 evaluation.device 与顶层 device 不一致；将使用顶层 device 作为唯一真源。")
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

    param_space_path = os.path.join(base_dir, 'configs', 'param_space.yaml')
    param_manager = ParamManager(param_space_path)
    constraint_manager = PhysicalConstraintManager(param_manager)
    data_processor = UnifiedDataProcessor(str(NORMALIZATION_CONFIG_PATH))

    # 加载代理模型权重路径
    pulse_model, injury_model = load_surrogate_models(config=config, device=device)
    
    surrogate = SurrogateAdapter(
        pulse_model=pulse_model, injury_model=injury_model, 
        param_manager=param_manager, config=config, data_processor=data_processor
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

    # 命令行覆盖：支持临时加载某个策略权重并强制启用直推
    if args.strategy_ckpt:
        if not os.path.isfile(args.strategy_ckpt):
            raise FileNotFoundError(f"指定的策略网络权重不存在: {args.strategy_ckpt}")

        # 若存在训练时保存的哈希文件，检查归一化配置是否一致
        # 目的：避免“训练和评估的归一化统计不一致”导致预测偏移。
        ckpt_dir = os.path.dirname(args.strategy_ckpt)
        sha_info_path = os.path.join(ckpt_dir, 'sha_checksums.yaml')
        if os.path.isfile(sha_info_path):
            try:
                with open(sha_info_path, 'r', encoding='utf-8') as f:
                    sha_info = yaml.safe_load(f) or {}
                expected_norm_sha = str(sha_info.get('normalization_config_sha1', '')).strip()
                if expected_norm_sha:
                    with open(str(NORMALIZATION_CONFIG_PATH), 'rb') as f:
                        current_norm_sha = hashlib.sha1(f.read()).hexdigest()
                    if current_norm_sha != expected_norm_sha:
                        logger.warning(
                            "当前 normalization_config 与策略网络训练时记录的哈希不一致，"
                            "评估结果可能出现分布偏移。"
                        )
            except Exception as e:
                logger.warning(f"读取或校验 sha_checksums.yaml 失败: {e}")

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

    # 1) 解析输入工况
    # 双模式：
    # - 指定 --input_csv：按本地 CSV 工况评估
    # - 未指定 --input_csv：自动读取 injury_test_indices.csv 对应测试集，附带真值标签
    truth_arrays = {}
    input_source_type = None
    input_source_path = None
    input_related_name = None
    test_ref_paths = {}
    if args.input_csv:
        logger.info(f"读取输入工况文件: {args.input_csv}")
        df_input = pd.read_csv(args.input_csv)
        input_source_type = 'input_csv'
        input_source_path = os.path.abspath(args.input_csv)
        input_related_name = _sanitize_name(os.path.splitext(os.path.basename(args.input_csv))[0])
        if all(col in df_input.columns for col in ['y_HIC', 'y_Dmax', 'y_Nij']):
            truth_arrays['y_HIC'] = df_input['y_HIC'].to_numpy(dtype=np.float32)
            truth_arrays['y_Dmax'] = df_input['y_Dmax'].to_numpy(dtype=np.float32)
            truth_arrays['y_Nij'] = df_input['y_Nij'].to_numpy(dtype=np.float32)
    else:
        pool_npz_path = RAW_DATA_DIR / 'raw_data_packed.npz'
        test_idx_path = SPLIT_INDICES_DIR / 'injury_test_indices.csv'
        if not pool_npz_path.exists():
            raise FileNotFoundError(f"测试集自动模式下未找到数据包: {pool_npz_path}")
        if not test_idx_path.exists():
            raise FileNotFoundError(f"测试集自动模式下未找到索引文件: {test_idx_path}")

        test_indices = load_int_vector_csv(test_idx_path)
        with np.load(pool_npz_path, allow_pickle=True) as data:
            if 'x_att_raw' not in data:
                raise KeyError("raw_data_packed.npz 缺失 x_att_raw，无法组装测试集工况。")

            x_att_raw = data['x_att_raw'][test_indices]
            if x_att_raw.ndim != 2 or x_att_raw.shape[1] != len(FEATURE_ORDER):
                raise ValueError(f"x_att_raw 形状异常，期望 (*, {len(FEATURE_ORDER)})，实际 {x_att_raw.shape}")

            case_ids = data['case_ids'][test_indices] if 'case_ids' in data else np.arange(len(test_indices))
            df_input = pd.DataFrame(x_att_raw, columns=FEATURE_ORDER)
            df_input.insert(0, 'case_id', case_ids)

            for key in ['y_HIC', 'y_Dmax', 'y_Nij', 'ais_head', 'ais_chest', 'ais_neck']:
                if key in data:
                    truth_arrays[key] = data[key][test_indices]

        input_source_type = 'test_split'
        input_source_path = str(test_idx_path.resolve())
        input_related_name = 'injury_test_split'
        test_ref_paths = {
            'test_indices_path': str(test_idx_path.resolve()),
            'raw_data_npz_path': str(pool_npz_path.resolve()),
        }

        logger.info(f"未提供 input_csv，已自动加载测试集工况: {len(df_input)} 条")

    # 2) 评估输出目录：统一放到 ARS_optim/saved_eval/<与输入相关>_<时间戳>/
    saved_eval_root = os.path.join(base_dir, 'saved_eval')
    os.makedirs(saved_eval_root, exist_ok=True)
    time_tag = datetime.now().strftime('%m%d_%H%M%S')
    run_folder_name = f"{input_related_name}_{time_tag}"
    save_dir = os.path.join(saved_eval_root, run_folder_name)
    os.makedirs(save_dir, exist_ok=False)

    output_csv_name = os.path.basename(args.output_csv) if args.output_csv else 'evaluation_results.csv'
    output_csv_path = os.path.join(save_dir, output_csv_name)

    logger.info(f"本次评估输出目录: {save_dir}")

    # 同步保存本次评估使用的配置快照，便于复现
    with open(os.path.join(save_dir, 'config_used.yaml'), 'w', encoding='utf-8') as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
    with open(param_space_path, 'r', encoding='utf-8') as f:
        param_space_cfg = yaml.safe_load(f)
    with open(os.path.join(save_dir, 'param_space_used.yaml'), 'w', encoding='utf-8') as f:
        yaml.safe_dump(param_space_cfg, f, allow_unicode=True, sort_keys=False)
    
    # 组装上下文特征（state + fixed-control）
    # 例如当前参数空间下，context 可能包含：
    # [impact_velocity, impact_angle, overlap, LL1, LL2, SP, SH, RA, is_driver_side, OT]
    # 缺失列回退 default，越界值直接报错，避免静默修正。
    context_params = param_manager.get_context_params()
    context_names = [p['name'] for p in context_params]
    fixed_params = param_manager.control_fixed_params
    fixed_names = [p['name'] for p in fixed_params]

    context_df = pd.DataFrame(index=df_input.index)
    missing_context_cols = []
    for p in context_params:
        name = p['name']
        if name in df_input.columns:
            context_df[name] = pd.to_numeric(df_input[name], errors='raise')
        else:
            if 'default' not in p:
                raise ValueError(f"输入 CSV 缺失上下文参数 '{name}'，且 param_space.yaml 未提供 default，无法评估。")
            context_df[name] = float(p['default'])
            missing_context_cols.append(name)

        if p.get('type') == 'continuous':
            p_min = float(p['min'])
            p_max = float(p['max'])
            out_of_range = (context_df[name] < p_min) | (context_df[name] > p_max)
            if out_of_range.any():
                bad_idx = context_df.index[out_of_range].tolist()[:10]
                raise ValueError(
                    f"输入 CSV 中上下文参数 '{name}' 存在越界值，合法范围 [{p_min}, {p_max}]，"
                    f"示例行索引: {bad_idx}"
                )

    if missing_context_cols:
        logger.warning(f"输入 CSV 缺失部分上下文参数列，已回退 default: {missing_context_cols}")

    context_tensor = torch.tensor(context_df[context_names].values, dtype=torch.float32, device=device)

    # 3) 计算 baseline 与优化结果
    logger.info("开始执行基线推断与联合局部精调...")
    
    # 构造 baseline 动作向量
    # 规则：若 CSV 给出可调列（如 BTF/LLATTF/AFT），按 CSV；否则回退 default。
    trainable_params = param_manager.control_trainable_params
    control_names = [p['name'] for p in trainable_params]
    baseline_trainable_df = pd.DataFrame(index=df_input.index)
    missing_trainable_cols = []

    for p in trainable_params:
        name = p['name']
        if name in df_input.columns:
            baseline_trainable_df[name] = pd.to_numeric(df_input[name], errors='raise')
        else:
            baseline_trainable_df[name] = float(p['default'])
            missing_trainable_cols.append(name)

        if p.get('type') == 'continuous':
            p_min = float(p['min'])
            p_max = float(p['max'])
            out_of_range = (baseline_trainable_df[name] < p_min) | (baseline_trainable_df[name] > p_max)
            if out_of_range.any():
                bad_idx = baseline_trainable_df.index[out_of_range].tolist()[:10]
                raise ValueError(
                    f"输入 CSV 中可调参数 '{name}' 存在越界值，合法范围 [{p_min}, {p_max}]，"
                    f"示例行索引: {bad_idx}"
                )

    if missing_trainable_cols:
        logger.warning(f"输入 CSV 未提供部分可调参数列，baseline 已回退 default: {missing_trainable_cols}")

    baseline_trainable = torch.tensor(
        baseline_trainable_df[control_names].values,
        dtype=torch.float32,
        device=device
    )

    # 固定控制参数按“每一行”保留，避免把整批样本错误地写成同一个固定值
    if fixed_names:
        fixed_context_tensor = torch.tensor(context_df[fixed_names].values, dtype=torch.float32, device=device)
        full_baseline = torch.cat([baseline_trainable, fixed_context_tensor], dim=1)
    else:
        fixed_context_tensor = torch.empty((context_tensor.shape[0], 0), dtype=torch.float32, device=device)
        full_baseline = baseline_trainable

    with torch.no_grad():
        pulse_norm = surrogate.generate_pulse(context_tensor)
        # baseline 评估
        # [Batch, D_context] + [Batch, D_trainable] + [Batch, 2, Seq_Len] -> 损失与损伤预测
        baseline_loss_batch, baseline_preds, baseline_info = surrogate.predict_injury_and_loss(context_tensor, baseline_trainable, pulse_norm)

    # 优化评估：支持复用 pulse_norm，避免重复推理波形模型
    optimized_actions, optimized_preds, opt_info = optimizer.optimize(context_tensor, pulse_norm=pulse_norm)

    # 日志一些简要指标
    logger.info(f"优化耗时: {opt_info.get('time_cost', float('nan')):.6f} s")
    if 'initial' in opt_info:
        logger.info(f"初始平均损伤: {opt_info['initial'].get('loss_mean', float('nan')):.4f}")
    if baseline_loss_batch is not None:
        mean_base = baseline_loss_batch.mean().item()
        logger.info(f"基线平均损伤: {mean_base:.4f}")

    # 输出批量评估摘要
    # 这里将 batch 结果拆成逐样本统计，便于后续定位具体 case 的改进幅度。
    tracker = MetricsTracker()
    init_actions = opt_info.get('initial', {}).get('actions')
    init_loss_batch = opt_info.get('initial', {}).get('loss_batch')
    final_loss_batch = opt_info.get('final_loss_batch')
    case_ids_series = df_input['case_id'] if 'case_id' in df_input.columns else pd.Series(range(len(df_input)))
    avg_time = float(opt_info.get('time_cost', 0.0)) / max(1, len(df_input))

    for i in range(len(df_input)):
        # 构造单样本结果结构，复用 MetricsTracker.update 接口
        case_result = {
            'initial': {'loss_mean': float(init_loss_batch[i].item()) if init_loss_batch is not None else float('nan')},
            'final_loss_batch': final_loss_batch[i:i+1].detach() if final_loss_batch is not None else None,
            'time_cost': avg_time,
            'trajectory': opt_info.get('trajectory', [])
        }
        init_action_i = init_actions[i].detach().cpu().numpy() if init_actions is not None else None
        opt_action_i = optimized_actions[i].detach().cpu().numpy()
        tracker.update(case_result, case_id=int(case_ids_series.iloc[i]), initial_action=init_action_i, optimized_action=opt_action_i)

    tracker.log_summary()

    # 4) 结果合并与输出
    # 输出列按 Base_/Init_/Opt_ 前缀分组，便于直接做横向对比。
    all_control_names = control_names + fixed_names
    df_base_ctrl = pd.DataFrame(full_baseline.cpu().numpy(), columns=[f"Base_{n}" for n in all_control_names])

    init_actions = opt_info.get('initial', {}).get('actions')
    if init_actions is not None:
        if fixed_context_tensor.numel() > 0:
            full_init = torch.cat([init_actions, fixed_context_tensor], dim=1)
        else:
            full_init = init_actions
        df_init_ctrl = pd.DataFrame(full_init.cpu().numpy(), columns=[f"Init_{n}" for n in all_control_names])
    else:
        df_init_ctrl = pd.DataFrame({f"Init_{n}": [float('nan')] * len(df_input) for n in all_control_names})

    if fixed_context_tensor.numel() > 0:
        full_optimized = torch.cat([optimized_actions, fixed_context_tensor], dim=1)
    else:
        full_optimized = optimized_actions
    df_opt_actions = pd.DataFrame(full_optimized.cpu().numpy(), columns=[f"Opt_{n}" for n in all_control_names])

    # 损伤对比（Baseline / Init / Optimized）
    df_baseline_inj = pd.DataFrame(baseline_preds.cpu().numpy(), columns=['Base_HIC', 'Base_Dmax', 'Base_Nij'])
    init_preds = opt_info.get('initial', {}).get('preds')
    if init_preds is not None:
        df_init_inj = pd.DataFrame(init_preds.cpu().numpy(), columns=['Init_HIC', 'Init_Dmax', 'Init_Nij'])
    else:
        df_init_inj = pd.DataFrame({'Init_HIC': [float('nan')] * len(df_input), 'Init_Dmax': [float('nan')] * len(df_input), 'Init_Nij': [float('nan')] * len(df_input)})

    df_opt_inj = pd.DataFrame(optimized_preds.cpu().numpy(), columns=['Opt_HIC', 'Opt_Dmax', 'Opt_Nij'])

    # 损失列
    df_base_loss = pd.DataFrame(baseline_loss_batch.cpu().numpy(), columns=['Base_Loss'])
    init_loss_batch = opt_info.get('initial', {}).get('loss_batch')
    if init_loss_batch is not None:
        df_init_loss = pd.DataFrame(init_loss_batch.cpu().numpy(), columns=['Init_Loss'])
    else:
        df_init_loss = pd.DataFrame({'Init_Loss': [float('nan')] * len(df_input)})

    opt_loss_batch = opt_info.get('final_loss_batch')
    if opt_loss_batch is not None:
        df_opt_loss = pd.DataFrame(opt_loss_batch.cpu().numpy(), columns=['Opt_Loss'])
    else:
        df_opt_loss = pd.DataFrame({'Opt_Loss': [float('nan')] * len(df_input)})

    # 风险概率列（头/胸/颈）
    df_base_prob = pd.DataFrame({
        'Base_Phead': baseline_info['p_head'].cpu().numpy(),
        'Base_Pchest': baseline_info['p_chest'].cpu().numpy(),
        'Base_Pneck': baseline_info['p_neck'].cpu().numpy(),
    })

    init_detail = opt_info.get('initial', {}).get('detail', {})
    if init_detail and all(k in init_detail for k in ['p_head', 'p_chest', 'p_neck']):
        df_init_prob = pd.DataFrame({
            'Init_Phead': init_detail['p_head'].cpu().numpy(),
            'Init_Pchest': init_detail['p_chest'].cpu().numpy(),
            'Init_Pneck': init_detail['p_neck'].cpu().numpy(),
        })
    else:
        df_init_prob = pd.DataFrame({
            'Init_Phead': [float('nan')] * len(df_input),
            'Init_Pchest': [float('nan')] * len(df_input),
            'Init_Pneck': [float('nan')] * len(df_input),
        })

    df_opt_prob = pd.DataFrame({
        'Opt_Phead': opt_info['p_head'].cpu().numpy(),
        'Opt_Pchest': opt_info['p_chest'].cpu().numpy(),
        'Opt_Pneck': opt_info['p_neck'].cpu().numpy(),
    })

    # AIS 等级列（Baseline / Init / Optimized）
    ot_array = context_df['OT'].to_numpy()
    base_hic = baseline_preds[:, 0].detach().cpu().numpy()
    base_dmax = baseline_preds[:, 1].detach().cpu().numpy()
    base_nij = baseline_preds[:, 2].detach().cpu().numpy()
    base_ais_head = AIS_cal_head(base_hic)
    base_ais_chest = AIS_cal_chest(base_dmax, ot_array)
    base_ais_neck = AIS_cal_neck(base_nij)

    if init_preds is not None:
        init_hic = init_preds[:, 0].detach().cpu().numpy()
        init_dmax = init_preds[:, 1].detach().cpu().numpy()
        init_nij = init_preds[:, 2].detach().cpu().numpy()
        init_ais_head = AIS_cal_head(init_hic)
        init_ais_chest = AIS_cal_chest(init_dmax, ot_array)
        init_ais_neck = AIS_cal_neck(init_nij)
    else:
        init_ais_head = np.full((len(df_input),), np.nan)
        init_ais_chest = np.full((len(df_input),), np.nan)
        init_ais_neck = np.full((len(df_input),), np.nan)

    opt_hic = optimized_preds[:, 0].detach().cpu().numpy()
    opt_dmax = optimized_preds[:, 1].detach().cpu().numpy()
    opt_nij = optimized_preds[:, 2].detach().cpu().numpy()
    opt_ais_head = AIS_cal_head(opt_hic)
    opt_ais_chest = AIS_cal_chest(opt_dmax, ot_array)
    opt_ais_neck = AIS_cal_neck(opt_nij)

    df_ais = pd.DataFrame({
        'Base_AIS_head': base_ais_head,
        'Base_AIS_chest': base_ais_chest,
        'Base_AIS_neck': base_ais_neck,
        'Base_AIS_max': np.maximum.reduce([base_ais_head, base_ais_chest, base_ais_neck]),
        'Init_AIS_head': init_ais_head,
        'Init_AIS_chest': init_ais_chest,
        'Init_AIS_neck': init_ais_neck,
        'Init_AIS_max': np.nanmax(np.vstack([init_ais_head, init_ais_chest, init_ais_neck]), axis=0),
        'Opt_AIS_head': opt_ais_head,
        'Opt_AIS_chest': opt_ais_chest,
        'Opt_AIS_neck': opt_ais_neck,
        'Opt_AIS_max': np.maximum.reduce([opt_ais_head, opt_ais_chest, opt_ais_neck]),
    })

    # 真值列（若可用）
    df_truth = pd.DataFrame(index=df_input.index)
    if all(k in truth_arrays for k in ['y_HIC', 'y_Dmax', 'y_Nij']):
        true_hic = np.asarray(truth_arrays['y_HIC'], dtype=np.float32)
        true_dmax = np.asarray(truth_arrays['y_Dmax'], dtype=np.float32)
        true_nij = np.asarray(truth_arrays['y_Nij'], dtype=np.float32)
        df_truth['True_HIC'] = true_hic
        df_truth['True_Dmax'] = true_dmax
        df_truth['True_Nij'] = true_nij
        df_truth['True_AIS_head'] = np.asarray(truth_arrays['ais_head']) if 'ais_head' in truth_arrays else AIS_cal_head(true_hic)
        df_truth['True_AIS_chest'] = np.asarray(truth_arrays['ais_chest']) if 'ais_chest' in truth_arrays else AIS_cal_chest(true_dmax, ot_array)
        df_truth['True_AIS_neck'] = np.asarray(truth_arrays['ais_neck']) if 'ais_neck' in truth_arrays else AIS_cal_neck(true_nij)
        df_truth['True_AIS_max'] = np.maximum.reduce([
            df_truth['True_AIS_head'].to_numpy(),
            df_truth['True_AIS_chest'].to_numpy(),
            df_truth['True_AIS_neck'].to_numpy(),
        ])

    # 降幅列：主要关注 Baseline -> Optimized 的绝对降幅与相对降幅
    eps = 1e-8
    base_loss_np = baseline_loss_batch.detach().cpu().numpy()
    opt_loss_np = opt_loss_batch.detach().cpu().numpy() if opt_loss_batch is not None else np.full((len(df_input),), np.nan)
    reduction_df = pd.DataFrame({
        'Reduction_HIC_abs': base_hic - opt_hic,
        'Reduction_Dmax_abs': base_dmax - opt_dmax,
        'Reduction_Nij_abs': base_nij - opt_nij,
        'Reduction_HIC_pct': (base_hic - opt_hic) / np.maximum(np.abs(base_hic), eps),
        'Reduction_Dmax_pct': (base_dmax - opt_dmax) / np.maximum(np.abs(base_dmax), eps),
        'Reduction_Nij_pct': (base_nij - opt_nij) / np.maximum(np.abs(base_nij), eps),
        'Reduction_Loss_abs': base_loss_np - opt_loss_np,
        'Reduction_Loss_pct': (base_loss_np - opt_loss_np) / np.maximum(np.abs(base_loss_np), eps),
        'Reduction_AIS_max_abs': df_ais['Base_AIS_max'].to_numpy() - df_ais['Opt_AIS_max'].to_numpy(),
    })

    # 宏观降损指标：均值（Baseline - Optimized）
    base_p_head = baseline_info['p_head'].detach().cpu().numpy()
    base_p_chest = baseline_info['p_chest'].detach().cpu().numpy()
    base_p_neck = baseline_info['p_neck'].detach().cpu().numpy()
    opt_p_head = opt_info['p_head'].detach().cpu().numpy()
    opt_p_chest = opt_info['p_chest'].detach().cpu().numpy()
    opt_p_neck = opt_info['p_neck'].detach().cpu().numpy()

    base_joint_risk = 1.0 - (1.0 - base_p_head) * (1.0 - base_p_chest) * (1.0 - base_p_neck)
    opt_joint_risk = 1.0 - (1.0 - opt_p_head) * (1.0 - opt_p_chest) * (1.0 - opt_p_neck)

    summary_metrics = {
        'mean_reduction_HIC': float(np.mean(base_hic - opt_hic)),
        'mean_reduction_Dmax': float(np.mean(base_dmax - opt_dmax)),
        'mean_reduction_Nij': float(np.mean(base_nij - opt_nij)),
        'mean_reduction_Phead': float(np.mean(base_p_head - opt_p_head)),
        'mean_reduction_Pchest': float(np.mean(base_p_chest - opt_p_chest)),
        'mean_reduction_Pneck': float(np.mean(base_p_neck - opt_p_neck)),
        'mean_reduction_joint_risk': float(np.mean(base_joint_risk - opt_joint_risk)),
        'mean_base_joint_risk': float(np.mean(base_joint_risk)),
        'mean_opt_joint_risk': float(np.mean(opt_joint_risk)),
        'n_samples': int(len(df_input)),
    }

    df_final = pd.concat([
        df_input,
        df_truth,
        df_base_ctrl,
        df_init_ctrl,
        df_opt_actions,
        df_baseline_inj,
        df_init_inj,
        df_opt_inj,
        df_ais,
        df_base_prob,
        df_init_prob,
        df_opt_prob,
        df_base_loss,
        df_init_loss,
        df_opt_loss,
        reduction_df,
    ], axis=1)
    df_final.to_csv(output_csv_path, index=False)

    # 评估信息记录：输入来源、策略权重、配置来源与快照路径
    evaluation_record = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'output_dir': save_dir,
        'output_csv_path': output_csv_path,
        'input_source': {
            'type': input_source_type,
            'path': input_source_path,
            **test_ref_paths,
        },
        'strategy_checkpoint_path': os.path.abspath(args.strategy_ckpt) if args.strategy_ckpt else None,
        'direct_inference': bool(config.get('optimization', {}).get('direct_inference', False)),
        'config_files': {
            'default_config_path': os.path.abspath(cfg_path),
            'param_space_path': os.path.abspath(param_space_path),
            'saved_default_config_snapshot': os.path.join(save_dir, 'config_used.yaml'),
            'saved_param_space_snapshot': os.path.join(save_dir, 'param_space_used.yaml'),
        },
        'evaluation_config': config.get('evaluation', {}),
        'optimization_config': config.get('optimization', {}),
    }
    with open(os.path.join(save_dir, 'evaluation_record.yaml'), 'w', encoding='utf-8') as f:
        yaml.safe_dump(evaluation_record, f, allow_unicode=True, sort_keys=False)

    # 结果汇总：输出用户要求的宏观降损指标
    evaluation_summary = {
        'summary_metrics': summary_metrics,
        'formulas': {
            'joint_risk': 'L_risk = 1 - Π_k (1 - P_k)',
            'reported_reduction': 'mean(Baseline - Optimized)',
        }
    }
    with open(os.path.join(save_dir, 'evaluation_summary.yaml'), 'w', encoding='utf-8') as f:
        yaml.safe_dump(evaluation_summary, f, allow_unicode=True, sort_keys=False)

    logger.info(f"评估完成！结果CSV已保存至: {output_csv_path}")
    logger.info(f"评估记录文件: {os.path.join(save_dir, 'evaluation_record.yaml')}")
    logger.info(f"评估汇总文件: {os.path.join(save_dir, 'evaluation_summary.yaml')}")

if __name__ == "__main__":
    main()