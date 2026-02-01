import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = 'T'
import warnings
warnings.filterwarnings('ignore')
import json
import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import pandas as pd

# 导入项目模块
import model.model as module_arch
from model.metric import ISORating
from parse_config import ConfigParser # 仅用于日志记录器
from utils.util import InputScaler, inverse_transform, plot_waveform_comparison

#==========================================================================================
# 1. 配置文件
#==========================================================================================
# 绘图中文正常显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False    # 负号正常显示

# 1.1. 文件路径配置
# --------------------------------------------------------------------------------------
# 指定要加载的模型检查点 (.pth) 文件路径
CHECKPOINT_PATH = (
    r"E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\LX_model_PulsePredict\saved\models\HybridPulseCNN\1213_095952\model_best.pth"
)

# 指定要分析的数据集 (.npz) 文件路径 (例如，测试集或包含所有工况的完整数据集)
DATASET_NPZ_PATH_LIST = [
    r"E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\仿真数据库相关\VCS波形数据集打包\acc_data_before1111_6134\packaged_data_test.npz",
    r"E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\仿真数据库相关\VCS波形数据集打包\acc_data_before1111_6134\packaged_data_train.npz"
]

# 1.2. 绘图轴配置
# --------------------------------------------------------------------------------------
# 定义散点图的 X 轴和 Y 轴分别对应哪个工况参数
# 索引: 0 = 速度 (velocity), 1 = 角度 (angle), 2 = 重叠率 (overlap)
X_AXIS_PARAM_INDEX = 2  # 例如：重叠率
Y_AXIS_PARAM_INDEX = 1  # 例如：角度

# 坐标轴标签
PARAM_NAMES = ["Velocity (km/h)", "Angle (deg)", "Overlap"]
X_AXIS_LABEL = PARAM_NAMES[X_AXIS_PARAM_INDEX]
Y_AXIS_LABEL = PARAM_NAMES[Y_AXIS_PARAM_INDEX]

# 1.3. 工况范围筛选配置 (类似于 test.py)
# --------------------------------------------------------------------------------------
# 定义要分析的特定工况范围
# 如果要分析 DATASET_NPZ_PATH 中的所有数据，请将 'conditions' 设置为空列表: []
SPECIFIC_CASE_CONFIG = {
    'description': "全部分析工况",
    'conditions': [
        # 示例 1: 仅分析角度绝对值 >= 15 度
        # {'param_index': 1, 'type': 'abs_range', 'range': [15, 60]},
        
        # 示例 2: 仅分析速度在 40 到 50 之间
        # {'param_index': 0, 'type': 'range', 'range': [40, 50]},
        
        # 示例 3: 小角度 & 大重叠率 (同 test.py)
        # {'param_index': 1, 'type': 'abs_range', 'range': [0, 15]},    # 条件1: 角度绝对值在[0, 15]度
        # {'param_index': 2, 'type': 'abs_range', 'range': [0.75, 1.0]}  # 条件2: 重叠率绝对值在[0.75, 1.0]
    ]
}

# 1.4. 其他配置
# --------------------------------------------------------------------------------------
# 用于推理的批量大小
BATCH_SIZE = 512

# ISO Rating 的颜色映射范围 (vmin, vmax)
ISO_RATING_RANGE_X = (0.5, 1.0)  # X 轴范围
ISO_RATING_RANGE_Y = (0.3, 0.9)  # Y 轴范围
ISO_RATING_RANGE_Z = (0.2, 0.8)  # Z 轴范围

PLOT_WAVEFORM_CONFIG = {
    # 'target_case_ids': [5254,6878,8172,4532,698,918,58,2542,3078,1153,1976,2068,4120,1043,7760],     # 指定要强制绘图的 case_id 列表，例如 [10, 25]
    'target_case_ids': [],
    'plot_low_score': True,    # 是否自动绘制低分案例
    'low_score_threshold': 0.55 # 低分阈值 (ISO Rating X < 0.5)
}

# EXPORT_EXCEL_CASE_IDS = [3249, 4024, 6561, 5254, 704, 2350] 
EXPORT_EXCEL_CASE_IDS = [] 

# 1.5. 组合波形绘图配置
# --------------------------------------------------------------------------------------
COMBINED_PLOT_CONFIG = {
    'enabled': False,  # 是否启用组合绘图
    'case_groups': [
        # 每个组将绘制在一张图上
        {
            'case_ids': list(np.arange(9027,9042)),  # 第一组工况ID
            'group_name': 'overlap change'  # 组名，用于文件命名
        },
        # {
        #     'case_ids': [4532, 698, 918],  # 第二组工况ID
        #     'group_name': 'Group_2'
        # },
        # 可以添加更多组
    ]
}

#==========================================================================================
# 2. 辅助函数
#==========================================================================================

def get_run_root_and_config_path(checkpoint_path):
    """
    根据检查点路径推断出实验的根目录和config.json路径。
    """
    cp_path = Path(checkpoint_path)
    # 假设 .pth 文件在 session 文件夹中 (如 resume_... 或 test_...)
    config_path = cp_path.parent / 'config.json'
    
    if config_path.exists():
        # .pth 在 session 文件夹中, config 也在
        run_root_dir = cp_path.parent
    else:
        # .pth 在 session 文件夹中, config 在上一级 (标准的 resume 场景)
        config_path = cp_path.parent.parent / 'config.json'
        if config_path.exists():
            run_root_dir = cp_path.parent.parent
        else:
            # .pth 可能就在 run_root_dir 中
            config_path = cp_path.parent / 'config.json'
            if config_path.exists():
                run_root_dir = cp_path.parent
            else:
                raise FileNotFoundError(
                    f"在 {cp_path.parent} 或其父目录中均未找到 'config.json'。"
                )
    
    return run_root_dir, config_path

def plot_iso_scatter(x_data, y_data, color_data, config, save_path, vmin, vmax):
    """
    绘制并保存散点图。
    (此版本接受 vmin 和 vmax 参数)
    """
    plt.figure(figsize=(12, 10))
    sc = plt.scatter(
        x_data, 
        y_data, 
        c=color_data, 
        cmap='viridis',  # 使用 viridis 颜色映射
        alpha=0.75,
        s=100,
        vmin=vmin,  # 使用传入的 vmin
        vmax=vmax   # 使用传入的 vmax
    )
    
    # 更新 colorbar 标签以显示动态范围
    cbar = plt.colorbar(sc, label=f"ISO Rating (Range: ({vmin:.2f}, {vmax:.2f}))") 
    cbar.ax.tick_params(labelsize=16)  # 添加这行,控制colorbar刻度数字大小
    cbar.set_label(f"ISO Rating (Range: ({vmin:.2f}, {vmax:.2f}))", fontsize=16)  # 控制colorbar标签字号
    
    plt.xlabel(config['x_label'], fontsize=18)
    plt.ylabel(config['y_label'], fontsize=18)
    plt.title(config['title'], fontsize=16, pad=10)
    plt.tick_params(axis='both', labelsize=16)  #控制刻度数字大小
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 在标题下方添加工况范围描述
    # plt.figtext(0.5, 0.95, f"Data Filter: {config['subtitle']}", ha="center", fontsize=10, style='italic')
    
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

def save_case_to_excel(pred_wave, true_wave, case_id, save_dir):
    """
    将指定 Case 的预测和真值波形保存为 Excel 文件。
    包含 3 个 Sheet (X, Y, Z)。
    """
    # 确保输入是 Numpy 数组
    if isinstance(pred_wave, torch.Tensor):
        pred_wave = pred_wave.cpu().numpy()
    if isinstance(true_wave, torch.Tensor):
        true_wave = true_wave.cpu().numpy()

    # 时间戳：0.001s ~ 0.150s (假设长度为 150)
    length = pred_wave.shape[1]
    time_seq = np.arange(1, length + 1) * 0.001

    # 定义三个方向的 Sheet 名称
    directions = ['X', 'Y', 'Z']
    
    # 构造保存路径 (保存到 save_dir 下的 excel_data 文件夹)
    excel_dir = save_dir / "excel_data"
    os.makedirs(excel_dir, exist_ok=True)
    
    file_name = f"case_{case_id}_data.xlsx"
    save_path = excel_dir / file_name

    # 使用 Pandas ExcelWriter 写入多个 Sheet
    try:
        with pd.ExcelWriter(save_path) as writer:
            for i, direction in enumerate(directions):
                # 构造 DataFrame
                df = pd.DataFrame({
                    'Time (s)': time_seq,
                    'Ground Truth': true_wave[i, :],
                    'Prediction': pred_wave[i, :]
                })
                # 写入 Sheet
                df.to_excel(writer, sheet_name=f'Direction_{direction}', index=False)
        
        print(f"  [Excel保存] Case {case_id} 已保存至: {save_path}")
    except Exception as e:
        print(f"  [Excel保存失败] Case {case_id}: {e}")

def plot_combined_waveforms(pred_waves_dict, true_waves_dict, params_dict, iso_dict, save_dir, group_name):
    """
    将多个工况的波形绘制在同一张图上进行对比。
    
    参数:
        pred_waves_dict: {case_id: pred_wave_array} 预测波形字典
        true_waves_dict: {case_id: true_wave_array} 真值波形字典
        params_dict: {case_id: {'vel': v, 'ang': a, 'ov': o}} 工况参数字典
        iso_dict: {case_id: {'x': score_x, 'y': score_y, 'z': score_z}} ISO评分字典
        save_dir: 保存目录
        group_name: 组名
    """
    # 创建保存目录
    combined_dir = save_dir / "combined_waveforms"
    os.makedirs(combined_dir, exist_ok=True)
    
    # 时间序列 (假设 150 个采样点, 0.001s 间隔)
    case_ids = list(pred_waves_dict.keys())
    n_samples = pred_waves_dict[case_ids[0]].shape[1] if isinstance(pred_waves_dict[case_ids[0]], np.ndarray) else pred_waves_dict[case_ids[0]].shape[1]
    time_seq = np.arange(n_samples) * 0.001
    
    # 方向标签
    direction_labels = ['X', 'Y', 'Z']
    colors = plt.cm.tab10(np.linspace(0, 1, len(case_ids)))  # 为每个 case 分配颜色
    
    # 创建 3 行 1 列的子图 (X, Y, Z)
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    fig.suptitle(f'Combined Waveform Comparison - {group_name}', fontsize=18, fontweight='bold')
    
    for direction_idx, direction in enumerate(direction_labels):
        ax = axes[direction_idx]
        
        # 遍历每个 case_id
        for color_idx, case_id in enumerate(case_ids):
            # 获取数据
            pred_wave = pred_waves_dict[case_id]
            true_wave = true_waves_dict[case_id]
            
            # 转换为 numpy (如果是 Tensor)
            if isinstance(pred_wave, torch.Tensor):
                pred_wave = pred_wave.cpu().numpy()
            if isinstance(true_wave, torch.Tensor):
                true_wave = true_wave.cpu().numpy()
            
            # 获取当前方向的波形
            pred_signal = pred_wave[direction_idx, :]
            true_signal = true_wave[direction_idx, :]
            
            # 获取工况参数和ISO评分
            params = params_dict[case_id]
            iso_scores = iso_dict[case_id]
            iso_score = iso_scores[direction.lower()]
            
            # 构造图例标签
            label_true = f'Case {case_id} GT (v={params["vel"]:.1f}, a={params["ang"]:.1f}, o={params["ov"]:.2f}, ISO={iso_score:.3f})'
            label_pred = f'Case {case_id} Pred'
            
            # 绘制真值 (实线)
            ax.plot(time_seq, true_signal, 
                   color=colors[color_idx], 
                   linestyle='-', 
                   linewidth=2.0,
                   label=label_true,
                   alpha=0.8)
            
            # 绘制预测值 (虚线)
            ax.plot(time_seq, pred_signal, 
                   color=colors[color_idx], 
                   linestyle='--', 
                   linewidth=2.0,
                   label=label_pred,
                   alpha=0.8)
        
        # 设置子图属性
        ax.set_xlabel('Time (s)', fontsize=14)
        ax.set_ylabel(f'Acceleration (g) - {direction}', fontsize=14)
        ax.set_title(f'Direction {direction}', fontsize=16, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.6)
        # ax.legend(loc='upper right', fontsize=10, ncol=2)
        ax.tick_params(axis='both', labelsize=12)
    
    plt.tight_layout()
    
    # 保存图片
    save_path = combined_dir / f"combined_{group_name}.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    return save_path

#==========================================================================================
# 3. 主执行函数
#==========================================================================================

def main():
    # --- 0. 初始化和日志 ---
    # 使用 ConfigParser 仅为了获取其 logger
    try:
        dummy_config = ConfigParser({}, resume=CHECKPOINT_PATH, is_test_run=True)
        logger = dummy_config.get_logger('plot_scatter')
        # 我们不使用 dummy_config.save_dir, 而是自己构造
    except Exception:
        # 如果ConfigParser失败，使用基本日志
        import logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger('plot_scatter')
        logger.warning("未能初始化 ConfigParser，使用基本日志。")

    logger.info("开始执行精度分布散点图绘制脚本...")
    
    # --- 1. 确定路径和加载配置 ---
    checkpoint_path = Path(CHECKPOINT_PATH)
    try:
        run_root_dir, config_path = get_run_root_and_config_path(checkpoint_path)
    except FileNotFoundError as e:
        logger.error(e)
        return

    logger.info(f"加载模型检查点: {checkpoint_path}")
    logger.info(f"加载配置文件: {config_path}")
    
    with open(config_path) as f:
        config = json.load(f)

    # 创建保存目录
    save_plot_dir = run_root_dir / "prediction_scatter_plots"
    os.makedirs(save_plot_dir, exist_ok=True)
    logger.info(f"图表将保存至: {save_plot_dir}")

    # --- 2. 加载模型 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_arch_type = config['arch']['type']
    model_arch_args = config['arch']['args']
    
    try:
        model = getattr(module_arch, model_arch_type)(**model_arch_args).to(device)
    except Exception as e:
        logger.error(f"加载模型架构 '{model_arch_type}' 失败: {e}")
        return
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    logger.info(f"模型 '{model_arch_type}' 加载成功并设置到 {device}。")

    # --- 3. 加载 Scalers ---
    bounds = config['data_loader_train']['args'].get('physics_bounds')
    if not bounds:
         raise ValueError("配置文件中未找到 'physics_bounds'")
    input_scaler = InputScaler(**bounds)

    target_scaler = None
    try:
        scaler_path_str = config['data_loader_train']['args'].get('scaler_path')
        if scaler_path_str:
            scaler_path = Path(scaler_path_str)
            if scaler_path.exists():
                target_scaler = joblib.load(scaler_path)
                logger.info(f"成功加载目标波形 Scaler: {scaler_path}")
            else:
                logger.warning(f"Scaler 文件 '{scaler_path}' 未找到。")
        else:
            logger.warning("配置中未指定 'scaler_path'。")
        
        if target_scaler is None:
             logger.warning("未加载 Target Scaler，将使用归一化尺度进行评估（可能不准确）。")
             
    except Exception as e:
        logger.error(f"加载 Scaler 时出错: {e}")
        return

    # --- 4. 加载并筛选数据 ---
    logger.info(f"正在加载并筛选数据集...")
    
    # <--- 修改: 循环加载多个文件并合并 --->
    all_raw_params_list = []
    all_waveforms_list = []
    all_case_ids_list = []
    all_source_files_list = []

    for npz_path_str in DATASET_NPZ_PATH_LIST:
        npz_path = Path(npz_path_str)
        if not npz_path.exists():
            logger.warning(f"文件不存在，跳过: {npz_path}")
            continue
            
        try:
            data = np.load(npz_path)
            # 读取数据
            params = data['params']
            waveforms = data['waveforms']
            case_ids = data['case_ids']
            
            all_raw_params_list.append(params)
            all_waveforms_list.append(waveforms)
            all_case_ids_list.append(case_ids)
            
            # 记录来源文件名 (扩展为与数据等长的列表)
            file_name = npz_path.name
            all_source_files_list.extend([file_name] * len(case_ids))
            
            logger.info(f"成功加载: {file_name} (样本数: {len(case_ids)})")
            
        except Exception as e:
            logger.error(f"加载 {npz_path} 失败: {e}")

    if not all_raw_params_list:
        logger.error("未成功加载任何数据，脚本终止。")
        return

    # 合并所有数据
    all_raw_params = np.concatenate(all_raw_params_list, axis=0)
    all_waveforms = np.concatenate(all_waveforms_list, axis=0)
    all_case_ids = np.concatenate(all_case_ids_list, axis=0)
    all_source_files = np.array(all_source_files_list)
    
    logger.info(f"数据合并完成，总样本数: {len(all_case_ids)}")

    # 应用工况筛选
    conditions = SPECIFIC_CASE_CONFIG.get('conditions', [])
    if not conditions:
        logger.info("未定义筛选条件，将分析数据集中的所有样本。")
        filtered_indices = np.arange(all_raw_params.shape[0])
    else:
        logger.info(f"根据 {len(conditions)} 个条件筛选工况...")
        combined_mask = np.full(all_raw_params.shape[0], True)
        for cond in conditions:
            param_index = cond['param_index']
            cond_type = cond['type']
            min_val, max_val = cond['range']
            
            param_to_check = all_raw_params[:, param_index]
            
            if cond_type == 'abs_range':
                current_mask = (np.abs(param_to_check) >= min_val) & (np.abs(param_to_check) <= max_val)
            else: # 'range'
                current_mask = (param_to_check >= min_val) & (param_to_check <= max_val)
            
            combined_mask &= current_mask
        
        filtered_indices = np.where(combined_mask)[0]

    if len(filtered_indices) == 0:
        logger.error("筛选后无任何样本，脚本终止。")
        return

    logger.info(f"筛选完毕，共 {len(filtered_indices)} / {all_raw_params.shape[0]} 个样本待处理。")

    # 提取筛选后的数据
    filtered_raw_params = all_raw_params[filtered_indices]
    filtered_true_waveforms = all_waveforms[filtered_indices]
    filtered_case_ids = all_case_ids[filtered_indices]
    filtered_source_files = all_source_files[filtered_indices]

    # --- 5. 准备模型输入 (归一化) ---
    normalized_params_list = []
    for params in filtered_raw_params:
        norm_vel, norm_ang, norm_ov = input_scaler.transform(params[0], params[1], params[2])
        normalized_params_list.append([norm_vel, norm_ang, norm_ov])
    
    normalized_params_np = np.array(normalized_params_list, dtype=np.float32)
    
    # 转换为 PyTorch Tensors
    norm_params_tensor = torch.from_numpy(normalized_params_np)
    true_waveforms_tensor = torch.from_numpy(filtered_true_waveforms).float()

    # --- 6. 执行批量推理 ---
    logger.info("开始执行模型批量推理...")
    all_pred_waveforms_orig = []
    
    # 使用 DataLoader 自动处理批次
    inference_dataset = TensorDataset(norm_params_tensor, true_waveforms_tensor)
    inference_loader = DataLoader(inference_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    GauNll_use = model_arch_args.get('GauNll_use', True) # 默认GauNll
    
    with torch.no_grad():
        for batch_data, batch_target in tqdm(inference_loader, desc="Inference"):
            batch_data = batch_data.to(device)
            batch_target = batch_target.to(device) # 目标也放上GPU，以便inverse_transform

            # ------------------- 模型推理 ----------------------
            output = model(batch_data)
            
            # 提取用于评估的输出 (通常是均值)
            metrics_output = model.get_metrics_output(output)
            # ---------------------------------------------------

            # 逆变换到物理尺度
            pred_orig, _ = inverse_transform(metrics_output, batch_target, target_scaler)
            
            all_pred_waveforms_orig.append(pred_orig.cpu())

    # 拼接所有批次的预测结果
    all_pred_waveforms_orig = torch.cat(all_pred_waveforms_orig, dim=0) # (N_filtered, 3, 150)
    
    logger.info("模型推理完成。")

    # --- 7. 计算 ISO Ratings ---
    logger.info("正在计算 ISO Ratings (逐样本)...")
    iso_ratings_x = []
    iso_ratings_y = []
    iso_ratings_z = []

    # 逐样本计算 (ISORating 类目前不支持批量)
    for i in tqdm(range(len(filtered_indices)), desc="Calculating ISO"):
        pred_wave = all_pred_waveforms_orig[i].numpy() # (3, 150)
        true_wave = filtered_true_waveforms[i]         # (3, 150)
        
        # X
        iso_x = ISORating(analyzed_signal=pred_wave[0, :], reference_signal=true_wave[0, :]).calculate()
        iso_ratings_x.append(iso_x)
        
        # Y
        iso_y = ISORating(analyzed_signal=pred_wave[1, :], reference_signal=true_wave[1, :]).calculate()
        iso_ratings_y.append(iso_y)
        
        # Z
        iso_z = ISORating(analyzed_signal=pred_wave[2, :], reference_signal=true_wave[2, :]).calculate()
        iso_ratings_z.append(iso_z)

    logger.info("ISO Ratings 计算完成。")

    # <--- 保存结果到 CSV 表格 --->
    import pandas as pd
    
    summary_df = pd.DataFrame({
        'case_id': filtered_case_ids,
        'source_file': filtered_source_files,
        'velocity': filtered_raw_params[:, 0],
        'angle': filtered_raw_params[:, 1],
        'overlap': filtered_raw_params[:, 2],
        'iso_rating_x': iso_ratings_x,
        'iso_rating_y': iso_ratings_y,
        'iso_rating_z': iso_ratings_z
    })

    csv_save_path = save_plot_dir / "evaluation_summary.csv"
    summary_df.to_csv(csv_save_path, index=False, float_format='%.4f')
    logger.info(f"评估结果汇总表已保存至: {csv_save_path}")

    # <--- 绘制波形比较图 --->
    logger.info("正在根据配置筛选并绘制波形对比图...")
    
    # 准备目标ID集合和阈值
    target_ids_set = set(PLOT_WAVEFORM_CONFIG['target_case_ids'])
    low_score_thresh = PLOT_WAVEFORM_CONFIG['low_score_threshold']
    plot_low_score = PLOT_WAVEFORM_CONFIG['plot_low_score']
    export_excel_ids_set = set(EXPORT_EXCEL_CASE_IDS)
    plot_count = 0
    for i in tqdm(range(len(filtered_indices)), desc="Plotting Waveforms"):
        c_id = filtered_case_ids[i]
        iso_x = iso_ratings_x[i]
        
        # 判断是否需要绘图: 指定ID 或 (开启低分检测 且 分数低于阈值)
        should_plot = (c_id in target_ids_set) or (plot_low_score and iso_x < low_score_thresh)
        
        if should_plot:
            # 准备参数字典
            raw_p = filtered_raw_params[i]
            params_dict = {'vel': raw_p[0], 'ang': raw_p[1], 'ov': raw_p[2]}
            
            # 准备 ISO 评分字典
            iso_dict = {
                'x': iso_ratings_x[i],
                'y': iso_ratings_y[i],
                'z': iso_ratings_z[i]
            }
            
            # 获取来源文件名
            src_file = filtered_source_files[i]
            
            # 调用绘图工具
            # 技巧: 将 source_file 传给 epoch 参数，使其显示在标题中 "Epoch: packaged_data_test.npz"
            # 图片将保存在 run_root_dir/fig/epoch_{src_file}/ 下，自动按来源文件分类文件夹
            plot_waveform_comparison(
                pred_wave=all_pred_waveforms_orig[i], # Tensor
                true_wave=filtered_true_waveforms[i], # Numpy
                params=params_dict,
                case_id=c_id,
                epoch=src_file, 
                batch_idx='', 
                sample_idx='',
                save_dir=run_root_dir, 
                iso_ratings=iso_dict
            )
            plot_count += 1

        # <--- 保存指定 Case 到 Excel --->
        if c_id in export_excel_ids_set:
            save_case_to_excel(
                pred_wave=all_pred_waveforms_orig[i], # Tensor
                true_wave=filtered_true_waveforms[i], # Numpy
                case_id=c_id,
                save_dir=run_root_dir
            )
    logger.info(f"波形绘图完成,共绘制 {plot_count} 张图片。保存在 {run_root_dir}/fig 目录下。")

    # <--- 新增: 组合波形绘图 --->
    if COMBINED_PLOT_CONFIG['enabled'] and COMBINED_PLOT_CONFIG['case_groups']:
        logger.info("正在绘制组合波形对比图...")
        
        for group_config in COMBINED_PLOT_CONFIG['case_groups']:
            group_case_ids = group_config['case_ids']
            group_name = group_config['group_name']
            
            # 收集该组的数据
            pred_waves_dict = {}
            true_waves_dict = {}
            params_dict_group = {}
            iso_dict_group = {}
            
            found_count = 0
            for case_id in group_case_ids:
                # 在筛选后的数据中查找该 case_id
                match_indices = np.where(filtered_case_ids == case_id)[0]
                
                if len(match_indices) > 0:
                    idx = match_indices[0]  # 取第一个匹配
                    
                    # 收集数据
                    pred_waves_dict[case_id] = all_pred_waveforms_orig[idx]
                    true_waves_dict[case_id] = filtered_true_waveforms[idx]
                    
                    raw_p = filtered_raw_params[idx]
                    params_dict_group[case_id] = {
                        'vel': raw_p[0], 
                        'ang': raw_p[1], 
                        'ov': raw_p[2]
                    }
                    
                    iso_dict_group[case_id] = {
                        'x': iso_ratings_x[idx],
                        'y': iso_ratings_y[idx],
                        'z': iso_ratings_z[idx]
                    }
                    
                    found_count += 1
                else:
                    logger.warning(f"Case ID {case_id} 未在筛选后的数据中找到，已跳过。")
            
            # 如果至少找到一个 case，则绘制
            if found_count > 0:
                save_path = plot_combined_waveforms(
                    pred_waves_dict=pred_waves_dict,
                    true_waves_dict=true_waves_dict,
                    params_dict=params_dict_group,
                    iso_dict=iso_dict_group,
                    save_dir=run_root_dir,
                    group_name=group_name
                )
                logger.info(f"组合图 '{group_name}' 已保存至: {save_path} (包含 {found_count}/{len(group_case_ids)} 个工况)")
            else:
                logger.warning(f"组 '{group_name}' 中没有找到任何有效工况，跳过绘图。")
        
        logger.info("组合波形绘图完成。")

    # --- 8. 准备绘图数据 ---
    plot_x_data = filtered_raw_params[:, X_AXIS_PARAM_INDEX]
    plot_y_data = filtered_raw_params[:, Y_AXIS_PARAM_INDEX]

    # --- 9. 绘图并保存 ---
    filter_desc = SPECIFIC_CASE_CONFIG['description']
    
    # 绘制 X-Rating
    logger.info("正在绘制 X-Rating 散点图...")
    plot_config_x = {
        'x_label': X_AXIS_LABEL,
        'y_label': Y_AXIS_LABEL,
        'title': f"ISO Rating (X-Axis) vs. {Y_AXIS_LABEL} and {X_AXIS_LABEL}",
        'subtitle': filter_desc
    }
    save_path_x = save_plot_dir / "iso_scatter_X.png"
    # 传入 X 轴的特定范围
    plot_iso_scatter(
        plot_x_data, plot_y_data, iso_ratings_x, plot_config_x, save_path_x,
        vmin=ISO_RATING_RANGE_X[0], vmax=ISO_RATING_RANGE_X[1]
    )

    # 绘制 Y-Rating
    logger.info("正在绘制 Y-Rating 散点图...")
    plot_config_y = {
        'x_label': X_AXIS_LABEL,
        'y_label': Y_AXIS_LABEL,
        'title': f"ISO Rating (Y-Axis) vs. {Y_AXIS_LABEL} and {X_AXIS_LABEL}",
        'subtitle': filter_desc
    }
    save_path_y = save_plot_dir / "iso_scatter_Y.png"
    # 传入 Y 轴的特定范围
    plot_iso_scatter(
        plot_x_data, plot_y_data, iso_ratings_y, plot_config_y, save_path_y,
        vmin=ISO_RATING_RANGE_Y[0], vmax=ISO_RATING_RANGE_Y[1]
    )

    # 绘制 Z-Rating
    logger.info("正在绘制 Z-Rating 散点图...")
    plot_config_z = {
        'x_label': X_AXIS_LABEL,
        'y_label': Y_AXIS_LABEL,
        'title': f"ISO Rating (Z-Axis) vs. {Y_AXIS_LABEL} and {X_AXIS_LABEL}",
        'subtitle': filter_desc
    }
    save_path_z = save_plot_dir / "iso_scatter_Z.png"
    # 传入 Z 轴的特定范围
    plot_iso_scatter(
        plot_x_data, plot_y_data, iso_ratings_z, plot_config_z, save_path_z,
        vmin=ISO_RATING_RANGE_Z[0], vmax=ISO_RATING_RANGE_Z[1]
    )

    logger.info(f"所有绘图已完成并保存至: {save_plot_dir}")

if __name__ == "__main__":
    main()