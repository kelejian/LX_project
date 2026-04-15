import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = 'T'
import warnings
warnings.filterwarnings('ignore')
import argparse
import numpy as np
import torch
from tqdm import tqdm

import PulsePredict.data_loader.data_loaders as module_data
import PulsePredict.model.loss as module_loss
import PulsePredict.model.metric as module_metric
import PulsePredict.model.model as module_arch
from PulsePredict.parse_config import ConfigParser
from PulsePredict.model.metric import ISORating # 导入 ISORating 类用于计算
from PulsePredict.utils import plot_waveform_comparison


def main(config):
    logger = config.get_logger('test')

    # --- 0. 定义绘图配置 ---
    PLOT_ISO_RATINGS_IN_TITLE = True # 设置为 True 以在标题中显示ISO-rating
    BATCH_IDX = 0              # 设置为要绘图的批次索引（从0开始）
    logger.info(f"绘图批次索引: {BATCH_IDX}")
    logger.info(f"绘图标题中是否显示ISO-rating: {PLOT_ISO_RATINGS_IN_TITLE}")

    # --- 1. 定义分组评估配置 ---
    grouping_config = {
        'param_name': 'velocity',  # 按哪个参数分组: 'velocity', 'angle', 'overlap'
        'param_index': 0,          # 参数在(N, 3)输入中的索引: 0=速度, 1=角度, 2=重叠率
        'ranges': {                # 定义区间的名字和 [min, max] 范围
            'low_speed': [23, 35],
            'mid_speed': [35, 48],
            'high_speed': [48, 65]
        }
    }
    logger.info(f"将根据参数 '{grouping_config['param_name']}' 的不同范围对测试结果进行分组统计。")
    
    # --- 2. 定义特定组合工况评估配置 ---
    specific_case_config = {
        'small_angle_large_overlap': {
            'description': "小角度(|ang|<=15) & 大重叠率(|ov|>=0.75)",
            'conditions': [
                {'param_index': 1, 'type': 'abs_range', 'range': [0, 15]},    # 条件1: 角度绝对值在[0, 15]度
                {'param_index': 2, 'type': 'abs_range', 'range': [0.75, 1.0]}  # 条件2: 重叠率绝对值在[0.75, 1.0]
            ]
        },
        'angle_overlap_same_sign': {
            'description': "角度(ang)与重叠率(ov)同号",
            'conditions': [
                # param_index: 1 (角度)
                # other_param_index: 2 (重叠率)
                {'param_index': 1, 'type': 'same_sign', 'other_param_index': 2}
            ]
        },
        'angle_gt15_and_full_or_samesign': {
            'description': "角度(abs)>15° 且 (重叠率=100% 或 角度重叠率同号)",
            # 注意：此 'conditions' 列表仅用于占位，
            # 实际逻辑在下面的循环中被硬编码 (hardcoded)
            'conditions': [ 
                {'type': 'custom_logic'}
            ]
        },
    }
    logger.info(f"将根据特定输入参数组合对测试结果进行额外评估。")
    # ---------------------------------------------

    # setup data_loader instances
    data_loader = config.init_obj('data_loader_test', module_data)
    # 强制 dataset 提供 UnifiedDataProcessor
    processor = getattr(data_loader, 'processor', None)
    if processor is None:
        raise RuntimeError("数据集缺少 `processor`，请先运行 `python -m prepare_data` 生成归一化配置。")
    # 打印测试集数据量
    logger.info(f"测试集数据量: {len(data_loader.train_test_indices)}")

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    criterion = config.init_obj('loss', module_loss)

    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)

    # prepare model for testing
    model = model.to(device)
    criterion = criterion.to(device)
    model.load_state_dict(state_dict)
    if 'criterion_state_dict' not in checkpoint:
        raise KeyError("检查点中缺少 `criterion_state_dict`，无法按训练时的任务权重进行测试。")
    criterion.load_state_dict(checkpoint['criterion_state_dict'])
    model.eval()

    # --- 2. 收集所有测试样本的结果 ---
    all_raw_params = []
    all_preds_orig = []
    all_targets_orig = []
    all_losses = []

    with torch.no_grad():
        for batch_idx, (data, target, case_ids) in enumerate(tqdm(data_loader, desc="testing")):
            data, target = data.to(device), target.to(device)
            # ------------------- 使用统一的模型接口 ----------------------
            output = model(data)
            loss, loss_components = criterion(output, target)
            metrics_output = model.get_metrics_output(output)
            # ------------------------------------------------------------
            
            # 收集每个样本的损失值；注意，损失函数reduction不是'none'，这里loss.item()是批次的平均损失；为简化，我们直接用批次平均损失代表该批次中每个样本的损失
            all_losses.extend([loss.item()] * data.shape[0])

            # 使用 UnifiedDataProcessor 进行逆归一化（tensor -> numpy -> processor -> tensor）
            metrics_output_np = processor.process_waveform(metrics_output.detach().cpu().numpy(), inverse=True)
            target_np = processor.process_waveform(target.detach().cpu().numpy(), inverse=True)
            pred_mean_orig = torch.from_numpy(metrics_output_np).to(metrics_output.device).type_as(metrics_output)
            target_orig = torch.from_numpy(target_np).to(target.device).type_as(target)

            # ---------------------------------------       

            # 收集逆变换后的工况、预测和目标（使用 UnifiedDataProcessor）
            for i in range(data.shape[0]):
                norm_params = data[i].cpu().numpy()
                raw_vals = processor.process_by_name(norm_params[None, :], ["impact_velocity", "impact_angle", "overlap"], inverse=True)[0]
                raw_vel, raw_ang, raw_ov = float(raw_vals[0]), float(raw_vals[1]), float(raw_vals[2])
                all_raw_params.append([raw_vel, raw_ang, raw_ov])

            all_preds_orig.append(pred_mean_orig.cpu())
            all_targets_orig.append(target_orig.cpu())

            # ------------------------------画图----------------------------------
            if batch_idx == BATCH_IDX:
                plot_samples(data, batch_idx, pred_mean_orig, target_orig, case_ids, processor, config, logger, plot_iso_ratings=PLOT_ISO_RATINGS_IN_TITLE)
            # --------------------------------------------------------------------

    # 将列表中的批次拼接成一个大的张量/数组
    all_preds_orig = torch.cat(all_preds_orig, dim=0)
    all_targets_orig = torch.cat(all_targets_orig, dim=0)
    all_raw_params = np.array(all_raw_params)
    all_losses = np.array(all_losses)

    # --- 3. 对全量和分组数据进行评估 ---
    logger.info("\n" + "="*50)
    logger.info(" 全量测试集评估结果 ".center(50, "="))
    logger.info("="*50)
    evaluate_subset(all_preds_orig, all_targets_orig, all_losses, metric_fns, logger)

    param_to_check = all_raw_params[:, grouping_config['param_index']]
    
    for range_name, (min_val, max_val) in grouping_config['ranges'].items():
        title = f" 分组评估: {range_name} ({grouping_config['param_name']}: [{min_val}, {max_val}]) "
        logger.info("\n" + "="*50)
        logger.info(title.center(50, "="))
        logger.info("="*50)
        
        # 找到在此区间的样本索引
        indices = np.where((param_to_check >= min_val) & (param_to_check <= max_val))[0]
        
        if len(indices) == 0:
            logger.info("该区间内无测试样本。")
            continue
            
        # 根据索引筛选子集
        subset_preds = all_preds_orig[indices]
        subset_targets = all_targets_orig[indices]
        subset_losses = all_losses[indices]
        
        evaluate_subset(subset_preds, subset_targets, subset_losses, metric_fns, logger, f"样本数: {len(indices)}")

    # --- 4. 对特定组合工况数据进行评估 ---
    for case_name, config_item in specific_case_config.items():
        title = f" 特定工况评估: {config_item['description']} "
        logger.info("\n" + "="*50)
        logger.info(title.center(50, "="))
        logger.info("="*50)

        if case_name == 'angle_gt15_and_full_or_samesign':
            # 手动构建复杂的布尔掩码
            
            # Condition A: 角度绝对值 > 15°
            param_angle = all_raw_params[:, 1]
            mask_A = np.abs(param_angle) > 15
            
            # Condition B: 重叠率 = 100% (即 |overlap| == 1.0)
            param_overlap = all_raw_params[:, 2]
            mask_B = np.isclose(np.abs(param_overlap), 1.0)
            
            # Condition C: 角度和重叠率同号
            mask_C = (param_angle * param_overlap) > 0
            
            # 最终逻辑: A AND (B OR C)
            combined_mask = mask_A & (mask_B | mask_C)

        else:
            # 其余工况均按“多个条件取交集”处理。
            combined_mask = np.full(all_raw_params.shape[0], True)

            for cond in config_item['conditions']:
                cond_type = cond['type']
                
                # 如果是占位符，则跳过
                if cond_type == 'custom_logic':
                    continue
                    
                param_index = cond['param_index']

                if cond_type == 'abs_range':
                    min_val, max_val = cond['range']
                    param_to_check = all_raw_params[:, param_index]
                    current_mask = (np.abs(param_to_check) >= min_val) & (np.abs(param_to_check) <= max_val)
                
                elif cond_type == 'range':
                    min_val, max_val = cond['range']
                    param_to_check = all_raw_params[:, param_index]
                    current_mask = (param_to_check >= min_val) & (param_to_check <= max_val)
                
                elif cond_type == 'same_sign':
                    other_param_index = cond['other_param_index']
                    param1_values = all_raw_params[:, param_index]
                    param2_values = all_raw_params[:, other_param_index]
                    current_mask = (param1_values * param2_values) > 0
                
                else:
                    logger.warning(f"未知的条件类型: {cond_type}，已跳过。")
                    continue
                
                combined_mask &= current_mask
            
        # 找到最终满足所有条件的样本索引
        indices = np.where(combined_mask)[0]

        if len(indices) == 0:
            logger.info("该特定工况下无测试样本。")
            continue

        # 根据索引筛选出数据子集
        subset_preds = all_preds_orig[indices]
        subset_targets = all_targets_orig[indices]
        subset_losses = all_losses[indices]
        
        # 调用评估函数计算并打印子集的指标
        evaluate_subset(subset_preds, subset_targets, subset_losses, metric_fns, logger, f"样本数: {len(indices)}")
    # ----------------------------------------------------

def plot_samples(data, batch_idx, pred_mean_orig, target_orig, case_ids, processor, config, logger, plot_iso_ratings=False):
    """
    为指定批次内的全部样本绘制预测与真实波形对比图。

    参数:
        processor: 用于输入工况反归一化的 UnifiedDataProcessor 实例。
    """
    num_samples_to_plot = data.shape[0]
    print(f"\nPlotting samples from batch {batch_idx}...")
    for j in range(num_samples_to_plot):
        # --- 从归一化输入中反算出原始工况参数 ---
        norm_params = data[j].cpu().numpy()
        proc_vals = processor.process_by_name(norm_params[None, :], ["impact_velocity", "impact_angle", "overlap"], inverse=True)[0]
        raw_vel, raw_ang, raw_ov = float(proc_vals[0]), float(proc_vals[1]), float(proc_vals[2])

        collision_params = {'vel': raw_vel, 'ang': raw_ang, 'ov': raw_ov}
        
        pred_sample = pred_mean_orig[j]
        target_sample = target_orig[j]
        sample_case_id = case_ids[j].item()
        
        iso_scores = None
        if plot_iso_ratings:
            # ISORating 需要 numpy array 作为输入
            pred_np = pred_sample.cpu().numpy()
            target_np = target_sample.cpu().numpy()
            
            # 分别计算三个通道的ISO Rating
            iso_x = ISORating(analyzed_signal=pred_np[0, :], reference_signal=target_np[0, :]).calculate()
            iso_y = ISORating(analyzed_signal=pred_np[1, :], reference_signal=target_np[1, :]).calculate()
            iso_z = ISORating(analyzed_signal=pred_np[2, :], reference_signal=target_np[2, :]).calculate()
            iso_scores = {'x': iso_x, 'y': iso_y, 'z': iso_z}

        # 使用被重定向后的 config.save_dir
        plot_waveform_comparison(
            pred_wave=pred_sample,
            true_wave=target_sample,
            params=collision_params,
            case_id=sample_case_id,
            epoch='test',
            batch_idx=batch_idx,
            sample_idx=j,
            save_dir=config.save_dir,
            iso_ratings=iso_scores # 将计算出的分数传递给绘图函数
        )
    logger.info(f"\n绘图结果已保存至 '{config.save_dir}' 目录下的 'fig' 子目录中。\n")

def evaluate_subset(preds, targets, losses, metric_fns, logger, header_info=None):
    """
    计算并打印给定数据子集的各项指标。
    """
    if header_info:
        logger.info(header_info)
        
    log = {'loss': np.mean(losses)}
    for met in metric_fns:
        log[met.__name__] = met(preds, targets)
    
    # 格式化输出
    for key, value in log.items():
        logger.info('    {:15s}: {}'.format(str(key), value))

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PulsePredict 测试入口')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to the checkpoint which to test (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
