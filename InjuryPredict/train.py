# -*- coding: utf-8 -*-
import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = 'T'
import warnings
warnings.filterwarnings('ignore')
import os, json
import time
from datetime import datetime
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, accuracy_score, r2_score

from common.metrics.injury_risk import AIS_cal_head, AIS_cal_chest, AIS_cal_neck
from common.tools.seeding import set_random_seed, GLOBAL_SEED
from common.settings import INJURY_PROCESSED_DIR, get_injury_processed_dataset_path

from InjuryPredict.utils import models
from InjuryPredict.Injurydata_prepare import InjuryPackedDataset, load_processed_subset
from InjuryPredict.utils.loss import weighted_loss
from InjuryPredict.utils.tools import get_parameter_groups, build_metric_trackers, round_to_significant, round_float_fields, convert_numpy_types, get_mais_3c_metrics
from InjuryPredict.config import RUNS_DIR, training_params, loss_params, model_params, val_metrics_to_track

# --- 合并 train 和 valid 为一个函数 ---
def run_one_epoch(model, loader, criterion, device, optimizer=None):
    """
    执行一个完整的训练或验证周期。
    如果提供了 optimizer，则为训练模式；否则为验证模式。

    参数:
        model: 模型实例。
        loader: 数据加载器。
        criterion: 损失函数。
        device: GPU 或 CPU。
        optimizer (optional): 优化器。默认为 None。

    返回:
        metrics (dict): 包含该周期所有指标的字典。
    """
    # 根据是否存在 optimizer 设置模式
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    loss_batch = []
    all_preds, all_trues = [], []
    all_true_ais_head, all_true_ais_chest, all_true_ais_neck = [], [], []
    all_true_mais = []
    all_ot = []
    
    # 根据模式选择是否启用梯度计算
    with torch.set_grad_enabled(is_train):
        for batch in loader:
            (batch_x_acc, batch_x_att_continuous, batch_x_att_discrete,
             batch_y_HIC, batch_y_Dmax, batch_y_Nij,
             batch_ais_head, batch_ais_chest, batch_ais_neck, batch_y_MAIS,
             batch_OT) = [d.to(device) for d in batch]
            
            if is_train:
                optimizer.zero_grad()

            batch_y_true = torch.stack([batch_y_HIC, batch_y_Dmax, batch_y_Nij], dim=1)

            # 前向传播
            batch_pred, _, _ = model(batch_x_acc, batch_x_att_continuous, batch_x_att_discrete)

            # 计算损失
            loss = criterion(batch_pred, batch_y_true, batch_OT)

            # 如果是训练模式，则执行反向传播和优化
            if is_train:
                loss.backward()
                optimizer.step()

            # 记录损失和用于计算指标的值
            loss_batch.append(loss.item())
            all_preds.append(batch_pred.detach().cpu().numpy())
            all_trues.append(batch_y_true.detach().cpu().numpy())
            all_true_ais_head.append(batch_ais_head.cpu().numpy())
            all_true_ais_chest.append(batch_ais_chest.cpu().numpy())
            all_true_ais_neck.append(batch_ais_neck.cpu().numpy())
            all_true_mais.append(batch_y_MAIS.cpu().numpy())
            all_ot.append(batch_OT.cpu().numpy())

    if not loss_batch:
        raise ValueError("DataLoader 为空，无法执行一个完整的 epoch。")

    # --- 指标计算部分 ---
    avg_loss = np.mean(loss_batch)
    ot = np.concatenate(all_ot)
    preds, trues = np.concatenate(all_preds), np.concatenate(all_trues)
    pred_hic, pred_dmax, pred_nij = preds[:, 0], preds[:, 1], preds[:, 2]
    true_hic, true_dmax, true_nij = trues[:, 0], trues[:, 1], trues[:, 2]
    
    ais_head_pred, ais_chest_pred, ais_neck_pred = AIS_cal_head(pred_hic), AIS_cal_chest(pred_dmax, ot), AIS_cal_neck(pred_nij)
    true_ais_head, true_ais_chest, true_ais_neck = np.concatenate(all_true_ais_head), np.concatenate(all_true_ais_chest), np.concatenate(all_true_ais_neck)
    true_mais = np.concatenate(all_true_mais)
    mais_pred = np.maximum.reduce([ais_head_pred, ais_chest_pred, ais_neck_pred])
    mais_metrics_3c = get_mais_3c_metrics(true_mais, mais_pred)
    
    metrics = {
        'loss': avg_loss,
        'accu_head': accuracy_score(true_ais_head, ais_head_pred) * 100,
        'accu_chest': accuracy_score(true_ais_chest, ais_chest_pred) * 100,
        'accu_neck': accuracy_score(true_ais_neck, ais_neck_pred) * 100,
        'accu_mais': accuracy_score(true_mais, mais_pred) * 100,
        'accu_mais_3c': mais_metrics_3c['accuracy'],
        'mae_hic': mean_absolute_error(true_hic, pred_hic), 'rmse_hic': root_mean_squared_error(true_hic, pred_hic),
        'mae_dmax': mean_absolute_error(true_dmax, pred_dmax), 'rmse_dmax': root_mean_squared_error(true_dmax, pred_dmax),
        'mae_nij': mean_absolute_error(true_nij, pred_nij), 'rmse_nij': root_mean_squared_error(true_nij, pred_nij),
        'r2_hic': r2_score(true_hic, pred_hic),
        'r2_dmax': r2_score(true_dmax, pred_dmax),
        'r2_nij': r2_score(true_nij, pred_nij),
    }
    return metrics

if __name__ == "__main__":
    set_random_seed()
    ''' 训练损伤预测模型 (TCN-based) 以进行多任务损伤预测 '''
    from torch.utils.tensorboard import SummaryWriter

    # 创建独立文件夹保存本次运行结果
    current_time = datetime.now().strftime("%m%d%H%M")
    run_dir = os.path.join(RUNS_DIR, f"InjuryPredictModel_{current_time}")
    os.makedirs(run_dir, exist_ok=True)

    # 初始化 TensorBoard
    writer = SummaryWriter(log_dir=run_dir)
    
    ############################################################################################
    ############################################################################################
    # --- 从导入的配置中加载超参数 ---
    # 1. 优化与训练相关
    Epochs = training_params['Epochs']
    Batch_size = training_params['Batch_size']
    Learning_rate = training_params['Learning_rate']
    Learning_rate_min = training_params['Learning_rate_min']
    weight_decay = training_params['weight_decay']
    early_stop_start_epochs = training_params['early_stop_start_epochs']
    Patience = training_params['Patience']
    
    # 2. 损失函数相关
    base_loss = loss_params['base_loss']
    weight_factor_classify = loss_params['weight_factor_classify']
    weight_factor_sample = loss_params['weight_factor_sample']
    loss_weights = loss_params['loss_weights']

    # 3. 模型结构相关
    Ksize_init = model_params['Ksize_init']
    Ksize_mid = model_params['Ksize_mid']
    num_blocks_of_tcn = model_params['num_blocks_of_tcn']
    tcn_channels_list = model_params['tcn_channels_list']
    num_layers_of_mlpE = model_params['num_layers_of_mlpE']
    num_layers_of_mlpD = model_params['num_layers_of_mlpD']
    mlpE_hidden = model_params['mlpE_hidden']
    mlpD_hidden = model_params['mlpD_hidden']
    tcn_output_dim = model_params['tcn_output_dim']
    mlp_encoder_output_dim = model_params['mlp_encoder_output_dim']
    mlp_decoder_output_dim = model_params['mlp_decoder_output_dim']
    dropout_MLP = model_params['dropout_MLP']
    dropout_TCN = model_params['dropout_TCN']
    use_channel_attention = model_params['use_channel_attention'] # 是否使用通道注意力机制
    fixed_channel_weight = model_params['fixed_channel_weight'] # 固定的通道注意力权重(None表示自适应学习)
    ############################################################################################
    ############################################################################################
    if Patience > Epochs: Patience = Epochs

    # 加载数据集对象
    print(f".pt 数据文件路径: {INJURY_PROCESSED_DIR}/*.pt")
    train_pt = get_injury_processed_dataset_path("train")
    val_pt = get_injury_processed_dataset_path("val")
    if not train_pt.exists():
        raise FileNotFoundError(
            f"找不到训练数据 ({train_pt})。请先运行: python -m InjuryPredict.Injurydata_prepare"
        )
    if not val_pt.exists():
        raise FileNotFoundError(
            f"找不到验证数据 ({val_pt})。请先运行: python -m InjuryPredict.Injurydata_prepare"
        )
    train_dataset = load_processed_subset(train_pt)
    if len(train_dataset) == 0:
        raise ValueError("train_dataset.pt 为空，InjuryPredict.train 不支持空训练集。")
    print(f"训练集大小: {len(train_dataset)}")
    val_dataset = load_processed_subset(val_pt)
    print(f"验证集大小: {len(val_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=Batch_size, shuffle=True, num_workers=0)
    val_enabled = len(val_dataset) > 0
    val_loader = DataLoader(val_dataset, batch_size=Batch_size, shuffle=False, num_workers=0) if val_enabled else None
    if not val_enabled:
        print("警告: 验证集为空，本次训练将跳过验证、best_val_* 权重保存和 early stop。")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    model = models.InjuryPredictModel(
        Ksize_init=Ksize_init,
        Ksize_mid=Ksize_mid,
        num_classes_of_discrete=train_dataset.dataset.num_classes_of_discrete, # 从加载的训练集中获取元数据
        tcn_channels_list=tcn_channels_list,
        num_layers_of_mlpE=num_layers_of_mlpE,
        num_layers_of_mlpD=num_layers_of_mlpD,
        mlpE_hidden=mlpE_hidden,
        mlpD_hidden=mlpD_hidden,
        tcn_output_dim=tcn_output_dim,
        mlp_encoder_output_dim=mlp_encoder_output_dim,
        mlp_decoder_output_dim=mlp_decoder_output_dim,
        dropout_MLP=dropout_MLP,
        dropout_TCN=dropout_TCN,
        use_channel_attention=use_channel_attention,
        fixed_channel_weight=fixed_channel_weight
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # 打印模型结构和参数信息
    print(model)
    print(f"模型总参数量: {total_params}, 可训练参数量: {trainable_params}")

    # 定义损失函数
    criterion = weighted_loss(base_loss, weight_factor_classify, weight_factor_sample, loss_weights)
    # 优化器（参数分组管理）和学习率调度器
    param_groups = get_parameter_groups(model, weight_decay=weight_decay, head_decay_ratio=0.05,head_keywords=('head',))   
    optimizer = optim.AdamW(param_groups, lr=Learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Epochs, eta_min=Learning_rate_min)

    # 初始化指标跟踪器（由 val_metrics_to_track 驱动）
    metric_trackers = {}
    metric_states = {}
    if val_enabled:
        metric_trackers = build_metric_trackers(
            val_metrics_to_track,
            model_filename_fn = lambda metric_name: f"best_val_{metric_name}.pth"
        )
        if not metric_trackers:
            raise ValueError("val_metrics_to_track 不能为空。")
        tracked_metric_names = [tracker['display_name'] for tracker in metric_trackers.values()]
        print(f"将跟踪以下验证指标: {tracked_metric_names}")
        metric_states = {
            metric_name: {
                'best_value': tracker_info['initial_value'],
                'best_epoch': 0,
                'is_better': tracker_info['is_better'],
                'model_filename': tracker_info['model_filename'],
            }
            for metric_name, tracker_info in metric_trackers.items()
        }

    # 保存初始配置到 JSON 文件
    record_path = os.path.join(run_dir, "TrainingRecord.json")
    initial_record = {
        'GLOBAL_SEED': GLOBAL_SEED,
        "Trainset_size": len(train_dataset),
        "Valset_size": len(val_dataset),
        "validation_enabled": val_enabled,
        "INJURY_PROCESSED_DIR": str(INJURY_PROCESSED_DIR),
        "data_interface": {
            "processed_dir": str(INJURY_PROCESSED_DIR.resolve()),
            "default_entrypoint_rule": (
                "train.py 在未显式传入 processed_dir 时，默认通过 "
                "common.settings.INJURY_PROCESSED_DIR 读取 processed .pt 子集。"
            ),
        },
        "model_params_count": {
            "total_params": total_params,
            "trainable_params": trainable_params
        },
        "hyperparameters": {
            "training": {
                "Epochs": Epochs, "Batch_size": Batch_size, "Learning_rate": Learning_rate,
                "Learning_rate_min": Learning_rate_min, "weight_decay": weight_decay,
                "early_stop_start_epochs": early_stop_start_epochs, "Patience": Patience,
                "val_metrics_to_track": val_metrics_to_track if val_enabled else [],
            },
            "loss": {
                "base_loss": base_loss, "weight_factor_classify": weight_factor_classify,
                "weight_factor_sample": weight_factor_sample, "loss_weights": loss_weights,
            },
            "model": {
                "Ksize_init": Ksize_init, "Ksize_mid": Ksize_mid, "num_blocks_of_tcn": num_blocks_of_tcn,
                "tcn_channels_list": tcn_channels_list,
                "num_layers_of_mlpE": num_layers_of_mlpE, "num_layers_of_mlpD": num_layers_of_mlpD,
                "mlpE_hidden": mlpE_hidden, "mlpD_hidden": mlpD_hidden,
                "tcn_output_dim": tcn_output_dim, "mlp_encoder_output_dim": mlp_encoder_output_dim, "mlp_decoder_output_dim": mlp_decoder_output_dim,
                "dropout_MLP": dropout_MLP, "dropout_TCN": dropout_TCN,
                "use_channel_attention": use_channel_attention,
                "fixed_channel_weight": fixed_channel_weight,
                "num_classes_of_discrete": train_dataset.dataset.num_classes_of_discrete
            }
        }
    }
    # 转换Numpy类型并保存
    initial_record = convert_numpy_types(initial_record)
    with open(record_path, "w") as f:
        json.dump(initial_record, f, indent=4)
    print(f"初始配置已保存至: {record_path}")

    # 主训练循环
    for epoch in range(Epochs):
        epoch_start_time = time.time()

        # 重置通道注意力权重记录（训练开始前）
        if use_channel_attention and hasattr(model.tcn, 'channel_attention'):
            model.tcn.channel_attention.reset_epoch_records()

        # --- 调用统一的函数进行训练 ---
        train_metrics = run_one_epoch(model, train_loader, criterion, device, optimizer=optimizer)

        # 记录训练epoch的权重统计
        train_attention_stats = None
        if use_channel_attention and hasattr(model.tcn, 'channel_attention'):
            mean_weights_train, std_weights_train = model.tcn.channel_attention.get_epoch_attention_stats()
            if mean_weights_train is not None:
                train_attention_stats = {
                    'mean': mean_weights_train, 
                    'std': std_weights_train
                }
            # 重置权重记录（验证开始前）
            model.tcn.channel_attention.reset_epoch_records()

        val_metrics = None
        val_attention_stats = None
        if val_enabled:
            # 验证集允许为空；为空时这里直接跳过，不再构造伪验证指标。
            val_metrics = run_one_epoch(model, val_loader, criterion, device, optimizer=None)

            if use_channel_attention and hasattr(model.tcn, 'channel_attention'):
                mean_weights_val, std_weights_val = model.tcn.channel_attention.get_epoch_attention_stats()
                if mean_weights_val is not None:
                    val_attention_stats = {
                        'mean': mean_weights_val,
                        'std': std_weights_val
                    }

            missing_metrics = [name for name in metric_trackers.keys() if name not in val_metrics]
            if missing_metrics:
                raise KeyError(f"val_metrics_to_track 中存在无效指标: {missing_metrics}")

        print(f"Epoch {epoch+1}/{Epochs} | Train Loss: {train_metrics['loss']:.4g}")
        if val_enabled:
            print(f"            | Val Loss: {val_metrics['loss']:.4g} | MAIS Acc 6C: {val_metrics['accu_mais']:.4g}% | MAIS Acc 3C: {val_metrics['accu_mais_3c']:.4g}%")
            print(f"            | Head Acc: {val_metrics['accu_head']:.4g}%, Chest Acc: {val_metrics['accu_chest']:.4g}%, Neck Acc: {val_metrics['accu_neck']:.4g}%")
            print(f"            | R2: HIC={val_metrics['r2_hic']:.4g}, Dmax={val_metrics['r2_dmax']:.4g}, Nij={val_metrics['r2_nij']:.4g}")
        else:
            print("            | Validation disabled because val_dataset.pt is empty")
        
        scheduler.step()

        # TensorBoard 记录 (训练)
        writer.add_scalar("Loss/Train", train_metrics['loss'], epoch)
        writer.add_scalar("Accuracy_Train/MAIS", train_metrics['accu_mais'], epoch)
        writer.add_scalar("Accuracy_Train/MAIS_3C", train_metrics['accu_mais_3c'], epoch)
        writer.add_scalar("Accuracy_Train/Head", train_metrics['accu_head'], epoch)
        writer.add_scalar("Accuracy_Train/Chest", train_metrics['accu_chest'], epoch)
        writer.add_scalar("Accuracy_Train/Neck", train_metrics['accu_neck'], epoch)
        writer.add_scalar("MAE_Train/Train_HIC", train_metrics['mae_hic'], epoch)
        writer.add_scalar("MAE_Train/Train_Dmax", train_metrics['mae_dmax'], epoch)
        writer.add_scalar("MAE_Train/Train_Nij", train_metrics['mae_nij'], epoch)
        writer.add_scalar("R2_Train/HIC", train_metrics['r2_hic'], epoch)
        writer.add_scalar("R2_Train/Dmax", train_metrics['r2_dmax'], epoch)
        writer.add_scalar("R2_Train/Nij", train_metrics['r2_nij'], epoch)

        # TensorBoard 记录训练时的通道注意力权重
        if train_attention_stats is not None:
            mean_weights = train_attention_stats['mean']
            std_weights = train_attention_stats['std']
            
            writer.add_scalar("ChannelAttention_Train/X_Direction_MeanWeight", mean_weights[0], epoch)
            writer.add_scalar("ChannelAttention_Train/Y_Direction_MeanWeight", mean_weights[1], epoch)
            # writer.add_scalar("ChannelAttention_Train/Z_Direction_MeanWeight", mean_weights[2], epoch)

            writer.add_scalar("ChannelAttention_Train/X_Direction_Std", std_weights[0], epoch)
            writer.add_scalar("ChannelAttention_Train/Y_Direction_Std", std_weights[1], epoch)
            # writer.add_scalar("ChannelAttention_Train/Z_Direction_Std", std_weights[2], epoch)
            
            weight_variance = torch.var(mean_weights).item()
            writer.add_scalar("ChannelAttention_Train/Weight_Variance", weight_variance, epoch)
            
            total_weight = torch.sum(mean_weights).item()
            x_ratio = (mean_weights[0] / total_weight).item() if total_weight > 0 else 0
            writer.add_scalar("ChannelAttention_Train/X_Direction_Ratio", x_ratio, epoch)

        if val_enabled:
            # TensorBoard 记录 (验证)
            writer.add_scalar("Loss/Val", val_metrics['loss'], epoch)
            writer.add_scalar("Accuracy_Val/MAIS", val_metrics['accu_mais'], epoch)
            writer.add_scalar("Accuracy_Val/MAIS_3C", val_metrics['accu_mais_3c'], epoch)
            writer.add_scalar("Accuracy_Val/Head", val_metrics['accu_head'], epoch)
            writer.add_scalar("Accuracy_Val/Chest", val_metrics['accu_chest'], epoch)
            writer.add_scalar("Accuracy_Val/Neck", val_metrics['accu_neck'], epoch)
            writer.add_scalar("MAE_Val/HIC", val_metrics['mae_hic'], epoch)
            writer.add_scalar("MAE_Val/Dmax", val_metrics['mae_dmax'], epoch)
            writer.add_scalar("MAE_Val/Nij", val_metrics['mae_nij'], epoch)
            writer.add_scalar("R2_Val/HIC", val_metrics['r2_hic'], epoch)
            writer.add_scalar("R2_Val/Dmax", val_metrics['r2_dmax'], epoch)
            writer.add_scalar("R2_Val/Nij", val_metrics['r2_nij'], epoch)

        # TensorBoard 记录验证时的通道注意力权重
        if val_attention_stats is not None:
            mean_weights = val_attention_stats['mean']
            std_weights = val_attention_stats['std']
            
            writer.add_scalar("ChannelAttention_Val/X_Direction_Weight", mean_weights[0], epoch)
            writer.add_scalar("ChannelAttention_Val/Y_Direction_Weight", mean_weights[1], epoch)
            # writer.add_scalar("ChannelAttention_Val/Z_Direction_Weight", mean_weights[2], epoch)
            
            writer.add_scalar("ChannelAttention_Val/X_Direction_Std", std_weights[0], epoch)
            writer.add_scalar("ChannelAttention_Val/Y_Direction_Std", std_weights[1], epoch)
            # writer.add_scalar("ChannelAttention_Val/Z_Direction_Std", std_weights[2], epoch)
            
            weight_variance = torch.var(mean_weights).item()
            writer.add_scalar("ChannelAttention_Val/Weight_Variance", weight_variance, epoch)
            
            total_weight = torch.sum(mean_weights).item()
            x_ratio = (mean_weights[0] / total_weight).item() if total_weight > 0 else 0
            writer.add_scalar("ChannelAttention_Val/X_Direction_Ratio", x_ratio, epoch)
            
            # 每10个epoch记录一次权重分布直方图
            if epoch % 10 == 0:
                epoch_weights = model.tcn.channel_attention.get_epoch_attention_weights()
                if epoch_weights is not None:
                    writer.add_histogram("ChannelAttention_Val/X_Direction_Distri", epoch_weights[:, 0], epoch)
                    writer.add_histogram("ChannelAttention_Val/Y_Direction_Distri", epoch_weights[:, 1], epoch)
                    # writer.add_histogram("ChannelAttention_Val/Z_Direction_Distri", epoch_weights[:, 2], epoch)
            
            # 打印权重信息到控制台
            if epoch % 50 == 0 or epoch == Epochs - 1:
                # print(f"            | Val Channel Weights: X={mean_weights[0]:.3f}, Y={mean_weights[1]:.3f}, Z={mean_weights[2]:.3f}")
                print(f"            | Val Channel Weights: X={mean_weights[0]:.4g}, Y={mean_weights[1]:.4g}")
                print(f"            | Val Weight Variance: {weight_variance:.4g}")


        if val_enabled:
            for metric_name, state in metric_states.items():
                current_value = val_metrics[metric_name]
                if state['is_better'](current_value, state['best_value']):
                    state['best_value'] = current_value
                    state['best_epoch'] = epoch + 1
                    torch.save(model.state_dict(), os.path.join(run_dir, state['model_filename']))
                    print(f"Best {metric_trackers[metric_name]['display_name']} model saved: {current_value:.4g} at epoch {epoch+1}")

        # 早停逻辑
        if val_enabled and epoch > early_stop_start_epochs and (epoch + 1) >= Patience:
            all_stagnant = all(
                (epoch + 1 - state['best_epoch']) >= Patience
                for state in metric_states.values()
            )
            if all_stagnant:
                print(f"Early Stop at epoch: {epoch+1}!")
                for metric_name, state in metric_states.items():
                    print(f"Best {metric_trackers[metric_name]['display_name']}: {state['best_value']:.4g} (at epoch {state['best_epoch']})")
                break

        print(f"            | Time: {time.time()-epoch_start_time:.2f}s")

    # 保存最后的模型
    torch.save(model.state_dict(), os.path.join(run_dir, "final_model.pth"))
    print("Final model saved.")

    writer.close()

    # --- 最终记录训练结果 (加载、更新、保存) ---
    print("训练完成，正在加载初始记录并添加训练结果...")
    
    # 1. 定义训练结果
    best_metrics_by_tracker = {
        metric_name: {
            "best_value": round_to_significant(float(state['best_value']), 4),
            "best_epoch": int(state['best_epoch']),
            "model_file": state['model_filename'],
        }
        for metric_name, state in metric_states.items()
    }

    metrics_source = val_metrics if val_enabled else train_metrics
    last_epoch_metrics = round_float_fields({
        "loss": float(metrics_source['loss']),
        "accu_mais": float(metrics_source['accu_mais']),
        "accu_mais_3c": float(metrics_source['accu_mais_3c']),
        "accu_head": float(metrics_source['accu_head']),
        "accu_chest": float(metrics_source['accu_chest']),
        "accu_neck": float(metrics_source['accu_neck']),
        "mae_hic": float(metrics_source['mae_hic']),
        "mae_dmax": float(metrics_source['mae_dmax']),
        "mae_nij": float(metrics_source['mae_nij']),
        "r2_hic": float(metrics_source['r2_hic']),
        "r2_dmax": float(metrics_source['r2_dmax']),
        "r2_nij": float(metrics_source['r2_nij']),
    }, digits=4)

    training_results = {
        "final_epoch": epoch + 1,
        "validation_enabled": val_enabled,
        "best_metrics_by_tracker": best_metrics_by_tracker,
        "last_epoch_metrics": last_epoch_metrics,
        "last_epoch_metrics_source": "val" if val_enabled else "train",
    }
    
    # 2. 加载现有记录
    try:
        with open(record_path, "r") as f:
            final_record = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"警告: 未找到或无法解析 {record_path}。将创建一个新的记录文件。")
        # 从 initial_record 重新构建，以防万一
        final_record = initial_record 

    # 3. 添加新结果并转换类型
    final_record['results'] = convert_numpy_types(training_results)
    
    # 4. 覆盖保存
    with open(record_path, "w") as f:
        json.dump(final_record, f, indent=4)
    
    print(f"训练结果已更新至: {record_path}")
