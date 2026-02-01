# -*- coding: utf-8 -*-
import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = 'T'
import warnings
warnings.filterwarnings('ignore')
import os, json
import time
from datetime import datetime
import torch
import random
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, accuracy_score, r2_score

from utils import models
from utils.weighted_loss import weighted_loss
from utils.dataset_prepare import CrashDataset
from utils.AIS_cal import AIS_cal_head, AIS_cal_chest, AIS_cal_neck 
from utils.set_random_seed import set_random_seed, GLOBAL_SEED
from utils.optimizer_utils import get_parameter_groups

# --- 从 config.py 导入超参数 ---
from config import training_params, loss_params, model_params
set_random_seed()

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
    
    metrics = {
        'loss': avg_loss,
        'accu_head': accuracy_score(true_ais_head, ais_head_pred) * 100,
        'accu_chest': accuracy_score(true_ais_chest, ais_chest_pred) * 100,
        'accu_neck': accuracy_score(true_ais_neck, ais_neck_pred) * 100,
        'accu_mais': accuracy_score(true_mais, mais_pred) * 100,
        'mae_hic': mean_absolute_error(true_hic, pred_hic), 'rmse_hic': root_mean_squared_error(true_hic, pred_hic),
        'mae_dmax': mean_absolute_error(true_dmax, pred_dmax), 'rmse_dmax': root_mean_squared_error(true_dmax, pred_dmax),
        'mae_nij': mean_absolute_error(true_nij, pred_nij), 'rmse_nij': root_mean_squared_error(true_nij, pred_nij),
        'r2_hic': r2_score(true_hic, pred_hic),
        'r2_dmax': r2_score(true_dmax, pred_dmax),
        'r2_nij': r2_score(true_nij, pred_nij),
    }
    return metrics

def convert_numpy_types(obj):
        """递归转换NumPy类型为Python原生类型"""
        if isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_numpy_types(item) for item in obj)
        elif isinstance(obj, (np.integer, np.int_)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
        
if __name__ == "__main__":
    ''' 训练损伤预测模型 (TCN-based) 以进行多任务损伤预测 '''
    from torch.utils.tensorboard import SummaryWriter

    # 创建独立文件夹保存本次运行结果
    current_time = datetime.now().strftime("%m%d%H%M")
    run_dir = os.path.join("./runs", f"InjuryPredictModel_{current_time}")
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
    train_dataset = torch.load("./data/train_dataset.pt", weights_only=False)
    val_dataset = torch.load("./data/val_dataset.pt", weights_only=False)
    train_loader = DataLoader(train_dataset, batch_size=Batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=Batch_size, shuffle=False, num_workers=0)

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

    # 初始化跟踪变量
    val_loss_history, val_mais_accu_history, val_chest_accu_history, val_head_accu_history, val_neck_accu_history = [], [], [], [], []
    Best_val_loss = float('inf')
    Best_mais_accu, Best_chest_accu, Best_head_accu, Best_neck_accu = 0, 0, 0, 0
    Best_dmax_mae, Best_hic_mae, Best_nij_mae = float('inf'), float('inf'), float('inf')
    Best_hic_r2, Best_dmax_r2, Best_nij_r2 = float('-inf'), float('-inf'), float('-inf')
    best_loss_epoch, best_MAIS_accu_epoch, best_dmax_epoch, best_nij_epoch, best_chest_epoch, best_hic_epoch, best_head_epoch, best_neck_epoch = 0, 0, 0, 0, 0, 0, 0, 0
    best_hic_r2_epoch, best_dmax_r2_epoch, best_nij_r2_epoch = 0, 0, 0

    # 保存初始配置到 JSON 文件
    record_path = os.path.join(run_dir, "TrainingRecord.json")
    initial_record = {
        'GLOBAL_SEED': GLOBAL_SEED,
        "Trainset_size": len(train_dataset),
        "Valset_size": len(val_dataset),
        "model_params_count": {
            "total_params": total_params,
            "trainable_params": trainable_params
        },
        "hyperparameters": {
            "training": {
                "Epochs": Epochs, "Batch_size": Batch_size, "Learning_rate": Learning_rate,
                "Learning_rate_min": Learning_rate_min, "weight_decay": weight_decay,
                "Patience": Patience,
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

        # --- 调用统一的函数进行验证 ---
        val_metrics = run_one_epoch(model, val_loader, criterion, device, optimizer=None)

        # 记录验证epoch的权重统计
        val_attention_stats = None
        if use_channel_attention and hasattr(model.tcn, 'channel_attention'):
            mean_weights_val, std_weights_val = model.tcn.channel_attention.get_epoch_attention_stats()
            if mean_weights_val is not None:
                val_attention_stats = {
                    'mean': mean_weights_val, 
                    'std': std_weights_val
                }
        
        val_loss_history.append(val_metrics['loss'])
        val_mais_accu_history.append(val_metrics['accu_mais'])
        val_head_accu_history.append(val_metrics['accu_head'])
        val_chest_accu_history.append(val_metrics['accu_chest'])
        val_neck_accu_history.append(val_metrics['accu_neck'])

        print(f"Epoch {epoch+1}/{Epochs} | Train Loss: {train_metrics['loss']:.3f}")
        print(f"            | Val Loss: {val_metrics['loss']:.3f} | MAIS Acc: {val_metrics['accu_mais']:.2f}%")
        print(f"            | Head Acc: {val_metrics['accu_head']:.2f}%, Chest Acc: {val_metrics['accu_chest']:.2f}%, Neck Acc: {val_metrics['accu_neck']:.2f}%")
        print(f"            | R²: HIC={val_metrics['r2_hic']:.4f}, Dmax={val_metrics['r2_dmax']:.4f}, Nij={val_metrics['r2_nij']:.4f}")
        
        scheduler.step()

        # TensorBoard 记录 (训练)
        writer.add_scalar("Loss/Train", train_metrics['loss'], epoch)
        writer.add_scalar("Accuracy_Train/MAIS", train_metrics['accu_mais'], epoch)
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

        # TensorBoard 记录 (验证)
        writer.add_scalar("Loss/Val", val_metrics['loss'], epoch)
        writer.add_scalar("Accuracy_Val/MAIS", val_metrics['accu_mais'], epoch)
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
                    writer.add_histogram("ChannelAttention_Val/X_Direction_Distribution", epoch_weights[:, 0], epoch)
                    writer.add_histogram("ChannelAttention_Val/Y_Direction_Distribution", epoch_weights[:, 1], epoch)
                    # writer.add_histogram("ChannelAttention_Val/Z_Direction_Distribution", epoch_weights[:, 2], epoch)
            
            # 打印权重信息到控制台
            if epoch % 50 == 0 or epoch == Epochs - 1:
                # print(f"            | Val Channel Weights: X={mean_weights[0]:.3f}, Y={mean_weights[1]:.3f}, Z={mean_weights[2]:.3f}")
                print(f"            | Val Channel Weights: X={mean_weights[0]:.3f}, Y={mean_weights[1]:.3f}")
                print(f"            | Val Weight Variance: {weight_variance:.3f}")


        model_save_configs = [
            # (metric_key, best_var_name, epoch_var_name, filename, format_str, compare_func)
            ('loss', 'Best_val_loss', 'best_loss_epoch', 'best_val_loss.pth', 'Val Loss: {:.3f}', min),
            ('accu_mais', 'Best_mais_accu', 'best_MAIS_accu_epoch', 'best_mais_accu.pth', 'MAIS accuracy: {:.2f}%', max),
            ('mae_dmax', 'Best_dmax_mae', 'best_dmax_epoch', 'best_dmax_mae.pth', 'Dmax MAE: {:.3f}', min),
            ('accu_chest', 'Best_chest_accu', 'best_chest_epoch', 'best_chest_accu.pth', 'Chest Acc: {:.2f}%', max),
            ('mae_hic', 'Best_hic_mae', 'best_hic_epoch', 'best_hic_mae.pth', 'HIC MAE: {:.3f}', min),
            ('accu_head', 'Best_head_accu', 'best_head_epoch', 'best_head_accu.pth', 'Head Acc: {:.2f}%', max),
            ('mae_nij', 'Best_nij_mae', 'best_nij_epoch', 'best_nij_mae.pth', 'Nij MAE: {:.3f}', min),
            ('accu_neck', 'Best_neck_accu', 'best_neck_epoch', 'best_neck_accu.pth', 'Neck Acc: {:.2f}%', max),
            ('r2_hic', 'Best_hic_r2', 'best_hic_r2_epoch', 'best_hic_r2.pth', 'HIC R²: {:.4f}', max),
            ('r2_dmax', 'Best_dmax_r2', 'best_dmax_r2_epoch', 'best_dmax_r2.pth', 'Dmax R²: {:.4f}', max),
            ('r2_nij', 'Best_nij_r2', 'best_nij_r2_epoch', 'best_nij_r2.pth', 'Nij R²: {:.4f}', max),
        ]
        
        for metric_key, best_var, epoch_var, filename, format_str, compare_func in model_save_configs:
            current_value = val_metrics[metric_key]
            best_value = globals()[best_var]
            
            is_better = (compare_func == max and current_value > best_value) or \
                       (compare_func == min and current_value < best_value)
            
            if is_better:
                globals()[best_var] = current_value
                globals()[epoch_var] = epoch + 1
                torch.save(model.state_dict(), os.path.join(run_dir, filename))
                print(f"Best model saved with val {format_str.format(current_value)} at epoch {epoch+1}")

        # 早停逻辑
        if epoch > Epochs * 0.4 and len(val_loss_history) >= Patience:
            # 检查最佳指标是否在最近 Patience 个 epoch 之外
            epochs_since_best_loss = epoch + 1 - best_loss_epoch
            epochs_since_best_mais = epoch + 1 - best_MAIS_accu_epoch
            epochs_since_best_head = epoch + 1 - best_head_epoch
            
            if (epochs_since_best_loss >= Patience and 
                epochs_since_best_mais >= Patience and 
                epochs_since_best_head >= Patience):
                print(f"Early Stop at epoch: {epoch+1}!")
                print(f"Best Val Loss: {Best_val_loss:.3f} (at epoch {best_loss_epoch})")
                print(f"Best MAIS accuracy: {Best_mais_accu:.2f}% (at epoch {best_MAIS_accu_epoch})")
                print(f"Best Head accuracy: {Best_head_accu:.2f}% (at epoch {best_head_epoch})")
                break

        print(f"            | Time: {time.time()-epoch_start_time:.2f}s")

    # 保存最后的模型
    torch.save(model.state_dict(), os.path.join(run_dir, "final_model.pth"))
    print("Final model saved.")

    writer.close()

    # --- 最终记录训练结果 (加载、更新、保存) ---
    print("训练完成，正在加载初始记录并添加训练结果...")
    
    # 1. 定义训练结果
    training_results = {
        "final_epoch": epoch + 1,
        "best_mais_accuracy": np.round(float(Best_mais_accu), 2),
        "best_mais_accuracy_epoch": int(best_MAIS_accu_epoch),
        "best_chest_accuracy": np.round(float(Best_chest_accu), 2),
        "best_chest_accuracy_epoch": int(best_chest_epoch),
        "best_head_accuracy": np.round(float(Best_head_accu), 2),
        "best_head_accuracy_epoch": int(best_head_epoch),
        "best_neck_accuracy": np.round(float(Best_neck_accu), 2),
        "best_neck_accuracy_epoch": int(best_neck_epoch),
        "best_dmax_mae": np.round(float(Best_dmax_mae), 2),
        "best_dmax_mae_epoch": int(best_dmax_epoch),
        "best_hic_mae": np.round(float(Best_hic_mae), 2),
        "best_hic_mae_epoch": int(best_hic_epoch),
        "best_nij_mae": np.round(float(Best_nij_mae), 3),
        "best_nij_mae_epoch": int(best_nij_epoch),
        "best_hic_r2": np.round(float(Best_hic_r2), 4),
        "best_hic_r2_epoch": int(best_hic_r2_epoch),
        "best_dmax_r2": np.round(float(Best_dmax_r2), 4),
        "best_dmax_r2_epoch": int(best_dmax_r2_epoch),
        "best_nij_r2": np.round(float(Best_nij_r2), 4),
        "best_nij_r2_epoch": int(best_nij_r2_epoch),

        "lowest_val_loss": np.round(float(Best_val_loss), 3),
        "lowest_val_loss_epoch": int(best_loss_epoch),

        "last_epoch_metrics": {
            "val_loss": np.round(float(val_metrics['loss']), 3),
            "accu_mais": np.round(float(val_metrics['accu_mais']), 2),
            "accu_head": np.round(float(val_metrics['accu_head']), 2),
            "accu_chest": np.round(float(val_metrics['accu_chest']), 2),
            "accu_neck": np.round(float(val_metrics['accu_neck']), 2),
            "mae_hic": np.round(float(val_metrics['mae_hic']), 2),
            "mae_dmax": np.round(float(val_metrics['mae_dmax']), 2),
            "mae_nij": np.round(float(val_metrics['mae_nij']), 3),
            "r2_hic": np.round(float(val_metrics['r2_hic']), 4),
            "r2_dmax": np.round(float(val_metrics['r2_dmax']), 4),
            "r2_nij": np.round(float(val_metrics['r2_nij']), 4),
        }
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