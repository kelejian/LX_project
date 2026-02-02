import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = 'T'
import warnings
warnings.filterwarnings('ignore')
import optuna
import torch
import json
import numpy as np
from torch import optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from optuna.storages import RDBStorage
import joblib

from utils import models
from utils.dataset_prepare import CrashDataset
from utils.AIS_cal import AIS_cal_head, AIS_cal_chest, AIS_cal_neck
from utils.weighted_loss import weighted_loss
from utils.set_random_seed import set_random_seed
from utils.optimizer_utils import get_parameter_groups

# 沿用 train.py 中标准化的单轮训练/验证函数
def run_one_epoch(model, loader, criterion, device, optimizer=None):
    """
    执行一个完整的训练或验证周期。
    """
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    loss_batch = []
    all_preds, all_trues = [], []
    all_true_ais_head, all_true_ais_chest, all_true_ais_neck, all_true_mais = [], [], [], []
    all_ot = []

    with torch.set_grad_enabled(is_train):
        for batch in loader:
            (batch_x_acc, batch_x_att_continuous, batch_x_att_discrete,
             batch_y_HIC, batch_y_Dmax, batch_y_Nij,
             batch_ais_head, batch_ais_chest, batch_ais_neck, batch_y_MAIS,
             batch_OT) = [d.to(device) for d in batch]
            
            if is_train:
                optimizer.zero_grad()

            batch_y_true = torch.stack([batch_y_HIC, batch_y_Dmax, batch_y_Nij], dim=1)
            batch_pred, _, _ = model(batch_x_acc, batch_x_att_continuous, batch_x_att_discrete)
            loss = criterion(batch_pred, batch_y_true, batch_OT)

            if is_train:
                loss.backward()
                optimizer.step()

            loss_batch.append(loss.item())
            all_preds.append(batch_pred.detach().cpu().numpy())
            all_trues.append(batch_y_true.detach().cpu().numpy())
            all_true_ais_head.append(batch_ais_head.cpu().numpy())
            all_true_ais_chest.append(batch_ais_chest.cpu().numpy())
            all_true_ais_neck.append(batch_ais_neck.cpu().numpy())
            all_true_mais.append(batch_y_MAIS.cpu().numpy())
            all_ot.append(batch_OT.cpu().numpy())

    avg_loss = np.mean(loss_batch)
    ot = np.concatenate(all_ot)
    preds = np.concatenate(all_preds)
    pred_hic, pred_dmax, pred_nij = preds[:, 0], preds[:, 1], preds[:, 2]

    ais_head_pred, ais_chest_pred, ais_neck_pred = AIS_cal_head(pred_hic), AIS_cal_chest(pred_dmax, ot), AIS_cal_neck(pred_nij)
    true_ais_head = np.concatenate(all_true_ais_head)
    true_ais_chest = np.concatenate(all_true_ais_chest)
    true_ais_neck = np.concatenate(all_true_ais_neck)
    true_mais = np.concatenate(all_true_mais)
    mais_pred = np.maximum.reduce([ais_head_pred, ais_chest_pred, ais_neck_pred])

    metrics = {
        'loss': avg_loss,
        'accu_mais': accuracy_score(true_mais, mais_pred) * 100,
        'accu_head': accuracy_score(true_ais_head, ais_head_pred) * 100,
        'accu_chest': accuracy_score(true_ais_chest, ais_chest_pred) * 100,
        'accu_neck': accuracy_score(true_ais_neck, ais_neck_pred) * 100,
    }
    return metrics


def objective(trial):
    """定义Optuna的多目标优化函数"""
    # --- 超参数搜索空间 (与之前保持一致) ---
    Batch_size = trial.suggest_int("Batch_size", 384, 768, step=128)
    learning_rate = trial.suggest_float("learning_rate", 0.012, 0.04, log=True)
    # eta_min = trial.suggest_float("eta_min", 1e-8, 1e-3, log=True)
    eta_min = 1e-7
    weight_decay = trial.suggest_float("weight_decay", 2e-5, 2e-3, log=True)
    # weight_decay = 5e-4
    weight_factor_classify = trial.suggest_float("weight_factor_classify", 1.0, 2.0, step=0.1)
    # weight_factor_classify = 1.2
    weight_factor_sample = trial.suggest_float("weight_factor_sample", 0, 1.0, step=0.1)
    # weight_factor_sample = 0.1
    
    Ksize_init = trial.suggest_int("Ksize_init", 4, 10, step=2)
    # Ksize_init = 8
    # Ksize_mid = trial.suggest_categorical("Ksize_mid", [3, 5])
    Ksize_mid = trial.suggest_int("Ksize_mid", 3, 7, step=2)
    # Ksize_mid = 5
    # num_blocks_of_tcn = trial.suggest_int("num_blocks_of_tcn", 2, 5)
    # num_blocks_of_tcn = 3
    tcn_channels_list = trial.suggest_categorical("tcn_channels_list", [
        [64, 128, 256], [64, 96, 128, 160]
    ])
    num_layers_of_mlpE = trial.suggest_int("num_layers_of_mlpE", 3, 4)
    # num_layers_of_mlpE = 3
    num_layers_of_mlpD = trial.suggest_int("num_layers_of_mlpD", 3, 4)
    # num_layers_of_mlpD = 3
    mlpE_hidden = trial.suggest_int("mlpE_hidden", 160, 256, step=32)
    # mlpE_hidden = 224
    mlpD_hidden = trial.suggest_int("mlpD_hidden", 96, 192, step=32)
    # mlpD_hidden = 160
    tcn_output_dim = trial.suggest_int("tcn_output_dim", 96, 160, step=32)
    mlp_encoder_output_dim = trial.suggest_int("mlp_encoder_output_dim", 96, 128, step=32)
    mlp_decoder_output_dim = trial.suggest_int("mlp_decoder_output_dim", 96, 256, step=32)
    # dropout_MLP = trial.suggest_float("dropout_MLP", 0.1, 0.45, step=0.05)
    dropout_MLP = trial.suggest_float("dropout_MLP", 0.05, 0.3, step=0.05)
    # dropout_MLP = 0.15
    dropout_TCN = trial.suggest_float("dropout_TCN", 0.05, 0.25, step=0.05)
    # dropout_TCN = 0.15
    # loss_weights_head = trial.suggest_float("loss_weights_head", 0.20, 0.3, step=0.02)
    loss_weights_head = trial.suggest_float("loss_weights_head", 0.15, 0.3, step=0.05)
    loss_weights_neck = trial.suggest_float("loss_weights_neck", 15, 40, step=5)

    # 固定参数
    Epochs = 260 # Optuna中通常使用较少的Epochs进行快速评估
    base_loss = "mae"
    loss_weights = (loss_weights_head, 1.0, loss_weights_neck)

    
    # 加载数据集
    train_dataset = torch.load("./data/train_dataset.pt", weights_only=False)
    val_dataset = torch.load("./data/val_dataset.pt", weights_only=False)
    train_loader = DataLoader(train_dataset, batch_size=Batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=Batch_size, shuffle=False, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 构建模型
    model = models.InjuryPredictModel(
        num_classes_of_discrete=train_dataset.dataset.num_classes_of_discrete,
        Ksize_init=Ksize_init, Ksize_mid=Ksize_mid,
        # num_blocks_of_tcn=num_blocks_of_tcn,
        tcn_channels_list=tcn_channels_list,
        num_layers_of_mlpE=num_layers_of_mlpE, num_layers_of_mlpD=num_layers_of_mlpD,
        mlpE_hidden=mlpE_hidden, mlpD_hidden=mlpD_hidden,
        tcn_output_dim=tcn_output_dim, mlp_encoder_output_dim=mlp_encoder_output_dim, mlp_decoder_output_dim=mlp_decoder_output_dim,
        dropout_MLP=dropout_MLP, dropout_TCN=dropout_TCN
    ).to(device)

    # 定义损失函数
    criterion = weighted_loss(base_loss, weight_factor_classify, weight_factor_sample, loss_weights)
    # 优化器（参数分组管理）和学习率调度器
    param_groups = get_parameter_groups(model, weight_decay=weight_decay, head_decay_ratio=0.05,head_keywords=('head',))   
    optimizer = optim.AdamW(param_groups, lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Epochs, eta_min=eta_min)

    val_metrics_list = []
    # 训练与验证循环
    for epoch in range(Epochs):
        run_one_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        val_metrics = run_one_epoch(model, val_loader, criterion, device, optimizer=None)
        scheduler.step()

        # # Optuna剪枝逻辑 (基于验证集损失)
        # trial.report(val_metrics['loss'], epoch)
        # if trial.should_prune():
        #     raise optuna.TrialPruned()

        # 为减少随机性，记录最后20轮的验证指标
        if epoch >= Epochs - 20:
            val_metrics_list.append(val_metrics)

    # --- 计算并返回新的多目标优化值 ---
    avg_mais_acc = np.mean([m['accu_mais'] for m in val_metrics_list])
    avg_head_acc = np.mean([m['accu_head'] for m in val_metrics_list])
    avg_chest_acc = np.mean([m['accu_chest'] for m in val_metrics_list])
    avg_neck_acc = np.mean([m['accu_neck'] for m in val_metrics_list])

    return avg_mais_acc, avg_head_acc, avg_chest_acc, avg_neck_acc

if __name__ == "__main__":
    set_random_seed()
    study_file = "./runs/optuna_study_multiobj_acc.pkl"
    study_name = "multiobj_acc_optimization"
    db_path = "sqlite:///./runs/optuna_study.db"
    storage = RDBStorage(db_path)

    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
        print(f"成功加载研究 '{study_name}' 从数据库。")
    except Exception as e:
        print(f"加载研究失败: {e}")
        print("创建新的多目标（准确率）优化研究...")
        # --- 配置新的多目标优化方向 ---
        study = optuna.create_study(
            sampler=optuna.samplers.NSGAIISampler(),
            study_name=study_name,
            storage=storage,
            directions=["maximize", "maximize", "maximize", "maximize"] # 对应 MAIS, Head, Chest, Neck 准确率
        )

    print(f'请在命令行运行optuna-dashboard sqlite:///runs/optuna_study.db 来监控优化过程。')

    # 定义回调函数，定期保存研究结果
    def save_study_callback(study, trial):
        if trial.number % 5 == 0:
            joblib.dump(study, study_file)
            print(f"\n研究已在第 {trial.number} 次试验后保存至 {study_file}\n")

    # 运行优化
    try:
        study.optimize(objective, n_trials=1000, callbacks=[save_study_callback])
    except KeyboardInterrupt:
        print("用户中断了优化。")
    finally:
        joblib.dump(study, study_file)
        print(f"最终研究状态已保存至 {study_file}")

    # --- 打印新的Pareto前沿结果 ---
    print("\n" + "="*50)
    print("           Pareto Front Results (Accuracy-focused)")
    print("="*50)
    for trial in study.best_trials:
        print(f"Trial Number: {trial.number}")
        print(f"  - MAIS Acc:   {trial.values[0]:.2f}%")
        print(f"  - Head Acc:   {trial.values[1]:.2f}%")
        print(f"  - Chest Acc:  {trial.values[2]:.2f}%")
        print(f"  - Neck Acc:   {trial.values[3]:.2f}%")
        print(f"  - Hyperparameters: {json.dumps(trial.params, indent=4)}")
        print("-" * 50)