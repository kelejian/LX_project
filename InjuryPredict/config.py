# -*- coding: utf-8 -*-
"""
集中管理模型训练、损失函数和网络结构的可调超参数。
"""

RUNS_DIR = "./InjuryPredict/runs"  # 模型训练结果的保存目录

# 1. 优化与训练相关
training_params = {
    "Epochs": 500,
    "Batch_size": 64,
    "Learning_rate": 0.005,
    "Learning_rate_min": 0,
    "weight_decay": 0.1,
    "early_stop_start_epochs": 300, # 早停开始轮数
    "Patience": 60, # 早停轮数
}

# 2. 损失函数相关
loss_params = {
    "base_loss": "mae",
    "weight_factor_classify": 1.05,
    "weight_factor_sample": 0.2,
    "loss_weights": (0.1, 1.0, 10.0), # HIC, Dmax, Nij 各自损失的权重
}

# 3. 模型结构相关
model_params = {
    "Ksize_init": 8,
    "Ksize_mid": 3,
    "num_blocks_of_tcn": 4,
    "tcn_channels_list": [32, 64, 128],  # 每个 TCN 块的输出通道数
    "tcn_output_dim": 128,  # TCN 编码器的输出特征维度
    "num_layers_of_mlpE": 3,
    "num_layers_of_mlpD": 2,
    "mlpE_hidden": 256,
    "mlpD_hidden": 160,
    "mlp_encoder_output_dim": 128,  # MLP 编码器的输出特征维度
    "mlp_decoder_output_dim": 128,  # MLP 解码器的输出特征维度
    "dropout_MLP": 0.1,
    "dropout_TCN": 0.1,
    "use_channel_attention": True,  # 是否使用通道注意力机制
    "fixed_channel_weight": [0.7, 0.3],  # X, Y 通道的固定权重
}

# 仅用于“验证集指标跟踪”的可选指标名（train.py / train_KFold.py 通用）
# 说明：
# 1) 下列指标名默认都表示 val 指标，不是 train 指标。
# 2) val_metrics_to_track 中可写 "loss" 或 "val_loss"（其余指标同理），内部会统一按 val 指标处理。
AVAILABLE_VAL_METRIC_NAMES = (
    'loss',
    'accu_head', 'accu_chest', 'accu_neck', 'accu_mais', 'accu_mais_3c',
    'mae_hic', 'mae_dmax', 'mae_nij',
    'rmse_hic', 'rmse_dmax', 'rmse_nij',
    'r2_hic', 'r2_dmax', 'r2_nij',
)


val_metrics_to_track = [
    # (指标名, 比较方式)
    # 指标名均为“验证集指标”语义：推荐写 "val_loss" / "val_accu_mais" 更直观。
    # 比较方式仅支持 "max" / "min"。
    # ("val_accu_mais", "max"),
    ("val_loss", "min"),
]

# K-Fold 专项设置
kfold_params = {
    "K": 5, # K-Fold 折数
    # KFold 使用的验证集指标跟踪配置（与 train.py 同语义）
    "val_metrics_to_track": val_metrics_to_track,
}