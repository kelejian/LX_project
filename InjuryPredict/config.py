# -*- coding: utf-8 -*-
"""
集中管理模型训练、损失函数和网络结构的可调超参数。
"""

# 1. 优化与训练相关
training_params = {
    "Epochs": 360,
    "Batch_size": 64,
    "Learning_rate": 0.005,
    "Learning_rate_min": 0,
    "weight_decay": 0.1,
    "Patience": 75, # 早停轮数
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
    "num_blocks_of_tcn": 3,
    "tcn_channels_list": [32, 64, 128],  # 每个 TCN 块的输出通道数
    "tcn_output_dim": 128,  # TCN 编码器的输出特征维度
    "num_layers_of_mlpE": 3,
    "num_layers_of_mlpD": 2,
    "mlpE_hidden": 256,
    "mlpD_hidden": 128,
    "mlp_encoder_output_dim": 128,  # MLP 编码器的输出特征维度
    "mlp_decoder_output_dim": 128,  # MLP 解码器的输出特征维度
    "dropout_MLP": 0.1,
    "dropout_TCN": 0.05,
    "use_channel_attention": True,  # 是否使用通道注意力机制
    "fixed_channel_weight": [0.7, 0.3],  # X, Y 通道的固定权重
}

# K-Fold 专项设置
kfold_params = {
    "K": 5, # K-Fold 折数
    "val_metrics_to_track": [
        # ("accu_mais", "max"),  # (指标名, 比较方式: "max" 或 "min")
        ("loss", "min"),
    ]
}