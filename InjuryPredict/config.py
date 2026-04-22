# -*- coding: utf-8 -*-
"""
集中管理模型训练、损失函数和网络结构的可调超参数。
"""

from pathlib import Path

from common.settings import PULSE_PREDICT_DIR

RUNS_DIR = "./InjuryPredict/runs"  # 模型训练结果的保存目录

DEFAULT_PULSE_RUN_DIR = PULSE_PREDICT_DIR / "saved" / "models" / "HybridPulseCNN" / "0415_161324"
DEFAULT_PULSE_CHECKPOINT = DEFAULT_PULSE_RUN_DIR / "model_best.pth"
DEFAULT_PULSE_CONFIG = DEFAULT_PULSE_RUN_DIR / "config.json"

# 1. 优化与训练相关
training_params = {
    "Epochs": 400,
    "Batch_size": 128,
    "Learning_rate": 0.01,
    "Learning_rate_min": 0,
    "weight_decay": 0.1,
    "early_stop_start_epochs": 400, # 早停开始轮数
    "Patience": 50, # 早停轮数
}

# 2. 损失函数相关
loss_params = {
    "base_loss": "mae",
    "weight_factor_classify": 1.05,
    "weight_factor_sample": 0.2,
    "task_prior_weights": (1.0, 1.2, 0.8), # HIC, Dmax, Nij 在 Kendall 主损失中的人工先验权重
}

# 3. 波形源课程学习相关
# 仅真值波形训练：phase_epochs={"warmup": Epochs, "transition": 0, "target_finetune": 0}。
# 仅预测波形训练：phase_epochs={"warmup": 0, "transition": 0, "target_finetune": Epochs}。
# 总epochs数 =  phase_epochs 各阶段之和 = training_params["Epochs"]
curriculum_params = {
    "phase_epochs": {
        "warmup": 80,
        "transition": 180,
        "target_finetune": 140,
    },
    "lambda_out_max": 0.1,
    "lambda_feat_max": 0.01,
    "bn_recalibration": True,
}

# 4. 模型结构相关
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

# 仅用于“验证集指标跟踪”的可选指标名（train.py / train_KFold.py 通用）。
# 说明：
# 1) 这里的每个名字都必须与 utils.training.run_one_epoch(...) 返回字典中的 key 一致；build_metric_trackers 只负责校验和去掉可选的 "val_" 前缀。
# 2) 下列指标名默认都表示 val 指标，不是 train 指标。
# 3) val_metrics_to_track 中可写 "loss" 或 "val_loss"（其余指标同理），内部会统一按 val 指标处理。
AVAILABLE_VAL_METRIC_NAMES = (
    'loss', 'main_pred_loss', 'pred_hic_loss', 'pred_dmax_loss', 'pred_nij_loss',
    'accu_head', 'accu_chest', 'accu_neck', 'accu_mais', 'accu_mais_3c',
    'g_mean_head', 'g_mean_chest', 'g_mean_neck', 'g_mean_mais', 'g_mean_mais_3c',
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
