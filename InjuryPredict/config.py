# -*- coding: utf-8 -*-
"""
集中管理模型训练、损失函数和网络结构的可调超参数。
"""

from pathlib import Path

from common.settings import PULSE_PREDICT_DIR

RUNS_DIR = "./InjuryPredict/runs"  # 模型训练结果的保存目录

DEFAULT_PULSE_RUN_DIR = PULSE_PREDICT_DIR / "saved" / "models" / "HybridPulseCNN" / "0502_123240"
DEFAULT_PULSE_CHECKPOINT = DEFAULT_PULSE_RUN_DIR / "model_best.pth"
DEFAULT_PULSE_CONFIG = DEFAULT_PULSE_RUN_DIR / "config.json"

# 1. 优化与训练相关
training_params = {
    "Epochs": 500,
    "Batch_size": 64,
    "Learning_rate": 5e-3,
    "Learning_rate_min": 0,
    "weight_decay": 0.1,
    "early_stop_start_epochs": 500, # 早停开始轮数
    "Patience": 50, # 早停轮数
    "write_epoch_metrics_csv": True, # 开发阶段可临时开启，用于导出紧凑 epoch 指标表；正式实验默认依赖 TensorBoard 与 TrainingRecord.json。
}

# 2. 损失函数相关
loss_params = {
    "base_loss": "huber", # "mse" 或 "huber" 或 "mae"
    "weight_factor_classify": 1.05,
    "weight_factor_sample": 0.2,
    "task_prior_weights": (1.0, 1.2, 0.8), # HIC, Dmax, Nij 在 Kendall 主损失中的人工先验权重
}

# 3. 波形源课程学习相关
# 仅真值波形训练：phase_epochs={"warmup": Epochs, "transition": 0, "target_finetune": 0}。
# 仅预测波形训练：phase_epochs={"warmup": 0, "transition": 0, "target_finetune": Epochs}。
# 总epochs数 =  phase_epochs 各阶段之和 = training_params["Epochs"]
curriculum_params = {
    # "phase_epochs": {
    #     "warmup": 125,
    #     "transition": 225,
    #     "target_finetune": 150,
    # },
    "phase_epochs": {
        "warmup": 150,
        "transition": 250,
        "target_finetune": 100,
    },
    # lambda_out_max/lambda_feat_max=0 表示不启用输出/特征一致性正则；正数表示在 transition 阶段按钟形调度逐步施加约束。
    "lambda_out_max": 1.0, # out_cons_loss 大约在阶段2末期尺度约为 main_pred_loss/main_gt_loss 的 5%~8%
    "lambda_feat_max": 1.0, # feat_cons_loss 大约在阶段2末期尺度约为 main_pred_loss/main_gt_loss 的 3%~4%

    # stop_gradient 用.detach()实现
    # 为False时一致性正则会同时推动真值分支和预测分支靠近彼此；
    # 为True时一致性正则为单向一致性 / teacher-anchor 约束：真值波形分支作为teacher，只推动预测波形分支向真值波形分支靠近
    "output_consistency": {
        "stop_gradient": True,
    },
    "feature_consistency": {
        "normalize": "sample_layernorm",
        "stop_gradient": False,
    },
    "bn_recalibration": False,
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

# 仅用于校验“验证集指标跟踪”配置中的指标名。
# 说明：
# 1) 这里的每个名字都必须与 utils.training.run_one_epoch(...) 返回字典中的 key 一致。
# 2) model_selection_params 中可写 "loss" 或 "val_loss"（其余指标同理），内部会统一按 val 指标处理。
AVAILABLE_VAL_METRIC_NAMES = (
    'loss', 'main_pred_loss', 'pred_hic_loss', 'pred_dmax_loss', 'pred_nij_loss',
    'accu_head', 'accu_chest', 'accu_neck', 'accu_mais', 'accu_mais_3c',
    'mae_hic', 'mae_dmax', 'mae_nij',
    'rmse_hic', 'rmse_dmax', 'rmse_nij',
    'r2_hic', 'r2_dmax', 'r2_nij',
)


# 可配置的验证集选模规则。single_metric_trackers 保留单指标诊断权重，composite_trackers 用固定优先级列表表达复合选模逻辑。
# 若需要把 val_accu_mais_3c 与 val_accu_mais 的优先级互换，只需调整 priority 列表顺序。
model_selection_params = {
    "primary_tracker": "dual_target",
    "single_metric_trackers": [
        {"name": "val_loss", "mode": "min", "filename": "best_val_loss.pth"},
        {"name": "val_accu_mais_3c", "mode": "max", "filename": "best_val_accu_mais_3c.pth"},
        {"name": "val_accu_mais", "mode": "max", "filename": "best_val_accu_mais.pth"},
    ],
    "composite_trackers": [
        {
            "name": "dual_target",
            "filename": "best_val_dual_target.pth",
            "priority": [
                {"metric": "val_accu_mais_3c", "mode": "max", "min_delta": 0.1},
                {"metric": "val_accu_mais", "mode": "max", "min_delta": 0.1},
                {"metric": "val_loss", "mode": "min", "min_delta": 0.0},
            ],
        }
    ],
}

# K-Fold 专项设置
kfold_params = {
    "K": 5, # K-Fold 折数
}
