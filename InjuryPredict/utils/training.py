"""InjuryPredict 训练流程公共工具。

该模块集中放置课程学习、指标计算、优化器构建和 BN 重校准逻辑，避免单次训练与 K-Fold
训练入口在关键训练语义上出现分叉。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, root_mean_squared_error
from torch.utils.data import ConcatDataset, Subset

from common.metrics.injury_risk import AIS_cal_head, AIS_cal_chest, AIS_cal_neck
from InjuryPredict.utils.loss import FeatureConsistencyLoss, OutputConsistencyLoss
from InjuryPredict.utils.tools import convert_mais_to_3c, get_parameter_groups


BATCH_KEYS = (
    "x_acc_gt",
    "x_acc_pred",
    "x_att_continuous",
    "x_att_discrete",
    "y_HIC",
    "y_Dmax",
    "y_Nij",
    "ais_head",
    "ais_chest",
    "ais_neck",
    "mais",
    "OT",
)

INJURY_METRIC_KEYS = (
    "accu_head",
    "accu_chest",
    "accu_neck",
    "accu_mais",
    "accu_mais_3c",
    "mae_hic",
    "mae_dmax",
    "mae_nij",
    "rmse_hic",
    "rmse_dmax",
    "rmse_nij",
    "r2_hic",
    "r2_dmax",
    "r2_nij",
)

LOSS_COEFFICIENT_KEYS = ("alpha", "lambda_out", "lambda_feat")


def unpack_batch(batch, device: torch.device) -> Dict[str, torch.Tensor]:
    """将 Injury batch 解包为命名张量字典。"""
    if len(batch) != len(BATCH_KEYS):
        raise ValueError(
            f"InjuryPredict batch 字段数应为 {len(BATCH_KEYS)}，实际为 {len(batch)}。"
            "请重新运行 `python -m InjuryPredict.Injurydata_prepare --overwrite` 生成新版 .pt。"
        )
    return {key: value.to(device) for key, value in zip(BATCH_KEYS, batch)}


def make_target(batch_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """构造三任务标签张量，列顺序为 HIC、Dmax、Nij。"""
    return torch.stack([batch_dict["y_HIC"], batch_dict["y_Dmax"], batch_dict["y_Nij"]], dim=1) # [B], [B], [B] -> [B, 3]


def _extract_subset_arrays(dataset, attr_name: str) -> np.ndarray:
    """从 Dataset / Subset / ConcatDataset 中按样本顺序提取底层数组字段。"""
    if isinstance(dataset, Subset):
        base = dataset.dataset
        return np.asarray(getattr(base, attr_name))[np.asarray(dataset.indices, dtype=np.int64)]
    if isinstance(dataset, ConcatDataset):
        return np.concatenate([_extract_subset_arrays(subset, attr_name) for subset in dataset.datasets])
    return np.asarray(getattr(dataset, attr_name))


def compute_output_consistency_weights(dataset, device: torch.device) -> torch.Tensor:
    """基于训练集标签标准差构造输出一致性的尺度归一化权重。"""
    labels = np.stack(
        [
            _extract_subset_arrays(dataset, "y_HIC"),
            _extract_subset_arrays(dataset, "y_Dmax"),
            _extract_subset_arrays(dataset, "y_Nij"),
        ],
        axis=1,
    ).astype(np.float64)
    std = labels.std(axis=0)
    if not np.all(np.isfinite(std)) or np.any(std <= 0):
        raise ValueError(f"训练集标签标准差非法，无法构造输出一致性权重: {std.tolist()}")
    return torch.as_tensor(1.0 / std, dtype=torch.float32, device=device).view(1, 3) # [3] -> [1, 3]


def validate_curriculum_params(total_epochs: int, curriculum_params: dict) -> Dict[str, int]:
    """校验三阶段 epoch 配置，允许某些阶段为 0 以支持消融设置。"""
    phase_epochs = dict(curriculum_params.get("phase_epochs", {}))
    required = ("warmup", "transition", "target_finetune")
    missing = [name for name in required if name not in phase_epochs]
    if missing:
        raise ValueError(f"curriculum_params.phase_epochs 缺少阶段: {missing}")

    phases = {name: int(phase_epochs[name]) for name in required}
    if any(value < 0 for value in phases.values()):
        raise ValueError(f"课程学习各阶段 epoch 不能为负数: {phases}")
    if sum(phases.values()) != int(total_epochs):
        raise ValueError(
            f"课程学习阶段总 epoch={sum(phases.values())} 与 training_params['Epochs']={total_epochs} 不一致。"
        )
    return phases


def smoothstep(u: float) -> float:
    """三次 smoothstep 调度函数，保证过渡两端一阶变化更平滑。"""
    u = min(1.0, max(0.0, float(u)))
    return 3.0 * u * u - 2.0 * u * u * u


def get_curriculum_state(epoch_idx: int, phase_epochs: Dict[str, int], curriculum_params: dict) -> dict:
    """根据 0-based epoch 生成当前课程阶段与损失调度系数。"""
    warmup = phase_epochs["warmup"]
    transition = phase_epochs["transition"]
    output_consistency = dict(curriculum_params.get("output_consistency", {}))
    feature_consistency = dict(curriculum_params.get("feature_consistency", {}))
    feature_consistency_enabled = bool(feature_consistency.get("enabled", True))

    # α=0 表示完全使用真值波形分支损失，α=1 表示完全使用预测波形分支损失，过渡阶段平滑调整两者权重以缓冲域切换冲击。
    if epoch_idx < warmup:
        phase = "warmup"
        alpha = 0.0
    elif epoch_idx < warmup + transition:
        phase = "transition"
        pos = epoch_idx - warmup
        u = 1.0 if transition <= 1 else pos / float(transition - 1)
        alpha = smoothstep(u) # α(u)=3u^2−2u^3, u范围[0,1], 从0平滑增加到1，且两端导数为0.
    else:
        phase = "target_finetune"
        alpha = 1.0 

    # 钟形曲线平滑调度，在 transition 中间达到最大值 1.0，在两端为 0
    # 因此一致性正则在过渡中段最强，在过渡两端为 0。含义是：早期以真值波形监督为主，后期以预测波形监督为主，中间用输出一致性和波形编码特征一致性降低域切换冲击。
    bell = 4.0 * alpha * (1.0 - alpha) if phase == "transition" else 0.0 
    return {
        "phase": phase,
        "alpha": float(alpha),
        "lambda_out": float(curriculum_params.get("lambda_out_max", 0.0)) * bell,
        "lambda_feat": (float(curriculum_params.get("lambda_feat_max", 0.0)) * bell) if feature_consistency_enabled else 0.0,
        "output_consistency": output_consistency,
        "feature_consistency": feature_consistency,
    }


def requires_phase3_metric_selection(phase_epochs: Dict[str, int]) -> bool:
    """判断当前阶段配置是否需要只在 Phase III 内保存最佳模型。"""
    phase12_epochs = int(phase_epochs["warmup"]) + int(phase_epochs["transition"])
    return phase12_epochs > 0 and int(phase_epochs["target_finetune"]) > 0


def get_metric_selection_start_epoch(phase_epochs: Dict[str, int]) -> int:
    """返回允许保存最佳模型的首个 1-based epoch。"""
    if requires_phase3_metric_selection(phase_epochs):
        return int(phase_epochs["warmup"]) + int(phase_epochs["transition"]) + 1
    return 1


def get_early_stop_anchor_epoch(early_stop_start_epochs: int, phase_epochs: Dict[str, int]) -> int:
    """返回早停 patience 重新开始计数的 1-based epoch 边界。"""
    anchor_epoch = max(0, int(early_stop_start_epochs))
    if requires_phase3_metric_selection(phase_epochs):
        phase12_epochs = int(phase_epochs["warmup"]) + int(phase_epochs["transition"])
        anchor_epoch = max(anchor_epoch, phase12_epochs)
    return anchor_epoch


def should_stop_early(metric_states: dict, current_epoch: int, patience: int, anchor_epoch: int) -> bool:
    """判断是否应早停，且不把 anchor_epoch 之前的未改善轮数计入 patience。"""
    current_epoch = int(current_epoch)
    patience = int(patience)
    anchor_epoch = int(anchor_epoch)
    if patience <= 0 or current_epoch <= anchor_epoch:
        return False
    if current_epoch - anchor_epoch < patience:
        return False

    for state in metric_states.values():
        best_epoch = int(state.get("best_epoch", 0))
        if best_epoch <= 0:
            return False
        stagnation_start = max(best_epoch, anchor_epoch)
        if current_epoch - stagnation_start < patience:
            return False
    return True


def build_injury_optimizer(
    model: nn.Module,
    criterion: nn.Module,
    lr: float,
    weight_decay: float,
    head_decay_ratio: float = 0.05,
) -> optim.Optimizer:
    """构建 AdamW，含参数分组管理；并将 Kendall log_vars 作为无权重衰减参数组纳入优化。"""
    param_groups = get_parameter_groups(
        model,
        weight_decay=weight_decay,
        head_decay_ratio=head_decay_ratio,
        head_keywords=('head',),
    )
    param_groups.append({"params": [criterion.log_vars], "weight_decay": 0.0})
    return optim.AdamW(param_groups, lr=lr)


def _mean_or_zero(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else 0.0


def _append_loss_components(storage: Dict[str, list], prefix: str, components: dict) -> None:
    for key, value in components.items():
        if key == "main_loss":
            continue
        # criterion 返回的 hic_loss/dmax_loss/nij_loss 是主任务的原始加权子损失；记录时只添加波形源前缀。
        storage.setdefault(f"{prefix}_{key}", []).append(float(value))


def log_injury_tensorboard_metrics(
    writer,
    split_name: str,
    metrics: dict,
    epoch: int,
    criterion_weight_state: Optional[dict] = None,
    optimizer: Optional[optim.Optimizer] = None,
) -> None:
    """按 Train/Val、Loss/InjuryMetrics 两级结构记录 TensorBoard 标量。"""
    split_name = split_name.strip().capitalize()

    if isinstance(metrics.get("loss"), (int, float)):
        writer.add_scalar(f"{split_name}/Loss/total_loss", metrics["loss"], epoch)

    for key, value in metrics.items():
        if key != "loss" and (key.endswith("_loss") or key in LOSS_COEFFICIENT_KEYS) and isinstance(value, (int, float)):
            writer.add_scalar(f"{split_name}/Loss/{key}", value, epoch)

    if optimizer is not None and optimizer.param_groups:
        # 学习率不是损失项本身，但它直接决定当前 loss 优化步长，因此放在 Loss 组内作为优化系数监控。
        writer.add_scalar(f"{split_name}/Loss/learning_rate", optimizer.param_groups[0]["lr"], epoch)

    if criterion_weight_state:
        for task_name, stats in criterion_weight_state.items():
            for stat_name, stat_value in stats.items():
                if isinstance(stat_value, (int, float)):
                    writer.add_scalar(f"{split_name}/Loss/Kendall/{task_name}/{stat_name}", stat_value, epoch)

    for key in INJURY_METRIC_KEYS:
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            writer.add_scalar(f"{split_name}/InjuryMetrics/{key}", value, epoch)


def _collect_common_outputs(
    all_preds,
    all_trues,
    all_true_ais_head,
    all_true_ais_chest,
    all_true_ais_neck,
    all_true_mais,
    all_ot,
) -> dict:
    """基于批次缓存计算回归与 AIS/MAIS 指标。"""
    ot = np.concatenate(all_ot)
    preds, trues = np.concatenate(all_preds), np.concatenate(all_trues)
    pred_hic, pred_dmax, pred_nij = preds[:, 0], preds[:, 1], preds[:, 2]
    true_hic, true_dmax, true_nij = trues[:, 0], trues[:, 1], trues[:, 2]

    ais_head_pred = AIS_cal_head(pred_hic)
    ais_chest_pred = AIS_cal_chest(pred_dmax, ot)
    ais_neck_pred = AIS_cal_neck(pred_nij)
    true_ais_head = np.concatenate(all_true_ais_head)
    true_ais_chest = np.concatenate(all_true_ais_chest)
    true_ais_neck = np.concatenate(all_true_ais_neck)
    true_mais = np.concatenate(all_true_mais)
    mais_pred = np.maximum.reduce([ais_head_pred, ais_chest_pred, ais_neck_pred])
    true_mais_3c = convert_mais_to_3c(true_mais)
    pred_mais_3c = convert_mais_to_3c(mais_pred)

    return {
        'accu_head': accuracy_score(true_ais_head, ais_head_pred) * 100,
        'accu_chest': accuracy_score(true_ais_chest, ais_chest_pred) * 100,
        'accu_neck': accuracy_score(true_ais_neck, ais_neck_pred) * 100,
        'accu_mais': accuracy_score(true_mais, mais_pred) * 100,
        'accu_mais_3c': accuracy_score(true_mais_3c, pred_mais_3c) * 100,
        'mae_hic': mean_absolute_error(true_hic, pred_hic),
        'rmse_hic': root_mean_squared_error(true_hic, pred_hic),
        'mae_dmax': mean_absolute_error(true_dmax, pred_dmax),
        'rmse_dmax': root_mean_squared_error(true_dmax, pred_dmax),
        'mae_nij': mean_absolute_error(true_nij, pred_nij),
        'rmse_nij': root_mean_squared_error(true_nij, pred_nij),
        'r2_hic': r2_score(true_hic, pred_hic),
        'r2_dmax': r2_score(true_dmax, pred_dmax),
        'r2_nij': r2_score(true_nij, pred_nij),
    }


def run_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[optim.Optimizer] = None,
    curriculum_state: Optional[dict] = None,
    output_consistency_weights: Optional[torch.Tensor] = None,
) -> dict:
    """执行一个训练或验证 epoch。

    训练阶段根据 curriculum_state 选择真值波形、预测波形或二者混合；验证阶段始终使用预测波形，使模型选择指标与真实部署域保持一致。
    验证 loss 不包含 warmup/transition 的课程混合项和一致性正则，始终表示预测波形源上的 Kendall 主任务损失，因此不同训练阶段的验证 loss 具有相同含义，但训练 loss 含义会因课程学习阶段而异。
    若 optimizer 为 None 则不执行反向传播和优化步骤，即对应验证阶段；否则执行完整训练步骤。

    返回:
        dict: 一个 epoch 级别的指标字典，所有回归和 AIS/MAIS 指标均先合并该 loader 覆盖的全部 batch 样本后统一计算。
        - loss: 一个 epoch 内各 batch 标量 loss 的算术平均；由于 batch loss 通常已在 batch 内取 mean，该值不是按样本数加权的严格全样本 loss。
        - accu_head/accu_chest/accu_neck/accu_mais/accu_mais_3c: 基于 pred_metric（预测标签值）与真实标签计算的分类准确率。
        - mae_hic/mae_dmax/mae_nij、rmse_hic/rmse_dmax/rmse_nij、r2_hic/r2_dmax/r2_nij: 基于 pred_metric（预测标签值） 与真实 [HIC, Dmax, Nij] 计算的回归指标。
        - main_gt_loss、gt_hic_loss、gt_dmax_loss、gt_nij_loss: 仅在训练 warmup 或 transition 中出现，表示真值波形源分支的 Kendall 主损失及第 1c 层原始加权子损失。
        - main_pred_loss、pred_hic_loss、pred_dmax_loss、pred_nij_loss: 在训练 transition/target_finetune 和验证中出现，表示预测波形源分支的 Kendall 主损失及第 1c 层原始加权子损失。
        - out_cons_loss、feat_cons_loss: 仅在训练 transition 中出现，表示尚未乘 lambda_out/lambda_feat 的输出一致性和特征一致性正则原始值。
        - phase、alpha、lambda_out、lambda_feat: 仅当传入 curriculum_state 时出现，记录当前课程阶段及其调度系数。
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    output_consistency_loss = OutputConsistencyLoss()
    output_consistency_params = (curriculum_state or {}).get("output_consistency", {})
    feature_consistency_params = (curriculum_state or {}).get("feature_consistency", {})
    feature_consistency_loss = FeatureConsistencyLoss(
        normalize=feature_consistency_params.get("normalize", None),
    )

    loss_batch = []
    component_values: Dict[str, list] = {}
    all_preds, all_trues = [], []
    all_true_ais_head, all_true_ais_chest, all_true_ais_neck = [], [], []
    all_true_mais, all_ot = [], []

    with torch.set_grad_enabled(is_train):
        for batch in loader:
            data = unpack_batch(batch, device) # 返回一个包含 x_acc_gt, x_acc_pred, x_att_continuous, x_att_discrete, y_HIC, y_Dmax, y_Nij, ais_head, ais_chest, ais_neck, mais, OT 的字典，且所有张量已转移到指定设备。
            y_true = make_target(data)

            if is_train:
                optimizer.zero_grad()
                state = curriculum_state or {"phase": "target_finetune", "alpha": 1.0, "lambda_out": 0.0, "lambda_feat": 0.0}
                phase = state["phase"]
                alpha = float(state["alpha"]) # α 是 transition 阶段的调度系数，控制两路主任务损失的线性混合权重；warmup 时 α=0 完全使用真值波形分支损失，target_finetune 时 α=1 完全使用预测波形分支损失，transition 时 α 从 0 平滑增加到 1。

                if phase == "warmup":
                    pred_metric, _, _ = model(data["x_acc_gt"], data["x_att_continuous"], data["x_att_discrete"]) # [B, 2, L], [B, C], [B, D] -> [B, 3]
                    loss, components = criterion(pred_metric, y_true, data["OT"], return_components=True)
                    # main_gt_loss 表示真值波形分支上的 Kendall 主任务损失，warmup 阶段总损失即为该项。
                    component_values.setdefault("main_gt_loss", []).append(float(loss.detach().item()))
                    _append_loss_components(component_values, "gt", components)
                elif phase == "transition":
                    if output_consistency_weights is None:
                        raise ValueError("transition 阶段必须提供 output_consistency_weights。")
                    pred_gt, enc_gt, _ = model(data["x_acc_gt"], data["x_att_continuous"], data["x_att_discrete"]) # [B, 2, L], [B, C], [B, D] -> [B, 3], [B, F]; 这里 pred_gt 指使用真值波形源得到的预测输出，enc_gt 是真值波形源对应的编码器特征
                    pred_pred, enc_pred, _ = model(data["x_acc_pred"], data["x_att_continuous"], data["x_att_discrete"]) # [B, 2, L], [B, C], [B, D] -> [B, 3], [B, F]; 这里 pred_pred 指使用预测波形源得到的预测输出，enc_pred 是预测波形源对应的编码器特征
                    main_gt_loss, comp_gt = criterion(pred_gt, y_true, data["OT"], return_components=True)
                    main_pred_loss, comp_pred = criterion(pred_pred, y_true, data["OT"], return_components=True)

                    wave_dim = int(getattr(model, "wave_feature_dim"))
                    # 训练流程负责选择“只对齐波形编码器输出”这一结构边界，loss.py 中的损失项只接收已选好的张量。
                    feat_gt_wave = enc_gt[:, :wave_dim] # [B, F] -> [B, wave_feature_dim]
                    feat_pred_wave = enc_pred[:, :wave_dim] # [B, F] -> [B, wave_feature_dim]
                    output_reference = pred_gt.detach() if output_consistency_params.get("stop_gradient", True) else pred_gt
                    out_cons_loss = output_consistency_loss(output_reference, pred_pred, output_consistency_weights)
                    feature_enabled = bool(feature_consistency_params.get("enabled", True))
                    if feature_enabled:
                        # 根据 feature_consistency_params 决定 feat_gt_wave 是否参与反向传播；如果 stop_gradient=True 则 detach 后的 feat_gt_wave 不会计算梯度，反之则参与梯度计算。
                        feature_reference = feat_gt_wave.detach() if feature_consistency_params.get("stop_gradient", False) else feat_gt_wave
                        feat_cons_loss = feature_consistency_loss(feature_reference, feat_pred_wave)
                    else:
                        feat_cons_loss = pred_pred.new_tensor(0.0)
                    # transition 阶段的总损失由两路主任务损失和两个一致性正则组成，其中 out_cons_loss 与 feat_cons_loss 记录的是尚未乘 lambda 的原始正则值。
                    loss = (
                        (1.0 - alpha) * main_gt_loss
                        + alpha * main_pred_loss
                        + float(state["lambda_out"]) * out_cons_loss
                        + float(state["lambda_feat"]) * feat_cons_loss
                    )
                    pred_metric = pred_pred
                    # main_pred_loss 表示预测波形分支上的 Kendall 主任务损失，Phase III 与部署评估均使用该波形源。
                    component_values.setdefault("main_gt_loss", []).append(float(main_gt_loss.detach().item()))
                    component_values.setdefault("main_pred_loss", []).append(float(main_pred_loss.detach().item()))
                    component_values.setdefault("out_cons_loss", []).append(float(out_cons_loss.detach().item()))
                    component_values.setdefault("feat_cons_loss", []).append(float(feat_cons_loss.detach().item()))
                    _append_loss_components(component_values, "gt", comp_gt)
                    _append_loss_components(component_values, "pred", comp_pred)
                elif phase == "target_finetune":
                    pred_metric, _, _ = model(data["x_acc_pred"], data["x_att_continuous"], data["x_att_discrete"]) # [B, 2, L], [B, C], [B, D] -> [B, 3]
                    loss, components = criterion(pred_metric, y_true, data["OT"], return_components=True)
                    # target_finetune 阶段只优化预测波形分支上的 Kendall 主任务损失。
                    component_values.setdefault("main_pred_loss", []).append(float(loss.detach().item()))
                    _append_loss_components(component_values, "pred", components)
                else:
                    raise ValueError(f"未知课程阶段: {phase}")

                loss.backward()
                optimizer.step()
            else:
                # 验证阶段固定评估预测波形源；这里不使用 curriculum_state，因此 val loss 在所有阶段均为预测波形分支的 Kendall 主任务损失。
                pred_metric, _, _ = model(data["x_acc_pred"], data["x_att_continuous"], data["x_att_discrete"]) # [B, 2, L], [B, C], [B, D] -> [B, 3]
                loss, components = criterion(pred_metric, y_true, data["OT"], return_components=True)
                component_values.setdefault("main_pred_loss", []).append(float(loss.detach().item()))
                _append_loss_components(component_values, "pred", components)

            loss_batch.append(float(loss.detach().item()))
            all_preds.append(pred_metric.detach().cpu().numpy())
            all_trues.append(y_true.detach().cpu().numpy())
            all_true_ais_head.append(data["ais_head"].detach().cpu().numpy())
            all_true_ais_chest.append(data["ais_chest"].detach().cpu().numpy())
            all_true_ais_neck.append(data["ais_neck"].detach().cpu().numpy())
            all_true_mais.append(data["mais"].detach().cpu().numpy())
            all_ot.append(data["OT"].detach().cpu().numpy())

    if not loss_batch:
        raise ValueError("DataLoader 为空，无法执行一个完整的 epoch。")

    metrics = {"loss": float(np.mean(loss_batch))} # loss_batch 是一个 epoch 内所有 batch 的总损失列表，取平均得到该 epoch 的平均损失（即 metrics["loss"] 是 batch 级等权平均）。
    # all_preds 收集的是当前阶段用于计算 epoch-level 指标的那一路预测值（pred_metric）：
    # 训练 Phase I：真值波形源预测；
    # 训练 Phase II/III：预测波形源预测；
    # 验证：始终预测波形源预测。
    metrics.update(_collect_common_outputs(
        all_preds,
        all_trues,
        all_true_ais_head,
        all_true_ais_chest,
        all_true_ais_neck,
        all_true_mais,
        all_ot,
    )) # _collect_common_outputs 基于整个 epoch（完整训练/验证集）的预测输出和真实标签计算回归与 AIS/MAIS 指标（不是 batch 平均）
    for key, values in component_values.items():
        metrics[key] = _mean_or_zero(values) # batch 级组分 loss 的等权平均。将 criterion 返回的各损失组件的平均值添加到 metrics 中，键名带有 gt/pred 前缀以区分两路主任务损失来源；如果某组件在当前阶段未计算过，则默认为 0。
    if curriculum_state is not None:
        metrics.update({
            "phase": curriculum_state["phase"],
            "alpha": float(curriculum_state["alpha"]),
            "lambda_out": float(curriculum_state["lambda_out"]),
            "lambda_feat": float(curriculum_state["lambda_feat"]),
        })
    return metrics


def bn_recalibrated_path(weight_path: Path) -> Path:
    """返回 BN 重校准权重的派生保存路径。"""
    weight_path = Path(weight_path)
    return weight_path.with_name(f"{weight_path.stem}_bn_recalibrated{weight_path.suffix}")


def recalibrate_batchnorm(model: nn.Module, loader, device: torch.device) -> int:
    """
    仅重估 BatchNorm 的 buffer： running_mean / running_var / (num_batches_tracked) ，不更新卷积、线性层、Kendall log_vars 等可训练参数。BatchNorm.weight 和 BatchNorm.bias 即 gamma 和 beta 因为是可训练参数，所以也不更新。

    该过程把模型主体保持在 eval 语义下，仅将 BatchNorm 层临时置为 train，使其在 no_grad 前向中根据预测波形源样本刷新运行统计量。
    因此重校准后的权重只改变 BN buffer，不改变模型学到的函数参数；它用于缓解训练阶段多波形源混合导致的 BN 统计量与部署输入域不一致。
    """
    original_modes = {module: module.training for module in model.modules()}
    original_requires_grad = [param.requires_grad for param in model.parameters()]
    bn_momentums = {}

    # 禁止所有参数梯度，并配合 torch.no_grad()，确保重校准不会产生反向传播或优化器更新。
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            # 重置后使用 cumulative moving average，避免默认 momentum=0.1 使统计量偏向遍历末尾的 batch。
            module.reset_running_stats()
            bn_momentums[module] = module.momentum
            module.momentum = None
            # 只有 BatchNorm 需要 train 模式来更新 running statistics；Dropout 等随机层仍保持 eval，避免引入额外随机性。
            module.train()
        elif isinstance(module, nn.Dropout):
            module.eval()

    processed_batches = 0
    with torch.no_grad():
        for batch in loader:
            data = unpack_batch(batch, device)
            if data["x_acc_pred"].shape[0] <= 1:
                # BatchNorm 对 batch 统计量敏感，单样本 batch 不用于刷新 running variance。
                continue
            # 重校准显式使用预测波形源 x_acc_pred，因为验证和部署阶段均以该输入域作为 InjuryPredict 的波形输入。
            model(data["x_acc_pred"], data["x_att_continuous"], data["x_att_discrete"]) # [B, 2, L], [B, C], [B, D] -> [B, 3]
            processed_batches += 1

    # 恢复调用前的训练/验证模式和参数 requires_grad 状态，避免重校准函数对外部训练流程产生副作用。
    for module, momentum in bn_momentums.items():
        module.momentum = momentum
    for module, was_training in original_modes.items():
        module.train(was_training)
    for param, requires_grad in zip(model.parameters(), original_requires_grad):
        param.requires_grad_(requires_grad)

    if processed_batches == 0:
        raise ValueError("BN 重校准未处理任何 batch；请确认训练集大小和 batch size 是否足以支持 BatchNorm。")
    return processed_batches


def save_bn_recalibrated_state(
    model: nn.Module,
    source_weight_path: Path,
    loader,
    device: torch.device,
) -> Dict[str, object]:
    """
    从指定权重文件派生一个 BN 重校准版本。

    源权重文件本身不会被覆盖；函数会先加载源权重，再执行 BatchNorm running statistics 重估，最后保存为 *_bn_recalibrated.pth。
    """
    source_weight_path = Path(source_weight_path)
    model.load_state_dict(torch.load(source_weight_path.as_posix(), map_location=device, weights_only=False))
    processed_batches = recalibrate_batchnorm(model, loader, device)
    target_path = bn_recalibrated_path(source_weight_path)
    torch.save(model.state_dict(), target_path.as_posix())
    return {
        "source": str(source_weight_path),
        "recalibrated": str(target_path),
        "processed_batches": int(processed_batches),
    }
