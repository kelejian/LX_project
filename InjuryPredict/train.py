# -*- coding: utf-8 -*-
import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = 'T' # 避免部分 Windows 控制台环境中的控制信号处理干扰训练进程。
import warnings
warnings.filterwarnings('ignore')
import json
import time
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

from common.settings import INJURY_PROCESSED_DIR, get_injury_processed_dataset_path
from common.tools.seeding import GLOBAL_SEED, set_random_seed

from InjuryPredict.Injurydata_prepare import load_processed_subset
from InjuryPredict.config import (
    RUNS_DIR,
    curriculum_params,
    loss_params,
    model_params,
    training_params,
    val_metrics_to_track,
)
from InjuryPredict.utils import models
from InjuryPredict.utils.loss import InjuryKendallMultiTaskLoss
from InjuryPredict.utils.tools import (
    build_metric_trackers,
    convert_numpy_types,
    round_float_fields,
    round_to_significant,
)
from InjuryPredict.utils.training import (
    build_injury_optimizer,
    compute_output_consistency_weights,
    get_early_stop_anchor_epoch,
    get_curriculum_state,
    get_metric_selection_start_epoch,
    log_injury_tensorboard_metrics,
    run_one_epoch,
    save_bn_recalibrated_state,
    should_stop_early,
    validate_curriculum_params,
)

def _write_training_record(path: str, record: dict) -> None:
    '''将训练记录保存为 JSON 文件，自动处理 numpy 数据类型转换。'''
    with open(path, "w", encoding="utf-8") as file:
        json.dump(convert_numpy_types(record), file, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    set_random_seed()
    # 创建独立文件夹保存本次运行结果
    current_time = datetime.now().strftime("%m%d%H%M")
    run_dir = os.path.join(RUNS_DIR, f"InjuryPredictModel_{current_time}")
    os.makedirs(run_dir, exist_ok=True)

    # 初始化 TensorBoard
    writer = SummaryWriter(log_dir=run_dir)

    Epochs = int(training_params['Epochs'])
    Batch_size = int(training_params['Batch_size'])
    Learning_rate = float(training_params['Learning_rate'])
    Learning_rate_min = float(training_params['Learning_rate_min'])
    weight_decay = float(training_params['weight_decay'])
    early_stop_start_epochs = int(training_params['early_stop_start_epochs'])
    Patience = min(int(training_params['Patience']), Epochs)
    phase_epochs = validate_curriculum_params(Epochs, curriculum_params)
    # 加载数据集对象
    print(f".pt 数据文件路径: {INJURY_PROCESSED_DIR}/*.pt")
    train_pt = get_injury_processed_dataset_path("train")
    val_pt = get_injury_processed_dataset_path("val")
    if not train_pt.exists():
        raise FileNotFoundError(f"找不到训练数据 ({train_pt})。请先运行: python -m InjuryPredict.Injurydata_prepare")
    if not val_pt.exists():
        raise FileNotFoundError(f"找不到验证数据 ({val_pt})。请先运行: python -m InjuryPredict.Injurydata_prepare")

    train_dataset = load_processed_subset(train_pt)
    if len(train_dataset) == 0:
        raise ValueError("train_dataset.pt 为空，InjuryPredict.train 不支持空训练集。")
    val_dataset = load_processed_subset(val_pt)

    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=Batch_size, shuffle=True, num_workers=0)
    # BN 重校准只用训练集预测波形源顺序前向刷新 BatchNorm buffer；shuffle=False 便于复现实验。
    bn_loader = DataLoader(train_dataset, batch_size=Batch_size, shuffle=False, num_workers=0)
    val_enabled = len(val_dataset) > 0
    val_loader = DataLoader(val_dataset, batch_size=Batch_size, shuffle=False, num_workers=0) if val_enabled else None
    if not val_enabled:
        # 无验证集时不启动 best_val_* 指标跟踪和 early stop；训练结束只保存 final_model.pth。
        print("警告: 验证集为空，本次训练将跳过验证、best_val_* 权重保存和 early stop。")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_consistency_weights = compute_output_consistency_weights(train_dataset, device)

    model = models.InjuryPredictModel(
        num_classes_of_discrete=train_dataset.dataset.num_classes_of_discrete,
        **model_params,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model)
    print(f"模型总参数量: {total_params}, 可训练参数量: {trainable_params}")

    criterion = InjuryKendallMultiTaskLoss(
        base_loss=loss_params['base_loss'],
        weight_factor_classify=loss_params['weight_factor_classify'],
        weight_factor_sample=loss_params['weight_factor_sample'],
        task_prior_weights=loss_params['task_prior_weights'],
    ).to(device)

    optimizer = build_injury_optimizer(model, criterion, Learning_rate, weight_decay)
    # 学习率调度与课程阶段解耦：优化器和余弦调度器只创建一次。
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Epochs, eta_min=Learning_rate_min)
    metric_selection_start_epoch = get_metric_selection_start_epoch(phase_epochs)
    early_stop_anchor_epoch = get_early_stop_anchor_epoch(early_stop_start_epochs, phase_epochs)

    # metric_trackers 是由 val_metrics_to_track 生成的静态规则表。key 为 run_one_epoch 返回的验证指标名；value 包含 compare_indicator、initial_value、is_better、model_filename、display_name。
    metric_trackers = {}
    # metric_states 是单次训练的动态状态表。key 与 metric_trackers 一致；value 包含当前 best_value、1-based best_epoch、判优函数 is_better 和对应 model_filename；无验证集时保持为空。
    metric_states = {}
    if val_enabled:
        # val_metrics_to_track 中的指标名必须能在 run_one_epoch(..., optimizer=None) 返回的 val_metrics 字典中找到。
        metric_trackers = build_metric_trackers(
            val_metrics_to_track,
            model_filename_fn=lambda metric_name: f"best_val_{metric_name}.pth",
        )
        if not metric_trackers:
            raise ValueError("val_metrics_to_track 不能为空。")
        print(f"将跟踪以下验证指标: {[tracker['display_name'] for tracker in metric_trackers.values()]}")
        if metric_selection_start_epoch > 1:
            print(f"最佳模型权重将只从 Phase III 开始保存，首个候选 epoch={metric_selection_start_epoch}。")
        print(f"早停 patience 将从 epoch={early_stop_anchor_epoch} 之后开始重新计数。")
        # 每个 metric_state 从对应 tracker 复制判优函数和文件名，并额外持有当前 run 内的动态最优值；best_epoch=0 表示该指标尚未产生可保存的最优模型。
        metric_states = {
            metric_name: {
                'best_value': tracker_info['initial_value'],
                'best_epoch': 0,
                'is_better': tracker_info['is_better'],
                'model_filename': tracker_info['model_filename'],
            }
            for metric_name, tracker_info in metric_trackers.items()
        }

    record_path = os.path.join(run_dir, "TrainingRecord.json")
    initial_record = {
        "GLOBAL_SEED": GLOBAL_SEED,
        "Trainset_size": len(train_dataset),
        "Valset_size": len(val_dataset),
        "validation_enabled": val_enabled,
        "INJURY_PROCESSED_DIR": str(INJURY_PROCESSED_DIR),
        "data_interface": {
            "processed_dir": str(INJURY_PROCESSED_DIR.resolve()),
            "waveform_fields": {
                "x_acc_gt": "共享归一化空间中的真值 XY 波形。",
                "x_acc_pred": "冻结 PulsePredict 输出的共享归一化 XY 预测波形，验证与部署默认使用该字段。",
            },
            "pulse_prediction": train_dataset.dataset.pulse_prediction_meta,
        },
        "model_params_count": {
            "total_params": total_params,
            "trainable_params": trainable_params,
        },
        "hyperparameters": {
            "training": {
                **training_params,
                "val_metrics_to_track": val_metrics_to_track if val_enabled else [],
            },
            "loss": loss_params,
            "curriculum": curriculum_params,
            "lr_scheduler": {
                "type": "CosineAnnealingLR",
                "T_max": Epochs,
                "eta_min": Learning_rate_min,
                "restart_on_phase_change": False,
            },
            "model_selection": {
                "phase3_only_best_model": metric_selection_start_epoch > 1,
                "best_model_start_epoch": metric_selection_start_epoch,
                "early_stop_anchor_epoch": early_stop_anchor_epoch,
            },
            "model": {
                **model_params,
                "num_classes_of_discrete": train_dataset.dataset.num_classes_of_discrete,
            },
        },
        "output_consistency_weights": output_consistency_weights.detach().cpu().numpy().reshape(-1).tolist(), # [1, 3] -> [3]
        "kendall_weight_state_initial": criterion.get_weight_state(),
    }
    _write_training_record(record_path, initial_record)
    print(f"初始配置已保存至: {record_path}")

    train_metrics = None
    val_metrics = None
    for epoch in range(Epochs):
        epoch_start_time = time.time()
        # 根据当前 epoch 获取当前课程阶段与损失调度系数
        curriculum_state = get_curriculum_state(epoch, phase_epochs, curriculum_params)

        train_metrics = run_one_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            curriculum_state=curriculum_state,
            output_consistency_weights=output_consistency_weights,
        )

        val_metrics = None
        if val_enabled:
            # 验证阶段不参与课程调度，固定使用预测波形源计算 Kendall 主任务损失，以保证各阶段 val/loss 可直接比较。
            val_metrics = run_one_epoch(model, val_loader, criterion, device, optimizer=None)
            missing_metrics = [name for name in metric_trackers.keys() if name not in val_metrics]
            if missing_metrics:
                raise KeyError(f"val_metrics_to_track 中存在无效指标: {missing_metrics}")

        log_injury_tensorboard_metrics(
            writer,
            "Train",
            train_metrics,
            epoch,
            criterion_weight_state=criterion.get_weight_state(),
            optimizer=optimizer,
        )
        if val_enabled:
            # Val/Loss/total_loss 表示预测波形源上的主任务损失，不包含课程学习的一致性正则项。
            log_injury_tensorboard_metrics(writer, "Val", val_metrics, epoch)
        scheduler.step()

        print(
            f"Epoch {epoch + 1}/{Epochs} | Phase: {curriculum_state['phase']} "
            f"| alpha={curriculum_state['alpha']:.4g} | Train Loss: {train_metrics['loss']:.4g}"
        )
        if val_enabled:
            print(
                f"            | Val Loss: {val_metrics['loss']:.4g} "
                f"| MAIS Acc 6C: {val_metrics['accu_mais']:.4g}% "
                f"| MAIS Acc 3C: {val_metrics['accu_mais_3c']:.4g}%"
            )
            print(
                f"            | R2: HIC={val_metrics['r2_hic']:.4g}, "
                f"Dmax={val_metrics['r2_dmax']:.4g}, Nij={val_metrics['r2_nij']:.4g}"
            )
        else:
            print("            | Validation disabled because val_dataset.pt is empty")

        if val_enabled:
            current_epoch = epoch + 1
            if current_epoch >= metric_selection_start_epoch:
                for metric_name, state in metric_states.items():
                    current_value = val_metrics[metric_name]
                    # metric_name 是 val_metrics 字典中的真实 key；metric_states[metric_name] 保存当前 run 中该指标已达到的最优值和对应 epoch。
                    if state['is_better'](current_value, state['best_value']):
                        state['best_value'] = current_value
                        state['best_epoch'] = current_epoch
                        torch.save(model.state_dict(), os.path.join(run_dir, state['model_filename']))
                        print(f"Best {metric_trackers[metric_name]['display_name']} model saved: {current_value:.4g} at epoch {current_epoch}")

        if val_enabled and should_stop_early(metric_states, epoch + 1, Patience, early_stop_anchor_epoch):
            print(f"Early Stop at epoch: {epoch + 1}!")
            for metric_name, state in metric_states.items():
                print(f"Best {metric_trackers[metric_name]['display_name']}: {state['best_value']:.4g} (at epoch {state['best_epoch']})")
            break

        print(f"            | Time: {time.time() - epoch_start_time:.2f}s")

    # 不论是否启用验证，训练结束都保存 final_model.pth；未启用验证时仅保存 final 的模型权重。
    final_model_path = os.path.join(run_dir, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print("Final model saved.")

    bn_recalibration_results = []
    if curriculum_params.get("bn_recalibration", True):
        # 对 final_model 和已保存的 best_val_* 权重分别派生 *_bn_recalibrated.pth；该步骤不覆盖源权重，也不改变此前的模型选择结果。
        candidate_paths = [final_model_path]
        if val_enabled:
            candidate_paths.extend(
                os.path.join(run_dir, state["model_filename"])
                for state in metric_states.values()
                if os.path.exists(os.path.join(run_dir, state["model_filename"]))
            )
        for path in sorted(set(candidate_paths)):
            result = save_bn_recalibrated_state(model, path, bn_loader, device)
            bn_recalibration_results.append(result)
            print(f"BN recalibrated model saved: {result['recalibrated']}")

    writer.close()

    # best_metrics_by_tracker 是写入 TrainingRecord.json 的结果摘要。key 与 metric_trackers 一致；value 只记录最终 best_value、best_epoch 和 model_file，不再参与训练过程更新。
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

    with open(record_path, "r", encoding="utf-8") as file:
        final_record = json.load(file)
    final_record["results"] = {
        "final_epoch": epoch + 1,
        "validation_enabled": val_enabled,
        "best_metrics_by_tracker": best_metrics_by_tracker,
        "last_epoch_metrics": last_epoch_metrics,
        "last_epoch_metrics_source": "val" if val_enabled else "train",
        "kendall_weight_state_final": criterion.get_weight_state(),
        "bn_recalibration": bn_recalibration_results,
    }
    _write_training_record(record_path, final_record)
    print(f"训练结果已更新至: {record_path}")
