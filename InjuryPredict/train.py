# -*- coding: utf-8 -*-
import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = 'T' # 避免部分 Windows 控制台环境中的控制信号处理干扰训练进程。
import warnings
warnings.filterwarnings('ignore')
import csv
import json
import time
from datetime import datetime
from numbers import Real

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
    model_selection_params,
    model_params,
    training_params,
)
from InjuryPredict.utils import models
from InjuryPredict.utils.loss import InjuryKendallMultiTaskLoss
from InjuryPredict.utils.tools import (
    build_composite_metric_trackers,
    build_single_metric_trackers,
    convert_numpy_types,
    is_composite_better,
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


def _snapshot_metrics(metrics: dict) -> dict:
    """复制当前验证指标快照，避免后续更新覆盖 best checkpoint 的判优依据。"""
    return {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, Real)
    }


def _make_epoch_csv_row(epoch: int, train_metrics: dict, val_metrics: dict | None, curriculum_state: dict) -> dict:
    """构造开发阶段可选的 epoch 级指标行；未启用验证时保留空值。"""
    row = {
        "epoch": int(epoch),
        "phase": curriculum_state["phase"],
        "alpha": float(curriculum_state["alpha"]),
        "train_loss": float(train_metrics["loss"]),
    }
    for key in (
        "loss",
        "accu_mais_3c",
        "accu_mais",
        "accu_head",
        "accu_chest",
        "accu_neck",
        "r2_hic",
        "r2_dmax",
        "r2_nij",
    ):
        row[f"val_{key}"] = "" if val_metrics is None else float(val_metrics[key])
    return row


def _append_epoch_metrics_csv(path: str, row: dict, fieldnames: list[str]) -> None:
    """追加写入紧凑 epoch 指标表；该文件仅用于开发阶段离线分析。"""
    file_exists = os.path.exists(path)
    with open(path, "a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _refresh_tracker_state(
    state: dict,
    model: torch.nn.Module,
    epoch: int,
    val_metrics: dict,
    criterion: InjuryKendallMultiTaskLoss,
    run_dir: str,
) -> None:
    """更新 tracker 的动态最优状态，并保存对应模型权重。"""
    state["best_epoch"] = int(epoch)
    state["best_metrics_snapshot"] = _snapshot_metrics(val_metrics)
    state["criterion_weight_state"] = criterion.get_weight_state()
    if state.get("kind") == "single":
        metric_key = state["metric_key"]
        state["best_value"] = float(val_metrics[metric_key])
    torch.save(model.state_dict(), os.path.join(run_dir, state["model_filename"]))


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
    write_epoch_metrics_csv = bool(training_params.get("write_epoch_metrics_csv", False))
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
    # BN 重校准使用固定随机种子打散训练集，避免顺序样本结构使 BatchNorm 统计量偏向末尾 batch。
    bn_generator = torch.Generator()
    bn_generator.manual_seed(GLOBAL_SEED)
    bn_loader = DataLoader(train_dataset, batch_size=Batch_size, shuffle=True, num_workers=0, generator=bn_generator)
    val_enabled = len(val_dataset) > 0
    val_loader = DataLoader(val_dataset, batch_size=Batch_size, shuffle=False, num_workers=0) if val_enabled else None
    if not val_enabled:
        # 无验证集时不启动 best_val_* 指标跟踪和 early stop；训练结束只保存 final_model.pth。
        print("警告: 验证集为空，本次训练将跳过验证、best_val_* 权重保存和 early stop。")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # OutputConsistencyLoss 使用的归一化参数来自训练集的真实损伤标签
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

    # single_metric_trackers 保留单指标 best 权重，composite_metric_trackers 用配置化优先级列表表达复合选模规则。
    single_metric_trackers = {}
    composite_metric_trackers = {}
    metric_trackers = {}
    required_val_metric_names = set()
    primary_tracker_name = None
    # metric_states 是单次训练的动态状态表；无验证集时保持为空。
    metric_states = {}
    if val_enabled:
        single_metric_configs = model_selection_params.get("single_metric_trackers", [])
        single_metric_trackers = build_single_metric_trackers(single_metric_configs)
        composite_metric_trackers = build_composite_metric_trackers(model_selection_params.get("composite_trackers", []))
        duplicated_tracker_names = set(single_metric_trackers) & set(composite_metric_trackers)
        if duplicated_tracker_names:
            raise ValueError(f"单指标与复合选模 tracker 名称重复: {sorted(duplicated_tracker_names)}")
        metric_trackers = {**single_metric_trackers, **composite_metric_trackers}
        if not metric_trackers:
            raise ValueError("至少需要配置一个验证集选模 tracker。")
        for tracker in single_metric_trackers.values():
            required_val_metric_names.add(tracker["metric_key"])
        for tracker in composite_metric_trackers.values():
            required_val_metric_names.update(rule["metric_key"] for rule in tracker["priority"])
        primary_tracker_name = model_selection_params.get("primary_tracker")
        if primary_tracker_name not in metric_trackers:
            raise ValueError(f"model_selection_params.primary_tracker={primary_tracker_name} 不在已配置的 tracker 中。")
        print(f"将跟踪以下验证指标: {[tracker['display_name'] for tracker in metric_trackers.values()]}")
        print(f"早停主 tracker: {metric_trackers[primary_tracker_name]['display_name']}")
        if metric_selection_start_epoch > 1:
            print(f"最佳模型权重将只从 Phase III 开始保存，首个候选 epoch={metric_selection_start_epoch}。")
        print(f"早停 patience 将从 epoch={early_stop_anchor_epoch} 之后开始重新计数。")
        for tracker_name, tracker_info in single_metric_trackers.items():
            metric_states[tracker_name] = {
                "kind": "single",
                "metric_key": tracker_info["metric_key"],
                "best_value": tracker_info["initial_value"],
                "best_epoch": 0,
                "is_better": tracker_info["is_better"],
                "model_filename": tracker_info["model_filename"],
                "best_metrics_snapshot": None,
                "criterion_weight_state": None,
            }
        for tracker_name, tracker_info in composite_metric_trackers.items():
            metric_states[tracker_name] = {
                "kind": "composite",
                "priority": tracker_info["priority"],
                "best_epoch": 0,
                "model_filename": tracker_info["model_filename"],
                "best_metrics_snapshot": None,
                "criterion_weight_state": None,
            }

    record_path = os.path.join(run_dir, "TrainingRecord.json")
    initial_record = {
        "GLOBAL_SEED": GLOBAL_SEED,
        "Trainset_size": len(train_dataset),
        "Valset_size": len(val_dataset),
        "validation_enabled": val_enabled,
        "INJURY_PROCESSED_DIR": str(INJURY_PROCESSED_DIR.resolve()),
        "data_interface": {
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
            },
            "loss": loss_params,
            "curriculum": curriculum_params,
            "model_selection_params": model_selection_params if val_enabled else {},
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
                "primary_tracker": primary_tracker_name,
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

    epoch_metrics_csv_path = os.path.join(run_dir, "epoch_metrics.csv")
    epoch_metrics_fieldnames = [
        "epoch",
        "phase",
        "alpha",
        "train_loss",
        "val_loss",
        "val_accu_mais_3c",
        "val_accu_mais",
        "val_accu_head",
        "val_accu_chest",
        "val_accu_neck",
        "val_r2_hic",
        "val_r2_dmax",
        "val_r2_nij",
    ]

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
            missing_metrics = [name for name in required_val_metric_names if name not in val_metrics]
            if missing_metrics:
                raise KeyError(f"模型选择规则中存在无效验证指标: {missing_metrics}")

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
        if write_epoch_metrics_csv:
            _append_epoch_metrics_csv(
                epoch_metrics_csv_path,
                _make_epoch_csv_row(epoch + 1, train_metrics, val_metrics, curriculum_state),
                epoch_metrics_fieldnames,
            )

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
                for tracker_name, state in metric_states.items():
                    if state["kind"] == "single":
                        metric_key = state["metric_key"]
                        current_value = val_metrics[metric_key]
                        is_better = state["is_better"](current_value, state["best_value"])
                    else:
                        is_better = is_composite_better(
                            val_metrics,
                            state["best_metrics_snapshot"],
                            state["priority"],
                        )
                    if is_better:
                        _refresh_tracker_state(state, model, current_epoch, val_metrics, criterion, run_dir)
                        display_name = metric_trackers[tracker_name]["display_name"]
                        print(f"Best {display_name} model saved at epoch {current_epoch}: {state['model_filename']}")

        primary_metric_states = {primary_tracker_name: metric_states[primary_tracker_name]} if val_enabled else {}
        if val_enabled and should_stop_early(primary_metric_states, epoch + 1, Patience, early_stop_anchor_epoch):
            print(f"Early Stop at epoch: {epoch + 1}!")
            state = metric_states[primary_tracker_name]
            print(f"Best {metric_trackers[primary_tracker_name]['display_name']} at epoch {state['best_epoch']}")
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

    # best_metrics_by_tracker 写入各 tracker 的最终最优快照；复合 tracker 记录完整验证指标快照，而不是压缩为单个标量。best_metrics_snapshot：某个 tracker 刷新最优时，同一个 epoch 的完整验证集指标快照
    best_metrics_by_tracker = {}
    for tracker_name, state in metric_states.items():
        entry = {
            "tracker_kind": state["kind"],
            "best_epoch": int(state["best_epoch"]),
            "model_file": state["model_filename"],
            "best_metrics_snapshot": round_float_fields(state["best_metrics_snapshot"], digits=4)
            if state["best_metrics_snapshot"] is not None else None,
            "criterion_weight_state": round_float_fields(state["criterion_weight_state"], digits=4)
            if state["criterion_weight_state"] is not None else None,
        }
        if state["kind"] == "single":
            entry["best_value"] = round_to_significant(float(state["best_value"]), 4)
            entry["metric_key"] = state["metric_key"]
        else:
            entry["priority"] = metric_trackers[tracker_name]["priority"]
        best_metrics_by_tracker[tracker_name] = entry

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
        "primary_tracker": primary_tracker_name,
        "last_epoch_metrics": last_epoch_metrics,
        "last_epoch_metrics_source": "val" if val_enabled else "train",
        "kendall_weight_state_final": criterion.get_weight_state(),
        "bn_recalibration": bn_recalibration_results,
    }
    _write_training_record(record_path, final_record)
    print(f"训练结果已更新至: {record_path}")
