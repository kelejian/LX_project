import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from imblearn.metrics import geometric_mean_score, classification_report_imbalanced
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, accuracy_score, confusion_matrix

from InjuryPredict.config import AVAILABLE_VAL_METRIC_NAMES


MAIS_3C_LABELS = [0, 1, 2]
MAIS_3C_DISPLAY_LABELS = ['0', '1-2', '3+']


def get_regression_metrics(y_true, y_pred):
    """计算并返回一组回归指标。"""
    return {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': root_mean_squared_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }


def get_classification_metrics(y_true, y_pred, labels, context_hint: str = "the data", warn_missing_labels: bool = True):
    """计算并返回一组分类指标。"""
    present_labels = set(np.unique(np.concatenate([y_true, y_pred])))
    missing_labels = set(labels) - present_labels

    if warn_missing_labels and missing_labels:
        print(f"\n*Warning: Labels {missing_labels} are not present in {context_hint}\n")

    return {
        'accuracy': accuracy_score(y_true, y_pred) * 100,
        'g_mean': geometric_mean_score(y_true, y_pred, labels=labels, average='multiclass'),
        'conf_matrix': confusion_matrix(y_true, y_pred, labels=labels),
        'report': classification_report_imbalanced(
            y_true, y_pred, labels=labels, digits=3,
            zero_division=0
        )
    }


def convert_mais_to_3c(y):
    """将 MAIS 0~5 映射到三分类: 0, 1-2, 3+。"""
    y = np.asarray(y, dtype=np.int64)
    return np.where(y <= 0, 0, np.where(y <= 2, 1, 2))


def get_mais_3c_metrics(y_true, y_pred, context_hint: str = "the data", warn_missing_labels: bool = True):
    """基于 MAIS 原始分级计算三分类指标。"""
    y_true_3c = convert_mais_to_3c(y_true)
    y_pred_3c = convert_mais_to_3c(y_pred)
    metrics = get_classification_metrics(
        y_true_3c,
        y_pred_3c,
        MAIS_3C_LABELS,
        context_hint=context_hint,
        warn_missing_labels=warn_missing_labels,
    )
    metrics['mapped_y_true'] = y_true_3c
    metrics['mapped_y_pred'] = y_pred_3c
    return metrics


def plot_scatter(y_true, y_pred, ais_true, title, xlabel, save_path):
    """绘制并保存散点图。"""
    plt.figure(figsize=(8, 7))
    colors = ['blue', 'green', 'yellow', 'orange', 'red', 'darkred']

    ais_indices = np.clip(ais_true, 0, 5).astype(int)
    ais_colors = [colors[i] for i in ais_indices]
    plt.scatter(y_true, y_pred, c=ais_colors, alpha=0.5, s=40)

    legend_elements = [Patch(facecolor=colors[i], label=f'AIS {i}') for i in range(6) if i in np.unique(ais_true)]

    max_val = max(np.max(y_true), np.max(y_pred)) * 1.05
    min_val = min(np.min(y_true), np.min(y_pred))
    min_val = min(0, min_val * 1.05)

    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Ideal Line")
    plt.xlabel(f"Ground Truth ({xlabel})", fontsize=16)
    plt.ylabel(f"Predictions ({xlabel})", fontsize=16)
    plt.title(f"Scatter Plot of Predictions vs Ground Truth\n({title})", fontsize=18)
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)

    first_legend = plt.legend(handles=legend_elements, title='AIS Level', loc='upper left')
    plt.gca().add_artist(first_legend)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix(cm, labels, title, save_path):
    """绘制并保存混淆矩阵图。"""
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, fontsize=12)
    plt.yticks(tick_marks, labels, fontsize=12)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)

    thresh = cm.max() / 2. if cm.max() > 0 else 0.5
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=12)
    plt.tight_layout(pad=0.5)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()


def get_compare_func(compare_mode: str):
    """根据 'max' 或 'min' 返回比较函数、初始值和判优函数。"""
    if compare_mode == 'max':
        return max, float('-inf'), lambda curr, best: curr > best # lambda curr, best: curr > best 是一个匿名函数，用于比较当前值和最佳值，返回 True 如果当前值更优（更大），否则返回 False。
    if compare_mode == 'min':
        return min, float('inf'), lambda curr, best: curr < best
    raise ValueError(f"无效的比较方式: {compare_mode}.")


def normalize_val_metric_name(metric_name: str) -> str:
    """将配置层指标名规范化为 run_one_epoch 返回的验证指标 key。"""
    metric_key = str(metric_name).strip()
    metric_key = metric_key[4:] if metric_key.startswith('val_') else metric_key
    if metric_key not in AVAILABLE_VAL_METRIC_NAMES:
        raise ValueError(
            f"无效的验证指标名: {metric_name}. "
            f"可选项: {list(AVAILABLE_VAL_METRIC_NAMES)}"
        )
    return metric_key


def normalize_compare_mode(compare_indicator: str) -> str:
    """将配置中的字符串比较方式统一为 'max' / 'min'。"""
    if isinstance(compare_indicator, str):
        compare_mode = compare_indicator.lower().strip()
        if compare_mode in ('max', 'min'):
            return compare_mode
    raise ValueError(f"无效的比较方式: {compare_indicator}. 仅支持 'max' 或 'min'.")


def _is_metric_value_better(current_value: float, best_value: float, compare_mode: str, min_delta: float = 0.0) -> bool:
    """按方向和最小改进幅度判断单个指标是否明确优于历史最优值。"""
    current_value = float(current_value)
    best_value = float(best_value)
    min_delta = float(min_delta)
    if compare_mode == 'max':
        return current_value - best_value > min_delta
    if compare_mode == 'min':
        return best_value - current_value > min_delta
    raise ValueError(f"无效的比较方式: {compare_mode}.")


def build_single_metric_trackers(metric_configs, model_filename_fn=None):
    """
    构建单指标验证集选模规则。

    metric_configs 中的每个条目必须包含 name 与 mode；filename 可选，未提供时按指标名生成默认权重文件名。
    """
    if model_filename_fn is None:
        model_filename_fn = lambda metric_name: f"best_val_{metric_name}.pth"

    trackers = {}
    for item in metric_configs:
        if not isinstance(item, dict):
            raise TypeError("single_metric_trackers 的每个条目必须是包含 name/mode/filename 的字典。")
        raw_metric_name = item["name"]
        compare_indicator = item["mode"]
        explicit_filename = item.get("filename")

        metric_key = normalize_val_metric_name(raw_metric_name)
        compare_mode = normalize_compare_mode(compare_indicator)
        _, initial_value, is_better = get_compare_func(compare_mode)
        trackers[metric_key] = {
            'kind': 'single',
            'metric_key': metric_key,
            'compare_indicator': compare_mode,
            'initial_value': initial_value,
            'is_better': is_better,
            'model_filename': explicit_filename or model_filename_fn(metric_key),
            'display_name': f"val/{metric_key}",
        }
    return trackers


def build_composite_metric_trackers(composite_configs):
    """根据优先级列表构建复合验证集选模规则。"""
    trackers = {}
    for item in composite_configs:
        tracker_name = str(item["name"]).strip()
        if not tracker_name:
            raise ValueError("复合选模规则的 name 不能为空。")
        if tracker_name in trackers:
            raise ValueError(f"复合选模规则名称重复: {tracker_name}")

        priority = []
        for rule in item.get("priority", []):
            priority.append({
                "metric_key": normalize_val_metric_name(rule["metric"]),
                "compare_indicator": normalize_compare_mode(rule["mode"]),
                "min_delta": float(rule.get("min_delta", 0.0)),
            })
        if not priority:
            raise ValueError(f"复合选模规则 {tracker_name} 必须包含至少一个 priority 条目。")

        trackers[tracker_name] = {
            'kind': 'composite',
            'name': tracker_name,
            'priority': priority,
            'model_filename': item.get("filename", f"best_val_{tracker_name}.pth"),
            'display_name': f"val/{tracker_name}",
        }
    return trackers


def is_composite_better(current_metrics: dict, best_metrics: dict | None, priority: list) -> bool:
    """
    按固定优先级列表判断当前验证指标是否优于历史最优快照。

    每一级指标只有超过 min_delta 时才产生明确优劣；若该级无明确差异，则继续比较下一优先级。
    """
    if best_metrics is None:
        return True

    for rule in priority:
        metric_key = rule["metric_key"]
        compare_mode = rule["compare_indicator"]
        min_delta = float(rule.get("min_delta", 0.0))
        if metric_key not in current_metrics:
            raise KeyError(f"当前验证指标缺少复合选模所需字段: {metric_key}")
        if metric_key not in best_metrics:
            raise KeyError(f"历史最优指标快照缺少复合选模所需字段: {metric_key}")

        current_value = float(current_metrics[metric_key])
        best_value = float(best_metrics[metric_key])
        if _is_metric_value_better(current_value, best_value, compare_mode, min_delta):
            return True
        if _is_metric_value_better(best_value, current_value, compare_mode, min_delta):
            return False
    return False

def get_parameter_groups(model, weight_decay=1e-2, head_decay_ratio=0.1, head_keywords=('head',), verbose=True):
    """
    精细化参数分组策略 (AdamW 最佳实践):
    1. Body Group (高 WD): 骨干网络权重 (Conv, Linear, Embedding)，维持正则化。
    2. Head Group (低 WD): 输出头权重，允许自由拟合物理量级 (针对回归任务优化)。
    3. No Decay Group (0 WD): 所有 Bias 和 Normalization 层参数 (1D tensor)，保持数值稳定性。

    :param model: 模型实例
    :param weight_decay: 全局(Body)的权重衰减系数
    :param head_decay_ratio: Head 部分的 WD 缩放比例
    :param head_keywords: 识别 Head 参数的关键词元组 (默认为 'head'，适配 InjuryPredictModel)
    :param verbose: 是否打印分组统计信息
    """
    decay_body_params = []
    decay_head_params = []
    no_decay_params = []

    param_ids = set()
    stats = {"body": 0, "head": 0, "no_decay": 0}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if id(param) in param_ids:
            continue
        param_ids.add(id(param))

        if param.ndim < 2:
            no_decay_params.append(param)
            stats["no_decay"] += param.numel()
        else:
            if any(k in name for k in head_keywords):
                decay_head_params.append(param)
                stats["head"] += param.numel()
            else:
                decay_body_params.append(param)
                stats["body"] += param.numel()

    if verbose:
        print(f"\n[Optimizer] 参数分组统计:")
        print(f"  - Body (WD={weight_decay}): {len(decay_body_params)} tensors, {stats['body']} params")
        print(f"  - Head (WD={weight_decay*head_decay_ratio}): {len(decay_head_params)} tensors, {stats['head']} params")
        print(f"  - No Decay (WD=0.0): {len(no_decay_params)} tensors, {stats['no_decay']} params")

    return [
        {'params': decay_body_params, 'weight_decay': weight_decay},
        {'params': decay_head_params, 'weight_decay': weight_decay * head_decay_ratio},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]


def round_to_significant(value, digits=4):
    """将数值按有效数字保留，默认 4 位。"""
    if isinstance(value, (bool, np.bool_)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if value == 0:
            return 0.0
        return float(f"{float(value):.{digits}g}")
    return value


def round_float_fields(data, digits=4):
    """递归地将容器中的浮点数按有效数字保留。"""
    if isinstance(data, dict):
        return {k: round_float_fields(v, digits=digits) for k, v in data.items()}
    if isinstance(data, list):
        return [round_float_fields(v, digits=digits) for v in data]
    if isinstance(data, tuple):
        return tuple(round_float_fields(v, digits=digits) for v in data)
    return round_to_significant(data, digits=digits)


def convert_numpy_types(obj):
    """递归转换 NumPy 类型为 Python 原生类型，便于 JSON 序列化。"""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    if isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    if isinstance(obj, (np.integer, np.int_)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
