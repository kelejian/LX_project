import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from imblearn.metrics import geometric_mean_score, classification_report_imbalanced
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, accuracy_score, confusion_matrix

from InjuryPredict.config import AVAILABLE_VAL_METRIC_NAMES


def get_regression_metrics(y_true, y_pred):
    """计算并返回一组回归指标。"""
    return {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': root_mean_squared_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }


def get_classification_metrics(y_true, y_pred, labels, context_hint: str = "the data"):
    """计算并返回一组分类指标。"""
    present_labels = set(np.unique(np.concatenate([y_true, y_pred])))
    missing_labels = set(labels) - present_labels

    if missing_labels:
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


def get_compare_func(func_indicator):
    """根据配置中的比较指示器返回比较函数、初始值和判优函数。"""
    if func_indicator == max or (isinstance(func_indicator, str) and func_indicator.lower() == 'max'):
        return max, float('-inf'), lambda curr, best: curr > best
    return min, float('inf'), lambda curr, best: curr < best


def build_metric_trackers(metrics_to_track, model_filename_fn=None):
    """根据配置构建指标跟踪器字典（仅跟踪验证集指标）。"""
    if model_filename_fn is None:
        model_filename_fn = lambda metric_name: f"best_val_{metric_name}.pth"

    trackers = {}
    for metric_name, compare_indicator in metrics_to_track:
        raw_metric_name = str(metric_name).strip()
        metric_key = raw_metric_name[4:] if raw_metric_name.startswith('val_') else raw_metric_name
        if metric_key not in AVAILABLE_VAL_METRIC_NAMES:
            raise ValueError(
                f"无效的验证指标名: {metric_name}. "
                f"可选项: {list(AVAILABLE_VAL_METRIC_NAMES)}"
            )

        if compare_indicator in (max, min):
            compare_mode = 'max' if compare_indicator == max else 'min'
        elif isinstance(compare_indicator, str):
            compare_mode = compare_indicator.lower().strip()
        else:
            raise ValueError(f"无效的比较方式: {compare_indicator}. 仅支持 'max' 或 'min'.")

        if compare_mode not in ('max', 'min'):
            raise ValueError(f"无效的比较方式: {compare_indicator}. 仅支持 'max' 或 'min'.")

        _, initial_value, is_better = get_compare_func(compare_indicator)
        trackers[metric_key] = {
            'compare_indicator': compare_mode,
            'initial_value': initial_value,
            'is_better': is_better,
            'model_filename': model_filename_fn(metric_key),
            'display_name': f"val/{metric_key}",
        }
    return trackers


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
