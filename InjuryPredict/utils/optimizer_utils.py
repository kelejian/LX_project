import torch
import torch.nn as nn

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
    
    # 集合用于去重检查（防止参数被重复添加）
    param_ids = set()
    
    # 统计信息
    stats = {"body": 0, "head": 0, "no_decay": 0}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # 严谨性检查：确保不重复添加
        if id(param) in param_ids:
            continue
        param_ids.add(id(param))

        # 策略 1: 不衰减组 (Bias, LayerNorm/BatchNorm 的 weight & bias)
        # 依据: PyTorch 中 Norm 参数和 Bias 都是 1 维的
        if param.ndim < 2:
            no_decay_params.append(param)
            stats["no_decay"] += param.numel()
        else:
            # 策略 2: 区分 Head 和 Body
            # 你的模型中 Head 命名为 HIC_head, Dmax_head, Nij_head，均包含 'head'
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