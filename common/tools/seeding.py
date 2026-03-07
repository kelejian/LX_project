import torch
import numpy as np
import random
import os

GLOBAL_SEED = 123

def set_random_seed(seed=GLOBAL_SEED):
    """
    设置全局随机种子以保证可复现性。
    设置范围包括: random, numpy, torch, torch.cuda。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 确保确定性行为
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except AttributeError:
        pass # 防止 cudnn 后端不可用的情况
