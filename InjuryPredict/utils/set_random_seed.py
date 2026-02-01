# 全局配置文件
GLOBAL_SEED = 123

def set_random_seed(seed=GLOBAL_SEED):
    """设置全局随机种子"""
    import torch
    import numpy as np
    import random
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # 确保CUDA操作的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False