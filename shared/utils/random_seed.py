# 全局配置文件
GLOBAL_SEED = 123

def set_random_seed(seed=GLOBAL_SEED):
    """设置全局随机种子"""
    import numpy as np
    import random
    
    # Set numpy and random seeds
    np.random.seed(seed)
    random.seed(seed)
    
    # Try to set PyTorch seeds if available
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        
        # 确保CUDA操作的确定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        # PyTorch not installed, skip torch-specific settings
        pass