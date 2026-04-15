import random

import numpy as np
import torch

GLOBAL_SEED = 123


def set_random_seed(seed=GLOBAL_SEED):
    """
    统一设置项目中常用随机源的种子。

    该函数面向训练、评估和实验脚本的入口阶段，负责同步设置
    Python `random`、NumPy 与 PyTorch 的随机状态，并关闭 cuDNN
    的 benchmark 以减少同一环境下的额外波动。

    Args:
        seed: 本次运行使用的随机种子。默认取 `GLOBAL_SEED`。
    """
    seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 优先保证同一脚本、同一环境下的可复现性。
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except AttributeError:
        pass
