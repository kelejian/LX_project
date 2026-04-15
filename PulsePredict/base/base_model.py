import torch.nn as nn
import numpy as np
from abc import abstractmethod


class BaseModel(nn.Module):
    """
    模型基类。
    """
    @abstractmethod
    def forward(self, *inputs):
        """
        定义前向传播接口。
        """
        raise NotImplementedError


    @abstractmethod
    def get_metrics_output(self, model_output):
        """
        提取用于计算评估指标的主输出张量。
        """
        raise NotImplementedError

    def __str__(self):
        """
        在模型字符串后附加可训练参数量，便于日志中快速核对模型规模。
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)
