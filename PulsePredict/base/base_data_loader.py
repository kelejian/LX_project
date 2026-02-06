import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler

class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, batch_size,  num_workers, 
                 collate_fn=default_collate,
                 training=True, train_test_indices=None, val_indices=None):
        """
        :param train_test_indices: 必须提供的训练集(或测试集)索引列表
        :param val_indices: 可选的验证集索引列表
        """
        self.val_indices = val_indices
        self.train_test_indices = train_test_indices
        self.training = training
        self.batch_idx = 0
        self.n_samples = len(dataset)

        # 获取 Sampler
        self.train_test_sampler, self.valid_sampler = self._split_sampler()

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        super().__init__(sampler=self.train_test_sampler, **self.init_kwargs) # 传入训练/测试 Sampler, 返回 相应DataLoader 实例

    def _split_sampler(self):
        """
        严格根据提供的 indices 构建 Sampler。不自行随机划分。
        """
        # 1. 严格检查：必须提供主索引 (Train or Test indices)
        if self.train_test_indices is None:
            raise ValueError("Strict Mode Error: 'train_test_indices' must be provided. "
                             "Random split logic has been removed.")

        # 2. 构建主 Sampler (用于当前 Loader 的迭代)
        train_test_sampler = SubsetRandomSampler(self.train_test_indices)
        self.n_samples = len(self.train_test_indices)

        # 3. 构建验证 Sampler（如果确定是训练模式且提供了验证索引且数量大于0）
        valid_sampler = None
        if self.training and self.val_indices is not None and len(self.val_indices) > 0:
            valid_sampler = SubsetRandomSampler(self.val_indices)
        
        return train_test_sampler, valid_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs) # 返回验证集 DataLoader