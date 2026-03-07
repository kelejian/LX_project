import logging
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch

from common.settings import FEATURE_ORDER, RAW_DATA_DIR, SPLIT_INDICES_DIR
from ARS_optim.src.core.param_manager import ParamManager
from ARS_optim.src.core.rule_engine import RuleEngine


class StateDataLoaderManager:
    """
    经验池驱动的数据流采样器。

    核心逻辑：
    - 从损伤预测训练数据经验池中按行抽样，天然复刻边际分布。
    - 输出上下文参数（state + trainable=False 的 control），作为策略输入与局部优化锚点。
    - 仅在本地做连续字段的小幅扰动；所有 step0 对应的硬规则收敛统一交给 RuleEngine。
    - 这样训练采样、评估输入修复、局部精调投影会共享同一套规则语义。

        输入/输出示例：
        - 经验池完整特征: [N, 13]
        - 输出 batch_context: [Batch, D_context]
            例如当前参数空间下，D_context 可为 10（state + 固定 control）。
    """

    def __init__(
        self,
        param_manager: ParamManager,
        batch_size: int,
        device: torch.device = torch.device('cpu'),
        seed: Optional[int] = None,
        pool_npz_path: Optional[str] = None,
        split_indices_path: Optional[str] = None,
        jitter_ratio: float = 0.01,
        jitter_prob: float = 1.0,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.param_manager = param_manager
        self.batch_size = int(batch_size)
        self.device = device
        self.seed = seed
        self.jitter_ratio = float(jitter_ratio)
        self.jitter_prob = float(jitter_prob)

        self.rng = torch.Generator(device=device)
        if seed is not None:
            self.rng.manual_seed(int(seed))

        self.context_params = self.param_manager.get_context_params()
        self.context_indices = self.param_manager.get_context_indices()
        self.rule_engine = RuleEngine(self.param_manager)

        pool_path = Path(pool_npz_path) if pool_npz_path else (RAW_DATA_DIR / 'raw_data_packed.npz')
        split_path = Path(split_indices_path) if split_indices_path else (SPLIT_INDICES_DIR / 'injury_train_indices.npy')

        full_features = self._load_feature_matrix_from_pool(pool_path)
        if split_path.exists():
            train_idx = np.load(split_path)
            full_features = full_features[train_idx]
            self.logger.info(f"经验池使用训练切分索引: {split_path}，样本数={full_features.shape[0]}")
        else:
            self.logger.warning(f"未找到训练切分索引: {split_path}，将使用全量经验池")

        if full_features.shape[0] == 0:
            raise ValueError("经验池为空，无法构建数据流")

        self.pool_full = torch.tensor(full_features, dtype=torch.float32, device=device)
        self.pool_context = torch.tensor(full_features[:, self.context_indices], dtype=torch.float32, device=device)
        self.pool_size = self.pool_context.size(0)

        self.context_cont_local_indices = [i for i, param in enumerate(self.context_params) if param.get('type') == 'continuous']
        if self.context_cont_local_indices:
            mins = [self.context_params[i]['min'] for i in self.context_cont_local_indices]
            maxs = [self.context_params[i]['max'] for i in self.context_cont_local_indices]
            self.cont_mins = torch.tensor(mins, dtype=torch.float32, device=device)
            self.cont_maxs = torch.tensor(maxs, dtype=torch.float32, device=device)
            self.cont_spans = torch.clamp(self.cont_maxs - self.cont_mins, min=1e-6)
        else:
            self.cont_mins = self.cont_maxs = self.cont_spans = None

    def _load_feature_matrix_from_pool(self, pool_path: Path) -> np.ndarray:
        if not pool_path.exists():
            raise FileNotFoundError(f"经验池文件不存在: {pool_path}")

        with np.load(pool_path, allow_pickle=True) as data:
            key_candidates = ['x_att_raw', 'att_raw', 'x_att', 'features']
            arr = None
            for key in key_candidates:
                if key in data.files:
                    arr = data[key]
                    break

            if arr is None:
                cols = []
                for name in FEATURE_ORDER:
                    if name not in data.files:
                        raise KeyError(f"经验池缺少特征列: {name}")
                    cols.append(np.asarray(data[name]).reshape(-1, 1))
                arr = np.concatenate(cols, axis=1)

        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"经验池特征矩阵维度异常: {arr.shape}")
        if arr.shape[1] < len(FEATURE_ORDER):
            raise ValueError(f"经验池特征维度不足: {arr.shape[1]} < {len(FEATURE_ORDER)}")
        return arr[:, :len(FEATURE_ORDER)]

    def _apply_bounded_jitter(self, batch_context: torch.Tensor) -> torch.Tensor:
        """对连续上下文字段施加有界扰动。"""
        if not self.context_cont_local_indices or self.jitter_ratio <= 0:
            return batch_context

        cont = batch_context[:, self.context_cont_local_indices]
        noise = torch.randn(cont.shape, generator=self.rng, device=self.device, dtype=torch.float32)
        noise = noise * (self.cont_spans.unsqueeze(0) * self.jitter_ratio)

        if self.jitter_prob < 1.0:
            mask = torch.rand(cont.shape, generator=self.rng, device=self.device) < self.jitter_prob
            noise = noise * mask.float()

        cont = cont + noise
        cont = torch.clamp(cont, self.cont_mins.unsqueeze(0), self.cont_maxs.unsqueeze(0))
        batch_context[:, self.context_cont_local_indices] = cont
        return batch_context

    def _generate_batch(self) -> torch.Tensor:
        idx = torch.randint(0, self.pool_size, (self.batch_size,), generator=self.rng, device=self.device)
        batch = self.pool_context[idx].clone()
        batch = self._apply_bounded_jitter(batch)
        return self.rule_engine.sanitize_context(batch)

    def get_infinite_generator(self) -> Iterator[torch.Tensor]:
        """返回无限数据流生成器，供训练循环持续调用 next()。"""
        while True:
            yield self._generate_batch()

    def get_dataset_tensor(self) -> torch.Tensor:
        """返回当前 split 对应的完整上下文张量。"""
        return self.pool_context

    def get_distribution_reference(
        self,
        max_samples: int = 0,
        shuffle: bool = False,
        feature_space: str = 'context',
        trainable_indices: Optional[list] = None,
    ) -> torch.Tensor:
        """获取用于训练分布偏离惩罚拟合的参考样本。"""
        mode = str(feature_space).lower()
        if mode == 'context':
            ref = self.pool_context
        elif mode == 'context_control':
            if trainable_indices is None:
                raise ValueError("feature_space=context_control 时必须提供 trainable_indices")
            if len(trainable_indices) == 0:
                ref = self.pool_context
            else:
                control_ref = self.pool_full[:, trainable_indices]
                ref = torch.cat([self.pool_context, control_ref], dim=1)
        else:
            raise ValueError(f"不支持的 feature_space: {feature_space}")

        if max_samples is None or int(max_samples) <= 0 or ref.shape[0] <= int(max_samples):
            return ref

        n = int(max_samples)
        if shuffle:
            idx = torch.randperm(ref.shape[0], generator=self.rng, device=self.device)[:n]
        else:
            idx = torch.arange(n, device=self.device)
        return ref[idx]

    def iter_dataset_batches(self, batch_size: Optional[int] = None, shuffle: bool = False) -> Iterator[torch.Tensor]:
        """按批遍历当前 split 的完整数据集。"""
        bs = int(batch_size) if batch_size is not None else int(self.batch_size)
        if bs <= 0:
            raise ValueError("batch_size 必须为正整数")

        n = self.pool_context.size(0)
        if n == 0:
            return

        if shuffle:
            order = torch.randperm(n, generator=self.rng, device=self.device)
        else:
            order = torch.arange(n, device=self.device)

        for start in range(0, n, bs):
            idx = order[start:start + bs]
            yield self.pool_context[idx]