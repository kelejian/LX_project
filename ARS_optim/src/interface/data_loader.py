import logging
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch

from common.settings import FEATURE_ORDER, RAW_DATA_DIR, SPLIT_INDICES_DIR
from ARS_optim.src.core.param_manager import ParamManager


class StateDataLoaderManager:
    """
    经验池驱动的数据流采样器。

    核心逻辑：
    - 从损伤预测训练数据经验池中按行抽样，天然复刻边际分布。
    - 输出上下文参数（state + trainable=False 的 control），作为策略输入与局部优化锚点。
    - 对连续上下文字段施加有界微扰，并强制裁剪到 param_space 定义范围内。
    - 额外保留 overlap-angle 的硬规则修复，避免微扰后违例。
    """

    def __init__(
        self,
        param_manager: ParamManager,
        batch_size: int,
        device: torch.device = torch.device('cpu'),
        seed: Optional[int] = None,
        pool_npz_path: Optional[str] = None,
        train_indices_path: Optional[str] = None,
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
        self.context_names = self.param_manager.get_context_names()

        pool_path = Path(pool_npz_path) if pool_npz_path else (RAW_DATA_DIR / 'raw_data_packed.npz')
        split_path = Path(train_indices_path) if train_indices_path else (SPLIT_INDICES_DIR / 'injury_train_indices.npy')

        full_features = self._load_feature_matrix_from_pool(pool_path)
        if split_path.exists():
            train_idx = np.load(split_path)
            full_features = full_features[train_idx]
            self.logger.info(f"经验池使用训练切分索引: {split_path}，样本数={full_features.shape[0]}")
        else:
            self.logger.warning(f"未找到训练切分索引: {split_path}，将使用全量经验池")

        if full_features.shape[0] == 0:
            raise ValueError("经验池为空，无法构建数据流")

        # [N, D_total] -> [N, D_context]
        self.pool_context = torch.tensor(full_features[:, self.context_indices], dtype=torch.float32, device=device)
        self.pool_size = self.pool_context.size(0)

        # 连续型上下文字段局部索引与边界（用于有界扰动）
        self.context_cont_local_indices = [i for i, p in enumerate(self.context_params) if p.get('type') == 'continuous']
        if self.context_cont_local_indices:
            mins = [self.context_params[i]['min'] for i in self.context_cont_local_indices]
            maxs = [self.context_params[i]['max'] for i in self.context_cont_local_indices]
            self.cont_mins = torch.tensor(mins, dtype=torch.float32, device=device)
            self.cont_maxs = torch.tensor(maxs, dtype=torch.float32, device=device)
            self.cont_spans = torch.clamp(self.cont_maxs - self.cont_mins, min=1e-6)
        else:
            self.cont_mins = self.cont_maxs = self.cont_spans = None

        # overlap-angle 规则相关列（在上下文局部坐标）
        self.local_overlap = self.context_names.index('overlap') if 'overlap' in self.context_names else None
        self.local_angle = self.context_names.index('impact_angle') if 'impact_angle' in self.context_names else None

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
                # 回退：按 FEATURE_ORDER 尝试逐列拼接
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

    def _enforce_overlap_angle_rule(self, batch_context: torch.Tensor) -> torch.Tensor:
        if self.local_overlap is None or self.local_angle is None:
            return batch_context

        overlap = batch_context[:, self.local_overlap]
        angle = batch_context[:, self.local_angle]
        mask = (overlap.abs() >= 0.25) & (overlap.abs() < 0.3)
        bad = mask & ((torch.sign(angle) == torch.sign(overlap)) | (angle.abs() <= 30.0))
        n_bad = int(bad.sum().item())
        if n_bad == 0:
            return batch_context

        u = torch.rand(n_bad, generator=self.rng, device=self.device)
        repaired_mag = 30.0 + 15.0 * u
        repaired_angle = torch.where(overlap[bad] > 0, -repaired_mag, repaired_mag)
        batch_context[bad, self.local_angle] = repaired_angle
        return batch_context

    def _generate_batch(self) -> torch.Tensor:
        idx = torch.randint(0, self.pool_size, (self.batch_size,), generator=self.rng, device=self.device)
        batch = self.pool_context[idx].clone()
        batch = self._apply_bounded_jitter(batch)
        batch = self._enforce_overlap_angle_rule(batch)
        return batch

    def get_infinite_generator(self) -> Iterator[torch.Tensor]:
        while True:
            yield self._generate_batch()