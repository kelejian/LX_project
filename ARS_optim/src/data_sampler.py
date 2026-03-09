import logging
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch

from common.settings import RAW_DATA_DIR, SPLIT_INDICES_DIR

from ARS_optim.src.constraints import ConstraintEngine
from ARS_optim.src.param_manager import ParamManager


class StateDataSampler:
    """从 InjuryPredict 训练经验池中按行采样上下文数据。

    训练期只保留两项职责：
    1. 从经验池复刻边际分布；
    2. 在连续 context 上加轻微扰动，再统一交给约束引擎收敛回可行域。
    """

    def __init__(
        self,
        param_manager: ParamManager,
        constraint_engine: ConstraintEngine,
        batch_size: int,
        device: torch.device = torch.device("cpu"),
        seed: Optional[int] = None,
        pool_npz_path: Optional[str] = None,
        split_indices_path: Optional[str] = None,
        jitter_ratio: float = 0.01,
        jitter_prob: float = 1.0,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.param_manager = param_manager
        self.constraint_engine = constraint_engine
        self.batch_size = int(batch_size)
        self.device = device
        self.jitter_ratio = float(jitter_ratio)
        self.jitter_prob = float(jitter_prob)

        self.rng = torch.Generator(device=device)
        if seed is not None:
            self.rng.manual_seed(int(seed))

        self.context_params = self.param_manager.get_context_params()
        self.context_indices = self.param_manager.get_context_indices()

        pool_path = Path(pool_npz_path) if pool_npz_path else (RAW_DATA_DIR / "raw_data_packed.npz")
        split_path = Path(split_indices_path) if split_indices_path else (SPLIT_INDICES_DIR / "injury_train_indices.npy")
        if not split_path.exists():
            raise FileNotFoundError(f"经验池切分索引不存在: {split_path}")
        self.pool_path = pool_path
        self.split_path = split_path
        self.using_split_indices = True

        full_features = self._load_feature_matrix_from_pool(pool_path)
        split_indices = np.load(split_path)
        full_features = full_features[split_indices]
        self.logger.info(f"经验池使用切分索引: {split_path}，样本数={full_features.shape[0]}")
        if full_features.shape[0] == 0:
            raise ValueError("经验池为空，无法构建数据流")

        self.pool_full = torch.tensor(full_features, dtype=torch.float32, device=device)
        self.pool_context = torch.tensor(full_features[:, self.context_indices], dtype=torch.float32, device=device)
        self.pool_size = self.pool_context.shape[0]

        self.context_cont_local_indices = [
            idx for idx, param in enumerate(self.context_params) if param.get("type") == "continuous"
        ]
        if self.context_cont_local_indices:
            mins = [self.context_params[idx]["min"] for idx in self.context_cont_local_indices]
            maxs = [self.context_params[idx]["max"] for idx in self.context_cont_local_indices]
            self.cont_mins = torch.tensor(mins, dtype=torch.float32, device=device)
            self.cont_maxs = torch.tensor(maxs, dtype=torch.float32, device=device)
            self.cont_spans = torch.clamp(self.cont_maxs - self.cont_mins, min=1e-6)
        else:
            self.cont_mins = None
            self.cont_maxs = None
            self.cont_spans = None

    def _load_feature_matrix_from_pool(self, pool_path: Path) -> np.ndarray:
        if not pool_path.exists():
            raise FileNotFoundError(f"经验池文件不存在: {pool_path}")
        with np.load(pool_path, allow_pickle=True) as data:
            if "x_att_raw" not in data:
                raise KeyError("经验池缺少 x_att_raw")
            features = np.asarray(data["x_att_raw"], dtype=np.float32)
        if features.ndim != 2 or features.shape[1] < self.param_manager.get_total_feature_dim():
            raise ValueError(f"经验池特征矩阵形状异常: {features.shape}")
        return features[:, : self.param_manager.get_total_feature_dim()]

    def _apply_bounded_jitter(self, batch_context: torch.Tensor) -> torch.Tensor:
        if not self.context_cont_local_indices or self.jitter_ratio <= 0:
            return batch_context
        # 只对连续 context 特征施加扰动：
        # 1. 离散变量没有连续噪声语义，直接加噪会破坏类别含义；
        # 2. trainable 控制量不属于经验池输入，它们由策略网络或局部精调阶段产生。
        continuous = batch_context[:, self.context_cont_local_indices]
        noise = torch.randn(continuous.shape, generator=self.rng, device=self.device, dtype=torch.float32)
        noise = noise * (self.cont_spans.unsqueeze(0) * self.jitter_ratio)
        if self.jitter_prob < 1.0:
            mask = torch.rand(continuous.shape, generator=self.rng, device=self.device) < self.jitter_prob
            noise = noise * mask.float()
        continuous = torch.clamp(continuous + noise, self.cont_mins.unsqueeze(0), self.cont_maxs.unsqueeze(0))
        batch_context[:, self.context_cont_local_indices] = continuous
        return batch_context

    def _generate_batch(self) -> torch.Tensor:
        indices = torch.randint(0, self.pool_size, (self.batch_size,), generator=self.rng, device=self.device)
        batch_context = self.pool_context[indices].clone()
        # 先对经验池样本做轻微连续扰动，再统一送入约束引擎净化。这样既能保持训练数据流接近真实边际分布，又能确保扰动后的 RA/座椅参数/碰撞工况重新落回 step0 定义的可行域中。
        batch_context = self._apply_bounded_jitter(batch_context)
        return self.constraint_engine.sanitize_context(batch_context)

    def get_infinite_generator(self) -> Iterator[torch.Tensor]:
        while True:
            yield self._generate_batch()

    def get_dataset_tensor(self) -> torch.Tensor:
        return self.pool_context

    def get_dataset_size(self) -> int:
        return int(self.pool_size)

    def get_source_info(self) -> dict:
        """返回当前采样器的数据流元信息。

        训练虽然按无限流取样，但数据来源并不是开放集合，而是固定的 injury 经验池切片。
        这里显式记录经验池路径、切分索引和扰动配置，便于训练摘要准确复现本次数据流语义。
        """
        return {
            "pool_npz_path": str(self.pool_path.resolve()),
            "split_indices_path": str(self.split_path.resolve()) if self.using_split_indices else None,
            "using_split_indices": bool(self.using_split_indices),
            "dataset_size": int(self.pool_size),
            "batch_size": int(self.batch_size),
            "jitter_ratio": float(self.jitter_ratio),
            "jitter_prob": float(self.jitter_prob),
            "continuous_context_names": [
                self.context_params[idx]["name"] for idx in self.context_cont_local_indices
            ],
        }

    def get_distribution_reference(self, max_samples: int = 0, shuffle: bool = False, feature_space: str = "context", trainable_indices: Optional[list] = None) -> torch.Tensor:
        if feature_space == "context":
            ref = self.pool_context
        elif feature_space == "context_control":
            if trainable_indices is None:
                raise ValueError("feature_space=context_control 时必须提供 trainable_indices")
            control_ref = self.pool_full[:, trainable_indices] if trainable_indices else self.pool_full[:, :0]
            ref = torch.cat([self.pool_context, control_ref], dim=1)
        else:
            raise ValueError(f"不支持的 feature_space: {feature_space}")

        if max_samples is None or int(max_samples) <= 0 or ref.shape[0] <= int(max_samples):
            return ref
        count = int(max_samples)
        if shuffle:
            indices = torch.randperm(ref.shape[0], generator=self.rng, device=self.device)[:count]
        else:
            indices = torch.arange(count, device=self.device)
        return ref[indices]

    def iter_dataset_batches(self, batch_size: Optional[int] = None, shuffle: bool = False) -> Iterator[torch.Tensor]:
        batch_size = int(batch_size) if batch_size is not None else self.batch_size
        if batch_size <= 0:
            raise ValueError("batch_size 必须为正整数")
        sample_count = self.pool_context.shape[0]
        order = (
            torch.randperm(sample_count, generator=self.rng, device=self.device)
            if shuffle
            else torch.arange(sample_count, device=self.device)
        )
        for start in range(0, sample_count, batch_size):
            indices = order[start : start + batch_size]
            yield self.pool_context[indices]
