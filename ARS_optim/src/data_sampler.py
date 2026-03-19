import logging
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch

from common.data_utils.split_io import load_int_vector_csv
from common.settings import RAW_DATA_DIR, SPLIT_INDICES_DIR

from ARS_optim.src.constraints import ConstraintEngine
from ARS_optim.src.param_manager import ParamManager


class StateDataSampler:
    """从 InjuryPredict 训练经验池中按行采样上下文数据。

    训练期只保留两项职责：
    1. 从经验池复刻边际分布；
    2. 在连续 context 上加轻微扰动，并对非法扰动做拒绝回退。
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
        self.seed = int(seed) if seed is not None else None
        self.jitter_ratio = float(jitter_ratio)
        self.jitter_prob = float(jitter_prob)
        self._last_rejection_rate = 0.0

        self.rng = torch.Generator(device=device)
        if self.seed is not None:
            self.rng.manual_seed(self.seed)

        self.context_params = self.param_manager.get_context_params()
        self.context_indices = self.param_manager.get_context_indices()

        pool_path = Path(pool_npz_path) if pool_npz_path else (RAW_DATA_DIR / "raw_data_packed.npz")
        split_path = Path(split_indices_path) if split_indices_path else (SPLIT_INDICES_DIR / "injury_train_indices.csv")
        if not split_path.exists():
            raise FileNotFoundError(f"经验池切分索引不存在: {split_path}")
        self.pool_path = pool_path
        self.split_path = split_path

        full_features = self._load_feature_matrix_from_pool(pool_path)
        split_indices = load_int_vector_csv(split_path)
        full_features = full_features[split_indices]
        self.logger.info(f"经验池使用切分索引: {split_path}，样本数={full_features.shape[0]}")
        if full_features.shape[0] == 0:
            raise ValueError("经验池为空，无法构建数据流")

        self.pool_full = torch.tensor(full_features, dtype=torch.float32, device=device)
        self.pool_context = torch.tensor(full_features[:, self.context_indices], dtype=torch.float32, device=device)
        self.pool_size = self.pool_context.shape[0]
        if self.pool_size == 0:
            raise ValueError("经验池为空，无法构建数据流")

        self.context_cont_local_indices = [
            idx for idx, param in enumerate(self.context_params) if param.get("type") == "continuous"
        ]
        if self.context_cont_local_indices:
            # 训练采样阶段属于输入端 context 扰动：
            # 即便某些 context 列来自当前不可调 control（如 LL1/LL2），
            # 这里的 base 范围只用于定义各列扰动尺度，不用于对试探样本做后置截断；
            # 真正的合法性判断统一交给约束引擎，若扰动后越界则整行回退原样本。
            mins = [
                float(self.context_params[idx]["base_min"])
                for idx in self.context_cont_local_indices
            ]
            maxs = [
                float(self.context_params[idx]["base_max"])
                for idx in self.context_cont_local_indices
            ]
            self.cont_mins = torch.tensor(mins, dtype=torch.float32, device=device)
            self.cont_maxs = torch.tensor(maxs, dtype=torch.float32, device=device)
            self.cont_spans = torch.clamp(self.cont_maxs - self.cont_mins, min=1e-6)
        else:
            self.cont_mins = None
            self.cont_maxs = None
            self.cont_spans = None

        cont_name_to_pos = {
            self.context_params[local_idx]["name"]: pos
            for pos, local_idx in enumerate(self.context_cont_local_indices)
        }
        self._overlap_cont_pos = cont_name_to_pos.get("overlap")
        self._angle_cont_pos = cont_name_to_pos.get("impact_angle")

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
            self._last_rejection_rate = 0.0
            return batch_context

        original_context = batch_context.clone()

        # 这里只对连续 context 列加噪声；离散参数 is_driver_side / OT 不在这个切片里，
        # 因而天然不会被扰动。这样可避免对离散标签做“先映射成浮点再回整”的伪连续处理。
        continuous = batch_context[:, self.context_cont_local_indices]

        feature_mask = torch.ones_like(continuous, dtype=torch.float32)
        if self._overlap_cont_pos is not None and self._angle_cont_pos is not None:
            # 当 overlap 处于特殊耦合带内时，impact_angle 的合法区间会随 overlap 分段变化。
            # 若在这一带继续对二者加独立高斯扰动，极易把样本推到非法区域，导致高拒绝率。
            # 因此这里直接冻结这两列，让采样器只在其余连续 context 维度上做轻扰动。
            protected_rows = self.constraint_engine.should_freeze_overlap_angle_jitter(batch_context)
            if protected_rows.any():
                feature_mask[protected_rows, self._overlap_cont_pos] = 0.0
                feature_mask[protected_rows, self._angle_cont_pos] = 0.0

        noise = torch.randn(
            continuous.shape,
            generator=self.rng,
            device=self.device,
            dtype=torch.float32,
        )
        noise = noise * (self.cont_spans.unsqueeze(0) * self.jitter_ratio)
        if self.jitter_prob < 1.0:
            prob_mask = torch.rand(
                continuous.shape,
                generator=self.rng,
                device=self.device,
            ) < self.jitter_prob
            feature_mask = feature_mask * prob_mask.to(dtype=torch.float32)

        tentative_continuous = continuous + noise * feature_mask
        batch_context[:, self.context_cont_local_indices] = tentative_continuous

        # 掩码后的扰动只生成“试探样本”；这里不做任何后置修补或边界截断。
        # 只要某行试探样本破坏硬约束，就整行拒绝并回退为原始经验池样本。
        # 训练采样阶段的输入端合法性只针对当前 context 列本身，
        # 不再补任何默认 trainable control 去“凑完整特征”后再校验。
        # 否则一旦未来切换 trainable 属性，采样器的语义就会被默认值污染。
        valid_mask = self.constraint_engine.validate_context_input(batch_context)
        self._last_rejection_rate = float((~valid_mask).to(dtype=torch.float32).mean().item())
        if (~valid_mask).any():
            batch_context[~valid_mask] = original_context[~valid_mask]
        return batch_context

    @property
    def last_rejection_rate(self) -> float:
        return self._last_rejection_rate

    def _generate_batch(self) -> torch.Tensor:
        indices = torch.randint(0, self.pool_size, (self.batch_size,), generator=self.rng, device=self.device)
        batch_context = self.pool_context[indices].clone()
        # 先尝试对经验池样本做轻微连续扰动；若扰动破坏了硬约束，则直接回退该样本，
        # 保留未经扰动的原始经验池样本，而不是生成一份后置修补过的伪新样本。
        # 训练链路只接受“经验池真样本”或“仍然物理合法的轻扰动样本”这两类数据，
        # 不再引入额外 sanitize 语义，避免采样器与约束层各自维护一套修复规则。
        return self._apply_bounded_jitter(batch_context)

    def get_infinite_generator(self) -> Iterator[torch.Tensor]:
        while True:
            yield self._generate_batch()

    def get_source_info(self) -> dict:
        """返回当前采样器的数据流元信息。

        训练虽然按无限流取样，但数据来源并不是开放集合，而是固定的 injury 经验池切片。
        这里显式记录经验池路径、切分索引和扰动配置，便于训练摘要准确复现本次数据流语义。
        """
        return {
            "pool_npz_path": str(self.pool_path.resolve()),
            "split_indices_path": str(self.split_path.resolve()),
            "dataset_size": int(self.pool_size),
            "seed": self.seed,
            "batch_size": int(self.batch_size),
            "jitter_ratio": float(self.jitter_ratio),
            "jitter_prob": float(self.jitter_prob),
            "continuous_context_names": [
                self.context_params[idx]["name"] for idx in self.context_cont_local_indices
            ],
        }

    def get_distribution_reference(
        self,
        max_samples: int = 0,
        feature_space: str = "context",
        trainable_indices: Optional[list] = None,
        sample_seed: Optional[int] = None,
    ) -> tuple[torch.Tensor, dict]:
        """返回分布惩罚参考集及其抽样元信息。

        训练和评估都必须复用同一条抽样语义：
        - 若未截断，直接使用完整训练经验池；
        - 若需要截断，则基于固定 seed 做一次性随机子采样。

        这里故意不用采样器主 RNG 的当前状态，也不采用“前 N 条样本”前缀切片。
        前者会让是否启用分布惩罚反过来污染训练数据流；后者则可能把参考分布偏成
        训练池存储顺序的前缀，而不再代表整体经验池。
        """
        # feature_space="context" 时，仅返回经验池中的 context 列，用于度量工况分布偏离。
        # feature_space="context_control" 时，返回 [context | trainable_control] 的拼接矩阵；
        # 其中 trainable_control 部分直接取自经验池原始样本中的对应控制参数列，而不是 default 值或策略网络输出。
        if feature_space == "context":
            full_reference = self.pool_context
        elif feature_space == "context_control":
            if trainable_indices is None:
                raise ValueError("feature_space=context_control 时必须提供 trainable_indices")
            control_ref = self.pool_full[:, trainable_indices] if trainable_indices else self.pool_full[:, :0]
            full_reference = torch.cat([self.pool_context, control_ref], dim=1)
        else:
            raise ValueError(f"不支持的 feature_space: {feature_space}")

        total_count = int(full_reference.shape[0])
        requested_max = int(max_samples or 0)
        if requested_max <= 0 or total_count <= requested_max:
            return full_reference, {
                "pool_size": total_count,
                "sample_count": total_count,
                "requested_max_samples": requested_max,
                "sampling_mode": "full_pool",
                "seed": None,
            }

        seed_value = self.seed if sample_seed is None else int(sample_seed)
        local_rng = torch.Generator(device=self.device)
        local_rng.manual_seed(0 if seed_value is None else seed_value)
        indices = torch.randperm(total_count, generator=local_rng, device=self.device)[:requested_max]
        return full_reference[indices], {
            "pool_size": total_count,
            "sample_count": int(indices.numel()),
            "requested_max_samples": requested_max,
            "sampling_mode": "seeded_random_subset",
            "seed": 0 if seed_value is None else seed_value,
        }

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
