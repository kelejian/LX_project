import logging
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch

from common.data_utils.split_io import load_int_vector_csv
from common.settings import RAW_DATA, get_split_indices_path

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
        jitter_max_attempts: int = 1,
        jitter_min_active_dims: int = 1,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.param_manager = param_manager
        self.constraint_engine = constraint_engine
        self.batch_size = int(batch_size)
        self.device = device
        self.seed = int(seed) if seed is not None else None
        self.jitter_ratio = float(jitter_ratio)
        self.jitter_prob = float(jitter_prob)
        self.jitter_max_attempts = int(jitter_max_attempts)
        self.jitter_min_active_dims = int(jitter_min_active_dims)
        if not (0.0 <= self.jitter_prob <= 1.0):
            raise ValueError("jitter_prob 必须位于 [0, 1] 区间")
        if self.jitter_max_attempts <= 0:
            raise ValueError("jitter_max_attempts 必须为正整数")
        if self.jitter_min_active_dims < 0:
            raise ValueError("jitter_min_active_dims 不能为负数")
        self._last_attempt_rejection_rate = 0.0
        self._last_final_fallback_rate = 0.0
        self._last_mean_changed_param_ratio = 0.0

        self.rng = torch.Generator(device=device)
        if self.seed is not None:
            self.rng.manual_seed(self.seed)

        self.context_params = self.param_manager.get_context_params()
        self.context_indices = self.param_manager.get_context_indices()

        pool_path = Path(pool_npz_path) if pool_npz_path else (RAW_DATA)
        split_path = Path(split_indices_path) if split_indices_path else get_split_indices_path("injury", "train")
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
        self.context_cont_names = [
            self.context_params[idx]["name"] for idx in self.context_cont_local_indices
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

    def _build_attempt_active_mask(self, allowed_mask: torch.Tensor) -> torch.Tensor:
        active_mask = allowed_mask.clone()
        if self.jitter_prob < 1.0:
            sampled_mask = torch.rand(
                allowed_mask.shape,
                generator=self.rng,
                device=self.device,
            ) < self.jitter_prob
            active_mask = allowed_mask & sampled_mask

        if self.jitter_min_active_dims <= 0:
            return active_mask

        allowed_counts = allowed_mask.sum(dim=1)
        target_counts = torch.minimum(
            allowed_counts,
            torch.full_like(allowed_counts, self.jitter_min_active_dims),
        )
        current_counts = active_mask.sum(dim=1)
        need_rows = current_counts < target_counts
        if not need_rows.any():
            return active_mask

        for row_idx in torch.nonzero(need_rows, as_tuple=False).flatten().cpu().tolist():
            need = int(target_counts[row_idx].item() - current_counts[row_idx].item())
            if need <= 0:
                continue
            candidates = torch.nonzero(
                allowed_mask[row_idx] & (~active_mask[row_idx]),
                as_tuple=False,
            ).flatten()
            if candidates.numel() == 0:
                continue
            if candidates.numel() <= need:
                active_mask[row_idx, candidates] = True
                continue
            chosen = candidates[
                torch.randperm(candidates.numel(), generator=self.rng, device=candidates.device)[:need]
            ]
            active_mask[row_idx, chosen] = True
        return active_mask

    def _apply_bounded_jitter(self, batch_context: torch.Tensor) -> torch.Tensor:
        if not self.context_cont_local_indices or self.jitter_ratio <= 0:
            self._last_attempt_rejection_rate = 0.0
            self._last_final_fallback_rate = 0.0
            self._last_mean_changed_param_ratio = 0.0
            return batch_context

        original_context = batch_context.clone()
        sampled_context = original_context.clone()
        accepted_mask = torch.zeros(original_context.shape[0], dtype=torch.bool, device=self.device)
        total_attempts = 0
        total_rejections = 0

        # 输入端仍保持“只接受合法扰动，否则回退原样本”的语义；
        # 区别只是每行允许做有限次独立重试，降低整样本完全重复率。
        for _ in range(self.jitter_max_attempts):
            pending_idx = torch.nonzero(~accepted_mask, as_tuple=False).flatten()
            if pending_idx.numel() == 0:
                break

            base_pending = original_context[pending_idx]
            allowed_mask_full = self.constraint_engine.build_context_jitter_mask(base_pending)
            allowed_mask = allowed_mask_full[:, self.context_cont_local_indices]
            active_mask = self._build_attempt_active_mask(allowed_mask)

            tentative = base_pending.clone()
            continuous = tentative[:, self.context_cont_local_indices]
            noise = torch.randn(
                continuous.shape,
                generator=self.rng,
                device=self.device,
                dtype=torch.float32,
            )
            noise = noise * (self.cont_spans.unsqueeze(0) * self.jitter_ratio)
            tentative[:, self.context_cont_local_indices] = continuous + noise * active_mask.to(dtype=torch.float32)

            changed_rows = active_mask.any(dim=1)
            valid_mask = changed_rows & self.constraint_engine.validate_context_input(tentative)
            total_attempts += int(pending_idx.numel())
            total_rejections += int((~valid_mask).sum().item())

            if valid_mask.any():
                accepted_idx = pending_idx[valid_mask]
                sampled_context[accepted_idx] = tentative[valid_mask]
                accepted_mask[accepted_idx] = True

        changed_mask = (
            sampled_context[:, self.context_cont_local_indices]
            - original_context[:, self.context_cont_local_indices]
        ).abs() > 1e-12
        self._last_attempt_rejection_rate = (
            float(total_rejections) / float(total_attempts) if total_attempts > 0 else 0.0
        ) # 所有扰动尝试中，不合法尝试的比例（不是“最终整样本是否回退”的比例）; 其大小反映了当前扰动机制碰到约束边界的频率
        self._last_final_fallback_rate = float((~accepted_mask).to(dtype=torch.float32).mean().item())
        # 单个bacth中回退为经验池原样本的占比; 若低于前者说明有限次重采样增加了扰动成功率
        self._last_mean_changed_param_ratio = float(changed_mask.to(dtype=torch.float32).mean().item())
        # “参数级”的变化比例，在 batch 的所有“连续 context 参数槽位”里，最终和基样本槽位不同的比例。反映最终 batch 和经验池相比有多“新”。如果较高，说明这批训练输入并不是大量原样本拷贝，而是大部分连续参数槽位都发生了变化。
        return sampled_context

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
            "jitter_max_attempts": int(self.jitter_max_attempts),
            "jitter_min_active_dims": int(self.jitter_min_active_dims),
            "continuous_context_names": list(self.context_cont_names),
        }

    def get_last_sampling_stats(self) -> dict:
        return {
            "attempt_rejection_rate": float(self._last_attempt_rejection_rate),
            "final_fallback_rate": float(self._last_final_fallback_rate),
            "mean_changed_param_ratio": float(self._last_mean_changed_param_ratio),
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
