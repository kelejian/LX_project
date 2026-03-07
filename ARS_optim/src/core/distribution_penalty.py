import logging
from typing import Optional

import torch


class DistributionPenalty:
    """
    训练分布偏离惩罚计算器。

    设计目标：
    1. 不改代理模型结构，仅在优化目标中增加稳健项。
    2. 支持 `mahalanobis` 与 `knn` 两种度量方式。
    3. 惩罚项默认保持小权重，避免策略网络过度保守。

    记号：
    - 输入批次特征记为 x。
    - 训练参考分布记为 D_train。
    - 输出逐样本惩罚 u(x) >= 0，供总损失组合。
    """

    def __init__(self, config: dict):
        self.logger = logging.getLogger(self.__class__.__name__)
        dist_cfg = config.get("optimization", {}).get("distribution_penalty", {})

        self.enabled = bool(dist_cfg.get("enabled", False))
        self.method = str(dist_cfg.get("method", "mahalanobis")).lower()
        self.feature_space = str(dist_cfg.get("feature_space", "context"))
        self.k = int(dist_cfg.get("k", 8))
        self.eps = float(dist_cfg.get("eps", 1e-6))
        self.clip_max = float(dist_cfg.get("clip_max", 10.0))
        self.normalize_by_train_stats = bool(dist_cfg.get("normalize_by_train_stats", True))

        if self.method not in {"mahalanobis", "knn"}:
            raise ValueError(f"distribution_penalty.method 不支持: {self.method}")
        if self.k <= 0:
            raise ValueError("distribution_penalty.k 必须为正整数")
        if self.eps <= 0:
            raise ValueError("distribution_penalty.eps 必须大于 0")
        if self.clip_max <= 0:
            raise ValueError("distribution_penalty.clip_max 必须大于 0")

        # 参考分布缓存（与 feature_space 对齐）
        self.ref_features: Optional[torch.Tensor] = None  # [N_ref, D]
        self.ref_mean: Optional[torch.Tensor] = None     # [D]
        self.ref_inv_cov: Optional[torch.Tensor] = None  # [D, D]

        # 归一化标尺（可选）
        self.scale_maha: Optional[float] = None
        self.scale_knn: Optional[float] = None

    @property
    def is_ready(self) -> bool:
        if not self.enabled:
            return False
        if self.method == "mahalanobis":
            return self.ref_mean is not None and self.ref_inv_cov is not None
        return self.ref_features is not None

    def fit(self, reference_features: torch.Tensor) -> None:
        """
        使用训练经验池参考特征拟合分布。

        参数:
        - reference_features: [N_ref, D]
        """
        if reference_features is None:
            raise ValueError("reference_features 不能为空")
        if reference_features.ndim != 2:
            raise ValueError(f"reference_features 维度异常: {reference_features.shape}")
        if reference_features.shape[0] < 2:
            raise ValueError("参考样本数量不足，至少需要 2 条")
        if self.method == "knn" and reference_features.shape[0] < self.k:
            raise ValueError(
                f"knn 模式下参考样本数({reference_features.shape[0]})不能小于 k({self.k})"
            )

        self.ref_features = reference_features.detach()

        if self.method == "mahalanobis":
            # x_c: [N_ref, D]
            x_c = self.ref_features - self.ref_features.mean(dim=0, keepdim=True)
            # 协方差 [D, D]
            cov = (x_c.T @ x_c) / max(1, x_c.shape[0] - 1)
            d = cov.shape[0]
            cov = cov + self.eps * torch.eye(d, device=cov.device, dtype=cov.dtype)
            inv_cov = torch.linalg.pinv(cov)

            self.ref_mean = self.ref_features.mean(dim=0)
            self.ref_inv_cov = inv_cov

            if self.normalize_by_train_stats:
                with torch.no_grad():
                    maha_ref = self._compute_mahalanobis(self.ref_features)
                    self.scale_maha = float(torch.clamp(maha_ref.mean(), min=self.eps).item())
            else:
                self.scale_maha = 1.0

        else:
            # knn 模式仅缓存参考点，归一化标尺基于参考点抽样估计。
            if self.normalize_by_train_stats:
                with torch.no_grad():
                    x = self.ref_features
                    n = x.shape[0]
                    # 为避免 O(N^2) 过重，最多采样 2048 个点估计尺度。
                    m = min(2048, n)
                    idx = torch.randperm(n, device=x.device)[:m]
                    x_sub = x[idx]
                    dist = torch.cdist(x_sub, x, p=2)
                    k_eff = min(self.k + 1, dist.shape[1])
                    knn_vals, _ = torch.topk(dist, k=k_eff, dim=1, largest=False)
                    if k_eff > 1:
                        knn_mean = knn_vals[:, 1:].mean(dim=1)
                    else:
                        knn_mean = knn_vals[:, 0]
                    self.scale_knn = float(torch.clamp(knn_mean.mean(), min=self.eps).item())
            else:
                self.scale_knn = 1.0

        self.logger.info(
            f"分布偏离惩罚参考分布已拟合: method={self.method}, "
            f"feature_space={self.feature_space}, "
            f"n_ref={self.ref_features.shape[0]}, dim={self.ref_features.shape[1]}"
        )

    def _compute_mahalanobis(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算逐样本马氏距离平方。

        x: [B, D]
        return: [B]
        """
        delta = x - self.ref_mean.unsqueeze(0)  # [B, D]
        # (x-mu)^T S^-1 (x-mu) -> [B]
        maha_sq = torch.einsum("bi,ij,bj->b", delta, self.ref_inv_cov, delta)
        maha_sq = torch.clamp(maha_sq, min=0.0)
        return maha_sq

    def _compute_knn(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算逐样本 kNN 均值距离。

        x: [B, D]
        return: [B]
        """
        dist = torch.cdist(x, self.ref_features, p=2)  # [B, N_ref]
        k_eff = min(self.k, dist.shape[1])
        knn_vals, _ = torch.topk(dist, k=k_eff, dim=1, largest=False)
        knn_mean = knn_vals.mean(dim=1)
        return knn_mean

    def compute(
        self,
        context_params: torch.Tensor,
        control_trainable: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        计算分布偏离惩罚（逐样本）。

        当前实现说明：
        - feature_space='context'：仅使用 context_params。
        - feature_space='context_control'：拼接可调控制参数。
        """
        if not self.enabled:
            return torch.zeros(context_params.shape[0], device=context_params.device, dtype=torch.float32)

        if not self.is_ready:
            # 训练主流程会在开始阶段拟合参考分布；若未拟合则返回 0 并提示。
            self.logger.warning("分布惩罚已启用但参考分布尚未拟合，当前批次返回0惩罚。")
            return torch.zeros(context_params.shape[0], device=context_params.device, dtype=torch.float32)

        if self.feature_space == "context_control":
            if control_trainable is None:
                raise ValueError("feature_space=context_control 时，control_trainable 不能为空")
            # [B, D_context] + [B, D_train] -> [B, D_all]
            x = torch.cat([context_params, control_trainable], dim=1)
        else:
            x = context_params

        if self.ref_features is not None and self.ref_features.shape[1] != x.shape[1]:
            raise ValueError(
                f"分布惩罚输入维度({x.shape[1]})与参考分布维度({self.ref_features.shape[1]})不一致，"
                "请检查 feature_space 与参考分布拟合配置。"
            )

        if self.method == "mahalanobis":
            penalty = self._compute_mahalanobis(x)
            if self.normalize_by_train_stats:
                denom = self.scale_maha if self.scale_maha is not None else 1.0
                penalty = penalty / max(denom, self.eps)
        else:
            penalty = self._compute_knn(x)
            if self.normalize_by_train_stats:
                denom = self.scale_knn if self.scale_knn is not None else 1.0
                penalty = penalty / max(denom, self.eps)

        penalty = torch.clamp(penalty, min=0.0, max=self.clip_max)
        return penalty
