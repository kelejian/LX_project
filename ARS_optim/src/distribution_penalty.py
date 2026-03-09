from typing import Optional

import torch


class DistributionPenalty:
    """训练分布偏离惩罚。

    该类只做两件事：
    - `fit` 根据训练参考样本缓存统计量；
    - `compute` 对当前 batch 返回逐样本惩罚。

    它不负责采样、也不负责动作修复，只负责度量“当前样本离训练分布有多远”。
    """

    def __init__(self, config: dict):
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

        self.ref_features: Optional[torch.Tensor] = None
        self.ref_mean: Optional[torch.Tensor] = None
        self.ref_inv_cov: Optional[torch.Tensor] = None
        self.ref_feature_dim: Optional[int] = None
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
        if reference_features is None or reference_features.ndim != 2 or reference_features.shape[0] < 2:
            raise ValueError("参考分布样本不足，无法拟合分布惩罚")
        if self.method == "knn" and reference_features.shape[0] < self.k:
            raise ValueError("knn 模式下参考样本数不能小于 k")

        self.ref_features = reference_features.detach()
        self.ref_feature_dim = int(reference_features.shape[1])
        if self.method == "mahalanobis":
            centered = self.ref_features - self.ref_features.mean(dim=0, keepdim=True)
            cov = (centered.T @ centered) / max(1, centered.shape[0] - 1)
            cov = cov + self.eps * torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
            self.ref_mean = self.ref_features.mean(dim=0)
            self.ref_inv_cov = torch.linalg.pinv(cov)
            if self.normalize_by_train_stats:
                with torch.no_grad():
                    ref_penalty = self._compute_mahalanobis(self.ref_features)
                    self.scale_maha = float(torch.clamp(ref_penalty.mean(), min=self.eps).item())
            else:
                self.scale_maha = 1.0
        else:
            if self.normalize_by_train_stats:
                with torch.no_grad():
                    ref = self.ref_features
                    sample_count = min(2048, ref.shape[0])
                    sample_indices = torch.randperm(ref.shape[0], device=ref.device)[:sample_count]
                    sample_ref = ref[sample_indices]
                    dist = torch.cdist(sample_ref, ref, p=2)
                    k_eff = min(self.k + 1, dist.shape[1])
                    knn_values, _ = torch.topk(dist, k=k_eff, dim=1, largest=False)
                    knn_mean = knn_values[:, 1:].mean(dim=1) if k_eff > 1 else knn_values[:, 0]
                    self.scale_knn = float(torch.clamp(knn_mean.mean(), min=self.eps).item())
            else:
                self.scale_knn = 1.0

    def _compute_mahalanobis(self, x: torch.Tensor) -> torch.Tensor:
        delta = x - self.ref_mean.unsqueeze(0)
        maha_sq = torch.einsum("bi,ij,bj->b", delta, self.ref_inv_cov, delta)
        return torch.clamp(maha_sq, min=0.0)

    def _compute_knn(self, x: torch.Tensor) -> torch.Tensor:
        distances = torch.cdist(x, self.ref_features, p=2)
        k_eff = min(self.k, distances.shape[1])
        knn_values, _ = torch.topk(distances, k=k_eff, dim=1, largest=False)
        return knn_values.mean(dim=1)

    def compute(self, context_params: torch.Tensor, control_trainable: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not self.enabled:
            return torch.zeros(context_params.shape[0], device=context_params.device, dtype=torch.float32)
        if not self.is_ready:
            raise RuntimeError("分布惩罚已启用，但尚未先调用 fit(reference_features)")

        if self.feature_space == "context_control":
            if control_trainable is None:
                raise ValueError("feature_space=context_control 时必须传入 control_trainable")
            x = torch.cat([context_params, control_trainable], dim=1)
        else:
            x = context_params

        if self.ref_feature_dim is not None and self.ref_feature_dim != x.shape[1]:
            raise ValueError(
                f"分布惩罚输入维度与参考分布不一致: current={x.shape[1]}, reference={self.ref_feature_dim}"
            )

        if self.method == "mahalanobis":
            penalty = self._compute_mahalanobis(x)
            if self.normalize_by_train_stats:
                penalty = penalty / max(self.scale_maha or 1.0, self.eps)
        else:
            penalty = self._compute_knn(x)
            if self.normalize_by_train_stats:
                penalty = penalty / max(self.scale_knn or 1.0, self.eps)
        return torch.clamp(penalty, min=0.0, max=self.clip_max)
