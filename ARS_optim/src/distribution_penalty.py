from typing import Optional

import torch


class DistributionPenalty:
    """训练分布偏离惩罚。

    该类只做两件事：
    - `fit` 根据训练参考样本缓存统计量；
    - `compute` 对当前 batch 返回逐样本惩罚。

    它不负责采样、前向投影或动作修复，只负责度量“当前样本离训练分布有多远”。

    工程语义上，它相当于给代理优化补一个“经验分布置信度”项：
    - ConstraintEngine 中的反向约束软惩罚回答“有没有越过可行域边界”
    - DistributionPenalty 的分布偏离惩罚回答“虽然合法，但是否跑到了训练集很少覆盖的稀疏区域”

    它与 GP / Kriging 中“预测方差”所承担的宏观职责相似，都是为了抑制代理模型在经验分布之外做过度自信的外推。
    """

    def __init__(self, config: dict):
        dist_cfg = config.get("optimization", {}).get("distribution_penalty", {})

        self.enabled = bool(dist_cfg.get("enabled", False))
        self.method = str(dist_cfg.get("method", "mahalanobis")).lower()
        # feature_space 决定了分布惩罚的输入特征是“工况 + 控制解”还是“仅工况”，后者更宽松，允许控制解在训练分布之外探索。一般使用前者。
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
        self.ref_feature_mean: Optional[torch.Tensor] = None
        self.ref_feature_scale: Optional[torch.Tensor] = None
        self.scale_maha: Optional[float] = None
        self.scale_knn: Optional[float] = None

    @property
    def is_ready(self) -> bool:
        """判断当前实例是否已经具备可计算 penalty 的参考统计量。"""
        if not self.enabled:
            return False
        if self.method == "mahalanobis":
            return self.ref_mean is not None and self.ref_inv_cov is not None
        return self.ref_features is not None

    def fit(self, reference_features: torch.Tensor) -> None:
        """根据训练参考样本拟合距离度量所需的统计量。

        这个方法不会直接返回 penalty，而是把参考分布的统计信息缓存到实例内部，供后续 `compute`
        阶段重复使用。`reference_features` 必须来自训练参考分布，而不是当前待评估 batch；
        否则惩罚会退化成“样本离它自己有多远”，失去限制域外外推的意义。
        """
        # 样本太少时，无论是协方差估计还是 kNN 距离统计都不可靠。
        if reference_features is None or reference_features.ndim != 2 or reference_features.shape[0] < 2:
            raise ValueError("参考分布样本不足，无法拟合分布惩罚")
        if self.method == "knn" and reference_features.shape[0] < self.k:
            raise ValueError("knn 模式下参考样本数不能小于 k")

        # 先把参考集从计算图中分离出来，再缓存逐维均值和标准差。
        # 后续无论使用 Mahalanobis 还是 kNN，距离都统一在标准化空间里计算，
        # 避免大尺度物理量完全主导距离。
        reference_features = reference_features.detach()
        self.ref_feature_mean = reference_features.mean(dim=0)
        self.ref_feature_scale = torch.clamp(
            reference_features.std(dim=0, unbiased=False),
            min=self.eps,
        )
        # 逐维标准化参考集特征
        self.ref_features = self._standardize(reference_features)
        self.ref_feature_dim = int(reference_features.shape[1])
        # 这里根据配置选择两种参考统计量：
        # - mahalanobis: 用全局协方差描述“整体形状”
        # - knn: 用局部邻域距离描述“附近有没有足够多训练样本支撑”
        if self.method == "mahalanobis":
            centered = self.ref_features - self.ref_features.mean(dim=0, keepdim=True)
            cov = (centered.T @ centered) / max(1, centered.shape[0] - 1)
            cov = cov + self.eps * torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
            self.ref_mean = self.ref_features.mean(dim=0)
            self.ref_inv_cov = torch.linalg.pinv(cov)
            if self.normalize_by_train_stats:
                with torch.no_grad():
                    # 用训练参考集自身的平均马氏距离做尺度归一化，
                    # 让 penalty 更接近“偏离训练集常见波动的多少倍”。
                    ref_penalty = self._compute_mahalanobis(self.ref_features) # 参考样本逐点的马氏平方距离: 把参考集中的每一个点，都拿来相对于“参考集自己的均值 + 协方差”计算一遍马氏平方距离
                    self.scale_maha = float(torch.clamp(ref_penalty.mean(), min=self.eps).item()) # 参考样本的平均马氏距离作为尺度
            else:
                self.scale_maha = 1.0
        elif self.method == "knn":
            # kNN 模式下，惩罚是样本到参考集最近 k 个邻居的平均距离。
            # fit 阶段只需要估一个“训练集内部的典型 kNN 距离尺度”，不必构造全量 NxN 距离矩阵。
            if self.normalize_by_train_stats:
                with torch.no_grad():
                    ref = self.ref_features # 完整的标准化后的参考集特征
                    sample_count = min(2048, ref.shape[0])
                    sample_indices = torch.randperm(ref.shape[0], device=ref.device)[:sample_count]
                    # 从 ref 里抽出来一部分子样本集
                    sample_ref = ref[sample_indices]
                    # 先算子样本集到全体参考集的两两欧氏距离，再提取最近邻距离。因此dist的形状是 [sample_count, ref.shape[0]]，每行是一个子样本到所有参考样本的距离列表。
                    dist = torch.cdist(sample_ref, ref, p=2)
                    k_eff = min(self.k + 1, dist.shape[1])
                    knn_values, _ = torch.topk(dist, k=k_eff, dim=1, largest=False)
                    # sample_ref 本身来自 ref，因此最近的一个点通常就是自己，要把这个 0 距离去掉。
                    knn_mean = knn_values[:, 1:].mean(dim=1) if k_eff > 1 else knn_values[:, 0]
                    # 用训练集内部的平均 kNN 距离作为尺度，避免 penalty 绝对量纲受数据规模影响过大。
                    self.scale_knn = float(torch.clamp(knn_mean.mean(), min=self.eps).item())
            else:
                self.scale_knn = 1.0
        else:
            raise ValueError(f"未知的 distribution_penalty.method: {self.method}")


    def _compute_mahalanobis(self, x: torch.Tensor) -> torch.Tensor:
        """返回逐样本 Mahalanobis 平方距离。

        它更偏向全局几何视角：样本若沿训练分布的主方向波动，惩罚不一定很大；
        若沿训练分布罕见方向偏离，惩罚会明显升高。
        """
        delta = x - self.ref_mean.unsqueeze(0)
        maha_sq = torch.einsum("bi,ij,bj->b", delta, self.ref_inv_cov, delta)
        return torch.clamp(maha_sq, min=0.0)

    def _standardize(self, x: torch.Tensor) -> torch.Tensor:
        """
        按参考分布逐维标准化，避免距离度量被大尺度参数主导。
        注意这里不是使用normalization_config.json里定义的全局归一化化参数，而是使用 fit 阶段从 reference_features 里计算得到的逐维均值和标准差。
        """
        if self.ref_feature_mean is None or self.ref_feature_scale is None:
            raise RuntimeError("distribution_penalty 尚未拟合，无法执行逐维标准化")
        return (x - self.ref_feature_mean.unsqueeze(0)) / self.ref_feature_scale.unsqueeze(0)

    def _compute_knn(self, x: torch.Tensor) -> torch.Tensor:
        """返回逐样本到参考集最近 k 个邻居的平均欧氏距离。

        它更偏向局部密度视角：
        - k 越大，越要求样本周围有一片稳定邻域支撑，惩罚更保守
        - k 越小，越接近“附近是否至少贴着几个训练样本”，探索性更强
        """
        distances = torch.cdist(x, self.ref_features, p=2)
        k_eff = min(self.k, distances.shape[1])
        knn_values, _ = torch.topk(distances, k=k_eff, dim=1, largest=False)
        return knn_values.mean(dim=1)

    def compute(self, context_params: torch.Tensor, control_trainable: Optional[torch.Tensor] = None) -> torch.Tensor:
        """计算逐样本分布偏离惩罚。

        当 `feature_space=context_control` 时，惩罚度量的是“当前工况 + 当前控制解”的联合分布偏离；
        当 `feature_space=context` 时，则只约束工况分布本身，不把控制变量纳入距离。
        """
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

        # compute 阶段使用和 fit 阶段完全一致的标准化规则；否则训练集统计量与运行期距离不可比。
        x = self._standardize(x)

        if self.method == "mahalanobis":
            penalty = self._compute_mahalanobis(x)
            if self.normalize_by_train_stats:
                penalty = penalty / max(self.scale_maha or 1.0, self.eps)
        else:
            penalty = self._compute_knn(x)
            if self.normalize_by_train_stats:
                penalty = penalty / max(self.scale_knn or 1.0, self.eps)
        # clip 只负责抑制极端 OOD 点对总目标的单点主导，不改变 penalty 的相对排序语义。
        return torch.clamp(penalty, min=0.0, max=self.clip_max)
