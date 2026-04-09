from typing import Dict, Tuple

import numpy as np
import torch
from matplotlib.path import Path as MplPath

from ARS_optim.src.param_manager import ParamManager


class ConstraintEngine:
    """统一管理 ARS 参数的输入端校验、输出端投影与反向约束软惩罚。

    约束来源：
    - 连续/离散参数定义、默认值和角色划分来自 `ParamManager`；
    - 额外耦合规则来自 `param_space.yaml` 中编码后的 `constraint_rules`。

    主要职责：
    - `validate_context_input`: 按当前 `context` 子张量口径做输入端校验；
    - `validate_full_input`: 按完整物理输入口径校验样本；
    - `validate_output_solution`: 校验寻优解是否满足输出端全部约束；
    - `project_forward`: 把候选解投影回合法域；
    - `compute_soft_penalty`: 对投影前原始解计算反向约束软惩罚。
    """

    def __init__(self, param_manager: ParamManager):
        self.param_manager = param_manager
        self.rules = param_manager.get_constraint_rules()

        self.total_dim = param_manager.get_total_feature_dim()
        self.context_indices = param_manager.get_context_indices()
        self.trainable_indices = param_manager.get_control_trainable_indices()
        self._trainable_idx_set = set(self.trainable_indices)
        self.context_params = param_manager.get_context_params()
        self.context_names = [param["name"] for param in self.context_params]

        all_params = param_manager.get_all_params()
        self.name_to_index = {param["name"]: param["index"] for param in all_params}
        self.continuous_indices = [
            param["index"] for param in all_params if param.get("type") == "continuous"
        ]
        self.base_continuous_bounds = {
            param["index"]: (float(param["base_min"]), float(param["base_max"]))
            for param in all_params
            if param.get("type") == "continuous"
        }
        self.base_continuous_scales = {
            idx: max(max_value - min_value, 1e-6)
            for idx, (min_value, max_value) in self.base_continuous_bounds.items()
        }
        self.output_continuous_bounds = {
            param["index"]: (
                float(param["opt_min"]) if param.get("role") == "control" and bool(param.get("trainable", False)) else float(param["base_min"]),
                float(param["opt_max"]) if param.get("role") == "control" and bool(param.get("trainable", False)) else float(param["base_max"]),
            )
            for param in all_params
            if param.get("type") == "continuous"
        }
        self.discrete_values = {
            idx: torch.tensor(values, dtype=torch.float32)
            for idx, values in param_manager.get_discrete_index_value_map().items()
        }

        trainable_bounds = param_manager.get_trainable_opt_bounds()
        self._trainable_mins_cpu = trainable_bounds[0]
        self._trainable_maxs_cpu = trainable_bounds[1]
        self._trainable_spans_cpu = torch.clamp(self._trainable_maxs_cpu - self._trainable_mins_cpu, min=1e-6)

        coupling = self.rules.get("coupling", {})
        extra_output_constraints = self.rules.get("extra_output_constraints", {})
        self.aft_btf_delta_max = float(coupling.get("aft_btf_delta_max", 25.0))
        self.llattf_btf_delta_min = float(coupling.get("llattf_btf_delta_min", 0.0))
        self.ll2_ll1_delta_max = float(coupling.get("ll2_ll1_delta_max", 0.0))
        self.aft_btf_extra_delta_min = float(extra_output_constraints.get("aft_btf_delta_min", 0.0))
        self.epsilon = float(coupling.get("epsilon", 1e-3))
        self._tol = 1e-5

        overlap_cfg = self.rules.get("overlap", {}).get("domain", {})
        self._overlap_neg_min, self._overlap_neg_max = map(
            float, overlap_cfg.get("negative", [-1.0, -0.25])
        )
        self._overlap_pos_min, self._overlap_pos_max = map(
            float, overlap_cfg.get("positive", [0.25, 1.0])
        )
        overlap_angle_rule = self.rules.get("overlap_angle", {})
        self._overlap_abs_min = float(overlap_angle_rule.get("overlap_abs_min", 0.25))
        self._overlap_abs_max = float(overlap_angle_rule.get("overlap_abs_max", 0.3))
        self._angle_segments = [
            tuple(map(float, segment))
            for segment in (
                list(overlap_angle_rule.get("angle_sampling", {}).get("positive_overlap", []))
                + list(overlap_angle_rule.get("angle_sampling", {}).get("negative_overlap", []))
            )
            if isinstance(segment, (list, tuple)) and len(segment) == 4
        ]

        self._impact_angle_idx = self.name_to_index["impact_angle"]
        self._overlap_idx = self.name_to_index["overlap"]
        self._ll1_idx = self.name_to_index.get("LL1")
        self._ll2_idx = self.name_to_index.get("LL2")
        self._btf_idx = self.name_to_index.get("BTF")
        self._llattf_idx = self.name_to_index.get("LLATTF")
        self._aft_idx = self.name_to_index.get("AFT")
        self._sp_idx = self.name_to_index.get("SP")
        self._sh_idx = self.name_to_index.get("SH")
        self._ra_idx = self.name_to_index.get("RA")
        self._side_idx = self.name_to_index.get("is_driver_side")
        self._ot_idx = self.name_to_index.get("OT")
        self._context_name_to_local_index = {name: idx for idx, name in enumerate(self.context_names)}
        self._context_overlap_local_idx = self._context_name_to_local_index.get("overlap")
        self._context_angle_local_idx = self._context_name_to_local_index.get("impact_angle")
        self._context_ll1_local_idx = self._context_name_to_local_index.get("LL1")
        self._context_ll2_local_idx = self._context_name_to_local_index.get("LL2")
        self._context_btf_local_idx = self._context_name_to_local_index.get("BTF")
        self._context_llattf_local_idx = self._context_name_to_local_index.get("LLATTF")
        self._context_aft_local_idx = self._context_name_to_local_index.get("AFT")
        self._context_sp_local_idx = self._context_name_to_local_index.get("SP")
        self._context_sh_local_idx = self._context_name_to_local_index.get("SH")
        self._context_ra_local_idx = self._context_name_to_local_index.get("RA")
        self._context_side_local_idx = self._context_name_to_local_index.get("is_driver_side")
        self._context_ot_local_idx = self._context_name_to_local_index.get("OT")

        self.context_continuous_bounds = {
            local_idx: (float(param["base_min"]), float(param["base_max"]))
            for local_idx, param in enumerate(self.context_params)
            if param.get("type") == "continuous"
        }
        self.context_discrete_values = {
            local_idx: torch.tensor([float(value) for value in param["values"]], dtype=torch.float32)
            for local_idx, param in enumerate(self.context_params)
            if param.get("type") == "discrete"
        }

        self._build_rule_caches()

    @staticmethod
    def _parse_side_ot_rule_key(rule_name: str, key: str) -> Tuple[int, int]:
        """把 `is_driver_side_OT` 形式的配置键解析成整数二元组。"""
        try:
            side_text, ot_text = key.split("_")
            return int(side_text), int(ot_text)
        except ValueError as exc:
            raise ValueError(f"{rule_name} 存在非法键 {key!r}，应为 'is_driver_side_OT' 形式") from exc

    def _build_rule_caches(self) -> None:
        """预解析与 `(is_driver_side, OT)` 相关的规则缓存。

        这些规则在训练采样、前向投影、输入校验和软惩罚里都会被高频查询，
        因此在初始化阶段先展开成 cache，避免运行期重复解析 YAML 结构。
        """
        self.seat_cache: Dict[Tuple[int, int], Dict[str, object]] = {}
        for key, points in (self.rules.get("seat_constraints", {}) or {}).items():
            side, ot = self._parse_side_ot_rule_key("seat_constraints", key)
            polygon = np.asarray(points, dtype=np.float32)
            if polygon.ndim != 2 or polygon.shape[1] != 2 or polygon.shape[0] < 3:
                raise ValueError(
                    f"seat_constraints[{key!r}] 必须是至少包含 3 个 [SP, SH] 点的二维多边形，实际形状为 {polygon.shape}"
                )
            sp_min, sh_min = np.min(polygon, axis=0)
            sp_max, sh_max = np.max(polygon, axis=0)
            shifted = np.roll(polygon, shift=-1, axis=0)
            # 用鞋带公式判断顶点方向；若输入是顺时针，则统一翻成逆时针。
            signed_area = 0.5 * float(np.sum(polygon[:, 0] * shifted[:, 1] - shifted[:, 0] * polygon[:, 1]))
            is_degenerate = abs(signed_area) <= 1e-8

            polygon_ccw = polygon
            if not is_degenerate and signed_area < 0.0:
                polygon_ccw = polygon[::-1].copy()

            polygon_tensor = torch.tensor(polygon_ccw, dtype=torch.float32)

            self.seat_cache[(side, ot)] = {
                "poly": polygon_ccw,
                "path": MplPath(polygon_ccw),
                "bbox": (float(sp_min), float(sp_max), float(sh_min), float(sh_max)),
                "polygon_tensor": polygon_tensor,
            }

        ra_ranges = self.rules.get("ra_ranges")
        if ra_ranges is None:
            raise ValueError("constraint_rules 缺少 ra_ranges 配置")

        self.ra_range_cache: Dict[Tuple[int, int], Tuple[float, float]] = {}
        for key, values in ra_ranges.items():
            side, ot = self._parse_side_ot_rule_key("ra_ranges", key)
            arr = np.asarray(values, dtype=np.float32).reshape(-1)
            if arr.size != 2:
                raise ValueError(f"ra_ranges[{key!r}] 必须是 [min, max] 形式")
            ra_min, ra_max = float(arr[0]), float(arr[1])
            if ra_min > ra_max:
                raise ValueError(f"ra_ranges[{key!r}] 中最小值不能大于最大值")
            self.ra_range_cache[(side, ot)] = (ra_min, ra_max)

    @staticmethod
    def _all_present(*indices: int) -> bool:
        """判断一组索引是否都已存在。"""
        return all(idx is not None for idx in indices)

    @staticmethod
    def _project_points_to_polygon_boundary(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
        """把二维点投到多边形边界上的最近位置。

        这是无梯度的 numpy 版本，主要服务于输入校验和几何判定；
        需要保留计算图的路径使用 `_project_points_to_polygon_boundary_torch`。
        """
        edge_start = polygon
        edge_end = np.roll(polygon, shift=-1, axis=0)
        projected = np.empty_like(points, dtype=np.float32)

        for point_idx, point in enumerate(points):
            best_point = None
            best_dist_sq = None
            for start, end in zip(edge_start, edge_end):
                segment = end - start
                denom = float(np.dot(segment, segment))
                if denom <= 1e-12:
                    candidate = start
                else:
                    t = float(np.dot(point - start, segment) / denom)
                    t = min(1.0, max(0.0, t))
                    candidate = start + t * segment
                dist_sq = float(np.sum((point - candidate) ** 2))
                if best_dist_sq is None or dist_sq < best_dist_sq:
                    best_dist_sq = dist_sq
                    best_point = candidate
            projected[point_idx] = best_point.astype(np.float32)
        return projected

    def _ensure_2d(self, tensor: torch.Tensor, dim: int, name: str) -> torch.Tensor:
        """检查张量是否为 `[N, dim]` 形状。"""
        if tensor.ndim != 2 or tensor.shape[1] != dim:
            raise ValueError(f"{name} 形状应为 [N, {dim}]，实际为 {tuple(tensor.shape)}")
        return tensor

    def compose_full_features(
        self,
        context_params: torch.Tensor,
        control_trainable: torch.Tensor = None,
    ) -> torch.Tensor:
        """把 `context` 与当前 trainable control 拼成完整特征矩阵。

        未显式提供的 trainable control 会保留 `ParamManager` 给出的 default，
        因此这个接口既可用于“完整输入校验”，也可用于“context + 当前动作”组合。
        """
        context_params = self._ensure_2d(context_params, len(self.context_indices), "context_params")
        full = self.param_manager.get_default_feature_matrix(
            context_params.shape[0], device=context_params.device
        )
        full[:, self.context_indices] = context_params
        if control_trainable is not None:
            control_trainable = self._ensure_2d(
                control_trainable, len(self.trainable_indices), "control_trainable"
            )
            full[:, self.trainable_indices] = control_trainable
        return full

    def split_from_full(self, full_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """从完整特征矩阵拆回 `context` 与 trainable control 子张量。"""
        full_features = self._ensure_2d(full_features, self.total_dim, "full_features")
        return full_features[:, self.context_indices], full_features[:, self.trainable_indices]

    def _get_overlap_angle_special_mask(self, overlap: torch.Tensor, lower_tol: float = 0.0) -> torch.Tensor:
        """判断样本是否落入 overlap-angle 的特殊约束带。

        训练采样阶段的“冻结扰动”和输入/输出端校验都复用这里，
        这样边界语义始终一致，不会一处按 `<0.3`、另一处按 `<=0.3`。
        """
        lower = self._overlap_abs_min - lower_tol
        return (overlap.abs() >= lower) & (overlap.abs() < self._overlap_abs_max)

    def build_context_jitter_mask(self, context_params: torch.Tensor) -> torch.Tensor:
        """构造 context 输入端允许被扰动的按列掩码。

        连续 context 列默认允许扰动；overlap-angle 特殊带和退化座椅自由度会被冻结。
        返回值与 context_params 同形状，离散列天然为 False。
        """
        x = self._ensure_2d(context_params, len(self.context_indices), "context_params")
        mask = torch.zeros_like(x, dtype=torch.bool)

        for local_idx, param in enumerate(self.context_params):
            if param.get("type") == "continuous":
                mask[:, local_idx] = True

        if self._context_overlap_local_idx is not None and self._context_angle_local_idx is not None:
            overlap = x[:, self._context_overlap_local_idx]
            protected_rows = self._get_overlap_angle_special_mask(overlap, lower_tol=0.0)
            if protected_rows.any():
                mask[protected_rows, self._context_overlap_local_idx] = False
                mask[protected_rows, self._context_angle_local_idx] = False

        if not self._all_present(
            self._context_side_local_idx,
            self._context_ot_local_idx,
            self._context_sp_local_idx,
            self._context_sh_local_idx,
        ):
            return mask

        side, ot = self._get_context_side_ot_codes(x)
        for (rule_side, rule_ot), info in self.seat_cache.items():
            row_mask = (side == rule_side) & (ot == rule_ot)
            if not row_mask.any():
                continue
            sp_min, sp_max, sh_min, sh_max = info["bbox"]
            # 这里故意按零宽线段单独处理，避免把 passenger 这类退化座椅区域误当成二维多边形。
            # 输入端采样只按当前座椅可行域的自由度维数决定是否冻结某一维。
            if abs(sh_max - sh_min) <= self._tol:
                mask[row_mask, self._context_sh_local_idx] = False
            if abs(sp_max - sp_min) <= self._tol:
                mask[row_mask, self._context_sp_local_idx] = False

        return mask

    def _replace_column(self, x: torch.Tensor, column_idx: int, new_column: torch.Tensor) -> torch.Tensor:
        """返回替换某一列后的新张量，避免在投影链中原地改值。"""
        x_new = x.clone()
        x_new[:, column_idx] = new_column
        return x_new

    def _get_side_ot_codes(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """从完整特征张量里取出 `(is_driver_side, OT)` 整数编码。"""
        side = torch.round(x[:, self._side_idx]).to(torch.int64)
        ot = torch.round(x[:, self._ot_idx]).to(torch.int64)
        return side, ot

    def _get_context_side_ot_codes(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """从 `context` 张量里取出 `(is_driver_side, OT)` 整数编码。"""
        side = torch.round(x[:, self._context_side_local_idx]).to(torch.int64)
        ot = torch.round(x[:, self._context_ot_local_idx]).to(torch.int64)
        return side, ot

    def _apply_trainable_bounds(self, x: torch.Tensor) -> torch.Tensor:
        """把所有 trainable control 裁剪到 `opt_min/opt_max` 区间。"""
        if not self.trainable_indices:
            return x
        mins = self._trainable_mins_cpu.to(device=x.device, dtype=x.dtype)
        maxs = self._trainable_maxs_cpu.to(device=x.device, dtype=x.dtype)
        x_new = x.clone()
        x_new[:, self.trainable_indices] = torch.clamp(x[:, self.trainable_indices], mins, maxs)
        return x_new

    def _project_upper_bound_pair(
        self,
        x: torch.Tensor,
        constrained_idx: int,
        reference_idx: int,
        delta: float,
        epsilon: float = 0.0,
    ) -> torch.Tensor:
        """对一对参数施加“上界型”耦合投影。

        目标约束为 `constrained <= reference + delta`。
        若左侧可调，则直接裁左侧；若左侧不可调但右侧可调，则反向裁右侧。
        """
        if constrained_idx in self._trainable_idx_set:
            constrained = torch.min(x[:, constrained_idx], x[:, reference_idx] + delta - epsilon)
            return self._replace_column(x, constrained_idx, constrained)
        elif reference_idx in self._trainable_idx_set:
            reference = torch.max(x[:, reference_idx], x[:, constrained_idx] - delta + epsilon)
            return self._replace_column(x, reference_idx, reference)
        return x

    def _project_lower_bound_pair(
        self,
        x: torch.Tensor,
        target_idx: int,
        reference_idx: int,
        delta: float,
    ) -> torch.Tensor:
        """对一对参数施加“下界型”耦合投影。

        目标约束为 `target >= reference + delta`。
        若左侧可调，则直接裁左侧；若左侧不可调但右侧可调，则反向裁右侧。
        """
        if target_idx in self._trainable_idx_set:
            target = torch.max(x[:, target_idx], x[:, reference_idx] + delta)
            return self._replace_column(x, target_idx, target)
        elif reference_idx in self._trainable_idx_set:
            reference = torch.min(x[:, reference_idx], x[:, target_idx] - delta)
            return self._replace_column(x, reference_idx, reference)
        return x

    def _project_control_couplings(self, x: torch.Tensor) -> torch.Tensor:
        """按单向依赖顺序投影 control 间的耦合约束。

        这里故意把每条规则写成单向更新：
        - AFT 上界依赖 BTF；
        - LLATTF 依赖 BTF；
        - LL2 依赖 LL1。

        这样训练与局部精调的前向链路就始终沿同一拓扑顺序收敛，不会因为多条规则互相“来回修正”而把语义变得不可预测。
        """
        x = self._project_upper_bound_pair(
            x,
            constrained_idx=self._aft_idx,
            reference_idx=self._btf_idx,
            delta=self.aft_btf_delta_max,
            epsilon=self.epsilon,
        )
        x = self._project_lower_bound_pair(
            x,
            target_idx=self._llattf_idx,
            reference_idx=self._btf_idx,
            delta=self.llattf_btf_delta_min,
        )
        x = self._project_upper_bound_pair(
            x,
            constrained_idx=self._ll2_idx,
            reference_idx=self._ll1_idx,
            delta=self.ll2_ll1_delta_max,
        )
        return x

    def _project_output_extra_constraints(self, x: torch.Tensor) -> torch.Tensor:
        """投影仅作用于寻优解的额外约束。

        这类规则不属于输入端数据流的物理采样定义域，因此不能混入 `coupling`。
        当前唯一规则是 AFT 不得早于 BTF，它只在策略直推与局部精调产出的
        trainable control 上生效。
        """
        return self._project_lower_bound_pair(
            x,
            target_idx=self._aft_idx,
            reference_idx=self._btf_idx,
            delta=self.aft_btf_extra_delta_min,
        )

    def _project_points_to_polygon_boundary_torch(
        self,
        points: torch.Tensor,
        polygon_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """把一批二维点投影到任意简单多边形边界上的最近位置。

        这是保留计算图的 torch 版本，供前向投影和软惩罚复用。
        这里直接比较点到所有边线段的最近距离，不依赖凸性，因此也能覆盖凹多边形座椅可行域。
        """
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f"points 形状应为 [N, 2]，实际为 {tuple(points.shape)}")

        # polygon_tensor 的每一行是一个顶点，按相邻顶点首尾相连后就得到多边形边界。
        polygon = polygon_tensor.to(device=points.device, dtype=points.dtype)
        edge_start = polygon
        edge_end = torch.roll(polygon, shifts=-1, dims=0)
        edges = edge_end - edge_start

        # diff[i, j] 表示第 i 个点相对第 j 条边起点的位移向量。
        diff = points.unsqueeze(1) - edge_start.unsqueeze(0)

        # 对每条边做线段投影：
        # t=0 对应边起点，t=1 对应边终点，0<t<1 对应边内部。
        # 先按“无限长直线”求投影参数，再 clamp 到 [0, 1]，就得到“投到线段上”的最近点。
        denom = torch.clamp(torch.sum(edges * edges, dim=1), min=1e-12)
        t = torch.sum(diff * edges.unsqueeze(0), dim=2) / denom.unsqueeze(0)
        t = torch.clamp(t, min=0.0, max=1.0)

        # candidates[i, j] 是第 i 个点投到第 j 条边线段上的候选最近点。
        candidates = edge_start.unsqueeze(0) + t.unsqueeze(2) * edges.unsqueeze(0)

        # 对每个点，比较它到所有候选边界点的距离，选出最近的一条边。
        dist_sq = torch.sum((points.unsqueeze(1) - candidates) ** 2, dim=2)
        best_idx = torch.argmin(dist_sq, dim=1)
        row_idx = torch.arange(points.shape[0], device=points.device)
        # 最后返回每个点投影到最近边界位置的坐标。
        return candidates[row_idx, best_idx]

    def _get_axis_interval_from_polygon(
        self,
        info: Dict[str, object],
        fixed_value: float,
        axis: str,
    ) -> Tuple[float, float] | None:
        """计算座椅可行域与水平/竖直截线的单维合法区间。

        当只训练 `SP` 或只训练 `SH` 时，另一维来自 context，不能被投影修改。
        因此这里不是做二维最近点投影，而是求“固定另一维后，当前可训练维度还能落到哪里”。
        """
        polygon = np.asarray(info["poly"], dtype=np.float32)
        intersections = []
        tol = self._tol

        for start, end in zip(polygon, np.roll(polygon, shift=-1, axis=0)):
            x1, y1 = float(start[0]), float(start[1])
            x2, y2 = float(end[0]), float(end[1])
            if axis == "x":
                # axis="x" 表示固定 SH=fixed_value，求截线 y=fixed_value 与多边形边界的交点 x。
                if abs(y1 - y2) <= tol:
                    # 若整条边本身就是水平边，并且恰好落在截线上，则这条边的两个端点都属于交点候选。
                    if abs(fixed_value - y1) <= tol:
                        intersections.extend([x1, x2])
                    continue
                # 只有当截线高度落在当前边的 y 范围内时，这条边才可能与截线相交。
                if min(y1, y2) - tol <= fixed_value <= max(y1, y2) + tol:
                    # t 是截线在当前边线段参数方程中的位置，t∈[0,1] 才表示交点在线段内部。
                    t = (fixed_value - y1) / (y2 - y1)
                    if -tol <= t <= 1.0 + tol:
                        intersections.append(x1 + t * (x2 - x1))
                continue

            if axis == "y":
                # axis="y" 表示固定 SP=fixed_value，求截线 x=fixed_value 与多边形边界的交点 y。
                if abs(x1 - x2) <= tol:
                    # 若整条边本身就是竖直边，并且恰好落在截线上，则这条边的两个端点都属于交点候选。
                    if abs(fixed_value - x1) <= tol:
                        intersections.extend([y1, y2])
                    continue
                if min(x1, x2) - tol <= fixed_value <= max(x1, x2) + tol:
                    t = (fixed_value - x1) / (x2 - x1)
                    if -tol <= t <= 1.0 + tol:
                        intersections.append(y1 + t * (y2 - y1))
                continue

            raise ValueError(f"axis 仅支持 'x' 或 'y'，实际为 {axis!r}")

        if not intersections:
            # 固定另一维后，截线与座椅可行域完全没有交集，说明当前样本不可能只靠单维调节进入合法域。
            return None
        # 当前项目里的座椅几何在做这种单维切片后都对应一个连续区间，因此直接取最小/最大交点即可。
        return float(min(intersections)), float(max(intersections))

    def _project_seat_feasible_region(self, x: torch.Tensor) -> torch.Tensor:
        """把 SP/SH 投影回当前 side/OT 对应的精确座椅可行域。

        这是输出端投影的一部分，职责是保证送入 surrogate 的候选解不离开
        当前 `(is_driver_side, OT)` 对应的座椅可行域：
        - `SP/SH` 都可调时做二维投影；
        - 只允许单维调节时，只沿该维做截断，保持另一维不变。
        """
        if not self.seat_cache:
            return x
        sp_trainable = self._sp_idx in self._trainable_idx_set
        sh_trainable = self._sh_idx in self._trainable_idx_set
        if not sp_trainable and not sh_trainable:
            return x

        x_new = x.clone()
        side, ot = self._get_side_ot_codes(x_new)
        for (rule_side, rule_ot), info in self.seat_cache.items():
            mask = (side == rule_side) & (ot == rule_ot)
            if not mask.any():
                continue
            indices = torch.nonzero(mask, as_tuple=False).squeeze(1)
            sp_min, sp_max, sh_min, sh_max = info["bbox"]

            # 退化情形本质上是线段，可直接按对应轴的区间截断。
            if abs(sh_max - sh_min) <= self._tol:
                if sp_trainable:
                    x_new[indices, self._sp_idx] = torch.clamp(x_new[indices, self._sp_idx], sp_min, sp_max)
                if sh_trainable:
                    x_new[indices, self._sh_idx] = sh_min
                continue
            if abs(sp_max - sp_min) <= self._tol:
                if sp_trainable:
                    x_new[indices, self._sp_idx] = sp_min
                if sh_trainable:
                    x_new[indices, self._sh_idx] = torch.clamp(x_new[indices, self._sh_idx], sh_min, sh_max)
                continue
            
            # 双变量都可调时，必须按二维几何域整体处理，不能把多边形误当成矩形分别裁两轴。
            if sp_trainable and sh_trainable:
                points = torch.stack([x_new[indices, self._sp_idx], x_new[indices, self._sh_idx]], dim=1)
                valid_local = self._validate_seat_points_numpy(points.detach().cpu().numpy(), info)
                outside_mask = ~torch.as_tensor(valid_local, dtype=torch.bool, device=x_new.device)
                if outside_mask.any():
                    projected = self._project_points_to_polygon_boundary_torch(
                        points[outside_mask],
                        info["polygon_tensor"],
                    )
                    outside_indices = indices[outside_mask]
                    x_new[outside_indices, self._sp_idx] = projected[:, 0]
                    x_new[outside_indices, self._sh_idx] = projected[:, 1]
                continue

            if sp_trainable:
                projected_sp = x_new[indices, self._sp_idx].clone()
                fixed_sh_list = x_new[indices, self._sh_idx].detach().cpu().tolist()
                for row_offset, fixed_sh in enumerate(fixed_sh_list):
                    interval = self._get_axis_interval_from_polygon(info, float(fixed_sh), axis="x")
                    if interval is None:
                        raise RuntimeError(
                            f"固定 SH={fixed_sh:.6g} 与 seat_constraints[{rule_side}_{rule_ot}] 不兼容，"
                            "当前样本不可能仅通过调节 SP 进入合法域。"
                        )
                    projected_sp[row_offset] = torch.clamp(projected_sp[row_offset], interval[0], interval[1])
                x_new[indices, self._sp_idx] = projected_sp
                continue

            if sh_trainable:
                projected_sh = x_new[indices, self._sh_idx].clone()
                fixed_sp_list = x_new[indices, self._sp_idx].detach().cpu().tolist()
                for row_offset, fixed_sp in enumerate(fixed_sp_list):
                    interval = self._get_axis_interval_from_polygon(info, float(fixed_sp), axis="y")
                    if interval is None:
                        raise RuntimeError(
                            f"固定 SP={fixed_sp:.6g} 与 seat_constraints[{rule_side}_{rule_ot}] 不兼容，"
                            "当前样本不可能仅通过调节 SH 进入合法域。"
                        )
                    projected_sh[row_offset] = torch.clamp(projected_sh[row_offset], interval[0], interval[1])
                x_new[indices, self._sh_idx] = projected_sh

        return x_new

    def _project_ra_ranges(self, x: torch.Tensor) -> torch.Tensor:
        """按 (is_driver_side, OT) 组合裁剪 RA 的合法子区间。

        RA 自身的全局范围还不够，真实合法区间还取决于当前乘员侧与体型，
        所以这里必须在完整特征张量里结合 side/OT 做条件裁剪。
        """
        # 若 RA 当前属于 context，则这里不应静默改值；那种情形应由输入端校验直接拦截。
        if not self.ra_range_cache or self._ra_idx not in self._trainable_idx_set:
            return x
        side, ot = self._get_side_ot_codes(x)
        ra_column = x[:, self._ra_idx].clone()
        for (rule_side, rule_ot), (ra_min, ra_max) in self.ra_range_cache.items():
            mask = (side == rule_side) & (ot == rule_ot)
            if not mask.any():
                continue
            ra_column = torch.where(mask, torch.clamp(ra_column, ra_min, ra_max), ra_column)
        return self._replace_column(x, self._ra_idx, ra_column)

    def _validate_seat_points_numpy(
        self,
        points: np.ndarray,
        info: Dict[str, object],
    ) -> np.ndarray:
        """在 numpy 侧校验 SP/SH 是否落在对应座椅可行域内。

        这里用于无梯度的输入校验和中间几何判定：
        - 线段退化情形直接按坐标范围判断；
        - 普通多边形先做 contains_points，再允许边界附近的数值误差通过投影距离容忍。
        """
        sp_min, sp_max, sh_min, sh_max = info["bbox"]
        tol = self._tol

        if abs(sh_min - sh_max) <= tol:
            return (
                (points[:, 0] >= sp_min - tol)
                & (points[:, 0] <= sp_max + tol)
                & (np.abs(points[:, 1] - sh_min) <= tol)
            )

        if abs(sp_min - sp_max) <= tol:
            return (
                (np.abs(points[:, 0] - sp_min) <= tol)
                & (points[:, 1] >= sh_min - tol)
                & (points[:, 1] <= sh_max + tol)
            )

        inside = info["path"].contains_points(points, radius=1e-9)
        if np.all(inside):
            return inside

        valid = inside.copy()
        outside_local = np.where(~inside)[0]
        snapped = self._project_points_to_polygon_boundary(points[outside_local], info["poly"])
        dist = np.linalg.norm(points[outside_local] - snapped, axis=1)
        valid[outside_local] = dist <= tol
        return valid

    def _seat_polygon_distance_penalty(
        self,
        sp: torch.Tensor,
        sh: torch.Tensor,
        info: Dict[str, object],
    ) -> torch.Tensor:
        """对任意简单多边形座椅可行域计算边界距离软惩罚。

        先复用 numpy 几何判定区分“域内/域外”样本；
        对域外点，再用 torch 版最近边投影计算到边界的距离。
        这样既兼容凹多边形，又能在域外保持可用梯度。
        """
        points = torch.stack([sp, sh], dim=1)
        valid_local = self._validate_seat_points_numpy(points.detach().cpu().numpy(), info)
        outside_mask = ~torch.as_tensor(valid_local, dtype=torch.bool, device=points.device)
        penalty = torch.zeros(points.shape[0], device=points.device, dtype=points.dtype)
        if not outside_mask.any():
            return penalty

        projected = self._project_points_to_polygon_boundary_torch(
            points[outside_mask],
            info["polygon_tensor"],
        )
        delta = (points[outside_mask] - projected).abs()
        penalty[outside_mask] = (
            delta[:, 0] / self.base_continuous_scales[self._sp_idx]
            + delta[:, 1] / self.base_continuous_scales[self._sh_idx]
        )
        return penalty

    def _validate_full_features(
        self,
        full_features: torch.Tensor,
        continuous_bounds: Dict[int, Tuple[float, float]],
    ) -> torch.Tensor:
        """按给定连续边界规则校验完整特征张量。

        这是 `validate_full_input` 与 `validate_output_solution` 共用的底层逻辑。
        二者差别只在于连续边界采用 `base_*` 还是 `opt_*` 口径，
        以及是否额外叠加输出端专属约束。
        """
        x = self._ensure_2d(full_features, self.total_dim, "full_features")
        tol = self._tol
        valid = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)

        for idx in self.continuous_indices:
            min_value, max_value = continuous_bounds[idx]
            valid &= (x[:, idx] >= min_value - tol) & (x[:, idx] <= max_value + tol)

        for idx, allowed_cpu in self.discrete_values.items():
            allowed = allowed_cpu.to(device=x.device, dtype=x.dtype)
            matches = (x[:, idx].unsqueeze(1) - allowed.unsqueeze(0)).abs() <= tol
            valid &= matches.any(dim=1)

        overlap = x[:, self._overlap_idx]
        overlap_valid = (
            ((overlap >= self._overlap_neg_min - tol) & (overlap <= self._overlap_neg_max + tol))
            | ((overlap >= self._overlap_pos_min - tol) & (overlap <= self._overlap_pos_max + tol))
        )
        valid &= overlap_valid

        trigger = self._get_overlap_angle_special_mask(overlap, lower_tol=tol)
        if trigger.any() and self._angle_segments:
            angle = x[:, self._impact_angle_idx]
            angle_valid = torch.zeros_like(trigger)
            for o_min, o_max, a_min, a_max in self._angle_segments:
                segment_mask = (
                    (overlap >= o_min - tol)
                    & (overlap <= o_max + tol)
                    & (angle >= a_min - tol)
                    & (angle <= a_max + tol)
                )
                angle_valid |= segment_mask
            valid &= (~trigger) | angle_valid

        valid &= x[:, self._ll2_idx] <= x[:, self._ll1_idx] + self.ll2_ll1_delta_max + tol
        valid &= x[:, self._llattf_idx] >= x[:, self._btf_idx] + self.llattf_btf_delta_min - tol
        valid &= x[:, self._aft_idx] <= x[:, self._btf_idx] + self.aft_btf_delta_max + tol

        side = None
        ot = None
        if self.ra_range_cache:
            side, ot = self._get_side_ot_codes(x)
            for (rule_side, rule_ot), (ra_min, ra_max) in self.ra_range_cache.items():
                mask = (side == rule_side) & (ot == rule_ot)
                if not mask.any():
                    continue
                valid[mask] &= (
                    (x[mask, self._ra_idx] >= ra_min - tol)
                    & (x[mask, self._ra_idx] <= ra_max + tol)
                )

        if self.seat_cache:
            if side is None or ot is None:
                side, ot = self._get_side_ot_codes(x)
            for (rule_side, rule_ot), info in self.seat_cache.items():
                mask = (side == rule_side) & (ot == rule_ot)
                if not mask.any():
                    continue
                points = torch.stack([x[mask, self._sp_idx], x[mask, self._sh_idx]], dim=1)
                valid_local = self._validate_seat_points_numpy(points.detach().cpu().numpy(), info)
                valid[mask] &= torch.as_tensor(valid_local, dtype=torch.bool, device=x.device)

        return valid

    def validate_output_solution(self, full_features: torch.Tensor) -> torch.Tensor:
        """校验寻优解是否满足输出端全部约束。"""
        x = self._ensure_2d(full_features, self.total_dim, "full_features")
        valid = self._validate_full_features(x, self.output_continuous_bounds)
        valid &= x[:, self._aft_idx] >= x[:, self._btf_idx] + self.aft_btf_extra_delta_min - self._tol
        return valid

    def validate_full_input(self, full_features: torch.Tensor) -> torch.Tensor:
        """按完整物理输入口径校验样本。

        这里故意不看 opt_min/opt_max，也不看仅输出端生效的额外约束。
        它只回答一个问题：这条完整输入是否落在代理模型训练数据所对应的
        base 定义域与共享耦合规则之内。
        """
        return self._validate_full_features(full_features, self.base_continuous_bounds)

    def validate_context_input(self, context_params: torch.Tensor) -> torch.Tensor:
        """只校验当前 context 张量本身，不再补任何默认 trainable control。

        这里故意只检查当前运行中被 ParamManager 划入 `context` 的列。
        若某条耦合约束在当前角色划分下横跨 `context` 与 `trainable control`，
        这一层不会补默认值去凑完整样本，而是把该约束留给完整输入校验或输出端约束处理。
        """
        x = self._ensure_2d(context_params, len(self.context_indices), "context_params")
        tol = self._tol
        valid = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)

        for local_idx, (min_value, max_value) in self.context_continuous_bounds.items():
            valid &= (x[:, local_idx] >= min_value - tol) & (x[:, local_idx] <= max_value + tol)

        for local_idx, allowed_cpu in self.context_discrete_values.items():
            allowed = allowed_cpu.to(device=x.device, dtype=x.dtype)
            matches = (x[:, local_idx].unsqueeze(1) - allowed.unsqueeze(0)).abs() <= tol
            valid &= matches.any(dim=1)

        if self._context_overlap_local_idx is not None:
            overlap = x[:, self._context_overlap_local_idx]
            overlap_valid = (
                ((overlap >= self._overlap_neg_min - tol) & (overlap <= self._overlap_neg_max + tol))
                | ((overlap >= self._overlap_pos_min - tol) & (overlap <= self._overlap_pos_max + tol))
            )
            valid &= overlap_valid

            if self._context_angle_local_idx is not None and self._angle_segments:
                trigger = self._get_overlap_angle_special_mask(overlap, lower_tol=tol)
                if trigger.any():
                    angle = x[:, self._context_angle_local_idx]
                    angle_valid = torch.zeros_like(trigger)
                    for o_min, o_max, a_min, a_max in self._angle_segments:
                        segment_mask = (
                            (overlap >= o_min - tol)
                            & (overlap <= o_max + tol)
                            & (angle >= a_min - tol)
                            & (angle <= a_max + tol)
                        )
                        angle_valid |= segment_mask
                    valid &= (~trigger) | angle_valid

        if self._all_present(self._context_ll1_local_idx, self._context_ll2_local_idx):
            valid &= (
                x[:, self._context_ll2_local_idx]
                <= x[:, self._context_ll1_local_idx] + self.ll2_ll1_delta_max + tol
            )
        if self._all_present(self._context_btf_local_idx, self._context_llattf_local_idx):
            valid &= (
                x[:, self._context_llattf_local_idx]
                >= x[:, self._context_btf_local_idx] + self.llattf_btf_delta_min - tol
            )
        if self._all_present(self._context_btf_local_idx, self._context_aft_local_idx):
            valid &= (
                x[:, self._context_aft_local_idx]
                <= x[:, self._context_btf_local_idx] + self.aft_btf_delta_max + tol
            )

        if self._all_present(self._context_side_local_idx, self._context_ot_local_idx, self._context_ra_local_idx):
            side, ot = self._get_context_side_ot_codes(x)
            for (rule_side, rule_ot), (ra_min, ra_max) in self.ra_range_cache.items():
                mask = (side == rule_side) & (ot == rule_ot)
                if not mask.any():
                    continue
                valid[mask] &= (
                    (x[mask, self._context_ra_local_idx] >= ra_min - tol)
                    & (x[mask, self._context_ra_local_idx] <= ra_max + tol)
                )

        if self._all_present(self._context_side_local_idx, self._context_ot_local_idx, self._context_sp_local_idx, self._context_sh_local_idx):
            side, ot = self._get_context_side_ot_codes(x)
            for (rule_side, rule_ot), info in self.seat_cache.items():
                mask = (side == rule_side) & (ot == rule_ot)
                if not mask.any():
                    continue
                points = torch.stack(
                    [x[mask, self._context_sp_local_idx], x[mask, self._context_sh_local_idx]],
                    dim=1,
                )
                valid_local = self._validate_seat_points_numpy(points.detach().cpu().numpy(), info)
                valid[mask] &= torch.as_tensor(valid_local, dtype=torch.bool, device=x.device)

        return valid

    def project_forward(self, full_features: torch.Tensor, strict: bool = False) -> torch.Tensor:
        """对完整特征张量做前向投影。

        中间迭代和最终输出共用同一套拓扑投影顺序，保证 surrogate
        在任何阶段都只看到已经回到合法域的候选解。
        """
        # 顺序固定为：opt 边界 -> control 耦合 -> 输出端额外约束 -> 座椅可行域 -> RA 条件区间。
        x = self._ensure_2d(full_features, self.total_dim, "full_features").clone()
        x = self._apply_trainable_bounds(x)
        x = self._project_control_couplings(x)
        x = self._project_output_extra_constraints(x)
        x = self._project_seat_feasible_region(x)
        x = self._project_ra_ranges(x)

        if strict:
            # `strict` 只保留“最终再做一次边界收口”的调用语义，不引入额外规则。
            x = self._apply_trainable_bounds(x)

        return x

    def compute_soft_penalty(
        self,
        full_features: torch.Tensor,
        include_opt_bounds: bool = False,
    ) -> torch.Tensor:
        """计算连续可导的约束软惩罚。

        这里的输入应是投影前的原始完整特征，用于给原始游离解提供补偿梯度。
        当前覆盖的约束项包括：
        - AFT >= BTF（仅输出端额外约束）
        - AFT <= BTF + 25
        - LLATTF >= BTF
        - LL2 <= LL1
        - 可选的 trainable opt 边界
        - RA 随 is_driver_side / OT 变化的条件区间
        - SP/SH 的座椅可行域：非退化多边形使用精确半平面惩罚，线段/矩形退化情形保留 bbox 惩罚
        """
        x = self._ensure_2d(full_features, self.total_dim, "full_features")
        penalty = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

        penalty += (
            torch.relu((x[:, self._btf_idx] + self.aft_btf_extra_delta_min) - x[:, self._aft_idx])
            / self.base_continuous_scales[self._aft_idx]
        )
        penalty += (
            torch.relu(x[:, self._aft_idx] - (x[:, self._btf_idx] + self.aft_btf_delta_max))
            / self.base_continuous_scales[self._aft_idx]
        )
        penalty += (
            torch.relu((x[:, self._btf_idx] + self.llattf_btf_delta_min) - x[:, self._llattf_idx])
            / self.base_continuous_scales[self._llattf_idx]
        )
        penalty += (
            torch.relu(x[:, self._ll2_idx] - (x[:, self._ll1_idx] + self.ll2_ll1_delta_max))
            / self.base_continuous_scales[self._ll2_idx]
        )

        if include_opt_bounds and self.trainable_indices:
            mins = self._trainable_mins_cpu.to(device=x.device, dtype=x.dtype)
            maxs = self._trainable_maxs_cpu.to(device=x.device, dtype=x.dtype)
            spans = self._trainable_spans_cpu.to(device=x.device, dtype=x.dtype)
            trainable = x[:, self.trainable_indices]
            penalty += (torch.relu(mins.unsqueeze(0) - trainable) / spans.unsqueeze(0)).sum(dim=1)
            penalty += (torch.relu(trainable - maxs.unsqueeze(0)) / spans.unsqueeze(0)).sum(dim=1)

        side = None
        ot = None
        if self.ra_range_cache:
            side, ot = self._get_side_ot_codes(x)
            for (rule_side, rule_ot), (ra_min, ra_max) in self.ra_range_cache.items():
                mask = (side == rule_side) & (ot == rule_ot)
                if not mask.any():
                    continue
                ra = x[mask, self._ra_idx]
                penalty[mask] += (
                    torch.relu(ra_min - ra) + torch.relu(ra - ra_max)
                ) / self.base_continuous_scales[self._ra_idx]

        if self.seat_cache:
            if side is None or ot is None:
                side, ot = self._get_side_ot_codes(x)
            for (rule_side, rule_ot), info in self.seat_cache.items():
                mask = (side == rule_side) & (ot == rule_ot)
                if not mask.any():
                    continue
                sp_min, sp_max, sh_min, sh_max = info["bbox"]
                sp = x[mask, self._sp_idx]
                sh = x[mask, self._sh_idx]
                # 软惩罚与输入校验/前向投影保持同一退化判定口径：只要某一轴零宽，就按线段 bbox 惩罚。
                if abs(sp_max - sp_min) <= self._tol or abs(sh_max - sh_min) <= self._tol:
                    penalty[mask] += (
                        torch.relu(sp_min - sp) / self.base_continuous_scales[self._sp_idx]
                        + torch.relu(sp - sp_max) / self.base_continuous_scales[self._sp_idx]
                        + torch.relu(sh_min - sh) / self.base_continuous_scales[self._sh_idx]
                        + torch.relu(sh - sh_max) / self.base_continuous_scales[self._sh_idx]
                    )
                else:
                    penalty[mask] += self._seat_polygon_distance_penalty(sp=sp, sh=sh, info=info)

        return penalty.to(dtype=torch.float32)
