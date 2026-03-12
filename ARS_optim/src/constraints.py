from typing import Dict, Tuple

import numpy as np
import torch
from matplotlib.path import Path as MplPath

from ARS_optim.src.param_manager import ParamManager


class ConstraintEngine:
    """统一管理 ARS 参数的只读校验、前向投影与软惩罚。

    该模块只保留三条职责清晰的路径：
    1. `is_valid_physics`：纯布尔校验，不修改输入；
    2. `project_forward`：把完整特征张量投影回可行域的连续子空间；
    3. `compute_soft_penalty`：为训练或局部精调提供连续可导的约束梯度。

    旧版 sanitize 风格的“读入后直接修正”已整体移除，避免同一份规则在多个入口被重复实现成不同语义。
    """

    def __init__(self, param_manager: ParamManager):
        self.param_manager = param_manager
        self.rules = param_manager.get_sampling_rules()

        self.total_dim = param_manager.get_total_feature_dim()
        self.context_indices = param_manager.get_context_indices()
        self.trainable_indices = param_manager.get_control_trainable_indices()
        self._trainable_idx_set = set(self.trainable_indices)

        all_params = param_manager.get_all_params()
        self.name_to_index = {param["name"]: param["index"] for param in all_params}
        self.continuous_indices = [
            param["index"] for param in all_params if param.get("type") == "continuous"
        ]
        self.continuous_bounds = {
            param["index"]: (float(param["min"]), float(param["max"]))
            for param in all_params
            if param.get("type") == "continuous"
        }
        self.discrete_values = {
            idx: torch.tensor(values, dtype=torch.float32)
            for idx, values in param_manager.get_discrete_index_value_map().items()
        }

        trainable_bounds = param_manager.get_trainable_bounds()
        self._trainable_mins_cpu = trainable_bounds[0]
        self._trainable_maxs_cpu = trainable_bounds[1]

        coupling = self.rules.get("coupling", {})
        self.aft_btf_delta_max = float(coupling.get("aft_btf_delta_max", 25.0))
        self.llattf_btf_delta_min = float(coupling.get("llattf_btf_delta_min", 0.0))
        self.ll2_ll1_delta_max = float(coupling.get("ll2_ll1_delta_max", 0.0))
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

        self._build_rule_caches()

    @staticmethod
    def _parse_side_ot_rule_key(rule_name: str, key: str) -> Tuple[int, int]:
        try:
            side_text, ot_text = key.split("_")
            return int(side_text), int(ot_text)
        except ValueError as exc:
            raise ValueError(f"{rule_name} 存在非法键 {key!r}，应为 'is_driver_side_OT' 形式") from exc

    def _build_rule_caches(self) -> None:
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
            signed_area = 0.5 * float(np.sum(polygon[:, 0] * shifted[:, 1] - shifted[:, 0] * polygon[:, 1]))
            is_degenerate = abs(signed_area) <= 1e-8

            polygon_ccw = polygon
            if not is_degenerate and signed_area < 0.0:
                polygon_ccw = polygon[::-1].copy()

            polygon_tensor = torch.tensor(polygon_ccw, dtype=torch.float32)
            normals_tensor = None
            if not is_degenerate:
                edge_start = polygon_tensor
                edge_end = torch.roll(polygon_tensor, shifts=-1, dims=0)
                edges = edge_end - edge_start
                # 先构造候选法线，再使用多边形质心统一校正方向，
                # 保证内部点在所有边上的有符号距离都不为正。
                normals_tensor = torch.stack([edges[:, 1], -edges[:, 0]], dim=1)
                normals_tensor = normals_tensor / torch.clamp(
                    torch.linalg.norm(normals_tensor, dim=1, keepdim=True),
                    min=1e-12,
                )
                centroid = polygon_tensor.mean(dim=0, keepdim=True)
                centroid_signed = torch.sum((centroid - edge_start) * normals_tensor, dim=1, keepdim=True)
                flip_mask = centroid_signed > 0
                normals_tensor = torch.where(flip_mask, -normals_tensor, normals_tensor)

            self.seat_cache[(side, ot)] = {
                "poly": polygon_ccw,
                "path": MplPath(polygon_ccw),
                "bbox": (float(sp_min), float(sp_max), float(sh_min), float(sh_max)),
                "is_degenerate": is_degenerate,
                "polygon_tensor": polygon_tensor,
                "normals_tensor": normals_tensor,
            }

        self.ra_range_cache: Dict[Tuple[int, int], Tuple[float, float]] = {}
        for key, values in (self.rules.get("ra_values", {}) or {}).items():
            side, ot = self._parse_side_ot_rule_key("ra_values", key)
            arr = np.asarray(values, dtype=np.float32).reshape(-1)
            if arr.size == 0:
                raise ValueError(f"ra_values[{key!r}] 不能为空")
            self.ra_range_cache[(side, ot)] = (float(np.min(arr)), float(np.max(arr)))

    @staticmethod
    def _project_points_to_polygon_boundary(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
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
        if tensor.ndim != 2 or tensor.shape[1] != dim:
            raise ValueError(f"{name} 形状应为 [N, {dim}]，实际为 {tuple(tensor.shape)}")
        return tensor

    def compose_full_features(
        self,
        context_params: torch.Tensor,
        control_trainable: torch.Tensor = None,
    ) -> torch.Tensor:
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
        full_features = self._ensure_2d(full_features, self.total_dim, "full_features")
        return full_features[:, self.context_indices], full_features[:, self.trainable_indices]

    def _replace_column(self, x: torch.Tensor, column_idx: int, new_column: torch.Tensor) -> torch.Tensor:
        x_new = x.clone()
        x_new[:, column_idx] = new_column
        return x_new

    def _get_side_ot_codes(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        side = torch.round(x[:, self._side_idx]).to(torch.int64)
        ot = torch.round(x[:, self._ot_idx]).to(torch.int64)
        return side, ot

    def _apply_trainable_bounds(self, x: torch.Tensor) -> torch.Tensor:
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
        if constrained_idx in self._trainable_idx_set:
            constrained = torch.min(
                x[:, constrained_idx], x[:, reference_idx] + delta - epsilon
            )
            return self._replace_column(x, constrained_idx, constrained)
        elif reference_idx in self._trainable_idx_set:
            reference = torch.max(
                x[:, reference_idx], x[:, constrained_idx] - delta + epsilon
            )
            return self._replace_column(x, reference_idx, reference)
        return x

    def _project_lower_bound_pair(
        self,
        x: torch.Tensor,
        target_idx: int,
        reference_idx: int,
        delta: float,
    ) -> torch.Tensor:
        if target_idx in self._trainable_idx_set:
            target = torch.max(x[:, target_idx], x[:, reference_idx] + delta)
            return self._replace_column(x, target_idx, target)
        elif reference_idx in self._trainable_idx_set:
            reference = torch.min(x[:, reference_idx], x[:, target_idx] - delta)
            return self._replace_column(x, reference_idx, reference)
        return x

    def _project_control_couplings(self, x: torch.Tensor) -> torch.Tensor:
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

    def _project_seat_bbox(self, x: torch.Tensor) -> torch.Tensor:
        if not self.seat_cache:
            return x
        if self._sp_idx not in self._trainable_idx_set and self._sh_idx not in self._trainable_idx_set:
            return x
        side, ot = self._get_side_ot_codes(x)
        for (rule_side, rule_ot), info in self.seat_cache.items():
            mask = (side == rule_side) & (ot == rule_ot)
            if not mask.any():
                continue
            sp_min, sp_max, sh_min, sh_max = info["bbox"]
            if self._sp_idx in self._trainable_idx_set:
                sp_column = x[:, self._sp_idx]
                bounded_sp = torch.where(mask, torch.clamp(sp_column, sp_min, sp_max), sp_column)
                x = self._replace_column(x, self._sp_idx, bounded_sp)
            if self._sh_idx in self._trainable_idx_set:
                sh_column = x[:, self._sh_idx]
                bounded_sh = torch.where(mask, torch.clamp(sh_column, sh_min, sh_max), sh_column)
                x = self._replace_column(x, self._sh_idx, bounded_sh)
        return x

    def _project_ra_ranges(self, x: torch.Tensor) -> torch.Tensor:
        if not self.ra_range_cache or self._ra_idx not in self._trainable_idx_set:
            return x
        side, ot = self._get_side_ot_codes(x)
        for (rule_side, rule_ot), (ra_min, ra_max) in self.ra_range_cache.items():
            mask = (side == rule_side) & (ot == rule_ot)
            if not mask.any():
                continue
            ra_column = x[:, self._ra_idx]
            bounded_ra = torch.where(mask, torch.clamp(ra_column, ra_min, ra_max), ra_column)
            x = self._replace_column(x, self._ra_idx, bounded_ra)
        return x

    def _project_seat_polygon(self, x: torch.Tensor) -> torch.Tensor:
        if not self.seat_cache:
            return x
        if self._sp_idx not in self._trainable_idx_set and self._sh_idx not in self._trainable_idx_set:
            return x

        x_new = x.clone()
        side, ot = self._get_side_ot_codes(x_new)
        for (rule_side, rule_ot), info in self.seat_cache.items():
            mask = (side == rule_side) & (ot == rule_ot)
            if not mask.any():
                continue
            indices = torch.nonzero(mask, as_tuple=False).squeeze(1)
            sp_min, sp_max, sh_min, sh_max = info["bbox"]

            if sh_min == sh_max:
                if self._sp_idx in self._trainable_idx_set:
                    x_new[indices, self._sp_idx] = torch.clamp(x_new[indices, self._sp_idx], sp_min, sp_max)
                if self._sh_idx in self._trainable_idx_set:
                    x_new[indices, self._sh_idx] = sh_min
                continue

            if sp_min == sp_max:
                if self._sp_idx in self._trainable_idx_set:
                    x_new[indices, self._sp_idx] = sp_min
                if self._sh_idx in self._trainable_idx_set:
                    x_new[indices, self._sh_idx] = torch.clamp(x_new[indices, self._sh_idx], sh_min, sh_max)
                continue

            if self._sp_idx not in self._trainable_idx_set or self._sh_idx not in self._trainable_idx_set:
                continue

            points = torch.stack([x_new[indices, self._sp_idx], x_new[indices, self._sh_idx]], dim=1)
            points_np = points.detach().cpu().numpy()
            inside = info["path"].contains_points(points_np, radius=1e-9)
            outside_local = np.where(~inside)[0]
            if outside_local.size == 0:
                continue

            snapped = self._project_points_to_polygon_boundary(points_np[outside_local], info["poly"])
            snapped_tensor = torch.tensor(snapped, dtype=x.dtype, device=x.device)
            outside_indices = torch.tensor(outside_local, dtype=torch.long, device=x.device)
            x_new[indices[outside_indices], self._sp_idx] = snapped_tensor[:, 0]
            x_new[indices[outside_indices], self._sh_idx] = snapped_tensor[:, 1]

        return x_new

    def _validate_seat_points_numpy(
        self,
        points: np.ndarray,
        info: Dict[str, object],
    ) -> np.ndarray:
        sp_min, sp_max, sh_min, sh_max = info["bbox"]
        tol = self._tol

        if sh_min == sh_max:
            return (
                (points[:, 0] >= sp_min - tol)
                & (points[:, 0] <= sp_max + tol)
                & (np.abs(points[:, 1] - sh_min) <= tol)
            )

        if sp_min == sp_max:
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

    def _polygon_halfplane_penalty(
        self,
        sp: torch.Tensor,
        sh: torch.Tensor,
        polygon_tensor: torch.Tensor,
        normals_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """基于凸多边形半平面约束计算可微软惩罚。

        对每条边，使用单位外法线计算点到该边支撑直线的有符号距离；
        多边形内部及边界上距离均不为正，外部点在至少一条边上距离为正。
        将所有正距离做 ReLU 后求和，既能区分 bbox 内但多边形外的区域，
        又能保持全程 torch 张量计算，供训练阶段反向传播使用。
        """
        points = torch.stack([sp, sh], dim=1)
        polygon = polygon_tensor.to(device=points.device, dtype=points.dtype)
        normals = normals_tensor.to(device=points.device, dtype=points.dtype)
        signed_distance = torch.einsum("bvd,vd->bv", points.unsqueeze(1) - polygon.unsqueeze(0), normals)
        return torch.relu(signed_distance).sum(dim=1)

    def is_valid_physics(self, full_features: torch.Tensor) -> torch.Tensor:
        x = self._ensure_2d(full_features, self.total_dim, "full_features")
        tol = self._tol
        valid = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)

        for idx in self.continuous_indices:
            min_value, max_value = self.continuous_bounds[idx]
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

        trigger = (overlap.abs() >= self._overlap_abs_min - tol) & (overlap.abs() < self._overlap_abs_max - tol)
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

    def project_forward(self, full_features: torch.Tensor, strict: bool = False) -> torch.Tensor:
        """对完整特征张量做前向投影。

        `strict=False` 用于训练和局部精调的中间迭代，做连续可微的耦合投影、
        bbox 裁剪及 RA 条件区间投影（均为纯 torch 可微操作，不阻断反向传播）。
        `strict=True` 只用于最终输出，在此基础上追加 numpy 多边形边界精确投影。
        """
        x = self._ensure_2d(full_features, self.total_dim, "full_features").clone()
        x = self._apply_trainable_bounds(x)
        x = self._project_control_couplings(x)
        x = self._project_seat_bbox(x)
        # RA 条件区间需在 strict=False 路径中执行：yaml 全局范围 [15, 40] 远宽于
        # 特定 (side, OT) 组合的合法子区间（如主驾 OT=1 上限 25°），若不在此处投影，
        # 训练/精调中间步会把超出条件区间的 RA 值送入 surrogate 模型，导致域外外推。
        x = self._project_ra_ranges(x)

        if strict:
            x = self._project_seat_polygon(x)
            x = self._apply_trainable_bounds(x)

        return x

    def compute_soft_penalty(
        self,
        full_features: torch.Tensor,
        include_yaml_bounds: bool = False,
    ) -> torch.Tensor:
        """计算连续可导的约束软惩罚。

        当前覆盖的约束项包括：
        - AFT <= BTF + 25
        - LLATTF >= BTF
        - LL2 <= LL1
        - 可选的 trainable yaml 边界
        - RA 随 is_driver_side / OT 变化的条件区间
        - SP/SH 的座椅可行域：非退化多边形使用精确半平面惩罚，线段/矩形退化情形保留 bbox 惩罚
        """
        x = self._ensure_2d(full_features, self.total_dim, "full_features")
        penalty = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

        penalty += torch.relu(x[:, self._aft_idx] - (x[:, self._btf_idx] + self.aft_btf_delta_max))
        penalty += torch.relu((x[:, self._btf_idx] + self.llattf_btf_delta_min) - x[:, self._llattf_idx])
        penalty += torch.relu(x[:, self._ll2_idx] - (x[:, self._ll1_idx] + self.ll2_ll1_delta_max))

        if include_yaml_bounds and self.trainable_indices:
            mins = self._trainable_mins_cpu.to(device=x.device, dtype=x.dtype)
            maxs = self._trainable_maxs_cpu.to(device=x.device, dtype=x.dtype)
            trainable = x[:, self.trainable_indices]
            penalty += torch.relu(mins.unsqueeze(0) - trainable).sum(dim=1)
            penalty += torch.relu(trainable - maxs.unsqueeze(0)).sum(dim=1)

        side = None
        ot = None
        if self.ra_range_cache:
            side, ot = self._get_side_ot_codes(x)
            for (rule_side, rule_ot), (ra_min, ra_max) in self.ra_range_cache.items():
                mask = (side == rule_side) & (ot == rule_ot)
                if not mask.any():
                    continue
                ra = x[mask, self._ra_idx]
                penalty[mask] += torch.relu(ra_min - ra) + torch.relu(ra - ra_max)

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
                if info["is_degenerate"]:
                    penalty[mask] += (
                        torch.relu(sp_min - sp)
                        + torch.relu(sp - sp_max)
                        + torch.relu(sh_min - sh)
                        + torch.relu(sh - sh_max)
                    )
                else:
                    penalty[mask] += self._polygon_halfplane_penalty(
                        sp=sp,
                        sh=sh,
                        polygon_tensor=info["polygon_tensor"],
                        normals_tensor=info["normals_tensor"],
                    )

        return penalty.to(dtype=torch.float32)
