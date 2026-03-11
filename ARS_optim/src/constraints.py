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
            self.seat_cache[(side, ot)] = {
                "poly": polygon,
                "path": MplPath(polygon),
                "bbox": (float(sp_min), float(sp_max), float(sh_min), float(sh_max)),
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

    def _apply_trainable_bounds(self, x: torch.Tensor) -> None:
        if not self.trainable_indices:
            return
        mins = self._trainable_mins_cpu.to(device=x.device, dtype=x.dtype)
        maxs = self._trainable_maxs_cpu.to(device=x.device, dtype=x.dtype)
        x[:, self.trainable_indices] = torch.clamp(x[:, self.trainable_indices], mins, maxs)

    def _project_upper_bound_pair(
        self,
        x: torch.Tensor,
        constrained_idx: int,
        reference_idx: int,
        delta: float,
        epsilon: float = 0.0,
    ) -> None:
        if constrained_idx in self._trainable_idx_set:
            x[:, constrained_idx] = torch.min(
                x[:, constrained_idx], x[:, reference_idx] + delta - epsilon
            )
        elif reference_idx in self._trainable_idx_set:
            x[:, reference_idx] = torch.max(
                x[:, reference_idx], x[:, constrained_idx] - delta + epsilon
            )

    def _project_lower_bound_pair(
        self,
        x: torch.Tensor,
        target_idx: int,
        reference_idx: int,
        delta: float,
    ) -> None:
        if target_idx in self._trainable_idx_set:
            x[:, target_idx] = torch.max(x[:, target_idx], x[:, reference_idx] + delta)
        elif reference_idx in self._trainable_idx_set:
            x[:, reference_idx] = torch.min(x[:, reference_idx], x[:, target_idx] - delta)

    def _project_control_couplings(self, x: torch.Tensor) -> None:
        self._project_upper_bound_pair(
            x,
            constrained_idx=self._aft_idx,
            reference_idx=self._btf_idx,
            delta=self.aft_btf_delta_max,
            epsilon=self.epsilon,
        )
        self._project_lower_bound_pair(
            x,
            target_idx=self._llattf_idx,
            reference_idx=self._btf_idx,
            delta=self.llattf_btf_delta_min,
        )
        self._project_upper_bound_pair(
            x,
            constrained_idx=self._ll2_idx,
            reference_idx=self._ll1_idx,
            delta=self.ll2_ll1_delta_max,
        )

    def _iter_side_ot_masks(self, x: torch.Tensor):
        if self._side_idx is None or self._ot_idx is None:
            return []
        side = torch.round(x[:, self._side_idx]).to(torch.int64)
        ot = torch.round(x[:, self._ot_idx]).to(torch.int64)
        return [
            ((side == rule_side) & (ot == rule_ot), info)
            for (rule_side, rule_ot), info in self.seat_cache.items()
        ]

    def _project_seat_bbox(self, x: torch.Tensor) -> None:
        if not self.seat_cache:
            return
        if self._sp_idx not in self._trainable_idx_set and self._sh_idx not in self._trainable_idx_set:
            return
        for mask, info in self._iter_side_ot_masks(x):
            if not mask.any():
                continue
            sp_min, sp_max, sh_min, sh_max = info["bbox"]
            if self._sp_idx in self._trainable_idx_set:
                x[mask, self._sp_idx] = torch.clamp(x[mask, self._sp_idx], sp_min, sp_max)
            if self._sh_idx in self._trainable_idx_set:
                x[mask, self._sh_idx] = torch.clamp(x[mask, self._sh_idx], sh_min, sh_max)

    def _project_ra_ranges(self, x: torch.Tensor) -> None:
        if not self.ra_range_cache or self._ra_idx not in self._trainable_idx_set:
            return
        side = torch.round(x[:, self._side_idx]).to(torch.int64)
        ot = torch.round(x[:, self._ot_idx]).to(torch.int64)
        for (rule_side, rule_ot), (ra_min, ra_max) in self.ra_range_cache.items():
            mask = (side == rule_side) & (ot == rule_ot)
            if not mask.any():
                continue
            x[mask, self._ra_idx] = torch.clamp(x[mask, self._ra_idx], ra_min, ra_max)

    def _project_seat_polygon(self, x: torch.Tensor) -> None:
        if not self.seat_cache:
            return
        if self._sp_idx not in self._trainable_idx_set and self._sh_idx not in self._trainable_idx_set:
            return

        side = torch.round(x[:, self._side_idx]).to(torch.int64)
        ot = torch.round(x[:, self._ot_idx]).to(torch.int64)
        for (rule_side, rule_ot), info in self.seat_cache.items():
            mask = (side == rule_side) & (ot == rule_ot)
            if not mask.any():
                continue
            indices = torch.nonzero(mask, as_tuple=False).squeeze(1)
            sp_min, sp_max, sh_min, sh_max = info["bbox"]

            if sh_min == sh_max:
                if self._sp_idx in self._trainable_idx_set:
                    x[indices, self._sp_idx] = torch.clamp(x[indices, self._sp_idx], sp_min, sp_max)
                if self._sh_idx in self._trainable_idx_set:
                    x[indices, self._sh_idx] = sh_min
                continue

            if sp_min == sp_max:
                if self._sp_idx in self._trainable_idx_set:
                    x[indices, self._sp_idx] = sp_min
                if self._sh_idx in self._trainable_idx_set:
                    x[indices, self._sh_idx] = torch.clamp(x[indices, self._sh_idx], sh_min, sh_max)
                continue

            if self._sp_idx not in self._trainable_idx_set or self._sh_idx not in self._trainable_idx_set:
                continue

            points = torch.stack([x[indices, self._sp_idx], x[indices, self._sh_idx]], dim=1)
            points_np = points.detach().cpu().numpy()
            inside = info["path"].contains_points(points_np, radius=1e-9)
            outside_local = np.where(~inside)[0]
            if outside_local.size == 0:
                continue

            snapped = self._project_points_to_polygon_boundary(points_np[outside_local], info["poly"])
            snapped_tensor = torch.tensor(snapped, dtype=x.dtype, device=x.device)
            outside_indices = torch.tensor(outside_local, dtype=torch.long, device=x.device)
            x[indices[outside_indices], self._sp_idx] = snapped_tensor[:, 0]
            x[indices[outside_indices], self._sh_idx] = snapped_tensor[:, 1]

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

        if self.ra_range_cache:
            side = torch.round(x[:, self._side_idx]).to(torch.int64)
            ot = torch.round(x[:, self._ot_idx]).to(torch.int64)
            for (rule_side, rule_ot), (ra_min, ra_max) in self.ra_range_cache.items():
                mask = (side == rule_side) & (ot == rule_ot)
                if not mask.any():
                    continue
                valid[mask] &= (
                    (x[mask, self._ra_idx] >= ra_min - tol)
                    & (x[mask, self._ra_idx] <= ra_max + tol)
                )

        if self.seat_cache:
            side = torch.round(x[:, self._side_idx]).to(torch.int64)
            ot = torch.round(x[:, self._ot_idx]).to(torch.int64)
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

        `strict=False` 用于训练和局部精调的中间迭代，只做连续可微的耦合投影和 bbox 级别安全裁剪。
        `strict=True` 只用于最终输出，在无梯度上下文中追加多边形边界投影和 RA 条件区间裁剪。
        """
        x = self._ensure_2d(full_features, self.total_dim, "full_features").clone()
        self._apply_trainable_bounds(x)
        self._project_control_couplings(x)
        self._project_seat_bbox(x)

        if strict:
            self._project_seat_polygon(x)
            self._project_ra_ranges(x)
            self._apply_trainable_bounds(x)

        return x

    def compute_soft_penalty(
        self,
        full_features: torch.Tensor,
        include_yaml_bounds: bool = False,
    ) -> torch.Tensor:
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

        if self.ra_range_cache:
            side = torch.round(x[:, self._side_idx]).to(torch.int64)
            ot = torch.round(x[:, self._ot_idx]).to(torch.int64)
            for (rule_side, rule_ot), (ra_min, ra_max) in self.ra_range_cache.items():
                mask = (side == rule_side) & (ot == rule_ot)
                if not mask.any():
                    continue
                ra = x[mask, self._ra_idx]
                penalty[mask] += torch.relu(ra_min - ra) + torch.relu(ra - ra_max)

        if self.seat_cache:
            side = torch.round(x[:, self._side_idx]).to(torch.int64)
            ot = torch.round(x[:, self._ot_idx]).to(torch.int64)
            for (rule_side, rule_ot), info in self.seat_cache.items():
                mask = (side == rule_side) & (ot == rule_ot)
                if not mask.any():
                    continue
                sp_min, sp_max, sh_min, sh_max = info["bbox"]
                sp = x[mask, self._sp_idx]
                sh = x[mask, self._sh_idx]
                penalty[mask] += (
                    torch.relu(sp_min - sp)
                    + torch.relu(sp - sp_max)
                    + torch.relu(sh_min - sh)
                    + torch.relu(sh - sh_max)
                )

        return penalty.to(dtype=torch.float32)
