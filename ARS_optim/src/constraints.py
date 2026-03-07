import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
from matplotlib.path import Path as MplPath

from ARS_optim.src.param_manager import ParamManager


class ConstraintEngine:
    """统一管理 ARS 参数硬约束与软惩罚。

    设计上只保留一套规则源：
    - `sanitize_*` 用于训练采样、baseline 输入和评估 CSV 的确定性修复；
    - `project_forward` 用于策略网络和局部精调阶段的可微投影；
    - `compute_soft_penalty` 用于把硬规则转成训练期的平滑惩罚。
    """

    def __init__(self, param_manager: ParamManager):
        self.param_manager = param_manager
        self.rules = param_manager.get_sampling_rules()

        self.total_dim = param_manager.get_total_feature_dim()
        self.context_indices = param_manager.get_context_indices()
        self.trainable_indices = param_manager.get_control_trainable_indices()
        context_params = param_manager.get_context_params()
        self.trainable_names = {param["name"]: idx for idx, param in enumerate(param_manager.control_trainable_params)}
        self.context_names = {param["name"]: idx for idx, param in enumerate(context_params)}
        self.fixed_defaults = {param["name"]: float(param["default"]) for param in param_manager.control_fixed_params}

        self.name_to_index = {param["name"]: param["index"] for param in param_manager.all_params}
        self.continuous_indices = [
            param["index"] for param in param_manager.all_params if param.get("type") == "continuous"
        ]
        self.continuous_bounds = {
            param["index"]: (float(param["min"]), float(param["max"]))
            for param in param_manager.all_params
            if param.get("type") == "continuous"
        }

        coupling = self.rules.get("coupling", {})
        self.aft_btf_delta_max = float(coupling.get("aft_btf_delta_max", 25.0))
        self.epsilon = float(coupling.get("epsilon", 1e-3))

        self._build_rule_caches()

    def _build_rule_caches(self) -> None:
        self.seat_cache: Dict[Tuple[int, int], Dict[str, object]] = {}
        for key, points in (self.rules.get("seat_constraints", {}) or {}).items():
            try:
                side_text, ot_text = key.split("_")
                side = int(side_text)
                ot = int(ot_text)
            except Exception:
                continue
            polygon = np.asarray(points, dtype=np.float32)
            if polygon.ndim != 2 or polygon.shape[1] != 2:
                continue
            sp_min, sh_min = np.min(polygon, axis=0)
            sp_max, sh_max = np.max(polygon, axis=0)
            self.seat_cache[(side, ot)] = {
                "poly": polygon,
                "path": MplPath(polygon),
                "bbox": (float(sp_min), float(sp_max), float(sh_min), float(sh_max)),
            }

        self.ra_cache: Dict[Tuple[int, int], torch.Tensor] = {}
        for key, values in (self.rules.get("ra_values", {}) or {}).items():
            try:
                side_text, ot_text = key.split("_")
                side = int(side_text)
                ot = int(ot_text)
            except Exception:
                continue
            arr = np.asarray(values, dtype=np.float32).reshape(-1)
            if arr.size == 0:
                continue
            self.ra_cache[(side, ot)] = torch.tensor(arr, dtype=torch.float32)

    def _ensure_2d(self, tensor: torch.Tensor, dim: int, name: str) -> torch.Tensor:
        if tensor.ndim != 2 or tensor.shape[1] != dim:
            raise ValueError(f"{name} 形状应为 [N, {dim}]，实际为 {tuple(tensor.shape)}")
        return tensor

    def compose_full_features(self, context_params: torch.Tensor, control_trainable: torch.Tensor = None) -> torch.Tensor:
        context_params = self._ensure_2d(context_params, len(self.context_indices), "context_params")
        full = self.param_manager.get_default_feature_matrix(context_params.shape[0], device=context_params.device)
        full[:, self.context_indices] = context_params
        if control_trainable is not None:
            control_trainable = self._ensure_2d(control_trainable, len(self.trainable_indices), "control_trainable")
            full[:, self.trainable_indices] = control_trainable
        return full

    def split_from_full(self, full_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        full_features = self._ensure_2d(full_features, self.total_dim, "full_features")
        return full_features[:, self.context_indices], full_features[:, self.trainable_indices]

    def sanitize_context(self, context_params: torch.Tensor) -> torch.Tensor:
        full = self.compose_full_features(context_params=context_params)
        full = self.sanitize_full_features(full, apply_control_couplings=True)
        context_params, _ = self.split_from_full(full)
        return context_params

    def sanitize_context_and_trainable(self, context_params: torch.Tensor, control_trainable: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        full = self.compose_full_features(context_params=context_params, control_trainable=control_trainable)
        full = self.sanitize_full_features(full, apply_control_couplings=True)
        return self.split_from_full(full)

    def sanitize_full_features(self, full_features: torch.Tensor, apply_control_couplings: bool = True) -> torch.Tensor:
        full_features = self._ensure_2d(full_features, self.total_dim, "full_features")
        x = full_features.clone()
        self._clamp_discrete(x)
        self._clamp_continuous_bounds(x)
        self._enforce_overlap_domain(x)
        self._enforce_overlap_angle_rule(x)
        if apply_control_couplings:
            self._enforce_control_couplings(x)
        self._enforce_ra_discrete(x)
        self._enforce_seat_polygon(x)
        self._clamp_continuous_bounds(x)
        return x

    def _get_param_col(self, name: str, cols: List[torch.Tensor], context_params: torch.Tensor, device: torch.device) -> torch.Tensor:
        if name in self.trainable_names:
            return cols[self.trainable_names[name]]
        if name in self.context_names:
            return context_params[:, self.context_names[name]]
        if name in self.fixed_defaults:
            return torch.full((context_params.shape[0],), self.fixed_defaults[name], device=device, dtype=torch.float32)
        raise KeyError(f"未知参数: {name}")

    def project_forward(self, actions: torch.Tensor, context_params: torch.Tensor) -> torch.Tensor:
        device = actions.device
        # Clone each column to avoid in-place version conflicts when the same
        # trainable variable participates in multiple coupled projections.
        cols = [actions[:, idx].clone() for idx in range(actions.shape[1])]

        aft = self._get_param_col("AFT", cols, context_params, device)
        btf = self._get_param_col("BTF", cols, context_params, device)
        if "AFT" in self.trainable_names:
            cols[self.trainable_names["AFT"]] = torch.min(aft, btf + self.aft_btf_delta_max - self.epsilon)
        aft = self._get_param_col("AFT", cols, context_params, device)
        if "BTF" in self.trainable_names:
            cols[self.trainable_names["BTF"]] = torch.max(btf, aft - self.aft_btf_delta_max + self.epsilon)

        llattf = self._get_param_col("LLATTF", cols, context_params, device)
        btf = self._get_param_col("BTF", cols, context_params, device)
        if "LLATTF" in self.trainable_names:
            cols[self.trainable_names["LLATTF"]] = torch.max(llattf, btf)
        llattf = self._get_param_col("LLATTF", cols, context_params, device)
        if "BTF" in self.trainable_names:
            cols[self.trainable_names["BTF"]] = torch.min(btf, llattf)

        ll1 = self._get_param_col("LL1", cols, context_params, device)
        ll2 = self._get_param_col("LL2", cols, context_params, device)
        if "LL2" in self.trainable_names:
            cols[self.trainable_names["LL2"]] = torch.min(ll2, ll1)
        ll2 = self._get_param_col("LL2", cols, context_params, device)
        if "LL1" in self.trainable_names:
            cols[self.trainable_names["LL1"]] = torch.max(ll1, ll2)

        projected_actions = torch.stack(cols, dim=1)
        min_bounds, max_bounds = self.param_manager.get_trainable_bounds(device=device)
        return torch.clamp(projected_actions, min_bounds.unsqueeze(0), max_bounds.unsqueeze(0))

    def compute_soft_penalty(self, actions: torch.Tensor, context_params: torch.Tensor) -> torch.Tensor:
        device = actions.device
        penalty = torch.zeros(actions.shape[0], device=device, dtype=torch.float32)
        cols = [actions[:, idx] for idx in range(actions.shape[1])]

        aft = self._get_param_col("AFT", cols, context_params, device)
        btf = self._get_param_col("BTF", cols, context_params, device)
        llattf = self._get_param_col("LLATTF", cols, context_params, device)
        ll1 = self._get_param_col("LL1", cols, context_params, device)
        ll2 = self._get_param_col("LL2", cols, context_params, device)

        penalty += torch.relu(aft - (btf + self.aft_btf_delta_max))
        penalty += torch.relu(btf - llattf)
        penalty += torch.relu(ll2 - ll1)

        if "is_driver_side" not in self.context_names or "OT" not in self.context_names:
            return penalty

        side = torch.round(context_params[:, self.context_names["is_driver_side"]]).to(torch.int64)
        ot = torch.round(context_params[:, self.context_names["OT"]]).to(torch.int64)

        if self.ra_cache and {"RA"} & (set(self.trainable_names) | set(self.context_names) | set(self.fixed_defaults)):
            ra = self._get_param_col("RA", cols, context_params, device)
            ra_penalty = torch.zeros_like(penalty)
            for (rule_side, rule_ot), allowed_cpu in self.ra_cache.items():
                mask = (side == rule_side) & (ot == rule_ot)
                if not mask.any():
                    continue
                allowed = allowed_cpu.to(device=device)
                ra_values = ra[mask]
                ra_penalty[mask] = (ra_values.unsqueeze(1) - allowed.unsqueeze(0)).abs().min(dim=1).values
            penalty += ra_penalty

        has_sp = "SP" in self.trainable_names or "SP" in self.context_names or "SP" in self.fixed_defaults
        has_sh = "SH" in self.trainable_names or "SH" in self.context_names or "SH" in self.fixed_defaults
        if has_sp and has_sh and self.seat_cache:
            sp = self._get_param_col("SP", cols, context_params, device)
            sh = self._get_param_col("SH", cols, context_params, device)
            seat_penalty = torch.zeros_like(penalty)
            for (rule_side, rule_ot), info in self.seat_cache.items():
                mask = (side == rule_side) & (ot == rule_ot)
                if not mask.any():
                    continue
                sp_min, sp_max, sh_min, sh_max = info["bbox"]
                sp_values = sp[mask]
                sh_values = sh[mask]
                seat_penalty[mask] = (
                    torch.relu(sp_min - sp_values)
                    + torch.relu(sp_values - sp_max)
                    + torch.relu(sh_min - sh_values)
                    + torch.relu(sh_values - sh_max)
                )
            penalty += seat_penalty
        return penalty

    def _clamp_discrete(self, x: torch.Tensor) -> None:
        side_idx = self.name_to_index.get("is_driver_side")
        ot_idx = self.name_to_index.get("OT")
        x[:, side_idx] = torch.clamp(torch.round(x[:, side_idx]), 0, 1)
        x[:, ot_idx] = torch.clamp(torch.round(x[:, ot_idx]), 1, 3)

    def _clamp_continuous_bounds(self, x: torch.Tensor) -> None:
        for idx in self.continuous_indices:
            min_value, max_value = self.continuous_bounds[idx]
            x[:, idx] = torch.clamp(x[:, idx], min_value, max_value)

    def _enforce_overlap_domain(self, x: torch.Tensor) -> None:
        overlap_idx = self.name_to_index["overlap"]
        overlap_cfg = self.rules.get("overlap", {})
        overlap = x[:, overlap_idx]

        special_abs_high = float(overlap_cfg.get("special_abs_high", 0.99))
        special_abs_low = float(overlap_cfg.get("special_abs_low", 0.02))
        force_to = float(overlap_cfg.get("force_to", 1.0))
        neg_min, neg_max = map(float, overlap_cfg.get("domain", {}).get("negative", [-1.0, -0.25]))
        pos_min, pos_max = map(float, overlap_cfg.get("domain", {}).get("positive", [0.25, 1.0]))

        special_mask = (overlap.abs() > special_abs_high) | (overlap.abs() < special_abs_low)
        overlap = torch.where(special_mask, torch.full_like(overlap, force_to), overlap)
        gap_mask = overlap.abs() < pos_min
        sign = torch.where(overlap >= 0.0, torch.ones_like(overlap), -torch.ones_like(overlap))
        overlap = torch.where(gap_mask, sign * pos_min, overlap)
        overlap = torch.where(
            overlap >= 0.0,
            torch.clamp(overlap, pos_min, pos_max),
            torch.clamp(overlap, neg_min, neg_max),
        )
        x[:, overlap_idx] = overlap

    def _enforce_overlap_angle_rule(self, x: torch.Tensor) -> None:
        overlap_idx = self.name_to_index["overlap"]
        angle_idx = self.name_to_index["impact_angle"]
        rule = self.rules.get("overlap_angle", {})

        overlap = x[:, overlap_idx]
        angle = x[:, angle_idx]
        abs_min = float(rule.get("overlap_abs_min", 0.25))
        abs_max = float(rule.get("overlap_abs_max", 0.3))
        angle_abs_min = float(rule.get("angle_abs_min", 30.0))
        trigger = (overlap.abs() >= abs_min) & (overlap.abs() < abs_max)
        if not trigger.any():
            return

        for segment in rule.get("angle_sampling", {}).get("positive_overlap", []):
            if not isinstance(segment, (list, tuple)) or len(segment) != 4:
                continue
            o_min, o_max, a_min, a_max = map(float, segment)
            mask = trigger & (overlap >= o_min) & (overlap <= o_max)
            if mask.any():
                angle[mask] = torch.clamp(angle[mask], a_min, a_max)

        for segment in rule.get("angle_sampling", {}).get("negative_overlap", []):
            if not isinstance(segment, (list, tuple)) or len(segment) != 4:
                continue
            o_min, o_max, a_min, a_max = map(float, segment)
            mask = trigger & (overlap >= o_min) & (overlap <= o_max)
            if mask.any():
                angle[mask] = torch.clamp(angle[mask], a_min, a_max)

        fallback = trigger & (angle.abs() < angle_abs_min)
        if fallback.any():
            signs = torch.where(overlap[fallback] > 0.0, -torch.ones_like(overlap[fallback]), torch.ones_like(overlap[fallback]))
            angle[fallback] = signs * angle_abs_min
        x[:, angle_idx] = torch.clamp(angle, -45.0, 45.0)

    def _enforce_control_couplings(self, x: torch.Tensor) -> None:
        ll1_idx = self.name_to_index.get("LL1")
        ll2_idx = self.name_to_index.get("LL2")
        btf_idx = self.name_to_index.get("BTF")
        llattf_idx = self.name_to_index.get("LLATTF")
        aft_idx = self.name_to_index.get("AFT")

        x[:, ll2_idx] = torch.min(x[:, ll2_idx], x[:, ll1_idx])
        x[:, llattf_idx] = torch.max(x[:, llattf_idx], x[:, btf_idx])
        x[:, aft_idx] = torch.min(x[:, aft_idx], x[:, btf_idx] + self.aft_btf_delta_max - self.epsilon)

    def _enforce_ra_discrete(self, x: torch.Tensor) -> None:
        if not self.ra_cache:
            return
        ra_idx = self.name_to_index["RA"]
        side_idx = self.name_to_index["is_driver_side"]
        ot_idx = self.name_to_index["OT"]
        side = torch.round(x[:, side_idx]).to(torch.int64)
        ot = torch.round(x[:, ot_idx]).to(torch.int64)
        ra = x[:, ra_idx]
        quantized = ra.clone()
        for (rule_side, rule_ot), allowed_cpu in self.ra_cache.items():
            mask = (side == rule_side) & (ot == rule_ot)
            if not mask.any():
                continue
            allowed = allowed_cpu.to(device=x.device)
            values = ra[mask]
            nearest = (values.unsqueeze(1) - allowed.unsqueeze(0)).abs().argmin(dim=1)
            quantized[mask] = allowed[nearest]
        x[:, ra_idx] = quantized

    def _enforce_seat_polygon(self, x: torch.Tensor) -> None:
        if not self.seat_cache:
            return
        sp_idx = self.name_to_index["SP"]
        sh_idx = self.name_to_index["SH"]
        side_idx = self.name_to_index["is_driver_side"]
        ot_idx = self.name_to_index["OT"]

        side = torch.round(x[:, side_idx]).to(torch.int64)
        ot = torch.round(x[:, ot_idx]).to(torch.int64)
        for (rule_side, rule_ot), info in self.seat_cache.items():
            mask = (side == rule_side) & (ot == rule_ot)
            if not mask.any():
                continue
            indices = torch.nonzero(mask, as_tuple=False).squeeze(1)
            sp = x[indices, sp_idx]
            sh = x[indices, sh_idx]
            polygon = info["poly"]
            path = info["path"]
            sp_min, sp_max, sh_min, sh_max = info["bbox"]

            sp = torch.clamp(sp, sp_min, sp_max)
            sh = torch.clamp(sh, sh_min, sh_max)

            if sp_min == sp_max and sh_min == sh_max:
                x[indices, sp_idx] = sp_min
                x[indices, sh_idx] = sh_min
                continue
            if sh_min == sh_max:
                x[indices, sp_idx] = sp
                x[indices, sh_idx] = sh_min
                continue
            if sp_min == sp_max:
                x[indices, sp_idx] = sp_min
                x[indices, sh_idx] = sh
                continue

            points = torch.stack([sp, sh], dim=1).detach().cpu().numpy()
            inside = path.contains_points(points, radius=1e-9)
            outside_local = np.where(~inside)[0]
            if outside_local.size > 0:
                vertices = torch.tensor(polygon, dtype=torch.float32, device=x.device)
                outside_indices = torch.tensor(outside_local, dtype=torch.long, device=x.device)
                outside_points = torch.stack([sp[outside_indices], sh[outside_indices]], dim=1)
                distances = torch.cdist(outside_points.unsqueeze(0), vertices.unsqueeze(0)).squeeze(0)
                nearest = distances.argmin(dim=1)
                snapped = vertices[nearest]
                sp[outside_indices] = snapped[:, 0]
                sh[outside_indices] = snapped[:, 1]

            x[indices, sp_idx] = sp
            x[indices, sh_idx] = sh
