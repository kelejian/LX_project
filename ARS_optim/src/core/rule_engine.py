import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
from matplotlib.path import Path as MplPath

from ARS_optim.src.core.param_manager import ParamManager


class RuleEngine:
    """统一规则入口。

    设计目标：
    1. 把 step0 中定义的硬规则集中在单处执行，避免训练/评估各写一套。
    2. 支持对完整 13 维特征进行规则化，再按需求拆回 context/trainable。
    3. 评估链路使用确定性修复（不做随机拒绝采样），保证结果可复现。
    """

    def __init__(self, param_manager: ParamManager):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.param_manager = param_manager
        self.rules = param_manager.get_sampling_rules()

        self.total_dim = self.param_manager.get_total_feature_dim()
        self.context_indices = self.param_manager.get_context_indices()
        self.trainable_indices = self.param_manager.get_control_trainable_indices()

        self.name_to_index = {p["name"]: p["index"] for p in self.param_manager.all_params}
        self.continuous_indices = [p["index"] for p in self.param_manager.all_params if p.get("type") == "continuous"]
        self.continuous_bounds = {
            p["index"]: (float(p["min"]), float(p["max"]))
            for p in self.param_manager.all_params
            if p.get("type") == "continuous"
        }

        self._build_rule_caches()

    def _build_rule_caches(self) -> None:
        seat_rules = self.rules.get("seat_constraints", {})
        self.seat_cache: Dict[Tuple[int, int], Dict] = {}
        if isinstance(seat_rules, dict):
            for key, pts in seat_rules.items():
                try:
                    side_s, ot_s = key.split("_")
                    side, ot = int(side_s), int(ot_s)
                except Exception:
                    continue
                poly = np.asarray(pts, dtype=np.float32)
                if poly.ndim != 2 or poly.shape[1] != 2:
                    continue
                sp_min, sh_min = np.min(poly, axis=0)
                sp_max, sh_max = np.max(poly, axis=0)
                self.seat_cache[(side, ot)] = {
                    "poly": poly,
                    "path": MplPath(poly),
                    "bbox": (float(sp_min), float(sp_max), float(sh_min), float(sh_max)),
                }

        ra_rules = self.rules.get("ra_values", {})
        self.ra_cache: Dict[Tuple[int, int], torch.Tensor] = {}
        if isinstance(ra_rules, dict):
            for key, vals in ra_rules.items():
                try:
                    side_s, ot_s = key.split("_")
                    side, ot = int(side_s), int(ot_s)
                except Exception:
                    continue
                arr = np.asarray(vals, dtype=np.float32).reshape(-1)
                if arr.size == 0:
                    continue
                self.ra_cache[(side, ot)] = torch.tensor(arr, dtype=torch.float32)

    def _ensure_2d(self, x: torch.Tensor, dim: int, name: str) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != dim:
            raise ValueError(f"{name} 形状应为 [N, {dim}]，实际为 {tuple(x.shape)}")
        return x

    def compose_full_features(
        self,
        context_params: torch.Tensor,
        control_trainable: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        组装完整 13 维特征张量。

        统一语义：
        - 先从 param_space.yaml 的 default 向量出发；
        - 再覆盖 context；
        - 最后按需覆盖 trainable control。

        这样可以保证训练、评估、局部精调对“缺省值”的理解完全一致，
        避免某些调用点把未覆盖列误填为 0。
        """
        context_params = self._ensure_2d(context_params, len(self.context_indices), "context_params")
        n = context_params.shape[0]
        device = context_params.device

        full = self.param_manager.get_default_feature_matrix(batch_size=n, device=device)
        full[:, self.context_indices] = context_params

        if control_trainable is not None:
            control_trainable = self._ensure_2d(control_trainable, len(self.trainable_indices), "control_trainable")
            full[:, self.trainable_indices] = control_trainable
        return full

    def split_from_full(self, full_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """从完整 13 维张量切回 context 与 trainable。"""
        full_features = self._ensure_2d(full_features, self.total_dim, "full_features")
        ctx = full_features[:, self.context_indices]
        trn = full_features[:, self.trainable_indices]
        return ctx, trn

    def sanitize_context(self, context_params: torch.Tensor) -> torch.Tensor:
        """对 context 参数做规则化。"""
        full = self.compose_full_features(context_params=context_params, control_trainable=None)
        full = self.sanitize_full_features(full)
        ctx, _ = self.split_from_full(full)
        return ctx

    def sanitize_context_and_trainable(
        self,
        context_params: torch.Tensor,
        control_trainable: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """对 context 与 trainable 联合规则化。"""
        full = self.compose_full_features(context_params=context_params, control_trainable=control_trainable)
        full = self.sanitize_full_features(full)
        return self.split_from_full(full)

    def sanitize_full_features(self, full_features: torch.Tensor) -> torch.Tensor:
        """统一硬规则处理入口（确定性）。"""
        full_features = self._ensure_2d(full_features, self.total_dim, "full_features")
        x = full_features.clone()

        self._clamp_discrete(x)
        self._clamp_continuous_bounds(x)
        self._enforce_overlap_domain(x)
        self._enforce_overlap_angle_rule(x)
        self._enforce_control_couplings(x)
        self._enforce_ra_discrete(x)
        self._enforce_seat_polygon(x)
        self._clamp_continuous_bounds(x)
        return x

    def _clamp_discrete(self, x: torch.Tensor) -> None:
        side_idx = self.name_to_index.get("is_driver_side")
        ot_idx = self.name_to_index.get("OT")
        if side_idx is not None:
            x[:, side_idx] = torch.clamp(torch.round(x[:, side_idx]), 0, 1)
        if ot_idx is not None:
            x[:, ot_idx] = torch.clamp(torch.round(x[:, ot_idx]), 1, 3)

    def _clamp_continuous_bounds(self, x: torch.Tensor) -> None:
        for idx in self.continuous_indices:
            mn, mx = self.continuous_bounds[idx]
            x[:, idx] = torch.clamp(x[:, idx], mn, mx)

    def _enforce_overlap_domain(self, x: torch.Tensor) -> None:
        overlap_idx = self.name_to_index.get("overlap")
        if overlap_idx is None:
            return
        overlap_cfg = self.rules.get("overlap", {})
        overlap = x[:, overlap_idx]

        special_abs_high = float(overlap_cfg.get("special_abs_high", 0.99))
        special_abs_low = float(overlap_cfg.get("special_abs_low", 0.02))
        force_to = float(overlap_cfg.get("force_to", 1.0))

        domain = overlap_cfg.get("domain", {})
        neg_min, neg_max = map(float, domain.get("negative", [-1.0, -0.25]))
        pos_min, pos_max = map(float, domain.get("positive", [0.25, 1.0]))

        special = (overlap.abs() > special_abs_high) | (overlap.abs() < special_abs_low)
        overlap = torch.where(special, torch.full_like(overlap, force_to), overlap)

        gap = overlap.abs() < pos_min
        sign = torch.where(overlap >= 0.0, torch.ones_like(overlap), -torch.ones_like(overlap))
        overlap = torch.where(gap, sign * pos_min, overlap)
        overlap = torch.where(overlap >= 0.0, torch.clamp(overlap, pos_min, pos_max), torch.clamp(overlap, neg_min, neg_max))
        x[:, overlap_idx] = overlap

    def _enforce_overlap_angle_rule(self, x: torch.Tensor) -> None:
        overlap_idx = self.name_to_index.get("overlap")
        angle_idx = self.name_to_index.get("impact_angle")
        if overlap_idx is None or angle_idx is None:
            return

        rule = self.rules.get("overlap_angle", {})
        overlap = x[:, overlap_idx]
        angle = x[:, angle_idx]
        abs_min = float(rule.get("overlap_abs_min", 0.25))
        abs_max = float(rule.get("overlap_abs_max", 0.3))
        ang_abs_min = float(rule.get("angle_abs_min", 30.0))

        trigger = (overlap.abs() >= abs_min) & (overlap.abs() < abs_max)
        if not trigger.any():
            return

        angle_sampling = rule.get("angle_sampling", {})
        pos_segments = [s for s in angle_sampling.get("positive_overlap", []) if isinstance(s, (list, tuple)) and len(s) == 4]
        neg_segments = [s for s in angle_sampling.get("negative_overlap", []) if isinstance(s, (list, tuple)) and len(s) == 4]

        for seg in pos_segments:
            o_lo, o_hi, a_lo, a_hi = map(float, seg)
            m = trigger & (overlap >= o_lo) & (overlap <= o_hi)
            if m.any():
                clipped = torch.clamp(angle[m], a_lo, a_hi)
                angle[m] = clipped

        for seg in neg_segments:
            o_lo, o_hi, a_lo, a_hi = map(float, seg)
            m = trigger & (overlap >= o_lo) & (overlap <= o_hi)
            if m.any():
                clipped = torch.clamp(angle[m], a_lo, a_hi)
                angle[m] = clipped

        # 分段未覆盖区域的兜底：只保证异号且 |angle| >= 30
        fallback = trigger & (angle.abs() < ang_abs_min)
        if fallback.any():
            signs = torch.where(overlap[fallback] > 0.0, -torch.ones_like(overlap[fallback]), torch.ones_like(overlap[fallback]))
            angle[fallback] = signs * ang_abs_min

        x[:, angle_idx] = torch.clamp(angle, -45.0, 45.0)

    def _enforce_control_couplings(self, x: torch.Tensor) -> None:
        ll1_idx = self.name_to_index.get("LL1")
        ll2_idx = self.name_to_index.get("LL2")
        btf_idx = self.name_to_index.get("BTF")
        llattf_idx = self.name_to_index.get("LLATTF")
        aft_idx = self.name_to_index.get("AFT")

        if ll1_idx is not None and ll2_idx is not None:
            x[:, ll2_idx] = torch.min(x[:, ll2_idx], x[:, ll1_idx])
        if btf_idx is not None and llattf_idx is not None:
            x[:, llattf_idx] = torch.max(x[:, llattf_idx], x[:, btf_idx])
        if btf_idx is not None and aft_idx is not None:
            coupling = self.rules.get("coupling", {})
            delta = float(coupling.get("aft_btf_delta_max", 25.0))
            eps = float(coupling.get("epsilon", 1e-3))
            x[:, aft_idx] = torch.min(x[:, aft_idx], x[:, btf_idx] + delta - eps)

    def _enforce_ra_discrete(self, x: torch.Tensor) -> None:
        ra_idx = self.name_to_index.get("RA")
        side_idx = self.name_to_index.get("is_driver_side")
        ot_idx = self.name_to_index.get("OT")
        if ra_idx is None or side_idx is None or ot_idx is None or not self.ra_cache:
            return

        side = torch.round(x[:, side_idx]).to(torch.int64)
        ot = torch.round(x[:, ot_idx]).to(torch.int64)
        ra = x[:, ra_idx]
        ra_out = ra.clone()

        for (s, o), allowed_cpu in self.ra_cache.items():
            m = (side == s) & (ot == o)
            if not m.any():
                continue
            allowed = allowed_cpu.to(device=x.device)
            vals = ra[m]
            idx = (vals.unsqueeze(1) - allowed.unsqueeze(0)).abs().argmin(dim=1)
            ra_out[m] = allowed[idx]

        x[:, ra_idx] = ra_out

    def _enforce_seat_polygon(self, x: torch.Tensor) -> None:
        sp_idx = self.name_to_index.get("SP")
        sh_idx = self.name_to_index.get("SH")
        side_idx = self.name_to_index.get("is_driver_side")
        ot_idx = self.name_to_index.get("OT")
        if None in (sp_idx, sh_idx, side_idx, ot_idx) or not self.seat_cache:
            return

        side = torch.round(x[:, side_idx]).to(torch.int64)
        ot = torch.round(x[:, ot_idx]).to(torch.int64)

        for (s, o), info in self.seat_cache.items():
            m = (side == s) & (ot == o)
            if not m.any():
                continue

            idx = torch.nonzero(m, as_tuple=False).squeeze(1)
            sp = x[idx, sp_idx]
            sh = x[idx, sh_idx]
            poly = info["poly"]
            path = info["path"]
            sp_min, sp_max, sh_min, sh_max = info["bbox"]

            sp = torch.clamp(sp, sp_min, sp_max)
            sh = torch.clamp(sh, sh_min, sh_max)

            if sp_min == sp_max and sh_min == sh_max:
                x[idx, sp_idx] = sp_min
                x[idx, sh_idx] = sh_min
                continue
            if sh_min == sh_max:
                x[idx, sp_idx] = sp
                x[idx, sh_idx] = sh_min
                continue
            if sp_min == sp_max:
                x[idx, sp_idx] = sp_min
                x[idx, sh_idx] = sh
                continue

            pts = torch.stack([sp, sh], dim=1).detach().cpu().numpy()
            inside = path.contains_points(pts, radius=1e-9)
            out_local = np.where(~inside)[0]
            if out_local.size > 0:
                verts = torch.tensor(poly, dtype=torch.float32, device=x.device)
                out_t = torch.tensor(out_local, dtype=torch.long, device=x.device)
                out_pts = torch.stack([sp[out_t], sh[out_t]], dim=1)
                d = torch.cdist(out_pts.unsqueeze(0), verts.unsqueeze(0)).squeeze(0)
                nearest = d.argmin(dim=1)
                snapped = verts[nearest]
                sp[out_t] = snapped[:, 0]
                sh[out_t] = snapped[:, 1]

            x[idx, sp_idx] = sp
            x[idx, sh_idx] = sh
