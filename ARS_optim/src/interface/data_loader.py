import logging
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch
from matplotlib.path import Path as MplPath

from common.settings import FEATURE_ORDER, RAW_DATA_DIR, SPLIT_INDICES_DIR
from ARS_optim.src.core.param_manager import ParamManager


class StateDataLoaderManager:
    """
    经验池驱动的数据流采样器。

    核心逻辑：
    - 从损伤预测训练数据经验池中按行抽样，天然复刻边际分布。
    - 输出上下文参数（state + trainable=False 的 control），作为策略输入与局部优化锚点。
    - 对连续上下文字段施加有界微扰，并强制裁剪到 param_space 定义范围内。
    - 额外保留 overlap-angle 的硬规则修复，避免微扰后违例。

        输入/输出示例：
        - 经验池完整特征: [N, 13]
        - 输出 batch_context: [Batch, D_context]
            例如当前参数空间下，D_context 可为 10（state + 固定 control）。
    """

    def __init__(
        self,
        param_manager: ParamManager,
        batch_size: int,
        device: torch.device = torch.device('cpu'),
        seed: Optional[int] = None,
        pool_npz_path: Optional[str] = None,
        train_indices_path: Optional[str] = None,
        jitter_ratio: float = 0.01,
        jitter_prob: float = 1.0,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.param_manager = param_manager
        self.batch_size = int(batch_size)
        self.device = device
        self.seed = seed
        self.jitter_ratio = float(jitter_ratio)
        self.jitter_prob = float(jitter_prob)

        self.rng = torch.Generator(device=device)
        if seed is not None:
            self.rng.manual_seed(int(seed))

        self.context_params = self.param_manager.get_context_params()
        self.context_indices = self.param_manager.get_context_indices()
        self.context_names = self.param_manager.get_context_names()
        self.rules = self.param_manager.get_sampling_rules() if hasattr(self.param_manager, 'get_sampling_rules') else {}

        pool_path = Path(pool_npz_path) if pool_npz_path else (RAW_DATA_DIR / 'raw_data_packed.npz')
        split_path = Path(train_indices_path) if train_indices_path else (SPLIT_INDICES_DIR / 'injury_train_indices.npy')

        full_features = self._load_feature_matrix_from_pool(pool_path)
        if split_path.exists():
            train_idx = np.load(split_path)
            full_features = full_features[train_idx]
            self.logger.info(f"经验池使用训练切分索引: {split_path}，样本数={full_features.shape[0]}")
        else:
            self.logger.warning(f"未找到训练切分索引: {split_path}，将使用全量经验池")

        if full_features.shape[0] == 0:
            raise ValueError("经验池为空，无法构建数据流")

        # 仅保留上下文列：
        # [N, D_total] -> [N, D_context]
        self.pool_context = torch.tensor(full_features[:, self.context_indices], dtype=torch.float32, device=device)
        self.pool_size = self.pool_context.size(0)

        # 连续型上下文字段局部索引与边界（用于有界扰动）
        self.context_cont_local_indices = [i for i, p in enumerate(self.context_params) if p.get('type') == 'continuous']
        if self.context_cont_local_indices:
            mins = [self.context_params[i]['min'] for i in self.context_cont_local_indices]
            maxs = [self.context_params[i]['max'] for i in self.context_cont_local_indices]
            self.cont_mins = torch.tensor(mins, dtype=torch.float32, device=device)
            self.cont_maxs = torch.tensor(maxs, dtype=torch.float32, device=device)
            self.cont_spans = torch.clamp(self.cont_maxs - self.cont_mins, min=1e-6)
        else:
            self.cont_mins = self.cont_maxs = self.cont_spans = None

        # overlap-angle 规则相关列（在上下文局部坐标）
        self.local_overlap = self.context_names.index('overlap') if 'overlap' in self.context_names else None
        self.local_angle = self.context_names.index('impact_angle') if 'impact_angle' in self.context_names else None
        self.local_is_driver_side = self.context_names.index('is_driver_side') if 'is_driver_side' in self.context_names else None
        self.local_ot = self.context_names.index('OT') if 'OT' in self.context_names else None
        self.local_sp = self.context_names.index('SP') if 'SP' in self.context_names else None
        self.local_sh = self.context_names.index('SH') if 'SH' in self.context_names else None
        self.local_ra = self.context_names.index('RA') if 'RA' in self.context_names else None
        self.local_ll1 = self.context_names.index('LL1') if 'LL1' in self.context_names else None
        self.local_ll2 = self.context_names.index('LL2') if 'LL2' in self.context_names else None
        self.local_btf = self.context_names.index('BTF') if 'BTF' in self.context_names else None
        self.local_llattf = self.context_names.index('LLATTF') if 'LLATTF' in self.context_names else None
        self.local_aft = self.context_names.index('AFT') if 'AFT' in self.context_names else None

        self.overlap_rule = self.rules.get('overlap', {}) if isinstance(self.rules, dict) else {}
        self.overlap_angle_rule = self.rules.get('overlap_angle', {}) if isinstance(self.rules, dict) else {}
        self.seat_rules = self.rules.get('seat_constraints', {}) if isinstance(self.rules, dict) else {}
        self.ra_rules = self.rules.get('ra_values', {}) if isinstance(self.rules, dict) else {}
        self.coupling_rules = self.rules.get('coupling', {}) if isinstance(self.rules, dict) else {}

        # overlap-angle 分段规则缓存（与 step0 的拒绝采样区间保持一致）
        angle_sampling = self.overlap_angle_rule.get('angle_sampling', {}) if isinstance(self.overlap_angle_rule, dict) else {}
        self.angle_sampling_positive = [
            (float(x[0]), float(x[1]), float(x[2]), float(x[3]))
            for x in angle_sampling.get('positive_overlap', [])
            if isinstance(x, (list, tuple)) and len(x) == 4
        ]
        self.angle_sampling_negative = [
            (float(x[0]), float(x[1]), float(x[2]), float(x[3]))
            for x in angle_sampling.get('negative_overlap', [])
            if isinstance(x, (list, tuple)) and len(x) == 4
        ]

        # 座椅多边形缓存：把规则键 "1_2" 解析为 (1, 2)，并预构建 Path 与包围盒
        self._seat_cache = {}
        if isinstance(self.seat_rules, dict):
            for key, pts in self.seat_rules.items():
                try:
                    side, ot = key.split('_')
                    side_i, ot_i = int(side), int(ot)
                except Exception:
                    continue
                poly_np = np.asarray(pts, dtype=np.float32)
                if poly_np.ndim != 2 or poly_np.shape[1] != 2:
                    continue
                sp_min, sh_min = np.min(poly_np, axis=0)
                sp_max, sh_max = np.max(poly_np, axis=0)
                self._seat_cache[(side_i, ot_i)] = {
                    'poly': poly_np,
                    'path': MplPath(poly_np),
                    'bbox': (float(sp_min), float(sp_max), float(sh_min), float(sh_max))
                }

        # RA 离散档位缓存（同样把键转成 tuple，后续按组批量映射）
        self._ra_cache = {}
        if isinstance(self.ra_rules, dict):
            for key, vals in self.ra_rules.items():
                try:
                    side, ot = key.split('_')
                    side_i, ot_i = int(side), int(ot)
                except Exception:
                    continue
                vals_np = np.asarray(vals, dtype=np.float32).reshape(-1)
                if vals_np.size == 0:
                    continue
                self._ra_cache[(side_i, ot_i)] = vals_np

    @staticmethod
    def _rule_key(is_driver_side: int, ot: int) -> str:
        return f"{int(is_driver_side)}_{int(ot)}"

    def _load_feature_matrix_from_pool(self, pool_path: Path) -> np.ndarray:
        if not pool_path.exists():
            raise FileNotFoundError(f"经验池文件不存在: {pool_path}")
        with np.load(pool_path, allow_pickle=True) as data:
            key_candidates = ['x_att_raw', 'att_raw', 'x_att', 'features']
            arr = None
            for key in key_candidates:
                if key in data.files:
                    arr = data[key]
                    break
            if arr is None:
                # 回退：按 FEATURE_ORDER 尝试逐列拼接
                cols = []
                for name in FEATURE_ORDER:
                    if name not in data.files:
                        raise KeyError(f"经验池缺少特征列: {name}")
                    cols.append(np.asarray(data[name]).reshape(-1, 1))
                arr = np.concatenate(cols, axis=1)

        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"经验池特征矩阵维度异常: {arr.shape}")
        if arr.shape[1] < len(FEATURE_ORDER):
            raise ValueError(f"经验池特征维度不足: {arr.shape[1]} < {len(FEATURE_ORDER)}")
        return arr[:, :len(FEATURE_ORDER)]

    def _apply_bounded_jitter(self, batch_context: torch.Tensor) -> torch.Tensor:
        """对连续上下文字段施加有界扰动。

        说明：
        - 扰动标准差 = 参数跨度 * jitter_ratio；
        - 每一维是否扰动由 jitter_prob 控制；
        - 扰动后立即 clamp 到 [min, max]。
        """
        if not self.context_cont_local_indices or self.jitter_ratio <= 0:
            return batch_context

        cont = batch_context[:, self.context_cont_local_indices]
        noise = torch.randn(cont.shape, generator=self.rng, device=self.device, dtype=torch.float32)
        noise = noise * (self.cont_spans.unsqueeze(0) * self.jitter_ratio)

        if self.jitter_prob < 1.0:
            mask = torch.rand(cont.shape, generator=self.rng, device=self.device) < self.jitter_prob
            noise = noise * mask.float()

        cont = cont + noise
        cont = torch.clamp(cont, self.cont_mins.unsqueeze(0), self.cont_maxs.unsqueeze(0))
        batch_context[:, self.context_cont_local_indices] = cont
        return batch_context

    def _enforce_overlap_angle_rule(self, batch_context: torch.Tensor) -> torch.Tensor:
        """修复 overlap-angle 联合规则（与 step0 对齐）。

        例子：
        - 当 overlap=0.255 时，angle 需落在 [-45, -40]；
        - 当 overlap=-0.27 时，angle 需落在 [35, 45]。
        """
        if self.local_overlap is None or self.local_angle is None:
            return batch_context

        overlap = batch_context[:, self.local_overlap]
        angle = batch_context[:, self.local_angle]
        overlap_abs_min = float(self.overlap_angle_rule.get('overlap_abs_min', 0.25))
        overlap_abs_max = float(self.overlap_angle_rule.get('overlap_abs_max', 0.3))
        angle_abs_min = float(self.overlap_angle_rule.get('angle_abs_min', 30.0))

        # 仅对 |overlap| ∈ [0.25, 0.3) 的样本执行修复
        trigger_mask = (overlap.abs() >= overlap_abs_min) & (overlap.abs() < overlap_abs_max)
        if not trigger_mask.any():
            return batch_context

        # 第一步：先做通用判定（异号且 |angle|>30）
        sign_valid = torch.sign(angle) != torch.sign(overlap)
        mag_valid = angle.abs() > angle_abs_min
        valid = sign_valid & mag_valid

        # 第二步：按子区间做更严格判定
        if self.angle_sampling_positive:
            pos = overlap > 0
            for o_lo, o_hi, a_lo, a_hi in self.angle_sampling_positive:
                seg = trigger_mask & pos & (overlap >= o_lo) & (overlap <= o_hi)
                if seg.any():
                    seg_valid = (angle >= a_lo) & (angle <= a_hi)
                    valid = torch.where(seg, valid & seg_valid, valid)

        if self.angle_sampling_negative:
            neg = overlap < 0
            for o_lo, o_hi, a_lo, a_hi in self.angle_sampling_negative:
                seg = trigger_mask & neg & (overlap >= o_lo) & (overlap <= o_hi)
                if seg.any():
                    seg_valid = (angle >= a_lo) & (angle <= a_hi)
                    valid = torch.where(seg, valid & seg_valid, valid)

        bad = trigger_mask & (~valid)
        if not bad.any():
            return batch_context

        repaired = torch.zeros_like(bad)

        # 命中分段规则的样本：直接在对应区间均匀重采样
        if self.angle_sampling_positive:
            pos = overlap > 0
            for o_lo, o_hi, a_lo, a_hi in self.angle_sampling_positive:
                seg_bad = bad & pos & (overlap >= o_lo) & (overlap <= o_hi)
                n = int(seg_bad.sum().item())
                if n > 0:
                    sampled = torch.empty(n, device=self.device, dtype=torch.float32).uniform_(a_lo, a_hi, generator=self.rng)
                    batch_context[seg_bad, self.local_angle] = sampled
                    repaired = repaired | seg_bad

        if self.angle_sampling_negative:
            neg = overlap < 0
            for o_lo, o_hi, a_lo, a_hi in self.angle_sampling_negative:
                seg_bad = bad & neg & (overlap >= o_lo) & (overlap <= o_hi)
                n = int(seg_bad.sum().item())
                if n > 0:
                    sampled = torch.empty(n, device=self.device, dtype=torch.float32).uniform_(a_lo, a_hi, generator=self.rng)
                    batch_context[seg_bad, self.local_angle] = sampled
                    repaired = repaired | seg_bad

        # 兜底：若规则表缺失某段，使用通用合法区间重采
        remain = bad & (~repaired)
        if remain.any():
            remain_pos = remain & (overlap > 0)
            remain_neg = remain & (overlap < 0)
            n_pos = int(remain_pos.sum().item())
            n_neg = int(remain_neg.sum().item())
            if n_pos > 0:
                batch_context[remain_pos, self.local_angle] = torch.empty(n_pos, device=self.device, dtype=torch.float32).uniform_(-45.0, -30.0, generator=self.rng)
            if n_neg > 0:
                batch_context[remain_neg, self.local_angle] = torch.empty(n_neg, device=self.device, dtype=torch.float32).uniform_(30.0, 45.0, generator=self.rng)

        return batch_context

    def _enforce_overlap_domain(self, batch_context: torch.Tensor) -> torch.Tensor:
        """
        强制 overlap 满足硬域约束：(-1, -0.25] ∪ [0.25, 1]，并复现 step0 的特殊值规则：
        - abs(overlap) > 0.99 或 abs(overlap) < 0.02 时置为 1.0
        """
        if self.local_overlap is None:
            return batch_context

        overlap = batch_context[:, self.local_overlap]

        overlap_domain = self.overlap_rule.get('domain', {}) if isinstance(self.overlap_rule, dict) else {}
        neg_range = overlap_domain.get('negative', [-1.0, -0.25])
        pos_range = overlap_domain.get('positive', [0.25, 1.0])

        neg_min, neg_max = float(neg_range[0]), float(neg_range[1])
        pos_min, pos_max = float(pos_range[0]), float(pos_range[1])
        special_abs_high = float(self.overlap_rule.get('special_abs_high', 0.99))
        special_abs_low = float(self.overlap_rule.get('special_abs_low', 0.02))
        force_to = float(self.overlap_rule.get('force_to', 1.0))

        # 特殊值规则（与 step0 对齐）
        special_mask = (overlap.abs() > special_abs_high) | (overlap.abs() < special_abs_low)
        overlap = torch.where(special_mask, torch.full_like(overlap, force_to), overlap)

        # 落入禁区 |overlap| < 0.25 的样本投影到边界 ±0.25
        gap_mask = overlap.abs() < pos_min
        sign = torch.where(overlap >= 0, torch.ones_like(overlap), -torch.ones_like(overlap))
        overlap = torch.where(gap_mask, sign * pos_min, overlap)

        # 分段夹紧到合法并集区间
        pos = overlap >= 0
        overlap = torch.where(pos, torch.clamp(overlap, pos_min, pos_max), torch.clamp(overlap, neg_min, neg_max))

        batch_context[:, self.local_overlap] = overlap
        return batch_context

    def _repair_control_constraints(self, batch_context: torch.Tensor) -> torch.Tensor:
        """修复 jitter 后的控制参数耦合关系。

        当前处理三条关系：
        1) LL2 <= LL1
        2) LLATTF >= BTF
        3) AFT <= BTF + delta - epsilon
        """
        # LL2 <= LL1
        if self.local_ll1 is not None and self.local_ll2 is not None:
            ll1 = batch_context[:, self.local_ll1]
            ll2 = batch_context[:, self.local_ll2]
            batch_context[:, self.local_ll2] = torch.min(ll2, ll1)

        # LLATTF >= BTF
        if self.local_btf is not None and self.local_llattf is not None:
            btf = batch_context[:, self.local_btf]
            llattf = batch_context[:, self.local_llattf]
            batch_context[:, self.local_llattf] = torch.max(llattf, btf)

        # AFT < BTF + delta
        if self.local_btf is not None and self.local_aft is not None:
            aft = batch_context[:, self.local_aft]
            btf = batch_context[:, self.local_btf]
            aft_btf_delta_max = float(self.coupling_rules.get('aft_btf_delta_max', 25.0))
            epsilon = float(self.coupling_rules.get('epsilon', 0.001))
            batch_context[:, self.local_aft] = torch.min(aft, btf + aft_btf_delta_max - epsilon)

        return batch_context

    def _repair_seat_and_ra_constraints(self, batch_context: torch.Tensor) -> torch.Tensor:
        """按 (is_driver_side, OT) 修复 SP/SH 与 RA。

        示例：
        - 主驾 50th 可映射到规则键 (1, 2)；
        - 副驾 95th 可映射到规则键 (0, 3)。
        """
        if self.local_is_driver_side is None or self.local_ot is None:
            return batch_context

        # 先把规则键离散化为 int，后续按组批量修复，减少逐样本 Python 循环
        is_driver_side = torch.round(batch_context[:, self.local_is_driver_side]).to(torch.int64)
        ot_vals = torch.round(batch_context[:, self.local_ot]).to(torch.int64)

        # RA 修复（按组批量映射到最近离散档位）
        if self.local_ra is not None and self._ra_cache:
            for (side, ot), options_np in self._ra_cache.items():
                group_mask = (is_driver_side == side) & (ot_vals == ot)
                if not group_mask.any():
                    continue
                options = torch.tensor(options_np, device=self.device, dtype=torch.float32)
                vals = batch_context[group_mask, self.local_ra]  # [N_group]
                # [N_group, 1] 与 [N_option] 广播，取最近档位
                nearest_idx = (vals.unsqueeze(1) - options.unsqueeze(0)).abs().argmin(dim=1)
                batch_context[group_mask, self.local_ra] = options[nearest_idx]

        # SP/SH 修复（按组批量进行点-in-多边形判定）
        if self.local_sp is not None and self.local_sh is not None and self._seat_cache:
            for (side, ot), info in self._seat_cache.items():
                group_mask = (is_driver_side == side) & (ot_vals == ot)
                if not group_mask.any():
                    continue

                idx = torch.nonzero(group_mask, as_tuple=False).squeeze(1)
                sp = batch_context[idx, self.local_sp]
                sh = batch_context[idx, self.local_sh]

                poly_np = info['poly']
                poly_path = info['path']
                sp_min, sp_max, sh_min, sh_max = info['bbox']

                # 退化多边形分支（线段/点）：与 step0 规则一致
                if sp_min == sp_max and sh_min == sh_max:
                    batch_context[idx, self.local_sp] = sp_min
                    batch_context[idx, self.local_sh] = sh_min
                    continue
                if sh_min == sh_max:
                    batch_context[idx, self.local_sp] = torch.clamp(sp, sp_min, sp_max)
                    batch_context[idx, self.local_sh] = sh_min
                    continue
                if sp_min == sp_max:
                    batch_context[idx, self.local_sp] = sp_min
                    batch_context[idx, self.local_sh] = torch.clamp(sh, sh_min, sh_max)
                    continue

                # 批量判定当前组是否落在多边形内
                pts = torch.stack([sp, sh], dim=1).detach().cpu().numpy()
                inside = poly_path.contains_points(pts, radius=1e-9)
                bad_local_np = np.where(~inside)[0]
                if bad_local_np.size == 0:
                    continue

                bad_local = torch.tensor(bad_local_np, device=self.device, dtype=torch.long)
                bad_idx = idx[bad_local]

                # 批量拒绝采样：每轮仅对未修复样本重采，最多 32 轮
                for _ in range(32):
                    if bad_idx.numel() == 0:
                        break
                    n_bad = bad_idx.numel()
                    sp_try = torch.empty(n_bad, device=self.device, dtype=torch.float32).uniform_(sp_min, sp_max, generator=self.rng)
                    sh_try = torch.empty(n_bad, device=self.device, dtype=torch.float32).uniform_(sh_min, sh_max, generator=self.rng)

                    trial_pts = torch.stack([sp_try, sh_try], dim=1).detach().cpu().numpy()
                    ok_np = poly_path.contains_points(trial_pts, radius=1e-9)
                    if not np.any(ok_np):
                        continue

                    ok = torch.tensor(ok_np, device=self.device, dtype=torch.bool)
                    ok_idx = bad_idx[ok]
                    batch_context[ok_idx, self.local_sp] = sp_try[ok]
                    batch_context[ok_idx, self.local_sh] = sh_try[ok]
                    bad_idx = bad_idx[~ok]

                # 兜底：仍未修复的样本回退到多边形首个顶点
                if bad_idx.numel() > 0:
                    batch_context[bad_idx, self.local_sp] = float(poly_np[0, 0])
                    batch_context[bad_idx, self.local_sh] = float(poly_np[0, 1])

        return batch_context

    def _generate_batch(self) -> torch.Tensor:
        # [Batch] 索引 -> [Batch, D_context] 样本 -> 依次执行规则修复
        idx = torch.randint(0, self.pool_size, (self.batch_size,), generator=self.rng, device=self.device)
        batch = self.pool_context[idx].clone()
        batch = self._apply_bounded_jitter(batch)
        batch = self._enforce_overlap_domain(batch)
        batch = self._enforce_overlap_angle_rule(batch)
        batch = self._repair_control_constraints(batch)
        batch = self._repair_seat_and_ra_constraints(batch)
        return batch

    def get_infinite_generator(self) -> Iterator[torch.Tensor]:
        """返回无限数据流生成器，供训练循环持续调用 next()。"""
        while True:
            yield self._generate_batch()

    def get_dataset_tensor(self) -> torch.Tensor:
        """返回当前 split 对应的完整上下文张量。

        返回:
            [N_split, D_context]，不做扰动与打乱。
        """
        return self.pool_context

    def iter_dataset_batches(self, batch_size: Optional[int] = None, shuffle: bool = False) -> Iterator[torch.Tensor]:
        """按批遍历当前 split 的完整数据集。

        典型用途：
        - 训练中做验证集全量评估（shuffle=False）
        - 离线统计（可选 shuffle=True）
        """
        bs = int(batch_size) if batch_size is not None else int(self.batch_size)
        if bs <= 0:
            raise ValueError("batch_size 必须为正整数")

        n = self.pool_context.size(0)
        if n == 0:
            return

        if shuffle:
            order = torch.randperm(n, generator=self.rng, device=self.device)
        else:
            order = torch.arange(n, device=self.device)

        for start in range(0, n, bs):
            idx = order[start:start + bs]
            yield self.pool_context[idx]