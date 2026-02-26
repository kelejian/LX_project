import torch
import logging
from typing import Iterator

# 严格采用项目的全局绝对路径规范，避免 CLI 调用时的 ModuleNotFoundError
from ARS_optim.src.core.param_manager import ParamManager

# ==============================================================================
# 由于这些采样规则、约束等，比较繁琐，不方便结构化表达，因此集中在此处硬编码：物理状态分布规则 (从 step0_params_sample 提取的真实边缘概率密度)
# 避免新增配置文件造成过度工程化，通过原生 Python 结构保证可读性与维护性
# ==============================================================================

# 1. 连续型状态变量直方图密度 [区间下限, 区间上限, 相对密度]
# 注意：速率（impact_velocity）属于不可调的状态变量，训练集偶尔出现极少值<25kph；但是采样流仅覆盖工程可执行范围 25~65 kph，这与参数空间中的 min=23.5 无冲突，因为仅用作训练数据统计。下方断言和生成逻辑均以 hist 本身的区间为准。
VELOCITY_HIST = [
    [25.0, 30.0, 9.0], [30.0, 35.0, 11.0], [35.0, 40.0, 12.0], [40.0, 45.0, 13.5],
    [45.0, 50.0, 13.5], [50.0, 55.0, 14.0], [55.0, 60.0, 14.0], [60.0, 65.0, 13.0]
]

ANGLE_HIST = [
    [-45.0, -35.0, 1.5], [-35.0, -30.0, 2.0], [-30.0, -25.0, 2.5], [-25.0, -20.0, 3.0],
    [-20.0, -15.0, 4.0], [-15.0, -10.0, 5.0], [-10.0, -5.0, 8.0], [-5.0, 0.0, 23.0],
    [0.0, 5.0, 23.0], [5.0, 10.0, 8.0], [10.0, 15.0, 5.0], [15.0, 20.0, 4.0],
    [20.0, 25.0, 3.0], [25.0, 30.0, 2.5], [30.0, 35.0, 2.0], [35.0, 45.0, 1.5]
]

OVERLAP_HIST = [
    [-1.0, -0.9, 11.5], [-0.9, -0.8, 8.5], [-0.8, -0.7, 7.0], [-0.7, -0.6, 6.0],
    [-0.6, -0.5, 5.0], [-0.5, -0.4, 4.0], [-0.4, -0.3, 3.5], [-0.3, -0.25, 2.0],
    [0.25, 0.3, 2.0], [0.3, 0.4, 3.5], [0.4, 0.5, 4.5], [0.5, 0.6, 5.5],
    [0.6, 0.7, 6.5], [0.7, 0.8, 7.5], [0.8, 0.9, 9.0], [0.9, 1.0, 13.0]
]

# 2. 离散型状态变量概率分布
OT_PROBS = [0.3, 0.4, 0.3]        # 对应 5th (30%), 50th (40%), 95th (30%)
DRIVER_SIDE_PROBS = [0.5, 0.5]    # 对应 副驾 (50%), 主驾 (50%)

# 3. 几何耦合约束与离散特征映射
SEAT_CONSTRAINTS = {
    # 格式: (is_driver_side, OT) -> 多边形顶点列表 [(SP_x, SH_y), ...]
    (1, 1): [(40, 0), (110, 110/20), (110, 60), (80, 60), (80, 30), (40, 30)], 
    (1, 2): [(-40, -40/18), (60, 60/18), (60, 60), (-40, 60)], 
    (1, 3): [(-60, -60/18), (30, 30/18), (30, 60), (-60, 60)],  
    (0, 1): [(-110, -10), (110, -10), (110, 70), (-110, 70)],  
    (0, 2): [(-110, -10), (110, -10), (110, 70), (-110, 70)], 
    (0, 3): [(-110, -10), (49, -10), (49, 70), (-110, 70)],    
}

RA_VALUES = {
    (1, 1): [15, 20, 25], 
    (1, 2): [15, 20, 25, 30], 
    (1, 3): [15, 20, 25, 30],   
    (0, 1): [20, 25, 30, 35, 40],  
    (0, 2): [20, 25, 30, 35, 40], 
    (0, 3): [20, 25, 30, 35, 40],    
}

# ==============================================================================
# 向量化算子定义 (Vectorized Operations)
# ==============================================================================

def sample_piecewise_tensor(hist_data: list, size: int, device: torch.device, rng: torch.Generator = None) -> torch.Tensor:
    """基于纯张量的高效分段密度加权采样器: 首先根据每个区间的宽度和密度计算出其被选中的概率，然后利用 torch.multinomial 挑选出 size 个区间，最后在这些被选中的区间内部进行均匀采样，生成最终的 size 个样本。

    可选地接收一个 torch.Generator 用于所有随机数，确保在循环生成流程中的
    随机性完全来自同一个 RNG，以便批次之间可复现。
    """
    bins = torch.tensor(hist_data, dtype=torch.float32, device=device)
    lows = bins[:, 0]
    highs = bins[:, 1]
    densities = bins[:, 2]

    # 计算每个区间的概率权重 = 宽度 * 密度
    areas = (highs - lows) * densities
    probs = areas / areas.sum()

    # 依据概率选取区间索引，并执行区间内均匀采样
    bin_indices = torch.multinomial(probs, size, replacement=True, generator=rng) if rng is not None else torch.multinomial(probs, size, replacement=True)
    chosen_lows = lows[bin_indices]
    chosen_highs = highs[bin_indices]

    if rng is not None:
        u = torch.rand(size, device=device, dtype=torch.float32, generator=rng)
    else:
        u = torch.rand(size, device=device, dtype=torch.float32)
    # [Batch] -> [Batch]
    return chosen_lows + u * (chosen_highs - chosen_lows)

def point_in_polygon_torch(x: torch.Tensor, y: torch.Tensor, poly_pts: torch.Tensor) -> torch.Tensor:
    """基于纯 PyTorch 张量操作的射线法 (Ray-Casting) 多边形内外判定"""
    num_pts = poly_pts.shape[0]
    inside = torch.zeros_like(x, dtype=torch.bool)
    p1x, p1y = poly_pts[0]
    for i in range(1, num_pts + 1):
        p2x, p2y = poly_pts[i % num_pts]
        # 判断射线是否穿过边，严格避免 0 作除数
        intersect = ((p1y > y) != (p2y > y)) & (x < (p2x - p1x) * (y - p1y) / (p2y - p1y + 1e-8) + p1x)
        inside ^= intersect
        p1x, p1y = p2x, p2y
    return inside

# ==============================================================================
# 核心类：状态数据加载与生成器
# ==============================================================================

class StateDataLoaderManager:
    """
    状态数据无限生成器 (Infinite State Stream Generator) - 高性能向量化版
    在目标设备(如 CUDA)内存中直接以 O(1) 循环复杂度生成满足多维非线性耦合约束的高纯度数据流。

    新增种子支持以保证可复现：
    - 所有随机调用均通过内部 torch.Generator 完成，且默认使用构造时传入的 seed。
    - 如果 device 为 CUDA，则生成器也设置在相同设备上，避免 host<->device 数据传输。
    """
    def __init__(self, param_manager: ParamManager, batch_size: int, device: torch.device = torch.device('cpu'), seed: int = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.param_manager = param_manager
        self.batch_size = batch_size
        self.device = device
        self.seed = seed
        # 随机数生成器，用于所有随机采样
        self.rng = torch.Generator(device=device)
        if seed is not None:
            self.rng.manual_seed(seed)
        
        self.total_dim = self.param_manager.get_total_feature_dim()
        self.state_indices = self.param_manager.get_state_indices()
        
        # 建立按特征名称检索 index 的映射表，确保特征在物理张量中绝对位置的正确性
        self.idx_map = {p['name']: p['index'] for p in self.param_manager.state_params}

        # sanity check: required state names must have sampling definitions
        required = {'impact_velocity', 'impact_angle', 'overlap'}
        missing = required - set(self.idx_map.keys())
        if missing:
            raise ValueError(f"StateDataLoaderManager 初始化失败，缺少必填状态参数: {missing}")

        # verify sampling hist ranges lie within param_space bounds
        # (param_space 好比训练数据、安全边界，但采样流对 speed 进行了上/下截断)
        # param_space 可通过 ParamManager 查询
        # 下面检查若 hist 超出将给出警告
        bounds = {p['name']:(p.get('min'), p.get('max')) for p in param_manager.state_params if p.get('type')=='continuous'}
        # compute hist-derived bounds
        hist_bounds = {
            'impact_velocity': (min(bin[0] for bin in VELOCITY_HIST), max(bin[1] for bin in VELOCITY_HIST)),
            'impact_angle': (min(bin[0] for bin in ANGLE_HIST), max(bin[1] for bin in ANGLE_HIST)),
            'overlap': (min(bin[0] for bin in OVERLAP_HIST), max(bin[1] for bin in OVERLAP_HIST)),
        }
        for name, (hmin, hmax) in hist_bounds.items():
            if name in bounds:
                pmin, pmax = bounds[name]
                if hmin < pmin - 1e-6 or hmax > pmax + 1e-6:
                    self.logger.warning(f"采样范围 {name} [{hmin},{hmax}] 超出 param_space 范围 [{pmin},{pmax}]。")
        
    def _apply_overlap_angle_rejection(self, ov: torch.Tensor, ang: torch.Tensor) -> torch.Tensor:
        """
        对 overlap 与 angle 施加临界区段排斥规则，等价于 step0_params_sample_1220 中的循环拒绝采样。
        输入均为 shape [Batch] 的张量，返回修改后的 angle。
        同时扁平化 RNG 使用以保证可复现。
        """
        mask_coup = (ov.abs() >= 0.25) & (ov.abs() < 0.3)
        num_coup = mask_coup.sum().item()
        if num_coup == 0:
            return ang

        coup_ov = ov[mask_coup]
        u = torch.rand(num_coup, device=self.device, generator=self.rng)
        new_ang = torch.zeros(num_coup, device=self.device)

        # 参照 step0_params_sample 中的区间划分，每个区间生成满足异号且|angle|>30的值。
        cond_p_1 = (coup_ov > 0) & (coup_ov <= 0.26)
        cond_p_2 = (coup_ov > 0.26) & (coup_ov <= 0.28)
        cond_p_3 = (coup_ov > 0.28)
        new_ang = torch.where(cond_p_1, -40.0 - u * 5.0, new_ang)
        new_ang = torch.where(cond_p_2, -35.0 - u * 10.0, new_ang)
        new_ang = torch.where(cond_p_3, -30.0 - u * 15.0, new_ang)

        cond_n_1 = (coup_ov < 0) & (coup_ov >= -0.26)
        cond_n_2 = (coup_ov < -0.26) & (coup_ov >= -0.28)
        cond_n_3 = (coup_ov < -0.28)
        new_ang = torch.where(cond_n_1, 40.0 + u * 5.0, new_ang)
        new_ang = torch.where(cond_n_2, 35.0 + u * 10.0, new_ang)
        new_ang = torch.where(cond_n_3, 30.0 + u * 15.0, new_ang)

        ang[mask_coup] = new_ang
        return ang

    def _generate_batch(self) -> torch.Tensor:
        """核心生成逻辑：在计算图外极速构建数据环境"""
        B = self.batch_size
        D = self.total_dim
        # 初始化完整维度的空白物理张量 (部分不属于 State 的特征位将空置)
        batch_phys = torch.zeros((B, D), device=self.device, dtype=torch.float32)
        
        # ---------------------------------------------------------
        # 1. 连续型工况特征分段密度采样
        #    使用内部 rng 以保证可复现
        # ---------------------------------------------------------
        # 使用同一个 RNG 以保证整批数据中的随机过程可复现
        vel = sample_piecewise_tensor(VELOCITY_HIST, B, self.device, rng=self.rng)
        ang = sample_piecewise_tensor(ANGLE_HIST, B, self.device, rng=self.rng)
        ov  = sample_piecewise_tensor(OVERLAP_HIST, B, self.device, rng=self.rng)
        
        # 极值保护：依据 step0 逻辑，极其贴近边界的值被收敛至纯粹的满载碰撞 (1.0)
        ov = torch.where(ov.abs() > 0.99, torch.sign(ov) * 1.0, ov)
        
        # ---------------------------------------------------------
        # 2. 重叠率临界区间内的角度排斥规则（分离为独立函数以便单元测试）
        # ---------------------------------------------------------
        ang = self._apply_overlap_angle_rejection(ov, ang)

        batch_phys[:, self.idx_map['impact_velocity']] = vel
        batch_phys[:, self.idx_map['impact_angle']] = ang
        batch_phys[:, self.idx_map['overlap']] = ov
        
        # ---------------------------------------------------------
        # 3. 离散类别特征采样
        #    使用内部 rng 保证可复现
        # ---------------------------------------------------------
        ot = torch.multinomial(torch.tensor(OT_PROBS, device=self.device), B, replacement=True, generator=self.rng) + 1.0 # 0,1,2 -> 1,2,3
        is_driver = torch.multinomial(torch.tensor(DRIVER_SIDE_PROBS, device=self.device), B, replacement=True, generator=self.rng).float()
        
        batch_phys[:, self.idx_map['OT']] = ot
        batch_phys[:, self.idx_map['is_driver_side']] = is_driver
        
        # ---------------------------------------------------------
        # 4. 基于乘员与侧位的空间几何约束向量化采样 (SP, SH, RA)
        # ---------------------------------------------------------
        ra_tensor = torch.zeros(B, device=self.device, dtype=torch.float32)
        sp_tensor = torch.zeros(B, device=self.device, dtype=torch.float32)
        sh_tensor = torch.zeros(B, device=self.device, dtype=torch.float32)
        
        # 遍历 6 种状态组合，进行并行分块约束映射 (O(1) 复杂度，常数 6)
        for drv_val in [0, 1]:
            for ot_val in [1, 2, 3]:
                mask = (is_driver == drv_val) & (ot == ot_val)
                num_in_group = mask.sum().item()
                if num_in_group == 0:
                    continue
                
                # A. 靠背角 RA 映射 (使用 rng 选索引)
                ra_opts = torch.tensor(RA_VALUES[(drv_val, ot_val)], device=self.device, dtype=torch.float32)
                ra_idx = torch.randint(0, len(ra_opts), (num_in_group,), device=self.device, generator=self.rng)
                ra_tensor[mask] = ra_opts[ra_idx]
                
                # B. 座椅位置 SP/SH 向量化拒绝采样 (Rejection Sampling within Polygon)
                poly_pts = torch.tensor(SEAT_CONSTRAINTS[(drv_val, ot_val)], device=self.device, dtype=torch.float32)
                sp_min, sh_min = poly_pts.min(dim=0)[0]
                sp_max, sh_max = poly_pts.max(dim=0)[0]
                
                sub_mask_valid = torch.zeros(num_in_group, dtype=torch.bool, device=self.device)
                sub_sp = torch.zeros_like(sub_mask_valid, dtype=torch.float32)
                sub_sh = torch.zeros_like(sub_mask_valid, dtype=torch.float32)
                
                # GPU 内极速迭代，直至少数越界点全部被重新抓回多边形内部
                while not sub_mask_valid.all():
                    num_invalid = (~sub_mask_valid).sum().item()
                    gen_sp = torch.rand(num_invalid, device=self.device, generator=self.rng) * (sp_max - sp_min) + sp_min
                    gen_sh = torch.rand(num_invalid, device=self.device, generator=self.rng) * (sh_max - sh_min) + sh_min
                    
                    inside = point_in_polygon_torch(gen_sp, gen_sh, poly_pts)
                    
                    sub_sp[~sub_mask_valid] = torch.where(inside, gen_sp, sub_sp[~sub_mask_valid])
                    sub_sh[~sub_mask_valid] = torch.where(inside, gen_sh, sub_sh[~sub_mask_valid])
                    sub_mask_valid[~sub_mask_valid] = inside
                
                sp_tensor[mask] = sub_sp
                sh_tensor[mask] = sub_sh
                
        batch_phys[:, self.idx_map['RA']] = ra_tensor
        batch_phys[:, self.idx_map['SP']] = sp_tensor
        batch_phys[:, self.idx_map['SH']] = sh_tensor
        
        # 断言：生成值不越出历史直方图边界（以 step0_params_sample 的上下限为准）
        vel = batch_phys[:, self.idx_map['impact_velocity']]
        ang = batch_phys[:, self.idx_map['impact_angle']]
        ov  = batch_phys[:, self.idx_map['overlap']]
        # bounds derived from histogram definitions rather than hardcoded constants
        vel_min = min(bin[0] for bin in VELOCITY_HIST)
        vel_max = max(bin[1] for bin in VELOCITY_HIST)
        ang_min = min(bin[0] for bin in ANGLE_HIST)
        ang_max = max(bin[1] for bin in ANGLE_HIST)
        ov_min  = min(bin[0] for bin in OVERLAP_HIST)
        ov_max  = max(bin[1] for bin in OVERLAP_HIST)
        assert vel.min() >= vel_min and vel.max() <= vel_max, f"velocity超出定义范围 [{vel_min},{vel_max}]"
        assert ang.min() >= ang_min and ang.max() <= ang_max, f"angle超出定义范围 [{ang_min},{ang_max}]"
        assert ov.min() >= ov_min and ov.max() <= ov_max, f"overlap超出定义范围 [{ov_min},{ov_max}]"
        
        # 裁剪并返回纯粹的状态参数矩阵
        # [Batch, Total_Dim] -> [Batch, D_State]
        return batch_phys[:, self.state_indices]

    def get_infinite_generator(self) -> Iterator[torch.Tensor]:
        """产出无限的自监督物理工况流"""
        while True:
            yield self._generate_batch()