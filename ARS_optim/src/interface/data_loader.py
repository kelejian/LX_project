import torch
import logging
from typing import Iterator

from src.core.param_manager import ParamManager

# 严格复刻 step0_params_sample_1220.py 中的多边形顶点约束与离散特征映射
SEAT_CONSTRAINTS = {
    (1, 1): [(40, 0), (110, 5.5), (110, 60), (80, 60), (80, 30), (40, 30)], 
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

def point_in_polygon_torch(x: torch.Tensor, y: torch.Tensor, poly_pts: torch.Tensor) -> torch.Tensor:
    """基于纯 PyTorch 张量操作的射线法 (Ray-Casting) 多边形内外判定"""
    num_pts = poly_pts.shape[0]
    inside = torch.zeros_like(x, dtype=torch.bool)
    p1x, p1y = poly_pts[0]
    for i in range(1, num_pts + 1):
        p2x, p2y = poly_pts[i % num_pts]
        # 判断射线是否穿过边
        intersect = ((p1y > y) != (p2y > y)) & (x < (p2x - p1x) * (y - p1y) / (p2y - p1y + 1e-8) + p1x)
        inside ^= intersect
        p1x, p1y = p2x, p2y
    return inside

class StateDataLoaderManager:
    """
    状态数据无限生成器 (Infinite State Stream Generator) - 精确约束版
    
    使用 PyTorch 向量化算子，在 GPU/CPU 内存中极速生成满足所有碰撞耦合与几何多边形约束的数据流。
    """
    def __init__(self, param_manager: ParamManager, batch_size: int, device: torch.device = torch.device('cpu')):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.param_manager = param_manager
        self.batch_size = batch_size
        self.device = device
        
        self.total_dim = self.param_manager.get_total_feature_dim()
        self.state_indices = self.param_manager.get_state_indices()
        self.idx_map = {p['name']: p['index'] for p in self.param_manager.state_params}
        
    def _generate_batch(self) -> torch.Tensor:
        B = self.batch_size
        D = self.total_dim
        batch_phys = torch.zeros((B, D), device=self.device, dtype=torch.float32)
        
        # 1. 速度: [25, 65]
        batch_phys[:, self.idx_map['impact_velocity']] = torch.rand(B, device=self.device) * 40.0 + 25.0
        
        # 2. 重叠率 & 角度
        # 重叠率 (-1, -0.25] ∪ [0.25, 1.0]
        u_ov = torch.rand(B, device=self.device)
        overlap = torch.where(u_ov < 0.5, torch.rand(B, device=self.device) * 0.75 - 1.0, torch.rand(B, device=self.device) * 0.75 + 0.25)
        batch_phys[:, self.idx_map['overlap']] = overlap
        
        # 初始角度 [-45, 45]
        angle = torch.rand(B, device=self.device) * 90.0 - 45.0
        
        # 极速角度纠正：重叠率绝对值在 0.25~0.3 时，角度需异号且绝对值 > 30
        mask_coup = (overlap.abs() >= 0.25) & (overlap.abs() < 0.3)
        num_coup = mask_coup.sum().item()
        if num_coup > 0:
            ang_sign = -torch.sign(overlap[mask_coup])
            # 重新采样在 [30, 45] 的绝对值
            ang_corrected = ang_sign * (torch.rand(num_coup, device=self.device) * 15.0 + 30.0)
            angle[mask_coup] = ang_corrected
        batch_phys[:, self.idx_map['impact_angle']] = angle
        
        # 3. 离散类别
        # OT: [1, 2, 3] 均匀生成
        ot = torch.randint(1, 4, (B,), device=self.device, dtype=torch.float32)
        batch_phys[:, self.idx_map['OT']] = ot
        
        # 侧位: [0, 1] 均匀生成
        is_driver = torch.randint(0, 2, (B,), device=self.device, dtype=torch.float32)
        batch_phys[:, self.idx_map['is_driver_side']] = is_driver
        
        # 4. 基于类别的几何约束采样 (SP, SH, RA)
        ra_tensor = torch.zeros(B, device=self.device, dtype=torch.float32)
        sp_tensor = torch.zeros(B, device=self.device, dtype=torch.float32)
        sh_tensor = torch.zeros(B, device=self.device, dtype=torch.float32)
        
        # 遍历 6 种状态组合，进行并行约束映射
        for drv_val in [0, 1]:
            for ot_val in [1, 2, 3]:
                mask = (is_driver == drv_val) & (ot == ot_val)
                num_in_group = mask.sum().item()
                if num_in_group == 0:
                    continue
                
                # A. 靠背角 RA 映射
                ra_opts = torch.tensor(RA_VALUES[(drv_val, ot_val)], device=self.device, dtype=torch.float32)
                ra_idx = torch.randint(0, len(ra_opts), (num_in_group,), device=self.device)
                ra_tensor[mask] = ra_opts[ra_idx]
                
                # B. 座椅位置 SP/SH 向量化拒绝采样
                poly_pts = torch.tensor(SEAT_CONSTRAINTS[(drv_val, ot_val)], device=self.device, dtype=torch.float32)
                sp_min, sh_min = poly_pts.min(dim=0)[0]
                sp_max, sh_max = poly_pts.max(dim=0)[0]
                
                sub_mask_valid = torch.zeros(num_in_group, dtype=torch.bool, device=self.device)
                sub_sp = torch.zeros_like(sub_mask_valid, dtype=torch.float32)
                sub_sh = torch.zeros_like(sub_mask_valid, dtype=torch.float32)
                
                # 循环直至少数不合规点全部被重新采样进多边形内部
                while not sub_mask_valid.all():
                    num_invalid = (~sub_mask_valid).sum().item()
                    gen_sp = torch.rand(num_invalid, device=self.device) * (sp_max - sp_min) + sp_min
                    gen_sh = torch.rand(num_invalid, device=self.device) * (sh_max - sh_min) + sh_min
                    
                    inside = point_in_polygon_torch(gen_sp, gen_sh, poly_pts)
                    
                    sub_sp[~sub_mask_valid] = torch.where(inside, gen_sp, sub_sp[~sub_mask_valid])
                    sub_sh[~sub_mask_valid] = torch.where(inside, gen_sh, sub_sh[~sub_mask_valid])
                    sub_mask_valid[~sub_mask_valid] = inside
                
                sp_tensor[mask] = sub_sp
                sh_tensor[mask] = sub_sh
                
        batch_phys[:, self.idx_map['RA']] = ra_tensor
        batch_phys[:, self.idx_map['SP']] = sp_tensor
        batch_phys[:, self.idx_map['SH']] = sh_tensor
        
        return batch_phys[:, self.state_indices]

    def get_infinite_generator(self) -> Iterator[torch.Tensor]:
        while True:
            yield self._generate_batch()