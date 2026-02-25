import torch
import logging
from src.core.param_manager import ParamManager

class PhysicalConstraintManager:
    """
    物理约束管理器 (Physical Constraint Manager)
    
    采用显式硬编码的形式收录所有已知的约束系统耦合规则。
    具备自适应能力：仅当规则中涉及的变量当前被设为 trainable=True 时，规则才会生效。
    """
    def __init__(self, param_manager: ParamManager):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.param_manager = param_manager
        
        # 提取所有开启了寻优（trainable=True）的参数及其局部索引（0~D_trainable-1）
        trainable_names = [p['name'] for p in param_manager.control_trainable_params]
        self.idx_map = {name: i for i, name in enumerate(trainable_names)}
        
        self.logger.info(f"已加载物理约束管理器，当前活跃寻优参数: {list(self.idx_map.keys())}")

    def _get_idx(self, name: str):
        return self.idx_map.get(name, None)

    def project_forward(self, actions: torch.Tensor) -> torch.Tensor:
        """
        前向硬投影（Hard Projection），供 StrategyNet 输出最后一步调用。
        使用 torch.min/max 等可导算子截断边界，保证梯度计算不断裂。
        
        参数: actions [Batch, D_trainable]
        """
        out = actions.clone()
        
        idx_AFT = self._get_idx('AFT')
        idx_BTF = self._get_idx('BTF')
        idx_LLATTF = self._get_idx('LLATTF')
        idx_LL1 = self._get_idx('LL1')
        idx_LL2 = self._get_idx('LL2')
        
        # 规则 1: AFT < BTF + 25 (在此留 0.1 的物理余量)
        if idx_AFT is not None and idx_BTF is not None:
            out[:, idx_AFT] = torch.min(out[:, idx_AFT], out[:, idx_BTF] + 24.9)
            
        # 规则 2: LLATTF >= BTF
        if idx_LLATTF is not None and idx_BTF is not None:
            out[:, idx_LLATTF] = torch.max(out[:, idx_LLATTF], out[:, idx_BTF])
            
        # 规则 3: 一级限力必须大于二级限力 (预留未来 LL1, LL2 可调的空间)
        if idx_LL1 is not None and idx_LL2 is not None:
            out[:, idx_LL1] = torch.max(out[:, idx_LL1], out[:, idx_LL2] + 0.1)
            
        return out

    def compute_soft_penalty(self, actions: torch.Tensor) -> torch.Tensor:
        """
        软惩罚计算 (Soft Penalty)，供 SurrogateAdapter 合并入最终 Loss。
        使用 ReLU 仅对越界量产生大于 0 的梯度惩罚。
        """
        penalty = torch.zeros(actions.shape[0], device=actions.device)
        
        idx_AFT = self._get_idx('AFT')
        idx_BTF = self._get_idx('BTF')
        idx_LLATTF = self._get_idx('LLATTF')
        idx_LL1 = self._get_idx('LL1')
        idx_LL2 = self._get_idx('LL2')
        
        # 规则 1: 若 AFT >= BTF + 25，产生惩罚
        if idx_AFT is not None and idx_BTF is not None:
            penalty += torch.relu(actions[:, idx_AFT] - (actions[:, idx_BTF] + 24.9))
            
        # 规则 2: 若 LLATTF < BTF，产生惩罚
        if idx_LLATTF is not None and idx_BTF is not None:
            penalty += torch.relu(actions[:, idx_BTF] - actions[:, idx_LLATTF])
            
        # 规则 3: 若 LL1 <= LL2，产生惩罚
        if idx_LL1 is not None and idx_LL2 is not None:
            penalty += torch.relu(actions[:, idx_LL2] + 0.1 - actions[:, idx_LL1])
            
        return penalty