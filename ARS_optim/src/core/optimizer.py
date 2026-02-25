import torch
import logging
from typing import Dict, Tuple

from src.interface.surrogate_adapter import SurrogateAdapter
from src.models.strategy_net import StrategyNet
from src.core.param_manager import ParamManager
from src.core.constraints import PhysicalConstraintManager

class ARSLocalOptimizer:
    """
    ARS 在线局部精调优化器 (Online Local Refinement Optimizer)
    
    工作流支持双模式:
    - 模式A (direct_inference=True): StrategyNet推断 -> 局部梯度精调 -> 硬投影。
    - 模式B (direct_inference=False): Default硬编码起点 -> 局部梯度精调 -> 硬投影。
    """
    def __init__(
        self, 
        config: dict, 
        param_manager: ParamManager, 
        constraint_manager: PhysicalConstraintManager, 
        surrogate: SurrogateAdapter, 
        strategy_net: StrategyNet = None # 当直推关闭时，可传入 None
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.param_manager = param_manager
        self.constraint_manager = constraint_manager
        self.surrogate = surrogate
        self.strategy_net = strategy_net
        
        # 解析在线优化的超参数
        opt_cfg = config.get('optimization', {})
        self.direct_inference = opt_cfg.get('direct_inference', False)
        self.refine_steps = int(opt_cfg.get('refine_steps', 50))
        self.lr = float(opt_cfg.get('lr', 0.05))

    def optimize(self, state_params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """执行在线寻优管线"""
        batch_size = state_params.shape[0]
        device = state_params.device
        
        self.surrogate.eval()
        
        # ==========================================
        # 阶段 1：获取初始解 (a_0)
        # ==========================================
        if self.direct_inference:
            if self.strategy_net is None:
                raise ValueError("[致命错误] 开启了 direct_inference=True，但未传入 strategy_net 实例！")
            self.strategy_net.eval()
            with torch.no_grad():
                init_actions = self.strategy_net(state_params)
        else:
            # 严格校验：提取可调参数的 default 值作为寻优起点
            try:
                defaults = [p['default'] for p in self.param_manager.control_trainable_params]
            except KeyError as e:
                raise ValueError(f"[致命配置错误] 当 direct_inference=False 时，所有可调参数(trainable=True)必须在 param_space.yaml 中配置 'default' 值！缺失: {e}")
            
            init_actions = torch.tensor(defaults, dtype=torch.float32, device=device)
            init_actions = init_actions.unsqueeze(0).expand(batch_size, -1) # 广播至 Batch 大小

        # 若不需要精调，直接返回初始解
        if self.refine_steps <= 0:
            with torch.no_grad():
                loss, preds, info = self.surrogate(state_params, init_actions)
            return init_actions.detach(), preds.detach(), info

        # ==========================================
        # 阶段 2：基于梯度的局部精调 (Local Refinement)
        # ==========================================
        actions = init_actions.clone().detach()
        actions.requires_grad = True
        
        optimizer = torch.optim.Adam([actions], lr=self.lr)
        min_b, max_b = self.param_manager.get_trainable_bounds(device=device)
        
        for step in range(self.refine_steps):
            optimizer.zero_grad()
            
            loss_batch, preds, info = self.surrogate(state_params, actions)
            loss_mean = loss_batch.mean()
            loss_mean.backward()
            optimizer.step()
            
            # 剥离计算图执行硬投影
            with torch.no_grad():
                actions.clamp_(min_b, max_b)
                projected_actions = self.constraint_manager.project_forward(actions)
                actions.copy_(projected_actions) # 原地替换，保留 requires_grad 属性
                
        # ==========================================
        # 阶段 3：最终评估与日志收尾
        # ==========================================
        with torch.no_grad():
            final_loss, final_preds, final_info = self.surrogate(state_params, actions)
            
        return actions.detach(), final_preds.detach(), final_info