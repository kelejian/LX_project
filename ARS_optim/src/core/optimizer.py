import torch
import logging
from typing import Dict, Tuple

# 严格执行绝对路径引用规范
from ARS_optim.src.interface.surrogate_adapter import SurrogateAdapter
from ARS_optim.src.models.strategy_net import StrategyNet
from ARS_optim.src.core.param_manager import ParamManager
from ARS_optim.src.core.constraints import PhysicalConstraintManager

class ARSLocalOptimizer:
    """
    ARS 在线局部精调优化器 (Online Local Refinement Optimizer)
    
    架构设计:
    本模块承载了摊销寻优与局部精确优化的桥接任务。
    工作流支持双模式:
    - 模式A (direct_inference=True): 缓存波形 -> StrategyNet联合推断 -> 损伤模型局部梯度精调 -> 混合状态硬投影。
    - 模式B (direct_inference=False): 缓存波形 -> Default硬编码起点 -> 损伤模型局部梯度精调 -> 混合状态硬投影。
    
    性能优化:
    得益于 SurrogateAdapter 的物理因果解耦，本模块实现了波形特征的“一次推断，全生命周期缓存”。
    在梯度下降迭代阶段，只针对计算量极小的损伤预测模块求解雅可比矩阵，将车规在线寻优的时间复杂度降低了一个数量级。
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
        # 是否在归一化空间中执行梯度精调，便于处理尺度不均问题
        self.optimize_in_normalized = bool(opt_cfg.get('optimize_in_normalized', False))

    def optimize(self, context_params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        执行在线寻优管线 (Online Optimization Pipeline)
        
        参数:
            context_params: [Batch, D_context] 物理尺度的上下文参数（state + fixed-control）
        返回:
            actions: [Batch, D_trainable] 寻优结束后的物理动作决策
            final_preds: [Batch, 3] 最终的损伤预测值 (HIC, Dmax, Nij)
            final_info: Dict 包含损失与违规惩罚等辅助信息的字典
        """
        import time
        batch_size = context_params.shape[0]
        device = context_params.device
        
        self.surrogate.eval()
        
        # ==========================================
        # 阶段 0：波形生成与常数级缓存 (O(1) 复杂度)
        # 物理意义：无论约束系统参数如何调整，碰撞波形在 T=0 时刻由工况决定且不可逆。
        # ==========================================
        with torch.no_grad():
            # pulse_norm: [Batch, 2, Seq_Len]
            pulse_norm = self.surrogate.generate_pulse(context_params)
        
        # ==========================================
        # 阶段 1：获取初始解 (a_0)
        # ==========================================
        if self.direct_inference:
            if self.strategy_net is None:
                raise ValueError("[致命错误] 开启了 direct_inference=True，但未传入 strategy_net 实例！")
            self.strategy_net.eval()
            with torch.no_grad():
                # [接口对齐]: 传入状态与波形进行多模态融合推断
                init_actions = self.strategy_net(context_params, pulse_norm)
        else:
            # 严格校验：提取可调参数的 default 值作为寻优起点
            try:
                defaults = [p['default'] for p in self.param_manager.control_trainable_params]
            except KeyError as e:
                raise ValueError(f"[致命配置错误] 当 direct_inference=False 时，所有可调参数必须具备 'default' 值！缺失: {e}")
            
            init_actions = torch.tensor(defaults, dtype=torch.float32, device=device)
            init_actions = init_actions.unsqueeze(0).expand(batch_size, -1)

        # 强制进行一次硬投影以确保合法（初始解已约束）
        with torch.no_grad():
            init_actions = self.constraint_manager.project_forward(init_actions, context_params)
        
        # 记录初始损失 / 预测值
        with torch.no_grad():
            init_loss_batch, init_preds, init_info = self.surrogate.predict_injury_and_loss(context_params, init_actions, pulse_norm)
        init_loss = init_loss_batch.mean()

        # 若不需要精调 (零次迭代)，直接评估并返回初始解
        if self.refine_steps <= 0:
            ret_info = {
                'initial': {
                    'loss_batch': init_loss_batch.detach(),
                    'loss_mean': init_loss.item(),
                    'preds': init_preds.detach()
                },
                'time_cost': 0.0,
                'trajectory': []
            }
            ret_info.update(init_info)
            return init_actions.detach(), init_preds.detach(), ret_info

        # ==========================================
        # 阶段 2：基于梯度的局部精调 (Local Refinement via Gradient Descent)
        # ==========================================
        # 准备优化变量：根据配置决定在归一化空间还是物理空间进行优化
        actions = init_actions.clone().detach()
        actions.requires_grad = True
        if self.optimize_in_normalized:
            norm_actions = self.surrogate._normalize_control(actions, device=device)
            norm_actions = norm_actions.clone().detach()
            norm_actions.requires_grad = True
            opt_var = norm_actions
        else:
            opt_var = actions

        # 仅对动作变量实例化优化器
        optimizer = torch.optim.Adam([opt_var], lr=self.lr)
        min_b, max_b = self.param_manager.get_trainable_bounds(device=device)
        
        trajectory = []
        start_time = time.time()
        for step in range(self.refine_steps):
            optimizer.zero_grad()

            # 物理尺度动作，用于损伤预测
            if self.optimize_in_normalized:
                phys = opt_var * (max_b - min_b).unsqueeze(0) + min_b.unsqueeze(0)
            else:
                phys = opt_var

            loss_batch, preds, info = self.surrogate.predict_injury_and_loss(context_params, phys, pulse_norm)
            loss_mean = loss_batch.mean()
            loss_mean.backward()
            optimizer.step()

            with torch.no_grad():
                if self.optimize_in_normalized:
                    opt_var.clamp_(0.0, 1.0)
                    phys_tmp = opt_var * (max_b - min_b).unsqueeze(0) + min_b.unsqueeze(0)
                    phys_tmp = self.constraint_manager.project_forward(phys_tmp, context_params)
                    opt_var.copy_((phys_tmp - min_b.unsqueeze(0)) / (max_b - min_b).unsqueeze(0))
                else:
                    opt_var.clamp_(min_b, max_b)
                    projected_actions = self.constraint_manager.project_forward(opt_var, context_params)
                    opt_var.copy_(projected_actions)
            trajectory.append(loss_mean.item())
        end_time = time.time()
        # ==========================================
        # 阶段 3：最终评估与日志收尾
        # ==========================================
        with torch.no_grad():
            if self.optimize_in_normalized:
                actions = opt_var * (max_b - min_b).unsqueeze(0) + min_b.unsqueeze(0)
                actions.clamp_(min_b, max_b)
                actions = self.constraint_manager.project_forward(actions, context_params)
            else:
                actions = opt_var
            final_loss, final_preds, final_info = self.surrogate.predict_injury_and_loss(context_params, actions, pulse_norm)
            final_info['final_loss_batch'] = final_loss.detach()
        
        final_info['initial'] = {
            'loss_batch': init_loss_batch.detach(),
            'loss_mean': init_loss.item(),
            'preds': init_preds.detach()
        }
        final_info['time_cost'] = end_time - start_time
        final_info['trajectory'] = trajectory
        return actions.detach(), final_preds.detach(), final_info