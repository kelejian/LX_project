import torch
import logging
from typing import Dict, Tuple, Optional

# 严格执行绝对路径引用规范
from ARS_optim.src.interface.surrogate_adapter import SurrogateAdapter
from ARS_optim.src.models.strategy_net import StrategyNet
from ARS_optim.src.core.param_manager import ParamManager
from ARS_optim.src.core.constraints import PhysicalConstraintManager

class ARSLocalOptimizer:
    """
    ARS 在线局部精调优化器。

    工作流支持双模式：
    - 模式A (direct_inference=True): 缓存波形 -> StrategyNet联合推断 -> 损伤模型局部梯度精调 -> 混合状态硬投影。
    - 模式B (direct_inference=False): 缓存波形 -> Default硬编码起点 -> 损伤模型局部梯度精调 -> 混合状态硬投影。

    说明：
    - 该模块只优化“可调控制参数”；
    - context 参数保持不变，作为每个样本的条件输入。
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

    def optimize(self, context_params: torch.Tensor, pulse_norm: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        执行在线寻优管线 (Online Optimization Pipeline)
        
        参数:
            context_params: [Batch, D_context] 物理尺度的上下文参数（state + fixed-control）
            pulse_norm: 可选，[Batch, 2, Seq_Len] 归一化波形缓存。若提供则复用，避免重复推理。
        返回:
            actions: [Batch, D_trainable] 寻优结束后的物理动作决策
            final_preds: [Batch, 3] 最终的损伤预测值 (HIC, Dmax, Nij)
            final_info: Dict 包含损失与违规惩罚等辅助信息的字典

        例子：
        - Batch=4, D_trainable=3 时，返回 actions 形状为 [4, 3]；
        - final_preds 每行依次为 [HIC, Dmax, Nij]。
        """
        import time
        batch_size = context_params.shape[0]
        device = context_params.device
        
        self.surrogate.eval()
        
        # 阶段 0：波形缓存
        # 同一批 context 在本次 optimize 中共享一份 pulse，避免重复计算。
        if pulse_norm is None:
            with torch.no_grad():
                # [Batch, D_context] -> [Batch, 2, Seq_Len]
                pulse_norm = self.surrogate.generate_pulse(context_params)
        
        # 阶段 1：获取初始解 a0
        if self.direct_inference:
            if self.strategy_net is None:
                raise ValueError("[致命错误] 开启了 direct_inference=True，但未传入 strategy_net 实例！")
            self.strategy_net.eval()
            with torch.no_grad():
                # [接口对齐]: 传入状态与波形进行多模态融合推断
                init_actions = self.strategy_net(context_params, pulse_norm)
        else:
            # 关闭直推时，使用配置里的 default 作为起点
            try:
                defaults = [p['default'] for p in self.param_manager.control_trainable_params]
            except KeyError as e:
                raise ValueError(f"[致命配置错误] 当 direct_inference=False 时，所有可调参数必须具备 'default' 值！缺失: {e}")
            
            init_actions = torch.tensor(defaults, dtype=torch.float32, device=device)
            init_actions = init_actions.unsqueeze(0).expand(batch_size, -1)

        # 对初始解做一次硬投影，保证满足参数耦合约束
        with torch.no_grad():
            init_actions = self.constraint_manager.project_forward(init_actions, context_params)
        
        # 记录初始损失 / 预测值
        with torch.no_grad():
            init_loss_batch, init_preds, init_info = self.surrogate.predict_injury_and_loss(context_params, init_actions, pulse_norm)
        init_loss = init_loss_batch.mean()

        # 若不需要精调（refine_steps<=0），直接返回初始解
        if self.refine_steps <= 0:
            ret_info = {
                'initial': {
                    'actions': init_actions.detach(),
                    'loss_batch': init_loss_batch.detach(),
                    'loss_mean': init_loss.item(),
                    'preds': init_preds.detach(),
                    'detail': {k: v.detach() for k, v in init_info.items()}
                },
                'time_cost': 0.0,
                'trajectory': []
            }
            ret_info.update(init_info)
            if 'joint_risk' in ret_info:
                ret_info['joint_risk_batch'] = ret_info['joint_risk']
            return init_actions.detach(), init_preds.detach(), ret_info

        # 阶段 2：局部精调
        # 可选择在归一化空间或物理空间更新参数：
        # - 归一化空间便于不同量纲参数共享学习率。
        actions = init_actions.clone().detach()
        if self.optimize_in_normalized:
            # normalized 变量作为可学习叶子张量；后续 optimizer 直接更新该变量
            norm_actions = self.surrogate._normalize_control(actions, device=device)
            norm_actions = norm_actions.clone().detach()
            norm_actions.requires_grad = True
            opt_var = norm_actions
        else:
            # 仅在物理空间优化时才需要把 actions 标记为可训练变量。
            # 在归一化模式下优化变量是 norm_actions，给 actions 开梯度属于冗余。
            actions.requires_grad = True
            opt_var = actions

        # 只对动作变量建优化器，不更新任何代理模型权重
        optimizer = torch.optim.Adam([opt_var], lr=self.lr)
        min_b, max_b = self.param_manager.get_trainable_bounds(device=device)
        
        trajectory = []
        start_time = time.time()
        for step in range(self.refine_steps):
            optimizer.zero_grad()

            # 将当前优化变量映射到物理尺度后计算损伤
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
                    # 先限制在 [0,1]，再映射到物理空间做约束投影，最后映射回 [0,1]
                    opt_var.clamp_(0.0, 1.0)
                    phys_tmp = opt_var * (max_b - min_b).unsqueeze(0) + min_b.unsqueeze(0)
                    phys_tmp = self.constraint_manager.project_forward(phys_tmp, context_params)
                    # copy_ 保持 opt_var 身份不变（优化器内部状态可持续复用）
                    opt_var.copy_((phys_tmp - min_b.unsqueeze(0)) / (max_b - min_b).unsqueeze(0))
                else:
                    # 物理空间下直接 clamp + 投影
                    opt_var.clamp_(min_b, max_b)
                    projected_actions = self.constraint_manager.project_forward(opt_var, context_params)
                    # 同理，用 copy_ 回写，避免替换张量对象
                    opt_var.copy_(projected_actions)
            # trajectory 记录每一步均值损失，可用于观察是否单调下降
            trajectory.append(loss_mean.item())
        end_time = time.time()
        # 阶段 3：最终评估
        with torch.no_grad():
            if self.optimize_in_normalized:
                actions = opt_var * (max_b - min_b).unsqueeze(0) + min_b.unsqueeze(0)
                actions.clamp_(min_b, max_b)
                actions = self.constraint_manager.project_forward(actions, context_params)
            else:
                actions = opt_var
            final_loss, final_preds, final_info = self.surrogate.predict_injury_and_loss(context_params, actions, pulse_norm)
            final_info['final_loss_batch'] = final_loss.detach()
            if 'joint_risk' in final_info:
                final_info['joint_risk_batch'] = final_info['joint_risk']
        
        final_info['initial'] = {
            'actions': init_actions.detach(),
            'loss_batch': init_loss_batch.detach(),
            'loss_mean': init_loss.item(),
            'preds': init_preds.detach(),
            'detail': {k: v.detach() for k, v in init_info.items()}
        }
        final_info['time_cost'] = end_time - start_time
        final_info['trajectory'] = trajectory
        return actions.detach(), final_preds.detach(), final_info