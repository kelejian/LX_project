import time
from typing import Dict, Optional, Tuple

import torch

from ARS_optim.src.constraints import ConstraintEngine
from ARS_optim.src.param_manager import ParamManager
from ARS_optim.src.strategy_net import StrategyNet
from ARS_optim.src.surrogate import SurrogateAdapter


class LocalRefiner:
    """局部精调优化器。

    评估阶段固定区分三层语义：
    - Base: 输入给定或 default 的 baseline；
    - Opt1: 仅当 direct_inference=True 时，记录策略网络直推结果；
    - Opt2: 仅当 refine_steps>0 时，记录局部精调结果。

    这里把“局部精调从哪里起步”和“哪些结果对外写出”分开处理：
    内部迭代可以从 default 或策略网络输出出发，但结果表只写入上面这三种已定义阶段，
    避免评估脚本再去推断中间状态含义。
    """

    def __init__(
        self,
        config: dict,
        param_manager: ParamManager,
        constraint_engine: ConstraintEngine,
        surrogate: SurrogateAdapter,
        strategy_net: StrategyNet = None,
    ):
        self.param_manager = param_manager
        self.constraint_engine = constraint_engine
        self.surrogate = surrogate
        self.strategy_net = strategy_net

        opt_cfg = config.get("optimization", {})
        self.direct_inference = bool(opt_cfg.get("direct_inference", False))
        self.refine_steps = int(opt_cfg.get("refine_steps", 50))
        self.lr = float(opt_cfg.get("lr", 0.05))

    def _resolve_initial_actions(self, context_params: torch.Tensor, pulse_norm: torch.Tensor) -> torch.Tensor:
        """生成局部精调的起始动作。

        - direct_inference=True 时，从策略网络直推得到初始解；
        - 否则从 trainable control 的 default 值出发。

        该函数只决定优化起点，不负责合法化，也不决定结果表里记录哪个阶段。
        """
        batch_size = context_params.shape[0]
        device = context_params.device
        if self.direct_inference:
            if self.strategy_net is None:
                raise ValueError("direct_inference=True 但未提供 strategy_net")
            self.strategy_net.eval()
            with torch.no_grad():
                return self.strategy_net(context_params, pulse_norm)
        return self.param_manager.get_trainable_defaults_tensor(device=device).unsqueeze(0).expand(batch_size, -1)

    def _legalize_actions(self, context_params: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """将动作收敛到最终可对外暴露的合法状态。

        这里固定采用两步：
        1. project_forward 处理可微的连续耦合约束；
        2. sanitize 处理离散规则和最终硬合法化。

        这样可以保证对外落盘的动作与训练/评估其余入口看到的是同一套最终合法定义，
        而不是把优化迭代过程中的中间变量直接暴露出去。
        """
        with torch.no_grad():
            projected = self.constraint_engine.project_forward(actions, context_params)
            _, sanitized = self.constraint_engine.sanitize_context_and_trainable(context_params, projected)
        return sanitized

    def _evaluate_stage(self, context_params: torch.Tensor, actions: torch.Tensor, pulse_norm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """评估一个已经合法化的动作阶段。"""
        loss_batch, preds, info, _ = self.surrogate.evaluate_actions(
            context_params=context_params,
            control_trainable=actions,
            pulse_norm=pulse_norm,
        )
        return loss_batch, preds, info

    def optimize(self, context_params: torch.Tensor, pulse_norm: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        self.surrogate.eval()

        if pulse_norm is None:
            with torch.no_grad():
                pulse_norm = self.surrogate.generate_pulse(context_params)
        init_actions = self._resolve_initial_actions(context_params, pulse_norm)
        init_actions = self._legalize_actions(context_params, init_actions)
        init_loss_batch, init_preds, init_info = self._evaluate_stage(context_params, init_actions, pulse_norm)
        direct_stage = {
            "enabled": bool(self.direct_inference),
            "actions": init_actions.detach() if self.direct_inference else None,
            "loss_batch": init_loss_batch.detach() if self.direct_inference else None,
            "preds": init_preds.detach() if self.direct_inference else None,
            "detail": {key: value.detach() for key, value in init_info.items()} if self.direct_inference else {},
        }

        if self.refine_steps <= 0:
            result = {
                "final_loss_batch": init_loss_batch.detach(),
                "direct_stage": direct_stage,
                "refine_stage_enabled": False,
                "time_cost": 0.0,
                "trajectory": [],
            }
            result.update(init_info)
            if "joint_risk" in result:
                result["joint_risk_batch"] = result["joint_risk"]
            return init_actions.detach(), init_preds.detach(), result

        opt_var = init_actions.clone().detach().requires_grad_(True)

        optimizer = torch.optim.Adam([opt_var], lr=self.lr)
        trajectory = []
        start = time.time()

        for _ in range(self.refine_steps):
            # 每一步先在当前物理参数上做一次 Adam 更新，再立即投影回连续约束子空间。
            # 不在循环内直接做 sanitize，是因为离散 snapping 会破坏这一路径上的梯度结构；
            # 最终硬合法化统一放到循环结束后执行一次。
            optimizer.zero_grad()
            loss_batch, _, _, _ = self.surrogate.evaluate_actions(
                context_params=context_params,
                control_trainable=opt_var,
                pulse_norm=pulse_norm,
            )
            loss_mean = loss_batch.mean()
            loss_mean.backward()
            optimizer.step()

            with torch.no_grad():
                opt_var.copy_(self.constraint_engine.project_forward(opt_var, context_params))
            trajectory.append(float(loss_mean.item()))

        end = time.time()
        final_actions = self._legalize_actions(context_params, opt_var)
        final_loss, final_preds, final_info = self._evaluate_stage(context_params, final_actions, pulse_norm)
        final_info["final_loss_batch"] = final_loss.detach()
        if "joint_risk" in final_info:
            final_info["joint_risk_batch"] = final_info["joint_risk"]
        final_info["direct_stage"] = direct_stage
        final_info["refine_stage_enabled"] = True
        final_info["time_cost"] = end - start
        final_info["trajectory"] = trajectory
        return final_actions.detach(), final_preds.detach(), final_info
