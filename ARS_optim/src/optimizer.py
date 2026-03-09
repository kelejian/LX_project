import time
from typing import Dict, Optional, Tuple

import torch

from ARS_optim.src.constraints import ConstraintEngine
from ARS_optim.src.param_manager import ParamManager
from ARS_optim.src.strategy_net import StrategyNet
from ARS_optim.src.surrogate import SurrogateAdapter


class LocalRefiner:
    """局部精调优化器。

    评估阶段按三层语义拆分：
    - Base: 输入给定或 default 的 baseline；
    - Opt1: 仅当 direct_inference=True 时，记录策略网络直推结果；
    - Opt2: 仅当 refine_steps>0 时，记录局部精调结果。

    注意：局部精调内部仍然可能从 default 或 Opt1 出发进行迭代，但只有满足阶段定义的结果才会写回评估表。
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

    def optimize(self, context_params: torch.Tensor, pulse_norm: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        batch_size = context_params.shape[0]
        device = context_params.device
        self.surrogate.eval()

        if pulse_norm is None:
            with torch.no_grad():
                pulse_norm = self.surrogate.generate_pulse(context_params)

        if self.direct_inference:
            if self.strategy_net is None:
                raise ValueError("direct_inference=True 但未提供 strategy_net")
            self.strategy_net.eval()
            with torch.no_grad():
                init_actions = self.strategy_net(context_params, pulse_norm)
        else:
            defaults = [param["default"] for param in self.param_manager.control_trainable_params]
            init_actions = torch.tensor(defaults, dtype=torch.float32, device=device).unsqueeze(0).expand(batch_size, -1)

        with torch.no_grad():
            init_actions = self.constraint_engine.project_forward(init_actions, context_params)
            # 评估阶段只应向外暴露最终合法动作，因此这里先做一次确定性 sanitize，后续 Opt1/Opt2 直接复用其预测结果，避免 run_eval 再重复跑一遍 surrogate。
            _, init_actions = self.constraint_engine.sanitize_context_and_trainable(context_params, init_actions)
            init_loss_batch, init_preds, init_info = self.surrogate.predict_injury_and_loss(context_params, init_actions, pulse_norm)
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
            # 这里采用 projected gradient descent 的实现形式：先在当前物理参数上做一次 Adam 更新，再立即用 project_forward把结果拉回连续可微约束子空间，避免迭代过程逐步漂出可行域。
            optimizer.zero_grad()
            loss_batch, _, _ = self.surrogate.predict_injury_and_loss(context_params, opt_var, pulse_norm)
            loss_mean = loss_batch.mean()
            loss_mean.backward()
            optimizer.step()

            with torch.no_grad():
                opt_var.copy_(self.constraint_engine.project_forward(opt_var, context_params))
            trajectory.append(float(loss_mean.item()))

        end = time.time()
        with torch.no_grad():
            final_actions = self.constraint_engine.project_forward(opt_var, context_params)
            _, final_actions = self.constraint_engine.sanitize_context_and_trainable(context_params, final_actions)
            final_loss, final_preds, final_info = self.surrogate.predict_injury_and_loss(context_params, final_actions, pulse_norm)
            final_info["final_loss_batch"] = final_loss.detach()
            if "joint_risk" in final_info:
                final_info["joint_risk_batch"] = final_info["joint_risk"]
        final_info["direct_stage"] = direct_stage
        final_info["refine_stage_enabled"] = True
        final_info["time_cost"] = end - start
        final_info["trajectory"] = trajectory
        return final_actions.detach(), final_preds.detach(), final_info
