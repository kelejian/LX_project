import logging
import time
from typing import Dict, Optional, Tuple

import torch

from ARS_optim.src.constraints import ConstraintEngine
from ARS_optim.src.param_manager import ParamManager
from ARS_optim.src.strategy_net import StrategyNet
from ARS_optim.src.surrogate import SurrogateAdapter


class LocalRefiner:
    """局部精调优化器。"""

    def __init__(
        self,
        config: dict,
        param_manager: ParamManager,
        constraint_engine: ConstraintEngine,
        surrogate: SurrogateAdapter,
        strategy_net: StrategyNet = None,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
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
            init_loss_batch, init_preds, init_info = self.surrogate.predict_injury_and_loss(context_params, init_actions, pulse_norm)
        init_loss = init_loss_batch.mean()

        if self.refine_steps <= 0:
            result = {
                "initial": {
                    "actions": init_actions.detach(),
                    "loss_batch": init_loss_batch.detach(),
                    "loss_mean": float(init_loss.item()),
                    "preds": init_preds.detach(),
                    "detail": {key: value.detach() for key, value in init_info.items()},
                },
                "time_cost": 0.0,
                "trajectory": [],
            }
            result.update(init_info)
            if "joint_risk" in result:
                result["joint_risk_batch"] = result["joint_risk"]
            return init_actions.detach(), init_preds.detach(), result

        min_bounds, max_bounds = self.param_manager.get_trainable_bounds(device=device)
        opt_var = init_actions.clone().detach().requires_grad_(True)

        optimizer = torch.optim.Adam([opt_var], lr=self.lr)
        trajectory = []
        start = time.time()

        for _ in range(self.refine_steps):
            optimizer.zero_grad()
            loss_batch, _, _ = self.surrogate.predict_injury_and_loss(context_params, opt_var, pulse_norm)
            loss_mean = loss_batch.mean()
            loss_mean.backward()
            optimizer.step()

            with torch.no_grad():
                opt_var.clamp_(min_bounds, max_bounds)
                opt_var.copy_(self.constraint_engine.project_forward(opt_var, context_params))
            trajectory.append(float(loss_mean.item()))

        end = time.time()
        with torch.no_grad():
            final_actions = self.constraint_engine.project_forward(opt_var, context_params)
            final_loss, final_preds, final_info = self.surrogate.predict_injury_and_loss(context_params, final_actions, pulse_norm)
            final_info["final_loss_batch"] = final_loss.detach()
            if "joint_risk" in final_info:
                final_info["joint_risk_batch"] = final_info["joint_risk"]

        final_info["initial"] = {
            "actions": init_actions.detach(),
            "loss_batch": init_loss_batch.detach(),
            "loss_mean": float(init_loss.item()),
            "preds": init_preds.detach(),
            "detail": {key: value.detach() for key, value in init_info.items()},
        }
        final_info["time_cost"] = end - start
        final_info["trajectory"] = trajectory
        return final_actions.detach(), final_preds.detach(), final_info
