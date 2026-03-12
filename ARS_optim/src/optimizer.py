import time
from typing import Dict, Tuple

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
    - Opt2: 仅当 refine_steps>0 时，记录局部精调结果。局部精调属于逐点优化。

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
        min_bounds, max_bounds = self.param_manager.get_trainable_bounds()
        self._trainable_mins_cpu = min_bounds
        self._trainable_maxs_cpu = max_bounds

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
                projected_actions, _ = self.strategy_net(context_params, pulse_norm)
                return projected_actions
        return self.param_manager.get_trainable_defaults_tensor(device=device).unsqueeze(0).expand(batch_size, -1)

    def _strict_project_actions(self, context_params: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            full_features = self.constraint_engine.compose_full_features(context_params, actions)
            projected_full = self.constraint_engine.project_forward(full_features, strict=True)
            _, projected_actions = self.constraint_engine.split_from_full(projected_full)
        return projected_actions

    def optimize(self, context_params: torch.Tensor, pulse_norm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """执行逐点局部精调。

        pulse_norm 必须由调用方显式提供，避免优化器内部再隐式触发一遍 pulse 生成，
        让训练、评估和局部精调都共享同一条明确的 pulse 来源语义。
        """
        self.surrogate.eval()

        init_actions = self._resolve_initial_actions(context_params, pulse_norm)
        init_actions = self._strict_project_actions(context_params, init_actions)
        init_loss_batch, init_preds, init_info = self.surrogate.predict_injury_and_loss(
            context_params=context_params,
            control_trainable=init_actions,
            pulse_norm=pulse_norm,
            include_yaml_bounds=False,
        )
        direct_stage = None
        if self.direct_inference:
            direct_stage = {
                "actions": init_actions.detach(),
                "loss_batch": init_loss_batch.detach(),
                "preds": init_preds.detach(),
                "detail": {
                    "p_head": init_info["p_head"].detach(),
                    "p_chest": init_info["p_chest"].detach(),
                    "p_neck": init_info["p_neck"].detach(),
                    "joint_risk": init_info["joint_risk"].detach(),
                },
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
        min_bounds = self._trainable_mins_cpu.to(device=context_params.device, dtype=context_params.dtype)
        max_bounds = self._trainable_maxs_cpu.to(device=context_params.device, dtype=context_params.dtype)
        # 局部精调直接在物理空间更新变量；若每步都硬夹回 yaml 边界，
        # 边界附近很容易因为 Adam 动量和投影共同作用而出现“贴边抖动”。
        # 这里允许优化变量在边界外保留 10% 的缓冲带，
        # 真正送入代理模型前仍会经过 forward projection 与软惩罚，
        # 最终输出阶段再由 strict projection 收回绝对合法域。
        relaxed_span = torch.clamp(max_bounds - min_bounds, min=1e-6) * 0.1
        relaxed_lower = min_bounds - relaxed_span
        relaxed_upper = max_bounds + relaxed_span

        for _ in range(self.refine_steps):
            optimizer.zero_grad()
            full_raw = self.constraint_engine.compose_full_features(context_params, opt_var)
            projected_full = self.constraint_engine.project_forward(full_raw, strict=False)
            _, projected_actions = self.constraint_engine.split_from_full(projected_full)

            _, _, risk_info = self.surrogate.predict_injury_and_loss(
                context_params=context_params,
                control_trainable=projected_actions,
                pulse_norm=pulse_norm,
                include_yaml_bounds=False,
                detach_info=False,
            )
            loss_risk = risk_info["loss_risk"]
            penalty = self.constraint_engine.compute_soft_penalty(full_raw, include_yaml_bounds=True)
            # 分布偏离惩罚应约束当前真正送入代理模型评估的候选解，
            # 否则会把置信度约束施加到投影前的松弛变量上，和实际优化轨迹错位。
            dist_penalty = self.surrogate.distribution_penalty.compute(context_params, projected_actions)
            total_batch = (
                loss_risk
                + self.surrogate.weight_penalty * penalty
                + self.surrogate.weight_distribution * dist_penalty
            )
            total_mean = total_batch.mean()
            total_mean.backward()
            optimizer.step()

            with torch.no_grad():
                opt_var.clamp_(relaxed_lower.unsqueeze(0), relaxed_upper.unsqueeze(0))
            trajectory.append(float(total_mean.item()))

        end = time.time()
        final_actions = self._strict_project_actions(context_params, opt_var)
        final_loss, final_preds, final_info = self.surrogate.predict_injury_and_loss(
            context_params=context_params,
            control_trainable=final_actions,
            pulse_norm=pulse_norm,
            include_yaml_bounds=False,
        )
        final_info["final_loss_batch"] = final_loss.detach()
        if "joint_risk" in final_info:
            final_info["joint_risk_batch"] = final_info["joint_risk"]
        final_info["direct_stage"] = direct_stage
        final_info["refine_stage_enabled"] = True
        final_info["time_cost"] = end - start
        final_info["trajectory"] = trajectory
        return final_actions.detach(), final_preds.detach(), final_info
