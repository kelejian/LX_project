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

    与策略网络的 Sigmoid 重参数化不同，这里的局部精调不会直接在物理量空间上用统一学习率更新。
    各控制参数量纲差异很大，若直接共用一个 Adam 步长，很难把学习率解释成稳定的一致尺度。
    因此这里先把 trainable 控制参数按各自 yaml 量程映射到无量纲潜空间，再在潜空间里优化；
    每轮前向时再还原回物理尺度，送入统一的投影与代理评估链路。
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
        min_bounds, max_bounds = self.param_manager.get_trainable_opt_bounds()
        self._trainable_mins_cpu = min_bounds
        self._trainable_maxs_cpu = max_bounds

    def _to_latent(self, actions_phys: torch.Tensor, mins: torch.Tensor, spans: torch.Tensor) -> torch.Tensor:
        """把物理尺度动作映射到无量纲潜空间。"""
        return (actions_phys - mins.unsqueeze(0)) / spans.unsqueeze(0)

    def _from_latent(self, latent_actions: torch.Tensor, mins: torch.Tensor, spans: torch.Tensor) -> torch.Tensor:
        """把无量纲潜变量还原回物理尺度动作。"""
        return mins.unsqueeze(0) + latent_actions * spans.unsqueeze(0)

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
        """把候选动作做最终严格合法化。

        局部精调内部每一步只走 strict=False，可保留纯 torch 的连续梯度链；
        但真正作为阶段输出写入结果表时，必须补做 strict=True，
        让座椅多边形边界这类 numpy 几何投影也生效，保证落盘结果是绝对合法的物理解。
        """
        with torch.no_grad():
            full_features = self.constraint_engine.compose_full_features(context_params, actions)
            projected_full = self.constraint_engine.project_forward(full_features, strict=True)
            _, projected_actions = self.constraint_engine.split_from_full(projected_full)
        return projected_actions

    def optimize(self, context_params: torch.Tensor, pulse_norm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """执行逐点局部精调。

        pulse_norm 必须由调用方显式提供，避免优化器内部再隐式触发一遍 pulse 生成，
        让训练、评估和局部精调都共享同一条明确的 pulse 来源语义。

        优化变量不是物理量本身，而是按 yaml 量程归一化后的无量纲潜变量 z：
        - z=0 对应 trainable 下界；
        - z=1 对应 trainable 上界；
        - 统一学习率表示“每步占各自量程的比例”。
        """
        self.surrogate.eval()

        init_actions = self._resolve_initial_actions(context_params, pulse_norm)
        init_actions = self._strict_project_actions(context_params, init_actions)
        init_loss_batch, init_preds, init_info = self.surrogate.predict_injury_and_loss(
            context_params=context_params,
            control_trainable=init_actions,
            pulse_norm=pulse_norm,
            include_opt_bounds=False,
        )
        direct_stage = None
        if self.direct_inference:
            # Opt1 的记录口径是“策略网络直推后、已经过 strict 投影的动作”。
            # 这样评估表里的 Opt1 与最终落地可执行的控制量保持同一物理语义，
            # 不会把投影前的 raw action 混进报告里造成歧义。
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
            return init_actions.detach(), init_preds.detach(), result

        min_bounds = self._trainable_mins_cpu.to(device=context_params.device, dtype=context_params.dtype)
        max_bounds = self._trainable_maxs_cpu.to(device=context_params.device, dtype=context_params.dtype)
        spans = torch.clamp(max_bounds - min_bounds, min=1e-6)
        latent_var = self._to_latent(init_actions, mins=min_bounds, spans=spans).detach().requires_grad_(True)

        optimizer = torch.optim.Adam([latent_var], lr=self.lr)
        trajectory = []
        start = time.time()
        # 潜空间 [0, 1] 与 yaml 边界一一对应；这里仍保留 10% 的松弛带，允许优化器在“合法边界附近”做少量越界探索，再通过前向投影和软惩罚把解拉回可行域，减少边界处的更新僵硬和振荡。
        # 缓冲带定义在无量纲 z 空间里，而不是物理尺度里。这样统一学习率的语义不会再被不同参数的量纲放大或缩小。
        relaxed_margin = 0.1
        relaxed_lower = -relaxed_margin
        relaxed_upper = 1.0 + relaxed_margin

        for _ in range(self.refine_steps):
            optimizer.zero_grad()
            actions_phys = self._from_latent(latent_var, mins=min_bounds, spans=spans)
            full_raw = self.constraint_engine.compose_full_features(context_params, actions_phys)
            projected_full = self.constraint_engine.project_forward(full_raw, strict=False)
            _, projected_actions = self.constraint_engine.split_from_full(projected_full)

            # 代理模型只看已经前向投影后的合法动作，避免把中间越界解送去做域外外推；
            # 但软惩罚仍作用在 full_raw 上，为“越界方向”保留梯度，把潜变量拉回可行域。
            _, _, risk_info = self.surrogate.predict_injury_and_loss(
                context_params=context_params,
                control_trainable=projected_actions,
                pulse_norm=pulse_norm,
                include_opt_bounds=False,
                detach_info=False,
            )
            loss_risk = risk_info["loss_risk"]
            penalty = self.constraint_engine.compute_soft_penalty(full_raw, include_opt_bounds=True)
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
                latent_var.clamp_(relaxed_lower, relaxed_upper)
            trajectory.append(float(total_mean.item()))

        end = time.time()
        final_actions_phys = self._from_latent(latent_var.detach(), mins=min_bounds, spans=spans)
        final_actions = self._strict_project_actions(context_params, final_actions_phys)
        final_loss, final_preds, final_info = self.surrogate.predict_injury_and_loss(
            context_params=context_params,
            control_trainable=final_actions,
            pulse_norm=pulse_norm,
            include_opt_bounds=False,
        )
        final_info["final_loss_batch"] = final_loss.detach()
        final_info["direct_stage"] = direct_stage
        final_info["refine_stage_enabled"] = True
        final_info["time_cost"] = end - start
        final_info["trajectory"] = trajectory
        return final_actions.detach(), final_preds.detach(), final_info
