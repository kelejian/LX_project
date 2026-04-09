import time
from typing import Any, Dict, List, Optional, Tuple

import torch

from ARS_optim.src.constraints import ConstraintEngine
from ARS_optim.src.param_manager import ParamManager
from ARS_optim.src.strategy_net import StrategyNet
from ARS_optim.src.surrogate import SurrogateAdapter


class Optimizer:
    """二阶段优化器，负责 `Opt1` 直推与 `Opt2` 局部精调。

    作用：
    - 统一管理策略直推、局部精调、收敛判定和 trace 记录；
    - 保证输出给评估脚本的阶段口径固定为 `Base / Opt1 / Opt2`。

    阶段定义：
    - `Base`: 给定 baseline control 后的直接评估结果；
    - `Opt1`: 策略网络直推并经过严格投影后的合法解；
    - `Opt2`: 在 `Opt1` 或 default 起点上继续局部精调后的合法解。

    注意：
    - 局部精调不直接在物理量空间上做统一学习率更新，而是先映射到各参数各自
      `[0, 1]` 的无量纲潜空间 `z` 再做 Adam。
    - 这样统一学习率表示的是“按各自可调量程的比例更新”，可避免不同物理量纲
      之间的步长语义失真。
    """

    def __init__(
        self,
        config: dict,
        param_manager: ParamManager,
        constraint_engine: ConstraintEngine,
        surrogate: SurrogateAdapter,
        strategy_net: StrategyNet = None,
    ):
        """读取局部精调所需的优化、调度与收敛配置。"""
        self.param_manager = param_manager
        self.constraint_engine = constraint_engine
        self.surrogate = surrogate
        self.strategy_net = strategy_net
        self.trainable_names = self.param_manager.get_trainable_names()

        opt_cfg = config.get("optimization", {}) or {}
        self.direct_inference = bool(opt_cfg.get("direct_inference", False))
        self.refine_steps = int(opt_cfg.get("refine_steps", 50))
        self.lr = float(opt_cfg.get("lr", 0.05))

        scheduler_cfg = opt_cfg.get("scheduler", {}) or {}
        self.scheduler_type = str(scheduler_cfg.get("type", "none")).lower()
        self.scheduler_eta_min = float(scheduler_cfg.get("eta_min", max(1.0e-6, self.lr * 0.1)))
        self.scheduler_step_size = max(1, int(scheduler_cfg.get("step_size", max(1, self.refine_steps // 4))))
        self.scheduler_gamma = float(scheduler_cfg.get("gamma", 0.5))
        self.scheduler_t_max_cfg = scheduler_cfg.get("T_max", "max_iters")

        convergence_cfg = opt_cfg.get("convergence", {}) or {}
        self.convergence_enabled = bool(convergence_cfg.get("enabled", True))
        self.convergence_action_tol_ratio = float(convergence_cfg.get("action_tol_ratio", 1.0e-4))
        self.convergence_loss_tol = float(convergence_cfg.get("loss_tol", 1.0e-6))
        self.convergence_patience = max(1, int(convergence_cfg.get("patience", 20)))
        if self.convergence_action_tol_ratio < 0.0 or self.convergence_loss_tol < 0.0:
            raise ValueError("局部精调 convergence 阈值必须非负")

        min_bounds, max_bounds = self.param_manager.get_trainable_opt_bounds()
        self._trainable_mins_cpu = min_bounds
        self._trainable_maxs_cpu = max_bounds

    def _build_refine_scheduler(self, optimizer: torch.optim.Optimizer):
        """按配置创建局部精调学习率调度器。"""
        if self.scheduler_type in {"none", ""}:
            return None
        if self.scheduler_type in {"cosine", "cosineannealinglr"}:
            t_max_cfg = self.scheduler_t_max_cfg
            if isinstance(t_max_cfg, str):
                t_max_text = t_max_cfg.strip().lower()
                if t_max_text == "max_iters":
                    t_max_value = self.refine_steps
                else:
                    try:
                        t_max_value = int(t_max_cfg)
                    except ValueError as exc:
                        raise ValueError("局部精调 scheduler.T_max 只能是整数或 'max_iters'") from exc
            else:
                t_max_value = self.refine_steps if t_max_cfg is None else int(t_max_cfg)
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, t_max_value),
                eta_min=self.scheduler_eta_min,
            )
        if self.scheduler_type in {"step", "steplr"}:
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.scheduler_step_size,
                gamma=self.scheduler_gamma,
            )
        raise ValueError(f"不支持的局部精调学习率调度器: {self.scheduler_type}")

    def _to_latent(self, actions_phys: torch.Tensor, mins: torch.Tensor, spans: torch.Tensor) -> torch.Tensor:
        """把物理动作映射到无量纲潜空间 `z`。"""
        return (actions_phys - mins.unsqueeze(0)) / spans.unsqueeze(0)

    def _from_latent(self, latent_actions: torch.Tensor, mins: torch.Tensor, spans: torch.Tensor) -> torch.Tensor:
        """把潜空间变量 `z` 还原回物理动作。"""
        return mins.unsqueeze(0) + latent_actions * spans.unsqueeze(0)

    def _resolve_initial_actions(self, context_params: torch.Tensor, pulse_norm: torch.Tensor) -> torch.Tensor:
        """确定局部精调的起点动作。

        作用：
        - 在开启 `direct_inference` 时，以策略网络直推解作为起点；
        - 否则，以 `param_space.yaml` 的 default 动作为起点。

        注意：
        - 返回值仍处于物理尺度下；
        - 后续会再经过一次 strict 投影，确保真正进入优化循环的起点绝对合法。
        """
        batch_size = context_params.shape[0]
        device = context_params.device
        if self.direct_inference:
            if self.strategy_net is None:
                raise ValueError("direct_inference=True 但未提供 strategy_net")
            self.strategy_net.eval()
            with torch.no_grad():
                # 返回的 projected_actions 已经是严格合法的动作解，且是原始物理尺度。
                projected_actions, _ = self.strategy_net(context_params, pulse_norm)
                return projected_actions
        return self.param_manager.get_trainable_defaults_tensor(device=device).unsqueeze(0).expand(batch_size, -1)

    def _strict_project_actions(self, context_params: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """对候选动作执行严格前向投影，并复核输出端合法性。

        作用：
        - 把当前 `context + actions` 投影回严格合法域；
        - 在 strict 投影后再做一次布尔校验，防止约束实现遗漏。

        输出：
        - 返回投影后的 trainable control，仍为物理尺度。
        """
        with torch.no_grad():
            full_features = self.constraint_engine.compose_full_features(context_params, actions)
            projected_full = self.constraint_engine.project_forward(full_features, strict=True)
            valid_mask = self.constraint_engine.validate_output_solution(projected_full)
            if not bool(valid_mask.all().item()):
                raise RuntimeError("strict 投影后的最终动作仍存在非法样本，请检查约束实现")
            _, projected_actions = self.constraint_engine.split_from_full(projected_full)
        return projected_actions

    def _build_trace_slots(
        self,
        batch_size: int,
        case_ids: Optional[List[str]],
        row_indices: Optional[List[int]],
        trace_case_ids: Optional[set[str]],
    ) -> List[Dict[str, Any]]:
        """为当前 batch 中需要导出 trace 的 case 预建记录槽位。

        作用：
        - 从当前 batch 中筛出命中 `trace_case_ids` 的样本；
        - 为每个命中样本准备一个固定槽位，后续迭代记录都追加到同一个 `rows` 列表里。

        输出：
        - 返回 `list[dict]`，每个元素对应一个 trace case，包含：
          `local_index / case_id / row_index / rows`。

        注意：
        - 这里不复制 case 数据，只建立轨迹容器；
        - 槽位顺序与当前 batch 中命中的 case 顺序一致。
        """
        if not trace_case_ids:
            return []
        if case_ids is None or row_indices is None:
            raise ValueError("trace_case_ids 非空时，optimize 必须显式提供 case_ids 和 row_indices")
        if len(case_ids) != batch_size or len(row_indices) != batch_size:
            raise ValueError("case_ids / row_indices 的长度必须与当前 batch 大小一致")
        slots: List[Dict[str, Any]] = []
        for local_index, (case_id, row_index) in enumerate(zip(case_ids, row_indices)):
            case_id_text = str(case_id)
            if case_id_text in trace_case_ids:
                # 每个槽位只创建一次；后续所有迭代记录都追加到同一个 `rows` 列表中。
                slots.append(
                    {
                        "local_index": int(local_index),
                        "case_id": case_id_text,
                        "row_index": int(row_index),
                        "rows": [],
                    }
                )
        return slots

    def _append_trace_rows(
        self,
        trace_slots: List[Dict[str, Any]],
        actions: torch.Tensor,
        total_loss: torch.Tensor,
        loss_info: Dict[str, torch.Tensor],
        *,
        stage: str,
        record_type: str,
        iteration: int,
        step_lr: Optional[float],
        max_abs_latent_update_ratio: Optional[torch.Tensor] = None,
        convergence_steps: Optional[torch.Tensor] = None,
    ) -> None:
        """把一组与 `trace_slots` 对齐的逐 case 结果追加到轨迹表。

        作用：
        - 将当前阶段、当前迭代步的动作值、损失分项、学习率和收敛状态写入 trace；
        - 每次调用追加一行，不覆盖之前已经记录的迭代历史。

        输入约束：
        - `actions`、`total_loss`、`loss_info` 必须已经只保留 trace case，且与
          `trace_slots` 的顺序严格一致。

        输出：
        - 无返回值；结果直接原地写入 `trace_slots[*]["rows"]`。
        """
        if not trace_slots:
            return

        actions_cpu = actions.detach().cpu()
        total_loss_cpu = total_loss.detach().cpu()
        loss_risk_cpu = loss_info["loss_risk"].detach().cpu()
        loss_constraint_cpu = loss_info["loss_constraint"].detach().cpu()
        loss_distribution_cpu = loss_info["loss_distribution"].detach().cpu()
        update_cpu = None if max_abs_latent_update_ratio is None else max_abs_latent_update_ratio.detach().cpu()
        convergence_cpu = None if convergence_steps is None else convergence_steps.detach().cpu()

        for trace_index, slot in enumerate(trace_slots):
            row = {
                "case_id": slot["case_id"],
                "row_index": slot["row_index"],
                "stage": stage,
                "record_type": record_type,
                "iteration": int(iteration),
                "step_lr": None if step_lr is None else float(step_lr),
                "loss_total": float(total_loss_cpu[trace_index].item()),
                "loss_risk": float(loss_risk_cpu[trace_index].item()),
                "loss_constraint": float(loss_constraint_cpu[trace_index].item()),
                "loss_distribution": float(loss_distribution_cpu[trace_index].item()),
                "max_abs_latent_update_ratio": None if update_cpu is None else float(update_cpu[trace_index].item()),
                "convergence_step": None,
            }
            if convergence_cpu is not None and torch.isfinite(convergence_cpu[trace_index]):
                row["convergence_step"] = int(convergence_cpu[trace_index].item())
            for param_index, param_name in enumerate(self.trainable_names):
                row[param_name] = float(actions_cpu[trace_index, param_index].item())
            slot["rows"].append(row)

    def optimize(
        self,
        context_params: torch.Tensor,
        pulse_norm: torch.Tensor,
        case_ids: Optional[List[str]] = None,
        row_indices: Optional[List[int]] = None,
        trace_case_ids: Optional[set[str]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """执行 `Opt1/Opt2` 优化，并返回最终动作与阶段信息。

        作用：
        - 先确定局部精调起点：`Opt1` 直推解或 default 动作；
        - 若启用 refine，则在潜空间 `z` 中执行逐 case Adam 优化；
        - 输出最终严格合法的动作、预测值以及阶段级附加信息。

        输入：
        - `context_params`: 已通过输入端校验的 `context` 张量。
        - `pulse_norm`: 与 `context_params` 对齐的归一化波形，由调用方显式传入。
        - `case_ids / row_indices`: 仅在导出指定 case 轨迹时使用。
        - `trace_case_ids`: 需要导出局部精调过程表的 case 标识集合。

        输出：
        - `final_actions`: 最终严格合法的 trainable control。
        - `final_preds`: `final_actions` 对应的损伤预测。
        - `info`: 阶段结果、损失分项、收敛步数和 trace 等附加信息。

        关键注意点：
        - `pulse_norm` 由调用方显式传入，保证训练、评估、局部精调三条链路共用同一份波形语义。
        - 局部精调虽然按 batch 并行执行，但收敛判定和 trace 记录都是逐 case 维护的。
        """
        self.surrogate.eval()

        batch_size = context_params.shape[0]
        # trace_slots 只记录当前 batch 中用户显式要求导出轨迹的 case。
        trace_slots = self._build_trace_slots(
            batch_size=batch_size,
            case_ids=case_ids,
            row_indices=row_indices,
            trace_case_ids=trace_case_ids,
        )
        trace_index_tensor = None
        if trace_slots:
            trace_index_tensor = torch.tensor(
                [slot["local_index"] for slot in trace_slots],
                dtype=torch.long,
                device=context_params.device,
            )

        # 先确定起点，再做一次 strict 投影，保证进入后续阶段的初始动作绝对合法。
        init_actions = self._resolve_initial_actions(context_params, pulse_norm)
        # 即使 default 或直推原解存在边界问题，这一步也会把起点收回合法域或直接报错。
        init_actions = self._strict_project_actions(context_params, init_actions)
        init_loss_batch, init_preds, init_info = self.surrogate.predict_injury_and_loss(
            context_params=context_params,
            control_trainable=init_actions,
            pulse_norm=pulse_norm,
            include_opt_bounds=False,
        )

        direct_stage = None
        # 开启策略直推时，起点动作本身就是 `Opt1` 结果。
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
            if trace_slots:
                traced_info = {
                    "loss_risk": init_info["loss_risk"].index_select(0, trace_index_tensor),
                    "loss_constraint": init_info["loss_constraint"].index_select(0, trace_index_tensor),
                    "loss_distribution": init_info["loss_distribution"].index_select(0, trace_index_tensor),
                }
                self._append_trace_rows(
                    trace_slots=trace_slots,
                    actions=init_actions.index_select(0, trace_index_tensor),
                    total_loss=init_loss_batch.index_select(0, trace_index_tensor),
                    loss_info=traced_info,
                    stage="Opt1",
                    record_type="direct_inference",
                    iteration=0,
                    step_lr=None,
                )

        if self.refine_steps <= 0:
            # 未启用 refine 时，初始严格合法解就是最终输出。
            result = {
                "final_loss_batch": init_loss_batch.detach(),
                "direct_stage": direct_stage,
                "refine_stage_enabled": False,
                "time_cost": 0.0,
                "convergence_steps": None,
                "trace_records": {
                    slot["row_index"]: {
                        "case_id": slot["case_id"],
                        "row_index": slot["row_index"],
                        "rows": slot["rows"],
                    }
                    for slot in trace_slots
                },
            }
            result.update(init_info)
            return init_actions.detach(), init_preds.detach(), result

        min_bounds = self._trainable_mins_cpu.to(device=context_params.device, dtype=context_params.dtype)
        max_bounds = self._trainable_maxs_cpu.to(device=context_params.device, dtype=context_params.dtype)
        spans = torch.clamp(max_bounds - min_bounds, min=1e-6)
        latent_var = self._to_latent(init_actions, mins=min_bounds, spans=spans).detach().requires_grad_(True)

        # 局部精调优化对象是每个样本的潜变量 `z`，不是模型权重，因此不使用 weight decay。
        optimizer = torch.optim.Adam([latent_var], lr=self.lr, weight_decay=0.0)
        scheduler = self._build_refine_scheduler(optimizer)

        convergence_steps = torch.full((batch_size,), float("nan"), device=context_params.device, dtype=torch.float32)  # 逐 case 收敛步数；NaN 表示尚未收敛
        convergence_streak = torch.zeros((batch_size,), device=context_params.device, dtype=torch.int64)  # 逐 case 连续稳定步数
        prev_total_batch = None

        if trace_slots:
            traced_info = {
                "loss_risk": init_info["loss_risk"].index_select(0, trace_index_tensor),
                "loss_constraint": init_info["loss_constraint"].index_select(0, trace_index_tensor),
                "loss_distribution": init_info["loss_distribution"].index_select(0, trace_index_tensor),
            }
            self._append_trace_rows(
                trace_slots=trace_slots,
                actions=init_actions.index_select(0, trace_index_tensor),
                total_loss=init_loss_batch.index_select(0, trace_index_tensor),
                loss_info=traced_info,
                stage="Opt2",
                record_type="init",
                iteration=0,
                step_lr=None,
            )

        start = time.time()
        relaxed_margin = 0.1
        relaxed_lower = -relaxed_margin
        relaxed_upper = 1.0 + relaxed_margin

        # 每一步都遵循“潜变量 -> 物理动作 -> 前向投影 -> surrogate 评估 -> 反传更新”的固定链路。
        for step_index in range(1, self.refine_steps + 1):
            # 当前步学习率会写入 trace，便于回看 scheduler 的实际行为。
            current_lr = float(optimizer.param_groups[0]["lr"])
            optimizer.zero_grad()
            # 保留更新前的潜变量，用于衡量当前步实际更新幅度。
            prev_latent = latent_var.detach().clone()

            # `latent_var` 是优化器真正更新的变量；`actions_phys` 是其对应的物理尺度候选解。
            actions_phys = self._from_latent(latent_var, mins=min_bounds, spans=spans)
            # `full_raw` 保留投影前的原始游离解，供软惩罚直接作用。
            full_raw = self.constraint_engine.compose_full_features(context_params, actions_phys)
            # surrogate 前向只喂合法解，因此先投影再拆回 trainable control。
            projected_full = self.constraint_engine.project_forward(full_raw, strict=False)
            _, projected_actions = self.constraint_engine.split_from_full(projected_full)

            # `risk_info` 针对投影后的合法解；约束软惩罚仍针对投影前原始解。
            _, _, risk_info = self.surrogate.predict_injury_and_loss(
                context_params=context_params,
                control_trainable=projected_actions,
                pulse_norm=pulse_norm,
                include_opt_bounds=False,
                detach_info=False,
            )
            loss_risk = risk_info["loss_risk"]
            # `soft_penalty` 处理显式可行域；`dist_penalty` 处理经验分布偏离。
            soft_penalty = self.constraint_engine.compute_soft_penalty(full_raw, include_opt_bounds=True)
            dist_penalty = self.surrogate.distribution_penalty.compute(context_params, projected_actions)
            total_batch = (
                loss_risk
                + self.surrogate.weight_penalty * soft_penalty
                + self.surrogate.weight_distribution * dist_penalty
            )

            # `total_batch` 是逐样本损失向量；反传前先求和成标量。
            total_batch.sum().backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            with torch.no_grad():
                # 给 `z` 保留一小段松弛带，避免过早贴边后完全失去更新空间。
                latent_var.clamp_(relaxed_lower, relaxed_upper)

            # 逐 case 取“最大单维潜变量改变量”，作为当前步更新幅度指标。
            latent_update = (latent_var.detach() - prev_latent).abs().amax(dim=1)
            if self.convergence_enabled and prev_total_batch is not None:
                # 收敛判定同时约束更新幅度和 loss 变化幅度，并要求连续若干步稳定。
                loss_change = (total_batch.detach() - prev_total_batch).abs()
                stable_mask = (
                    (latent_update <= self.convergence_action_tol_ratio)
                    & (loss_change <= self.convergence_loss_tol)
                )
                convergence_streak = torch.where(
                    stable_mask,
                    convergence_streak + 1,
                    torch.zeros_like(convergence_streak),
                )
                # 只给首次满足 patience 的样本写入收敛步数。
                newly_converged = torch.isnan(convergence_steps) & (convergence_streak >= self.convergence_patience)
                convergence_steps[newly_converged] = float(step_index)
            prev_total_batch = total_batch.detach()
            if trace_slots:
                with torch.no_grad():
                    # trace 只重算被指定 case，避免为整个 batch 重复组装明细。
                    traced_context = context_params.index_select(0, trace_index_tensor) 
                    traced_pulse = pulse_norm.index_select(0, trace_index_tensor)
                    traced_latent = latent_var.index_select(0, trace_index_tensor)
                    traced_actions_phys = self._from_latent(traced_latent, mins=min_bounds, spans=spans)
                    traced_full_raw = self.constraint_engine.compose_full_features(traced_context, traced_actions_phys)
                    traced_projected_full = self.constraint_engine.project_forward(traced_full_raw, strict=False)
                    _, traced_actions = self.constraint_engine.split_from_full(traced_projected_full)
                    # trace 重新走一遍代理前向，是为了把各分项 loss 一并落盘。
                    traced_total, _, traced_info = self.surrogate.predict_injury_and_loss(
                        context_params=traced_context,
                        control_trainable=traced_actions,
                        pulse_norm=traced_pulse,
                        include_opt_bounds=True,
                        penalty_features=traced_full_raw,
                    )
                    traced_updates = latent_update.index_select(0, trace_index_tensor)
                    traced_convergence = convergence_steps.index_select(0, trace_index_tensor)
                self._append_trace_rows(
                    trace_slots=trace_slots,
                    actions=traced_actions,
                    total_loss=traced_total,
                    loss_info=traced_info,
                    stage="Opt2",
                    record_type="iter",
                    iteration=step_index,
                    step_lr=current_lr,
                    max_abs_latent_update_ratio=traced_updates,
                    convergence_steps=traced_convergence,
                )

        end = time.time()
        # 最终输出再做一次 strict 投影，确保落盘结果使用的是绝对合法解。
        final_actions_phys = self._from_latent(latent_var.detach(), mins=min_bounds, spans=spans)
        final_actions = self._strict_project_actions(context_params, final_actions_phys)
        final_loss, final_preds, final_info = self.surrogate.predict_injury_and_loss(
            context_params=context_params,
            control_trainable=final_actions,
            pulse_norm=pulse_norm,
            include_opt_bounds=False,
        )

        if trace_slots:
            traced_final_info = {
                "loss_risk": final_info["loss_risk"].index_select(0, trace_index_tensor),
                "loss_constraint": final_info["loss_constraint"].index_select(0, trace_index_tensor),
                "loss_distribution": final_info["loss_distribution"].index_select(0, trace_index_tensor),
            }
            self._append_trace_rows(
                trace_slots=trace_slots,
                actions=final_actions.index_select(0, trace_index_tensor),
                total_loss=final_loss.index_select(0, trace_index_tensor),
                loss_info=traced_final_info,
                stage="Opt2",
                record_type="final_strict",
                iteration=self.refine_steps,
                step_lr=None,
                convergence_steps=convergence_steps.index_select(0, trace_index_tensor),
            )

        final_info["final_loss_batch"] = final_loss.detach()
        final_info["direct_stage"] = direct_stage
        final_info["refine_stage_enabled"] = True
        final_info["time_cost"] = end - start
        final_info["convergence_steps"] = convergence_steps.detach()
        final_info["trace_records"] = {
            slot["row_index"]: {
                "case_id": slot["case_id"],
                "row_index": slot["row_index"],
                "rows": slot["rows"],
            }
            for slot in trace_slots
        }
        
        return final_actions.detach(), final_preds.detach(), final_info
