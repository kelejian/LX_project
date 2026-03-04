import torch
import logging
from typing import List

# 严格执行绝对路径引用规范
from ARS_optim.src.core.param_manager import ParamManager

class PhysicalConstraintManager:
    """
    物理约束管理器 (Differentiable Physical Constraint Manager)
    
    功能设计:
    本模块负责在连续参数空间中执行刚性物理法则的数学约束。通过纯张量操作（如 torch.min, torch.max）
    替代传统的控制流（if-else/原地赋值），确保约束过程全流程可微，从而允许损失函数的梯度无损穿透
    约束层，指导上层策略网络或底层决策变量进行分布优化。
    
    混合状态支持 (Mixed-State Resolution):
    系统支持任意参数的可调性组合。通过动态上下文寻址，即使约束公式中的某些参数被固化（trainable=False）
    或属于不可控的环境状态（State），系统也能自动从状态张量或默认值字典中提取有效物理值进行边界判定，
    彻底消除了“仅当所有相关参数均可调时才执行约束”的同源可调性局限与静默失效风险。
    """
    def __init__(self, param_manager: ParamManager):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.param_manager = param_manager
        self.rules = param_manager.get_sampling_rules() if hasattr(param_manager, 'get_sampling_rules') else {}
        coupling = self.rules.get('coupling', {}) if isinstance(self.rules, dict) else {}
        self.aft_btf_delta_max = float(coupling.get('aft_btf_delta_max', 25.0))
        self.epsilon = float(coupling.get('epsilon', 1e-3))
        
        # 建立参数类别映射表：用于跨越张量边界进行动态数据检索
        # 核心逻辑：记录可调参数在 actions 张量中的具体列索引，以及状态参数在 state 张量中的列索引
        self.trainable_names = {p['name']: i for i, p in enumerate(param_manager.control_trainable_params)}
        self.context_names = {p['name']: i for i, p in enumerate(param_manager.get_context_params())}
        self.fixed_defaults = {p['name']: p['default'] for p in param_manager.control_fixed_params}

    def _get_param_col(self, name: str, cols: List[torch.Tensor], context_params: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        动态张量列寻址器 (Dynamic Tensor Column Resolver)
        
        接口设计:
        根据特征名称的注册属性，动态路由并提取对应的物理列张量。
        - 若为可调参数 (Trainable)，直接从解包后的 cols 列表中提取，维持前向计算图与级联修改。
        - 若为状态参数 (State)，从输入的 state_params 中切片提取。
        - 若为固定参数 (Fixed)，则动态构建形状为 [Batch] 的常数张量。
        """
        if name in self.trainable_names:
            return cols[self.trainable_names[name]]
        elif name in self.context_names:
            return context_params[:, self.context_names[name]]
        elif name in self.fixed_defaults:
            batch_size = context_params.shape[0]
            return torch.full((batch_size,), self.fixed_defaults[name], device=device, dtype=torch.float32)
        else:
            raise ValueError(f"[物理约束异常] 试图检索未知的系统参数: {name}")

    def project_forward(self, actions: torch.Tensor, context_params: torch.Tensor) -> torch.Tensor:
        """
        前向硬投影 (Hard Projection)
        
        功能设计:
        应用可微的次梯度算子强制纠正违反物理耦合关系的参数。
        采用“对称双向投影”：对于不等式 A < B，不仅在 A 可调时限制 A，当 B 可调时也会反向限制 B。
        
        参数:
            actions: [Batch, D_trainable] 策略网络输出的绝对尺度决策参数
            context_params: [Batch, D_context] 物理尺度上下文参数（state + fixed-control）
        返回:
            projected_actions: [Batch, D_trainable] 修正后且保留梯度的决策参数
        """
        device = actions.device
        # 将 [Batch, D] 拆成列列表，后续逐列替换时更直观。
        # 注意：这里不直接对 actions 原地改列，避免复杂的梯度别名问题。
        cols = [actions[:, i] for i in range(actions.shape[1])]
        
        # =========================================================
        # 约束 1: 气囊点爆时刻 (AFT) 必须早于 预紧器点火时刻 (BTF) + 25ms
        # 数学表达: AFT <= BTF + 25 - epsilon
        # =========================================================
        aft = self._get_param_col('AFT', cols, context_params, device)
        btf = self._get_param_col('BTF', cols, context_params, device)
        
        # 正向截断：如果 AFT 可调，其上限受制于 BTF
        if 'AFT' in self.trainable_names:
            # torch.min 是逐元素可微算子，适合做“上界投影”
            cols[self.trainable_names['AFT']] = torch.min(aft, btf + self.aft_btf_delta_max - self.epsilon)
            
        # 级联更新与反向截断：当一端改为 fixed/context 时，该分支可避免另一端失控
        # 注：在双侧都可调且前向已满足约束时，这里通常是恒等操作
        aft = self._get_param_col('AFT', cols, context_params, device)
        if 'BTF' in self.trainable_names:
            # 反向投影对应“下界投影”
            cols[self.trainable_names['BTF']] = torch.max(btf, aft - self.aft_btf_delta_max + self.epsilon)

        # =========================================================
        # 约束 2: 二级限力切换时刻 (LLATTF) 必须晚于或等于 预紧器点火时刻 (BTF)
        # 数学表达: LLATTF >= BTF
        # =========================================================
        llattf = self._get_param_col('LLATTF', cols, context_params, device)
        btf = self._get_param_col('BTF', cols, context_params, device)
        
        if 'LLATTF' in self.trainable_names:
            cols[self.trainable_names['LLATTF']] = torch.max(llattf, btf)
            
        # 同上：该分支主要用于 mixed-state 场景的稳定约束闭环
        llattf = self._get_param_col('LLATTF', cols, context_params, device)
        if 'BTF' in self.trainable_names:
            cols[self.trainable_names['BTF']] = torch.min(btf, llattf)

        # =========================================================
        # 约束 3: 二级限力值 (LL2) 必须小于等于 一级限力值 (LL1)
        # 数学表达: LL2 <= LL1
        # =========================================================
        ll1 = self._get_param_col('LL1', cols, context_params, device)
        ll2 = self._get_param_col('LL2', cols, context_params, device)
        
        if 'LL2' in self.trainable_names:
            cols[self.trainable_names['LL2']] = torch.min(ll2, ll1)
            
        # 同上：保留反向分支以兼容未来 trainable 配置调整
        ll2 = self._get_param_col('LL2', cols, context_params, device)
        if 'LL1' in self.trainable_names:
            cols[self.trainable_names['LL1']] = torch.max(ll1, ll2)

        # 梯度安全重组：重新堆叠回张量 [Batch, D_trainable]
        projected_actions = torch.stack(cols, dim=1)

        # 轻量一致性检查：若投影后仍有耦合约束残差，提示存在不可满足的参数组合
        with torch.no_grad():
            residual = self.compute_soft_penalty(projected_actions, context_params)
            # 阈值 1e-6 主要用于滤掉浮点误差，避免日志噪声
            if torch.any(residual > 1e-6):
                max_residual = float(residual.max().item())
                self.logger.warning(
                    f"检测到投影后仍有约束残差（max={max_residual:.6f}）。"
                    "请检查 fixed/state 参数是否导致耦合关系不可同时满足。"
                )
        return projected_actions

    def compute_soft_penalty(self, actions: torch.Tensor, context_params: torch.Tensor) -> torch.Tensor:
        """
        软惩罚计算 (Soft Penalty for Objective Function)
        
        功能设计:
        在摊销寻优的目标函数中提供平滑的导数指引。对于超出物理边界的部分施加 ReLU 惩罚，
        驱使神经网络在训练阶段自发地将参数分布收敛至安全的物理可行域内部。
        
        参数:
            actions: [Batch, D_trainable]
            context_params: [Batch, D_context]
        返回:
            penalty: [Batch] 当前批次内各个样本的违规程度总和
        """
        device = actions.device
        batch_size = actions.shape[0]
        penalty = torch.zeros(batch_size, device=device, dtype=torch.float32)
        
        # 软惩罚阶段直接使用原始 actions，无需考虑张量原地修改问题
        cols = [actions[:, i] for i in range(actions.shape[1])]
        
        aft = self._get_param_col('AFT', cols, context_params, device)
        btf = self._get_param_col('BTF', cols, context_params, device)
        llattf = self._get_param_col('LLATTF', cols, context_params, device)
        ll1 = self._get_param_col('LL1', cols, context_params, device)
        ll2 = self._get_param_col('LL2', cols, context_params, device)
        
        # 惩罚项 1: 约束 AFT < BTF + 25 越界
        penalty += torch.relu(aft - (btf + self.aft_btf_delta_max))
        
        # 惩罚项 2: 约束 LLATTF >= BTF 越界
        penalty += torch.relu(btf - llattf)

        # 惩罚项 3: 约束 LL2 <= LL1 越界
        penalty += torch.relu(ll2 - ll1)

        return penalty