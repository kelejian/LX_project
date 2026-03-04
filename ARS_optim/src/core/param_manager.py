import json
import yaml
import torch
import logging
from typing import List, Tuple

# 引入全局项目设置中的特征顺序常量进行硬校验
from common.settings import FEATURE_ORDER

class ParamManager:
    """
    参数空间管理器 (Parameter Space Manager) - 严格校验版
    负责解析子项目专属的 param_space.yaml，并与全局的 normalization_config.json 及 common.settings 进行严格的一致性校验。

    构造器支持两种用法：
      1. 传入文件路径：
         ParamManager('path/to/param_space.yaml', 'path/to/norm.json')
      2. 传入已解析的 dict（通常用于测试或交互式场景）
         ParamManager(param_space_dict, norm_config_path=None)
    若第二个参数省略，将使用 common.settings.NORMALIZATION_CONFIG_PATH。
    """
    def __init__(self, param_space_path_or_dict, norm_config_path: str = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.param_space_config = {}
        self.sampling_rules = {}

        # --------------- 解析参数空间 ----------------
        if isinstance(param_space_path_or_dict, dict):
            # 直接给定了加载后的 dict（适用于单元测试）
            self.param_space_config = param_space_path_or_dict
            self.param_space_raw = param_space_path_or_dict.get('parameters', [])
            self.sampling_rules = param_space_path_or_dict.get('sampling_rules', {})
        else:
            # 视为文件路径
            with open(param_space_path_or_dict, 'r', encoding='utf-8') as f:
                self.param_space_config = yaml.safe_load(f)
            self.param_space_raw = self.param_space_config['parameters']
            self.sampling_rules = self.param_space_config.get('sampling_rules', {})

        # --------------- 解析归一化配置 ----------------
        if norm_config_path is None:
            # 使用全局常量作为默认值
            from common.settings import NORMALIZATION_CONFIG_PATH
            norm_config_path = str(NORMALIZATION_CONFIG_PATH)

        with open(norm_config_path, 'r', encoding='utf-8') as f:
            self.norm_config = json.load(f)
            
        self.all_params = []
        self.state_params = []
        self.control_trainable_params = []
        self.control_fixed_params = []
        
        # 依次执行解析与极其严格的校验机制
        self._parse_parameters()
        self._validate_feature_order()
        self._validate_and_override_bounds()
        # 额外校验：每个连续变量在归一化配置中必须存在统计边界
        self._check_norm_presence()

    def _parse_parameters(self):
        """
        解析并按照 index 升序排序参数。
        严格强制校验重复索引以及必备的 default 键。
        """
        # 严格校验1：禁止重复的 index
        indices = [p.get('index') for p in self.param_space_raw]
        if len(indices) != len(set(indices)):
            raise ValueError("[致命配置错误] param_space.yaml 中存在重复的 index 设定！")

        sorted_params = sorted(self.param_space_raw, key=lambda x: x['index'])
        
        for p in sorted_params:
            self.all_params.append(p)
            role = p.get('role')
            
            # 状态参数（不可控的环境/乘员变量）
            if role == 'state':
                self.state_params.append(p)
            # 决策参数（约束系统的可控变量）
            elif role == 'control':
                trainable = p.get('trainable', False)
                if trainable:
                    self.control_trainable_params.append(p)
                else:
                    # 严格校验2：固定控制参数必须具备 default 键，严禁使用隐性默认值(如0.0)
                    if 'default' not in p:
                        raise ValueError(f"[致命配置错误] 不可调控制参数 (trainable=False) '{p['name']}' 缺失 'default' 键！")
                    self.control_fixed_params.append(p)
            else:
                raise ValueError(f"[致命配置错误] 参数 '{p['name']}' 的 role 属性异常: '{role}' (仅支持 state 或 control)")

    def _validate_feature_order(self):
        """
        严格校验 param_space.yaml 中的参数定义是否与 common.settings.FEATURE_ORDER 完全一致。
        任何名称或顺序的错位都会破坏代理模型的输入张量结构。
        """
        # 严格校验3：特征总数必须一致
        if len(self.all_params) != len(FEATURE_ORDER):
            raise ValueError(
                f"[致命特征错位] 参数数量不匹配！\n"
                f"param_space.yaml 定义了 {len(self.all_params)} 个参数，\n"
                f"但 common.settings.FEATURE_ORDER 规定为 {len(FEATURE_ORDER)} 个。"
            )
            
        for i, p in enumerate(self.all_params):
            # 严格校验4：Index 的连续性与一致性
            if p['index'] != i:
                raise ValueError(f"[致命特征错位] 参数 '{p['name']}' 的 index ({p['index']}) 与其实际排序位置 ({i}) 不符！")
            
            # 严格校验5：特征名称必须与全局常量完全吻合
            if p['name'] != FEATURE_ORDER[i]:
                raise ValueError(
                    f"[致命特征错位] 特征语义顺序不一致！\n"
                    f"param_space.yaml 中 index={i} 的特征命名为 '{p['name']}'，\n"
                    f"但 settings.FEATURE_ORDER 中严格约束该列应为 '{FEATURE_ORDER[i]}'。"
                )

    def _validate_and_override_bounds(self):
        """
        校验 param_space.yaml 与 normalization_config.json 物理边界的一致性。
        当存在差异时，抛出警告，并逻辑上强制以 param_space.yaml (局部寻优约束) 为准。
        """
        minmax_stats = self.norm_config.get("continuous", {}).get("minmax", {}).get("stats", {})
        maxabs_stats = self.norm_config.get("continuous", {}).get("maxabs", {}).get("stats", {})

        for p in self.all_params:
            if p.get("type") != "continuous":
                continue

            name = p["name"]
            p_min = p.get("min")
            p_max = p.get("max")
            
            if p_min is None or p_max is None:
                raise ValueError(f"[致命配置错误] 连续变量 '{name}' 必须在 param_space.yaml 中明确配置 'min' 和 'max' 极值。")

            norm_min, norm_max = None, None
            # 从全局 json 中提取该参数的统计边界
            if name in minmax_stats:
                norm_min = minmax_stats[name]["min"]
                norm_max = minmax_stats[name]["max"]
            elif name in maxabs_stats:
                norm_min = -maxabs_stats[name]["abs_max"]
                norm_max = maxabs_stats[name]["abs_max"]

            # 若比对发现差异（考虑浮点数精度截断，设阈值 1e-4）
            if norm_min is not None and norm_max is not None:
                if abs(p_min - norm_min) > 1e-4 or abs(p_max - norm_max) > 1e-4:
                    self.logger.warning(
                        f"[参数边界冲突] {name}: 全局归一化边界 [{norm_min}, {norm_max}] "
                        f"与 ARS寻优边界 [{p_min}, {p_max}] 不一致。将强制以 ARS寻优边界 为准。"
                    )
                    
    def get_total_feature_dim(self) -> int:
        return len(self.all_params)

    def get_state_dim(self) -> int:
        return len(self.state_params)

    def get_context_params(self) -> List[dict]:
        """
        获取上下文参数列表：state + trainable=False 的 control，按全局 index 升序。
        该顺序用于策略网络输入与经验池切片，必须稳定。
        """
        params = self.state_params + self.control_fixed_params
        return sorted(params, key=lambda x: x['index'])

    def get_context_dim(self) -> int:
        return len(self.get_context_params())

    def get_context_indices(self) -> List[int]:
        return [p['index'] for p in self.get_context_params()]

    def get_context_names(self) -> List[str]:
        return [p['name'] for p in self.get_context_params()]
        
    def get_trainable_dim(self) -> int:
        return len(self.control_trainable_params)

    def get_state_indices(self) -> List[int]:
        return [p['index'] for p in self.state_params]

    def get_control_trainable_indices(self) -> List[int]:
        return [p['index'] for p in self.control_trainable_params]

    def get_control_fixed_indices(self) -> List[int]:
        return [p['index'] for p in self.control_fixed_params]

    def get_trainable_bounds(self, device: torch.device = torch.device('cpu')) -> Tuple[torch.Tensor, torch.Tensor]:
        mins = [p['min'] for p in self.control_trainable_params]
        maxs = [p['max'] for p in self.control_trainable_params]
        return torch.tensor(mins, dtype=torch.float32, device=device), \
               torch.tensor(maxs, dtype=torch.float32, device=device)

    def get_control_fixed_defaults(self, device: torch.device = torch.device('cpu')) -> Tuple[List[int], torch.Tensor]:
        indices = [p['index'] for p in self.control_fixed_params]
        # 由于在 _parse_parameters 中已做硬校验，此处 p['default'] 必然存在，直接提取
        defaults = [p['default'] for p in self.control_fixed_params]
        return indices, torch.tensor(defaults, dtype=torch.float32, device=device)

    def get_sampling_rules(self) -> dict:
        """获取参数空间中集中定义的采样/约束规则。"""
        return self.sampling_rules

    # ------------------------------------------------------------------
    def _check_norm_presence(self):
        """
        验证每个连续参数在全局归一化配置中有对应的统计边界。
        如果发现某参数既不在 minmax 统计也不在 maxabs 统计中，则抛出异常。
        """
        cont_names = [p['name'] for p in self.all_params if p.get('type') == 'continuous']
        minmax_stats = self.norm_config.get('continuous', {}).get('minmax', {}).get('stats', {})
        maxabs_stats = self.norm_config.get('continuous', {}).get('maxabs', {}).get('stats', {})
        missing = [n for n in cont_names if n not in minmax_stats and n not in maxabs_stats]
        if missing:
            raise ValueError(f"[规范错误] 以下连续参数在归一化配置中缺失统计边界: {missing}")