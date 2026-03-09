from pathlib import Path
from typing import List, Tuple, Union

import torch
import yaml

from common.settings import FEATURE_ORDER


class ParamManager:
    """管理 ARS_optim 的参数定义与默认值。

    这里只保留两类职责：
    1. 读取并校验 param_space.yaml 中真正影响运行的参数定义；
    2. 向训练、评估和约束模块提供统一的索引、默认值与角色划分。

    不在这里额外引入与运行主链路无关的配置比对或兼容逻辑，避免参数层继续膨胀。
    """

    def __init__(self, param_space_path_or_dict: Union[str, Path, dict]):
        if isinstance(param_space_path_or_dict, dict):
            config = param_space_path_or_dict
        else:
            with open(param_space_path_or_dict, "r", encoding="utf-8") as file:
                config = yaml.safe_load(file)

        self.sampling_rules = config.get("sampling_rules", {})
        self.all_params = self._parse_parameters(config.get("parameters", []))

        self.state_params = [param for param in self.all_params if param["role"] == "state"]
        self.control_trainable_params = [
            param for param in self.all_params if param["role"] == "control" and bool(param.get("trainable", False))
        ]
        self.control_fixed_params = [
            param for param in self.all_params if param["role"] == "control" and not bool(param.get("trainable", False))
        ]

        self.params_by_name = {param["name"]: param for param in self.all_params}

        self._validate_feature_order()

    def _parse_parameters(self, raw_params: List[dict]) -> List[dict]:
        if not raw_params:
            raise ValueError("param_space.yaml 缺少 parameters 配置")

        indices = [param.get("index") for param in raw_params]
        if len(indices) != len(set(indices)):
            raise ValueError("param_space.yaml 存在重复 index")

        params = sorted(raw_params, key=lambda item: item["index"])
        for param in params:
            role = param.get("role")
            if role not in {"state", "control"}:
                raise ValueError(f"参数 {param.get('name')} 的 role 非法: {role}")
            if "default" not in param:
                raise ValueError(f"参数 {param.get('name')} 缺少 default")
            if param.get("type") == "continuous" and ("min" not in param or "max" not in param):
                raise ValueError(f"连续参数 {param.get('name')} 缺少 min/max")
            if param.get("type") == "discrete":
                values = param.get("values")
                if not isinstance(values, list) or not values:
                    raise ValueError(f"离散参数 {param.get('name')} 缺少非空 values 列表")
                if param["default"] not in values:
                    raise ValueError(f"离散参数 {param.get('name')} 的 default={param['default']} 不在 values={values} 中")
        return params

    def _validate_feature_order(self) -> None:
        if len(self.all_params) != len(FEATURE_ORDER):
            raise ValueError(
                f"参数数量与 FEATURE_ORDER 不一致: {len(self.all_params)} != {len(FEATURE_ORDER)}"
            )

        for expected_index, param in enumerate(self.all_params):
            if param["index"] != expected_index:
                raise ValueError(f"参数 {param['name']} 的 index 不连续")
            if param["name"] != FEATURE_ORDER[expected_index]:
                raise ValueError(
                    f"FEATURE_ORDER 与 param_space.yaml 不一致: index={expected_index}, "
                    f"expect={FEATURE_ORDER[expected_index]}, actual={param['name']}"
                )

    def get_total_feature_dim(self) -> int:
        return len(self.all_params)

    def get_all_params(self) -> List[dict]:
        return list(self.all_params)

    def get_param(self, name: str) -> dict:
        if name not in self.params_by_name:
            raise KeyError(f"未知参数: {name}")
        return self.params_by_name[name]

    def get_discrete_index_value_map(self) -> dict[int, List[float]]:
        return {
            param["index"]: [float(value) for value in param["values"]]
            for param in self.all_params
            if param.get("type") == "discrete"
        }

    def get_context_params(self) -> List[dict]:
        params = self.state_params + self.control_fixed_params
        return sorted(params, key=lambda item: item["index"])

    def get_context_dim(self) -> int:
        return len(self.get_context_params())

    def get_context_indices(self) -> List[int]:
        return [param["index"] for param in self.get_context_params()]

    def get_context_names(self) -> List[str]:
        return [param["name"] for param in self.get_context_params()]

    def get_trainable_dim(self) -> int:
        return len(self.control_trainable_params)

    def get_trainable_params(self) -> List[dict]:
        return list(self.control_trainable_params)

    def get_trainable_names(self) -> List[str]:
        return [param["name"] for param in self.control_trainable_params]

    def get_control_trainable_indices(self) -> List[int]:
        return [param["index"] for param in self.control_trainable_params]

    def get_fixed_control_names(self) -> List[str]:
        return [param["name"] for param in self.control_fixed_params]

    def get_sampling_rules(self) -> dict:
        return self.sampling_rules

    def get_default_values(self, params: List[dict]) -> List[float]:
        return [float(param["default"]) for param in params]

    def get_trainable_bounds(self, device: torch.device = torch.device("cpu")) -> Tuple[torch.Tensor, torch.Tensor]:
        mins = [float(param["min"]) for param in self.control_trainable_params]
        maxs = [float(param["max"]) for param in self.control_trainable_params]
        return (
            torch.tensor(mins, dtype=torch.float32, device=device),
            torch.tensor(maxs, dtype=torch.float32, device=device),
        )

    def get_trainable_defaults_tensor(self, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        defaults = self.get_default_values(self.control_trainable_params)
        return torch.tensor(defaults, dtype=torch.float32, device=device)

    def get_default_feature_vector(self, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        defaults = [float(param["default"]) for param in self.all_params]
        return torch.tensor(defaults, dtype=torch.float32, device=device)

    def get_default_feature_matrix(self, batch_size: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        base = self.get_default_feature_vector(device=device)
        return base.unsqueeze(0).expand(batch_size, -1).clone()
