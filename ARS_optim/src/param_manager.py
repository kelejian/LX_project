import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union

import torch
import yaml

from common.settings import FEATURE_ORDER, NORMALIZATION_CONFIG_PATH


class ParamManager:
    """管理 ARS_optim 的参数定义与默认值。

    这里保持两条原则：
    1. 只保留运行流程真正依赖的配置校验，避免把配置层写成过重的框架；
    2. 参数分类全部源于 param_space.yaml，训练与评估只通过这里取索引和默认值。
    """

    def __init__(self, param_space_path_or_dict: Union[str, Path, dict], norm_config_path: Union[str, Path, None] = None):
        self.logger = logging.getLogger(self.__class__.__name__)

        if isinstance(param_space_path_or_dict, dict):
            config = param_space_path_or_dict
        else:
            with open(param_space_path_or_dict, "r", encoding="utf-8") as file:
                config = yaml.safe_load(file)

        self.param_space_config = config
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
        self.params_by_index = {param["index"]: param for param in self.all_params}

        self._validate_feature_order()
        self._warn_if_bounds_conflict(norm_config_path or NORMALIZATION_CONFIG_PATH)

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

    def _warn_if_bounds_conflict(self, norm_config_path: Union[str, Path]) -> None:
        try:
            with open(norm_config_path, "r", encoding="utf-8") as file:
                norm_config = json.load(file)
        except Exception as exc:
            self.logger.warning(f"读取 normalization_config 失败，跳过边界比对: {exc}")
            return

        minmax_stats = norm_config.get("continuous", {}).get("minmax", {}).get("stats", {})
        maxabs_stats = norm_config.get("continuous", {}).get("maxabs", {}).get("stats", {})
        for param in self.all_params:
            if param.get("type") != "continuous":
                continue

            name = param["name"]
            cfg_min = float(param["min"])
            cfg_max = float(param["max"])
            ref_min = None
            ref_max = None
            if name in minmax_stats:
                ref_min = float(minmax_stats[name]["min"])
                ref_max = float(minmax_stats[name]["max"])
            elif name in maxabs_stats:
                ref_max = float(maxabs_stats[name]["abs_max"])
                ref_min = -ref_max

            if ref_min is None or ref_max is None:
                continue
            if abs(cfg_min - ref_min) > 1e-4 or abs(cfg_max - ref_max) > 1e-4:
                self.logger.warning(
                    f"参数 {name} 的 ARS 边界 [{cfg_min}, {cfg_max}] 与归一化配置 [{ref_min}, {ref_max}] 不一致，"
                    "运行时将以 ARS 边界为准。"
                )

    def get_total_feature_dim(self) -> int:
        return len(self.all_params)

    def get_state_params(self) -> List[dict]:
        return list(self.state_params)

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

    def get_control_trainable_indices(self) -> List[int]:
        return [param["index"] for param in self.control_trainable_params]

    def get_sampling_rules(self) -> dict:
        return self.sampling_rules

    def get_trainable_bounds(self, device: torch.device = torch.device("cpu")) -> Tuple[torch.Tensor, torch.Tensor]:
        mins = [float(param["min"]) for param in self.control_trainable_params]
        maxs = [float(param["max"]) for param in self.control_trainable_params]
        return (
            torch.tensor(mins, dtype=torch.float32, device=device),
            torch.tensor(maxs, dtype=torch.float32, device=device),
        )

    def get_default_feature_vector(self, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        defaults = [float(param["default"]) for param in self.all_params]
        return torch.tensor(defaults, dtype=torch.float32, device=device)

    def get_default_feature_matrix(self, batch_size: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        base = self.get_default_feature_vector(device=device)
        return base.unsqueeze(0).expand(batch_size, -1).clone()
