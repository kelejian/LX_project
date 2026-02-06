"""统一数据归一化处理器模块。

提供 UnifiedDataProcessor 类，支持连续/离散特征归一化及波形缩放，
所有归一化参数从 JSON 配置文件加载，确保三个子项目统一使用。

设计原则:
    1. 所有归一化参数从 data/normalization_config.json 加载，不依赖 .joblib
    2. 所有方法支持 inverse 参数，实现归一化/反归一化双向转换
    3. process_by_name 为通用入口，支持任意特征子集的处理（适配 ARS_optim 项目的可能需求）
"""
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List, Union

from common.settings import NORMALIZATION_CONFIG_PATH
from common.settings import (
    FEATURE_ORDER, CONTINUOUS_INDICES, DISCRETE_INDICES,
    DISCRETE_VALUE_TO_INDEX, MAXABS_INDICES_IN_CONTINUOUS, MINMAX_INDICES_IN_CONTINUOUS,
)

# 模块级常量：特征名映射
FEATURE_NAME_TO_INDEX = {name: idx for idx, name in enumerate(FEATURE_ORDER)}
CONTINUOUS_FEATURE_NAMES = [FEATURE_ORDER[i] for i in CONTINUOUS_INDICES]
DISCRETE_FEATURE_NAMES = [FEATURE_ORDER[i] for i in DISCRETE_INDICES]

class UnifiedDataProcessor:
    """统一数据归一化处理器，支持连续/离散特征归一化及波形缩放。
    
    核心方法:
        - process_waveform(data, inverse): 波形归一化/反归一化
        - process_continuous(data, feature_names, inverse): 连续特征归一化/反归一化
        - process_discrete(data, feature_names, inverse): 离散特征编码/解码
        - process_by_name(values, feature_names, inverse): 通用接口，按特征名处理任意子集
        - process_all_features(x_raw, inverse): 处理完整的FEATURE_ORDER中的特征
    
    使用示例:
        >>> processor = UnifiedDataProcessor()
        >>> processor.load_config()
        >>> 
        >>> # 波形归一化
        >>> waveform_norm = processor.process_waveform(waveform_raw, inverse=False)
        >>> 
        >>> # 按特征名处理任意子集（适配 ARS_optim 策略网络）
        >>> values = np.array([[50.0, 30.0, 3.0]])  # [velocity, angle, LL1]
        >>> names = ["impact_velocity", "impact_angle", "LL1"]
        >>> normalized = processor.process_by_name(values, names, inverse=False)
    """
    
    def __init__(self, config_path: Union[str, Path] = NORMALIZATION_CONFIG_PATH):
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        
        # 特征数量（从 settings.py 动态推导）
        self.n_features = len(FEATURE_ORDER)
        self.n_continuous = len(CONTINUOUS_INDICES)
        self.n_discrete = len(DISCRETE_INDICES)
        
        # 离散特征元信息
        self.discrete_feature_names = DISCRETE_FEATURE_NAMES
        self.discrete_allowed_values = {
            name: {int(k) for k in DISCRETE_VALUE_TO_INDEX[name].keys()}
            for name in self.discrete_feature_names
        }
        self.discrete_num_classes = {
            name: len(DISCRETE_VALUE_TO_INDEX[name])
            for name in self.discrete_feature_names
        }
        
        # 缓存：加载配置后构建的查找表（避免重复解析）
        self._minmax_params: Dict[str, Tuple[float, float]] = {}  # name -> (min, max)
        self._maxabs_params: Dict[str, float] = {}  # name -> abs_max
        self._discrete_mappings: Dict[str, Dict[int, int]] = {}  # name -> {value: index}
        self._discrete_inv_mappings: Dict[str, Dict[int, int]] = {}  # name -> {index: value}

    # ========================================================================
    # 配置管理
    # ========================================================================
    
    def load_config(self) -> bool:
        """从 JSON 加载归一化配置并构建内部查找表。"""
        if not self.config_path.exists():
            print(f"[Processor] 配置文件不存在: {self.config_path}")
            return False
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            self._build_lookup_tables()
            return True
        except Exception as e:
            print(f"[Processor] 配置文件加载错误 {self.config_path}: {e}")
            return False
    
    def _build_lookup_tables(self) -> None:
        """根据加载的配置构建内部查找表，加速归一化计算。"""
        # MinMax 参数
        minmax_stats = self.config.get("continuous", {}).get("minmax", {}).get("stats", {})
        for name, stats in minmax_stats.items():
            self._minmax_params[name] = (stats["min"], stats["max"])
        
        # MaxAbs 参数
        maxabs_stats = self.config.get("continuous", {}).get("maxabs", {}).get("stats", {})
        for name, stats in maxabs_stats.items():
            self._maxabs_params[name] = stats["abs_max"]
        
        # 离散特征映射
        discrete_cfg = self.config.get("discrete", {})
        for name in self.discrete_feature_names:
            if name in discrete_cfg:
                v2i = discrete_cfg[name].get("mapping", {}).get("value_to_index", {})
                self._discrete_mappings[name] = {int(k): v for k, v in v2i.items()}
                self._discrete_inv_mappings[name] = {v: int(k) for k, v in v2i.items()}

    def save_config(self, force_overwrite: bool = False) -> bool:
        """保存配置到 JSON 文件。"""
        if self.config_path.exists() and not force_overwrite:
            print(f"[Processor] 配置文件已存在，跳过: {self.config_path}")
            return False
        
        self.config.setdefault('meta', {})
        self.config['meta']['updated_at'] = datetime.now().isoformat()
        self.config['meta'].setdefault('source', "UnifiedDataProcessor.fit()")
        
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=4, ensure_ascii=False)
        print(f"[Processor] ✅ 配置已保存至 {self.config_path}")
        return True

    def validate_config(self, raise_on_error: bool = True) -> bool:
        """校验配置结构是否完整。"""
        required = ["meta", "feature_order", "continuous", "discrete", "waveform"]
        missing = [k for k in required if k not in self.config]
        if missing:
            if raise_on_error:
                raise ValueError(f"归一化配置缺少字段: {missing}")
            return False
        if self.config.get("feature_order") != FEATURE_ORDER:
            if raise_on_error:
                raise ValueError("归一化配置 feature_order 与 settings.py 不一致")
            return False
        return True

    def _ensure_config(self) -> None:
        """确保配置已加载，若未加载则自动加载。"""
        if not self.config:
            if not self.load_config():
                raise RuntimeError(f"无法加载归一化配置: {self.config_path}")

    # ========================================================================
    # 核心归一化/反归一化方法
    # ========================================================================
    
    def process_waveform(
        self,
        data: np.ndarray,
        inverse: bool = False
    ) -> np.ndarray:
        """波形归一化/反归一化。
        
        归一化: data / scale_factor
        反归一化: data * scale_factor
        
        Args:
            data: 波形数组，任意形状（如 [N, C, T] 或 [N, T]）
            inverse: False=归一化, True=反归一化
        
        Returns:
            处理后的波形数组，形状与输入相同
        """
        self._ensure_config()
        scale_factor = self.config.get("waveform", {}).get("scale_factor", 1.0)
        
        if scale_factor < 1e-9:
            scale_factor = 1.0  # 防止除零
        
        if inverse:
            return data * scale_factor
        else:
            return data / scale_factor
    
    def process_continuous(
        self,
        data: np.ndarray,
        feature_names: Optional[List[str]] = None,
        inverse: bool = False
    ) -> np.ndarray:
        """连续特征归一化/反归一化。
        
        MinMax 特征: x' = (x - min) / (max - min) → [0, 1]
        MaxAbs 特征: x' = x / abs_max → [-1, 1]
        
        Args:
            data: 连续特征数组，形状 [N, D] 或 [D,]
            feature_names: 特征名列表，长度必须等于 D。若为 None，则使用全部 n_continuous 个连续特征
            inverse: False=归一化, True=反归一化
        
        Returns:
            处理后的数组，形状与输入相同
        """
        self._ensure_config()
        
        data = np.atleast_2d(np.asarray(data, dtype=np.float64))
        n_samples, n_cols = data.shape
        
        if feature_names is None:
            feature_names = CONTINUOUS_FEATURE_NAMES
        
        if len(feature_names) != n_cols:
            raise ValueError(f"feature_names 长度 ({len(feature_names)}) 与数据列数 ({n_cols}) 不匹配")
        
        result = data.copy()
        
        for col_idx, name in enumerate(feature_names):
            if name in self._minmax_params:
                min_val, max_val = self._minmax_params[name]
                range_val = max_val - min_val
                if range_val < 1e-9:
                    range_val = 1.0  # 防止除零（如 SH 的 min=max=0）
                if inverse:
                    # 反归一化: x = x' * range + min
                    result[:, col_idx] = result[:, col_idx] * range_val + min_val
                else:
                    # 归一化: x' = (x - min) / range
                    result[:, col_idx] = (result[:, col_idx] - min_val) / range_val
            
            elif name in self._maxabs_params:
                abs_max = self._maxabs_params[name]
                if abs_max < 1e-9:
                    abs_max = 1.0
                if inverse:
                    result[:, col_idx] = result[:, col_idx] * abs_max
                else:
                    result[:, col_idx] = result[:, col_idx] / abs_max
            
            else:
                # 未知特征名：不处理，保持原值（并发出警告）
                print(f"[Processor] 警告: 连续特征 '{name}' 未在配置中找到归一化参数，保持原值")
        
        return result.squeeze() if n_samples == 1 and data.ndim == 1 else result
    
    def process_discrete(
        self,
        data: np.ndarray,
        feature_names: Optional[List[str]] = None,
        inverse: bool = False
    ) -> np.ndarray:
        """离散特征编码/解码。
        
        编码: 原始值 → 索引（如 OT: 1→0, 2→1, 3→2）
        解码: 索引 → 原始值
        
        Args:
            data: 离散特征数组，形状 [N, D] 或 [D,]，整数类型
            feature_names: 特征名列表，长度必须等于 D。若为 None，则使用全部离散特征
            inverse: False=编码(value→index), True=解码(index→value)
        
        Returns:
            处理后的整数数组，形状与输入相同
        """
        self._ensure_config()
        
        data = np.atleast_2d(np.asarray(data, dtype=np.int64))
        n_samples, n_cols = data.shape
        
        if feature_names is None:
            feature_names = DISCRETE_FEATURE_NAMES
        
        if len(feature_names) != n_cols:
            raise ValueError(f"feature_names 长度 ({len(feature_names)}) 与数据列数 ({n_cols}) 不匹配")
        
        result = data.copy()
        
        for col_idx, name in enumerate(feature_names):
            if name not in self._discrete_mappings:
                print(f"[Processor] 警告: 离散特征 '{name}' 未在配置中找到映射，保持原值")
                continue
            
            mapping = self._discrete_inv_mappings[name] if inverse else self._discrete_mappings[name]
            
            for row_idx in range(n_samples):
                val = int(result[row_idx, col_idx])
                if val in mapping:
                    result[row_idx, col_idx] = mapping[val]
                else:
                    raise ValueError(f"离散特征 '{name}' 存在非法值: {val}，允许值: {list(mapping.keys())}")
        
        return result.squeeze() if n_samples == 1 and data.ndim == 1 else result
    
    def process_by_name(
        self,
        values: np.ndarray,
        feature_names: List[str],
        inverse: bool = False
    ) -> np.ndarray:
        """通用接口：按特征名处理任意子集（自动分发到连续/离散处理）。
        
        此方法适用于 ARS_optim 策略网络输出的任意参数组合，无需预设固定的特征子集。
        
        Args:
            values: 特征值数组，形状 [N, D] 或 [D,]
            feature_names: 特征名列表，长度必须等于 D
            inverse: False=归一化/编码, True=反归一化/解码
        
        Returns:
            处理后的数组，形状与输入相同。
            注意：若同时包含连续和离散特征，返回 float64 类型（离散值会被转为浮点）
        """
        self._ensure_config()
        
        values = np.atleast_2d(np.asarray(values, dtype=np.float64))
        n_samples, n_cols = values.shape
        
        if len(feature_names) != n_cols:
            raise ValueError(f"feature_names 长度 ({len(feature_names)}) 与数据列数 ({n_cols}) 不匹配")
        
        result = values.copy()
        
        # 分离连续和离散特征的索引
        cont_indices = []
        cont_names = []
        disc_indices = []
        disc_names = []
        
        for idx, name in enumerate(feature_names):
            if name in CONTINUOUS_FEATURE_NAMES:
                cont_indices.append(idx)
                cont_names.append(name)
            elif name in DISCRETE_FEATURE_NAMES:
                disc_indices.append(idx)
                disc_names.append(name)
            else:
                raise ValueError(f"未知特征名: '{name}'，不在 FEATURE_ORDER 中")
        
        # 处理连续特征
        if cont_indices:
            cont_data = result[:, cont_indices]
            cont_processed = self.process_continuous(cont_data, cont_names, inverse=inverse)
            result[:, cont_indices] = cont_processed
        
        # 处理离散特征
        if disc_indices:
            disc_data = result[:, disc_indices].astype(np.int64)
            disc_processed = self.process_discrete(disc_data, disc_names, inverse=inverse)
            result[:, disc_indices] = disc_processed.astype(np.float64)
        
        return result.squeeze() if n_samples == 1 and values.ndim == 1 else result
    
    def process_all_features(
        self,
        x_raw: np.ndarray,
        inverse: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """处理完整的 n_features 维特征向量（InjuryPredict 主要使用此接口）。
        
        Args:
            x_raw: 原始特征数组，形状 [N, n_features]
            inverse: False=归一化, True=反归一化
        
        Returns:
            (x_continuous, x_discrete): 
                - x_continuous: 归一化后的连续特征 [N, n_continuous], float64
                - x_discrete: 编码后的离散特征 [N, n_discrete], int64
        """
        self._ensure_config()
        
        x_raw = np.atleast_2d(np.asarray(x_raw))
        if x_raw.shape[1] != self.n_features:
            raise ValueError(f"输入特征维度 ({x_raw.shape[1]}) 与预期 ({self.n_features}) 不匹配")
        
        # 分离连续和离散
        x_cont = x_raw[:, CONTINUOUS_INDICES].astype(np.float64)
        x_disc = x_raw[:, DISCRETE_INDICES].astype(np.int64)
        
        # 处理
        x_cont_processed = self.process_continuous(x_cont, CONTINUOUS_FEATURE_NAMES, inverse=inverse)
        x_disc_processed = self.process_discrete(x_disc, DISCRETE_FEATURE_NAMES, inverse=inverse)
        
        return x_cont_processed, x_disc_processed
    
    def get_waveform_scale_factor(self) -> float:
        """获取波形缩放因子（供外部构建可微归一化层使用）。"""
        self._ensure_config()
        return float(self.config.get("waveform", {}).get("scale_factor", 1.0))
    
    def get_continuous_params(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取连续特征的归一化参数（供构建可微归一化层使用）。
        
        Returns:
            (scale, offset): 归一化公式为 x' = (x - offset) / scale
                - scale: [n_continuous,] 数组
                - offset: [n_continuous,] 数组
            对于 MinMax 特征: scale = max - min, offset = min
            对于 MaxAbs 特征: scale = abs_max, offset = 0
        """
        self._ensure_config()
        
        scale = np.ones(self.n_continuous, dtype=np.float64)
        offset = np.zeros(self.n_continuous, dtype=np.float64)
        
        for local_idx, name in enumerate(CONTINUOUS_FEATURE_NAMES):
            if name in self._minmax_params:
                min_val, max_val = self._minmax_params[name]
                scale[local_idx] = max_val - min_val if (max_val - min_val) > 1e-9 else 1.0
                offset[local_idx] = min_val
            elif name in self._maxabs_params:
                abs_max = self._maxabs_params[name]
                scale[local_idx] = abs_max if abs_max > 1e-9 else 1.0
                offset[local_idx] = 0.0
        
        return scale, offset
    
    def get_discrete_num_classes(self) -> Dict[str, int]:
        """获取离散特征的类别数（供构建 Embedding 层使用）。"""
        return self.discrete_num_classes.copy()

    # ========================================================================
    # 归一化统计相关: 生成归一化参数配置文件或当该文件存在时仅打印统计量
    # ========================================================================

    def fit(
        self,
        dataset_dict: Dict[str, Any],
        top_k_waveform: int = 50,
        dataset_id: Optional[str] = None,
        fit_split: Optional[str] = None
    ) -> None:
        """
        根据训练集计算归一化统计量，结果存入 self.config。      
        注: 目前暂仅通过 generate_config_if_absent 调用，调用时配置文件必定不存在。
        
        Args:
            dataset_dict: 包含训练数据的字典，至少包含 'x_att_raw' 和波形键
            top_k_waveform: 计算波形缩放因子的 top-k 参数
            dataset_id: 数据集标识符，存储在配置元信息中
            fit_split: 拟合所用数据划分名称，存储在配置元信息中
        """
        # 直接初始化配置结构（目前调用时配置文件不存在，无需 load_config）
        self.config = {
            'meta': {"dataset_id": dataset_id, "fitted_on_split": fit_split},
            'feature_order': FEATURE_ORDER,
            'continuous': {},
            'discrete': {},
            'waveform': {}
        }

        # 1. 波形统计
        wave_key = 'x_acc_xy' if 'x_acc_xy' in dataset_dict else ('x_acc' if 'x_acc' in dataset_dict else None)
        if wave_key:
            flat_abs = np.abs(dataset_dict[wave_key]).flatten()
            scale_factor = float(np.mean(np.sort(flat_abs)[-top_k_waveform:])) if flat_abs.size else 1.0
            if scale_factor < 1e-6:
                scale_factor = 1.0
            self.config['waveform'] = {"method": "topk_abs_mean", "top_k": top_k_waveform, "scale_factor": scale_factor}

        # 2. 连续特征
        if 'x_att_raw' in dataset_dict:
            x_att_raw = np.asarray(dataset_dict['x_att_raw'])
            if x_att_raw.ndim != 2 or x_att_raw.shape[1] != self.n_features:
                raise ValueError(f"x_att_raw 必须是形状 (N, {self.n_features}) 的数组")
            
            cont_raw = x_att_raw[:, CONTINUOUS_INDICES].astype(float)
            cont_cfg = {
                'feature_order': CONTINUOUS_FEATURE_NAMES,
                'minmax': {'indices_in_continuous': MINMAX_INDICES_IN_CONTINUOUS, 'stats': {}},
                'maxabs': {'indices_in_continuous': MAXABS_INDICES_IN_CONTINUOUS, 'stats': {}}
            }
            
            for j in MINMAX_INDICES_IN_CONTINUOUS:
                name = FEATURE_ORDER[j]
                col = cont_raw[:, j]
                cont_cfg['minmax']['stats'][name] = {"min": float(np.min(col)), "max": float(np.max(col))}
            for j in MAXABS_INDICES_IN_CONTINUOUS:
                name = FEATURE_ORDER[j]
                col = cont_raw[:, j]
                cont_cfg['maxabs']['stats'][name] = {"abs_max": float(np.max(np.abs(col)))}
            self.config['continuous'] = cont_cfg

        # 3. 离散特征
        if 'x_att_raw' in dataset_dict:
            disc_raw = x_att_raw[:, DISCRETE_INDICES].astype(int)
            for local_idx, name in enumerate(self.discrete_feature_names):
                vals = set(np.unique(disc_raw[:, local_idx]).tolist())
                allowed = self.discrete_allowed_values[name]
                if not vals.issubset(allowed):
                    raise ValueError(f"{name} 存在非法取值: {sorted(vals - allowed)}")
            
            discrete_cfg = {"feature_order": self.discrete_feature_names}
            for local_idx, name in enumerate(self.discrete_feature_names):
                discrete_cfg[name] = {
                    "idx_in_att": DISCRETE_INDICES[local_idx],
                    "mapping": {"value_to_index": DISCRETE_VALUE_TO_INDEX[name]},
                    "stats": {"num_classes": self.discrete_num_classes[name]}
                }
            self.config['discrete'] = discrete_cfg

    def generate_config_if_absent(
        self,
        dataset_dict: Dict[str, Any],
        top_k_waveform: int = 50,
        dataset_id: Optional[str] = None,
        fit_split: Optional[str] = "train"
    ) -> bool:
        """若配置文件不存在，则生成并保存。"""
        if self.config_path.exists():
            return False
        print(f"[Processor] 配置文件不存在, 即将根据训练集生成, 且波形归一化参数top_k_waveform={top_k_waveform}")
        self.fit(dataset_dict, top_k_waveform, dataset_id, fit_split)
        return self.save_config(force_overwrite=True)  # 文件必不存在，直接写入

    # ========================================================================
    # 辅助方法
    # ========================================================================

    def print_computed_stats(self, dataset_dict: Dict[str, Any], top_k_waveform: int = 50) -> None:
        """打印当前数据的统计量（不保存）。"""
        print("="*60)
        print("[Processor] 当前数据的统计量（仅供参考）:")
        print("="*60)
        
        wave_key = 'x_acc_xy' if 'x_acc_xy' in dataset_dict else ('x_acc' if 'x_acc' in dataset_dict else None)
        if wave_key:
            flat_abs = np.abs(dataset_dict[wave_key]).flatten()
            if flat_abs.size > 0:
                print(f"  波形 scale_factor (top-{top_k_waveform}): {np.mean(np.sort(flat_abs)[-top_k_waveform:]):.4f}")
        
        if 'x_att_raw' in dataset_dict:
            x = np.asarray(dataset_dict['x_att_raw'])
            if x.ndim == 2 and x.shape[1] == self.n_features:
                cont = x[:, CONTINUOUS_INDICES].astype(float)
                print("  连续特征 (MinMax):")
                for j in MINMAX_INDICES_IN_CONTINUOUS:
                    print(f"    {FEATURE_ORDER[j]}: min={np.min(cont[:, j]):.4f}, max={np.max(cont[:, j]):.4f}")
                print("  连续特征 (MaxAbs):")
                for j in MAXABS_INDICES_IN_CONTINUOUS:
                    print(f"    {FEATURE_ORDER[j]}: abs_max={np.max(np.abs(cont[:, j])):.4f}")
                
                disc = x[:, DISCRETE_INDICES].astype(int)
                print("  离散特征分布:")
                for i, name in enumerate(self.discrete_feature_names):
                    unique, counts = np.unique(disc[:, i], return_counts=True)
                    print(f"    {name}: {dict(zip(unique.tolist(), counts.tolist()))}")
        print("="*60)

