import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

from common.settings import NORMALIZATION_CONFIG_PATH
from common.settings import (
    FEATURE_ORDER, CONTINUOUS_INDICES, DISCRETE_INDICES,
    DISCRETE_VALUE_TO_INDEX, MAXABS_INDICES_IN_CONTINUOUS, MINMAX_INDICES_IN_CONTINUOUS,
)

class UnifiedDataProcessor:
    """统一数据归一化处理器，支持连续/离散特征归一化及波形缩放。
    
    离散特征元信息从 settings.py 的 FEATURE_ORDER, DISCRETE_INDICES, 
    DISCRETE_VALUE_TO_INDEX 动态推导。
    """
    def __init__(self, config_path=NORMALIZATION_CONFIG_PATH):
        self.config_path = Path(config_path)
        self.config = {}
        
        # 特征数量
        self.n_features = len(FEATURE_ORDER)
        self.n_continuous = len(CONTINUOUS_INDICES)
        self.n_discrete = len(DISCRETE_INDICES)
        
        # 离散特征元信息
        self.discrete_feature_names = [FEATURE_ORDER[i] for i in DISCRETE_INDICES]
        self.discrete_allowed_values = {
            name: {int(k) for k in DISCRETE_VALUE_TO_INDEX[name].keys()}
            for name in self.discrete_feature_names
        }
        self.discrete_num_classes = {
            name: len(DISCRETE_VALUE_TO_INDEX[name])
            for name in self.discrete_feature_names
        }

        
    def load_config(self):
        """从 JSON 加载归一化配置。"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
                return True
            except Exception as e:
                print(f"配置文件加载错误 {self.config_path}: {e}")
                return False
        return False

    def save_config(self, force_overwrite: bool = False) -> bool:
        """保存配置到 JSON 文件。若文件已存在且 force_overwrite=False，则拒绝保存。"""
        if self.config_path.exists() and not force_overwrite:
            print(f"[Processor] 配置文件已存在，跳过: {self.config_path}")
            return False
        
        self.config.setdefault('meta', {})
        self.config['meta'].update({
            "updated_at": datetime.now().isoformat(),
            "source": self.config['meta'].get("source", "UnifiedDataProcessor.fit() 基于训练集统计")
        })
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=4, ensure_ascii=False)
        print(f"[Processor] ✅ 配置已保存至 {self.config_path}")
        return True

    def validate_config(self, raise_on_error: bool = True) -> bool:
        """校验配置结构是否完整。"""
        required_top = ["meta", "feature_order", "continuous", "discrete", "waveform"]
        missing = [k for k in required_top if k not in self.config]
        if missing:
            if raise_on_error:
                raise ValueError(f"归一化配置缺少字段: {missing}")
            return False
        if self.config.get("feature_order") != FEATURE_ORDER:
            if raise_on_error:
                raise ValueError("归一化配置 feature_order 与 CrashDataset 约定不一致")
            return False
        return True

    def fit(
        self,
        dataset_dict: Dict[str, Any],
        top_k_waveform: int = 50,
        dataset_id: Optional[str] = None,
        fit_split: Optional[str] = None,
        force_update: bool = False
    ) -> None:
        """根据训练集计算归一化统计量，更新内存配置（不自动保存）。
        
        Args:
            dataset_dict: 数据字典
                - 'x_att_raw': ndarray (N, n_features) - 连续 + 离散特征
                - 'x_acc_xy' / 'x_acc': ndarray (N, C, T) [可选]
            top_k_waveform: 波形 scale_factor 计算的 top-K 值数量
            dataset_id: 数据集标识，记录到 meta
            fit_split: 数据划分标识，记录到 meta
            force_update: 是否强制覆盖已有统计量
        
        Raises:
            ValueError: x_att_raw 形状错误或离散特征取值非法
        """
        self.load_config()
        self.config.setdefault('meta', {})
        self.config.setdefault('feature_order', FEATURE_ORDER)
        self.config.setdefault('continuous', {})
        self.config.setdefault('discrete', {})
        self.config.setdefault('waveform', {})

        self.config['meta'].update({
            "dataset_id": dataset_id,
            "fitted_on_split": fit_split
        })

        self.config['feature_order'] = FEATURE_ORDER

        # 1. 波形统计
        wave_key = None
        if 'x_acc_xy' in dataset_dict:
            wave_key = 'x_acc_xy'
        elif 'x_acc' in dataset_dict:
            wave_key = 'x_acc'

        if wave_key is not None:
            waveforms = dataset_dict[wave_key]
            flat_abs = np.abs(waveforms).flatten()
            if flat_abs.size == 0:
                scale_factor = 1.0
            else:
                top_k_vals = np.sort(flat_abs)[-top_k_waveform:]
                scale_factor = float(np.mean(top_k_vals))
                if scale_factor < 1e-6:
                    scale_factor = 1.0

            existing_factor = self.config.get('waveform', {}).get('scale_factor')
            if existing_factor is None or force_update:
                self.config['waveform'] = {
                    "method": "topk_abs_mean",
                    "top_k": int(top_k_waveform),
                    "scale_factor": scale_factor
                }
        # 2. 连续特征
        if 'x_att_raw' in dataset_dict:
            x_att_raw = np.asarray(dataset_dict['x_att_raw'])
            if x_att_raw.ndim != 2 or x_att_raw.shape[1] != self.n_features:
                raise ValueError(f"x_att_raw 必须是形状 (N, {self.n_features}) 的数组")

            cont_raw = x_att_raw[:, CONTINUOUS_INDICES].astype(float)

            cont_cfg = self.config.get('continuous', {})
            cont_cfg.setdefault('feature_order', [FEATURE_ORDER[i] for i in CONTINUOUS_INDICES])
            cont_cfg.setdefault('minmax', {})
            cont_cfg.setdefault('maxabs', {})
            cont_cfg['minmax']['indices_in_continuous'] = MINMAX_INDICES_IN_CONTINUOUS
            cont_cfg['maxabs']['indices_in_continuous'] = MAXABS_INDICES_IN_CONTINUOUS

            for j in MINMAX_INDICES_IN_CONTINUOUS:
                name = FEATURE_ORDER[j]
                col = cont_raw[:, j]
                cont_cfg['minmax'].setdefault('stats', {})[name] = {
                    "min": float(np.min(col)),
                    "max": float(np.max(col))
                }

            for j in MAXABS_INDICES_IN_CONTINUOUS:
                name = FEATURE_ORDER[j]
                col = cont_raw[:, j]
                cont_cfg['maxabs'].setdefault('stats', {})[name] = {
                    "abs_max": float(np.max(np.abs(col)))
                }

            self.config['continuous'] = cont_cfg

        # 3. 离散特征
        if 'x_att_raw' in dataset_dict:
            x_att_raw = np.asarray(dataset_dict['x_att_raw'])
            disc_raw = x_att_raw[:, DISCRETE_INDICES].astype(int)
            
            # 校验fit所用数据源(一般是训练集)中是否出现了settings.py中预定义映射关系之外的离散取值, 如果有则报错
            for local_idx, name in enumerate(self.discrete_feature_names):
                vals_in_data = set(np.unique(disc_raw[:, local_idx]).tolist()) # 提取数据中实际出现的所有唯一值
                allowed_vals = self.discrete_allowed_values[name] # 获取允许的值（来自 settings.py）
                if not vals_in_data.issubset(allowed_vals): # 检查数据中的值是否都在允许范围内
                    raise ValueError(f"{name} 存在非法取值: {sorted(vals_in_data - allowed_vals)}")
                
            # 将 settings.py 中的固定映射复制到配置文件中
            discrete_cfg = {
                "feature_order": self.discrete_feature_names,
            }
            for local_idx, name in enumerate(self.discrete_feature_names):
                discrete_cfg[name] = {
                    "idx_in_att": DISCRETE_INDICES[local_idx],
                    "mapping": {"value_to_index": DISCRETE_VALUE_TO_INDEX[name]},
                    "stats": {"num_classes": self.discrete_num_classes[name]}
                }
            self.config['discrete'] = discrete_cfg

    def fit_transform(
        self,
        dataset_dict: Dict[str, Any],
        top_k_waveform: int = 50,
        dataset_id: Optional[str] = None,
        fit_split: Optional[str] = None,
        force_update: bool = False
    ) -> Dict[str, Any]:
        """先 fit 后 transform，返回转换后的数据字典。"""
        self.fit(
            dataset_dict=dataset_dict,
            top_k_waveform=top_k_waveform,
            dataset_id=dataset_id,
            fit_split=fit_split,
            force_update=force_update
        )
        return self.transform(dataset_dict)

    def generate_config_if_absent(
        self,
        dataset_dict: Dict[str, Any],
        top_k_waveform: int = 50,
        dataset_id: Optional[str] = None,
        fit_split: Optional[str] = "train"
    ) -> bool:
        """若配置文件不存在，则基于训练集生成并保存。
        
        Args:
            dataset_dict: 训练集数据字典 {'x_att_raw': (N, n_features), ...}
            top_k_waveform: 波形 scale_factor 计算参数
            dataset_id: 数据集标识
            fit_split: 数据划分标识
            
        Returns:
            bool: True=成功生成, False=文件已存在跳过
        """
        if self.config_path.exists():
            print(f"[Processor] 配置文件已存在: {self.config_path}")
            print(f"[Processor] 跳过生成，如需重新生成请手动删除该文件")
            return False
        
        # 执行 fit 计算统计量
        self.fit(
            dataset_dict=dataset_dict,
            top_k_waveform=top_k_waveform,
            dataset_id=dataset_id,
            fit_split=fit_split,
            force_update=True  # 内存中强制更新
        )
        
        # 保存到文件（此时文件不存在，save_config 会成功）
        return self.save_config(force_overwrite=False)

    def print_computed_stats(
        self,
        dataset_dict: Dict[str, Any],
        top_k_waveform: int = 50
    ) -> None:
        """计算当前数据的统计量并打印（不保存），用于配置验证和调试。
        
        Args:
            dataset_dict: {'x_att_raw': (N, n_features), ...}
            top_k_waveform: 波形统计时取 top-K 绝对值均值
        """
        print("\n" + "="*60)
        print("[Processor] 当前数据的统计量（仅供参考，不会保存）:")
        print("="*60)
        
        # 波形统计
        wave_key = 'x_acc_xy' if 'x_acc_xy' in dataset_dict else ('x_acc' if 'x_acc' in dataset_dict else None)
        if wave_key is not None:
            waveforms = dataset_dict[wave_key]
            flat_abs = np.abs(waveforms).flatten()
            if flat_abs.size > 0:
                top_k_vals = np.sort(flat_abs)[-top_k_waveform:]
                scale_factor = float(np.mean(top_k_vals))
                print(f"  波形 scale_factor (top-{top_k_waveform} abs mean): {scale_factor:.4f}")
        
        # 连续特征统计
        if 'x_att_raw' in dataset_dict:
            x_att_raw = np.asarray(dataset_dict['x_att_raw'])
            if x_att_raw.ndim == 2 and x_att_raw.shape[1] == self.n_features:
                cont_raw = x_att_raw[:, CONTINUOUS_INDICES].astype(float)
                
                print("  连续特征 (MinMax):")
                for j in MINMAX_INDICES_IN_CONTINUOUS:
                    name = FEATURE_ORDER[j]
                    col = cont_raw[:, j]
                    print(f"    {name}: min={np.min(col):.4f}, max={np.max(col):.4f}")
                
                print("  连续特征 (MaxAbs):")
                for j in MAXABS_INDICES_IN_CONTINUOUS:
                    name = FEATURE_ORDER[j]
                    col = cont_raw[:, j]
                    print(f"    {name}: abs_max={np.max(np.abs(col)):.4f}")
        # 离散特征统计
        if 'x_att_raw' in dataset_dict:
            x_att_raw = np.asarray(dataset_dict['x_att_raw'])
            if x_att_raw.ndim == 2 and x_att_raw.shape[1] == self.n_features:
                disc_raw = x_att_raw[:, DISCRETE_INDICES].astype(int)
                
                print("  离散特征 取值分布:")
                for local_idx, name in enumerate(self.discrete_feature_names):
                    col = disc_raw[:, local_idx]
                    unique, counts = np.unique(col, return_counts=True)
                    value_counts = dict(zip(unique.tolist(), counts.tolist()))
                    print(f"    {name}: {value_counts}")
        
        print("="*60)

    def transform(
        self,
        dataset_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """使用已加载的配置转换数据（推理阶段使用）。
        
        Args:
            dataset_dict: 输入数据字典
                - 'x_att_raw': ndarray (N, n_features) - 连续 + 离散特征的二维数组
                - 'x_acc_xy' / 'x_acc_xyz' / 'x_acc': ndarray (N, C, T) [可选]
        
        Returns:
            Dict[str, Any]: 转换后的数据字典
                - 'x_att_continuous': ndarray (N, n_continuous), float32
                - 'x_att_discrete': ndarray (N, n_discrete), int64
                - 波形键原样保留，值为归一化后的结果
        
        Raises:
            ValueError: 配置未加载或结构不完整
        """
        if not self.config:
            self.load_config()
        self.validate_config(raise_on_error=True)
        output = dict(dataset_dict)

        # 以 CrashDataset 的输入为主：x_att_raw -> (x_att_continuous, x_att_discrete)
        if 'x_att_raw' in dataset_dict:
            x_att_raw = np.asarray(dataset_dict['x_att_raw'])
            # 这里的2代表数组是二维，n_features代表列数
            if x_att_raw.ndim != 2 or x_att_raw.shape[1] != self.n_features:
                raise ValueError(f"x_att_raw 必须是形状 (N, {self.n_features}) 的数组")

            cont_raw = x_att_raw[:, CONTINUOUS_INDICES].astype(float)
            disc_raw = x_att_raw[:, DISCRETE_INDICES].astype(int)

            output['x_att_continuous'] = self.transform_continuous_crashdataset(cont_raw)
            output['x_att_discrete'] = self.transform_discrete_crashdataset(disc_raw)

        # 波形
        if 'x_acc_xy' in dataset_dict:
            output['x_acc_xy'] = self.transform_waveform(dataset_dict['x_acc_xy'])
        if 'x_acc_xyz' in dataset_dict:
            output['x_acc_xyz'] = self.transform_waveform(dataset_dict['x_acc_xyz'])
        if 'x_acc' in dataset_dict and 'x_acc_xy' not in dataset_dict and 'x_acc_xyz' not in dataset_dict:
            output['x_acc'] = self.transform_waveform(dataset_dict['x_acc'])

        return output

    def transform_waveform(self, waveforms: np.ndarray) -> np.ndarray:
        """对波形进行全局缩放: normalized = waveforms / scale_factor。
        
        Args:
            waveforms: ndarray (N, C, T) - N样本, C通道, T时间步
        Returns:
            ndarray (N, C, T), float32
        """
        if not self.config: self.load_config()
        scale_factor = self.config.get('waveform', {}).get('scale_factor', 1.0)
        return waveforms / scale_factor

    def get_discrete_info(self) -> Dict[str, Dict[str, Any]]:
        """提取离散特征元信息，供 Embedding 层使用。
        
        Returns:
            Dict[str, Dict]: {特征名: {num_classes}}
        """
        if not self.config: self.load_config()
        discrete = self.config.get('discrete', {})
        
        result = {}
        for name in self.discrete_feature_names:
            feat_cfg = discrete.get(name, {})
            result[name] = {
                "num_classes": feat_cfg.get('stats', {}).get(
                    'num_classes', self.discrete_num_classes[name]
                )
            }
        return result

    def transform_continuous_crashdataset(self, cont_raw: np.ndarray) -> np.ndarray:
        """对连续特征进行归一化 (MinMax + MaxAbs 混合策略)。
        
        Args:
            cont_raw: ndarray (N, n_continuous) - 原始连续特征
        Returns:
            ndarray (N, n_continuous), float32 - MinMax特征→[0,1], MaxAbs特征→[-1,1]
        """
        cfg = self.config.get('continuous', {})
        out = cont_raw.astype(np.float32).copy()

        # MinMaxScaler: (x - min) / (max - min)
        for j in cfg.get('minmax', {}).get('indices_in_continuous', MINMAX_INDICES_IN_CONTINUOUS):
            name = FEATURE_ORDER[j]
            stats = cfg.get('minmax', {}).get('stats', {}).get(name, {})
            d_min = stats.get('min')
            d_max = stats.get('max')
            if d_min is None or d_max is None:
                raise ValueError(f"连续特征 {name} 缺少min/max统计量")
            denom = (d_max - d_min)
            if abs(denom) < 1e-12:
                out[:, j] = 0.0
            else:
                out[:, j] = (out[:, j] - d_min) / denom

        # MaxAbsScaler: x / abs_max
        for j in cfg.get('maxabs', {}).get('indices_in_continuous', MAXABS_INDICES_IN_CONTINUOUS):
            name = FEATURE_ORDER[j]
            stats = cfg.get('maxabs', {}).get('stats', {}).get(name, {})
            abs_max = stats.get('abs_max')
            if abs_max is None:
                raise ValueError(f"连续特征 {name} 缺少abs_max统计量")
            if abs(abs_max) < 1e-12:
                out[:, j] = 0.0
            else:
                out[:, j] = out[:, j] / abs_max

        return out

    def transform_discrete_crashdataset(self, disc_raw: np.ndarray) -> np.ndarray:
        """对离散特征进行整数编码 (LabelEncoder)。
        
        映射规则固定在 settings.py 的 DISCRETE_VALUE_TO_INDEX。
        
        Args:
            disc_raw: ndarray (N, num_discrete_features) - 原始离散特征
        Returns:
            ndarray (N, num_discrete_features), int64 - 编码后的离散特征
        Raises:
            ValueError: 出现非法取值时
        """
        disc_cfg = self.config.get('discrete', {})
        
        def _map_vectorized(col: np.ndarray, mapping: Dict[str, int], name: str) -> np.ndarray:
            """Vectorized mapping via lookup table."""
            col_int = col.astype(np.int64)
            keys = np.array([int(k) for k in mapping.keys()], dtype=np.int64)
            vals = np.array([mapping[str(k)] for k in keys], dtype=np.int64)
            
            # 检查是否有未知值
            unique_in_col = np.unique(col_int)
            unknown = set(unique_in_col.tolist()) - set(keys.tolist())
            if unknown:
                raise ValueError(f"离散特征 {name} 出现未见值: {sorted(unknown)}")
            
            max_key = int(keys.max()) + 1
            lut = np.full(max_key, -1, dtype=np.int64)
            lut[keys] = vals
            return lut[col_int]
        
        encoded_cols = []
        for local_idx, name in enumerate(self.discrete_feature_names):
            mapping = disc_cfg.get(name, {}).get('mapping', {}).get(
                'value_to_index', DISCRETE_VALUE_TO_INDEX[name]
            )
            enc = _map_vectorized(disc_raw[:, local_idx], mapping, name)
            encoded_cols.append(enc)
        
        return np.stack(encoded_cols, axis=1).astype(np.int64)
