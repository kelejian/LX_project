import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

from common.settings import NORMALIZATION_CONFIG_PATH


# 与 InjuryPredict/utils/dataset_prepare.py 中 CrashDataset/DataProcessor 的约定保持一致
FEATURE_ORDER = [
    "impact_velocity", "impact_angle", "overlap",
    "LL1", "LL2", "BTF", "LLATTF", "AFT", "SP", "SH", "RA",
    "is_driver_side", "OT"
]

CONTINUOUS_INDICES_11 = list(range(11))
DISCRETE_INDICES_2 = [11, 12]

# 在连续子向量(11维)中的索引（对应 impact_angle, overlap）
MAXABS_INDICES_IN_CONTINUOUS = [1, 2]

# 在连续子向量(11维)中的索引（对应 impact_velocity + 其余8个连续特征）
MINMAX_INDICES_IN_CONTINUOUS = [0, 3, 4, 5, 6, 7, 8, 9, 10]

# 固定离散映射（等价于LabelEncoder在这些取值上的编码）
DISCRETE_VALUE_TO_INDEX = {
    "is_driver_side": {"0": 0, "1": 1},
    "OT": {"1": 0, "2": 1, "3": 2}
}
class UnifiedDataProcessor:
    """
    统一数据归一化处理器。
    支持:
    - 连续特征 (MinMax, MaxAbs)
    - 离散特征 (通过手动映射进行标签编码)
    - 波形缩放 (全局缩放因子)
    
    读写 JSON 文件以实现“白盒化”配置能力。
    """
    def __init__(self, config_path=NORMALIZATION_CONFIG_PATH):
        self.config_path = Path(config_path)
        self.config = {}
        self.stats = {
            "meta": {},
            "features": {},
            "waveform": {},
            "discrete": {}
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

    def save_config(self):
        self.config.setdefault('meta', {})
        self.config['meta'].update({
            "updated_at": datetime.now().isoformat(),
            "source": self.config['meta'].get("source", "UnifiedDataProcessor auto-stat")
        })
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=4, ensure_ascii=False)
        print(f"[Processor] 配置已保存至 {self.config_path}")

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
        feature_names=None,
        top_k_waveform: int = 50,
        dataset_id: Optional[str] = None,
        fit_split: Optional[str] = None,
        force_update: bool = False
    ):
        """
        根据数据计算统计信息并更新配置（仅拟合，不自动保存）。
        仅应传入训练集数据以避免泄漏。
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

        # 固化feature_order（全局共用，禁止漂移）
        self.config['feature_order'] = FEATURE_ORDER

        # 1. 波形统计信息（按现状：topK 绝对值均值）
        # 约定：可传入 x_acc_xy 或 x_acc（通道在前）
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
                print(f"[Processor] 计算得到的波形缩放因子: {scale_factor:.4f}")
            else:
                print(f"[Processor] 使用现有的波形缩放因子: {existing_factor:.4f} (计算值为 {scale_factor:.4f})")

        # 2. 连续特征（严格按CrashDataset：x_att_raw[:,0:11]）
        if 'x_att_raw' in dataset_dict:
            x_att_raw = np.asarray(dataset_dict['x_att_raw'])
            if x_att_raw.ndim != 2 or x_att_raw.shape[1] != 13:
                raise ValueError("x_att_raw 必须是形状 (N,13) 的数组")

            cont_raw = x_att_raw[:, CONTINUOUS_INDICES_11].astype(float)

            cont_cfg = self.config.get('continuous', {})
            cont_cfg.setdefault('feature_order_11', FEATURE_ORDER[:11])
            cont_cfg.setdefault('minmax', {})
            cont_cfg.setdefault('maxabs', {})
            cont_cfg['minmax']['indices_in_continuous'] = MINMAX_INDICES_IN_CONTINUOUS
            cont_cfg['maxabs']['indices_in_continuous'] = MAXABS_INDICES_IN_CONTINUOUS

            # minmax 统计量
            for j in MINMAX_INDICES_IN_CONTINUOUS:
                name = FEATURE_ORDER[j]
                col = cont_raw[:, j]
                cont_cfg['minmax'].setdefault('stats', {})[name] = {
                    "min": float(np.min(col)),
                    "max": float(np.max(col))
                }

            # maxabs 统计量
            for j in MAXABS_INDICES_IN_CONTINUOUS:
                name = FEATURE_ORDER[j]
                col = cont_raw[:, j]
                cont_cfg['maxabs'].setdefault('stats', {})[name] = {
                    "abs_max": float(np.max(np.abs(col)))
                }

            self.config['continuous'] = cont_cfg

        # 3. 离散特征（严格按CrashDataset：x_att_raw[:,11:13]）
        if 'x_att_raw' in dataset_dict:
            x_att_raw = np.asarray(dataset_dict['x_att_raw'])
            disc_raw = x_att_raw[:, DISCRETE_INDICES_2].astype(int)
            # 仅用于校验取值范围（映射固定，避免训练集缺类导致不可复现）
            is_driver_vals = set(np.unique(disc_raw[:, 0]).tolist())
            ot_vals = set(np.unique(disc_raw[:, 1]).tolist())
            allowed_driver = {0, 1}
            allowed_ot = {1, 2, 3}
            if not is_driver_vals.issubset(allowed_driver):
                raise ValueError(f"is_driver_side 存在非法取值: {sorted(is_driver_vals - allowed_driver)}")
            if not ot_vals.issubset(allowed_ot):
                raise ValueError(f"OT 存在非法取值: {sorted(ot_vals - allowed_ot)}")

            self.config['discrete'] = {
                "feature_order_2": ["is_driver_side", "OT"],
                "is_driver_side": {
                    "idx_in_att": 11,
                    "mapping": {
                        "value_to_index": DISCRETE_VALUE_TO_INDEX["is_driver_side"],
                        "unknown_policy": "error",
                        "unknown_index": -1
                    },
                    "stats": {"num_classes": 2}
                },
                "OT": {
                    "idx_in_att": 12,
                    "mapping": {
                        "value_to_index": DISCRETE_VALUE_TO_INDEX["OT"],
                        "unknown_policy": "error",
                        "unknown_index": -1
                    },
                    "stats": {"num_classes": 3}
                }
            }

    def fit_transform(self, dataset_dict, feature_names=None, top_k_waveform=50, dataset_id=None,
                      fit_split=None, force_update=False):
        """先拟合再转换，返回转换后的数据字典。"""
        self.fit(dataset_dict, feature_names, top_k_waveform, dataset_id, fit_split, force_update)
        return self.transform(dataset_dict, feature_names)

    def fit_and_update_config(self, dataset_dict, feature_names=None, top_k_waveform=50, save=True, force_update=False):
        """兼容旧接口：拟合并按需保存。"""
        self.fit(dataset_dict, feature_names, top_k_waveform, dataset_id=None, fit_split=None, force_update=force_update)
        if save:
            self.save_config()

    def fit_and_update_config(self, dataset_dict, feature_names=None, top_k_waveform=50, save=True, force_update=False):
        """
        根据数据计算统计信息并更新配置。
        除非 force_update=True 或配置缺失，否则不会覆盖现有配置。
        
        Args:
            dataset_dict: 包含 'x_att_continuous', 'x_att_discrete', 'x_acc_raw' (波形) 的字典。
                          预期形状: (N, D_cont), (N, D_disc), (N, C, T) 或 (N, T)。
            feature_names: 字典或列表，提供特征名称。
                           格式: {index: "name", ...} 或 ["name1", "name2", ...]
            top_k_waveform: 用于波形缩放计算的顶部绝对值的数量。
            save: 是否立即保存到磁盘。
            force_update: 如果为 True，覆盖现有配置值。
        """
        self.load_config()
        
        # 1. 波形统计信息
        if 'x_acc' in dataset_dict:
            waveforms = dataset_dict['x_acc'] # Shape (N, ...)
            flat_abs = np.abs(waveforms).flatten()
            top_k_vals = np.sort(flat_abs)[-top_k_waveform:]
            scale_factor = float(np.mean(top_k_vals))
            if scale_factor < 1e-6: scale_factor = 1.0
            
            # 检查现有配置
            existing_factor = self.config.get('waveform', {}).get('scale_factor')
            
            if existing_factor is None or force_update:
                self.config.setdefault('waveform', {})['scale_factor'] = scale_factor
                print(f"[Processor] 计算得到的波形缩放因子: {scale_factor:.4f}")
            else:
                print(f"[Processor] 使用现有的波形缩放因子: {existing_factor:.4f} (计算值为 {scale_factor:.4f})")

        # 2. 连续特征
        if 'x_att_continuous' in dataset_dict:
            cont_data = dataset_dict['x_att_continuous']
            # 假设 2D: (N, D)
            num_features = cont_data.shape[1]
            
            self.config.setdefault('features', {})
            
            for i in range(num_features):
                # 识别特征名称
                f_name = f"feat_{i}"
                if feature_names and isinstance(feature_names, dict):
                    f_name = feature_names.get(i, f_name)
                elif feature_names and isinstance(feature_names, list) and i < len(feature_names):
                    f_name = feature_names[i]
                
                # 计算统计量
                f_min = float(np.min(cont_data[:, i]))
                f_max = float(np.max(cont_data[:, i]))
                f_abs_max = float(np.max(np.abs(cont_data[:, i])))
                
                # 确定策略 (默认 MinMax 0-1, 如果需要可为 Angle/Overlap 覆盖为 MaxAbs)
                # 这里只存储统计量。策略由用户需求（配置）定义。
                # 为了方便起见，如果缺失我们可以初始化一个默认策略。
                
                feat_conf = self.config['features'].get(f_name)
                
                if feat_conf is None or force_update:
                    # 启发式规则: 如果 min < 0, 也许使用 MaxAbs? 或者坚持使用 MinMax [0,1]。
                    # PulsePredict 逻辑: Velocity -> MinMax, Angle/Overlap -> MaxAbs.
                    # 我们将存储原始统计量，用户可以验证方法。
                    new_conf = {
                        "idx": i, # 存储索引仅供参考，但依赖用户映射名称
                        "stats": {
                            "min": f_min,
                            "max": f_max,
                            "abs_max": f_abs_max
                        },
                        "method": "minmax" # 默认值
                    }
                    if "angle" in f_name.lower() or "overlap" in f_name.lower():
                        new_conf["method"] = "maxabs"
                    
                    self.config['features'][f_name] = new_conf
                    print(f"[Processor] 已初始化 {f_name} 的配置: {new_conf['stats']}")
                else:
                     # 仅更新统计量检查（可选日志记录）
                     pass

        # 3. 离散特征
        if 'x_att_discrete' in dataset_dict:
            disc_data = dataset_dict['x_att_discrete']
            num_features = disc_data.shape[1]
            
            # 对于离散特征，我们需要映射关系。
            # 假设输入已经被某种程度编码或是原始整数？
            # 如果是原始整数，LabelEncoding 本质上是恒等映射，除非非连续。
            # 为了安全起见，我们构建映射。
            
            for i in range(num_features):
                f_name = f"disc_{i}"
                if feature_names: # 如果需要处理离散名称逻辑
                     # 如果 feature_names 先包含连续特征，需要偏移量
                     pass 
                
                # 我们假设调用者分别处理离散名称，或者我们仅使用索引
                unique_vals = np.unique(disc_data[:, i])
                mapping = {str(val): int(idx) for idx, val in enumerate(unique_vals)}
                
                # 仅保存记录，尽管 transform 通常接受原始输入。
                # 如果原始输入是字符串，我们需要映射。如果是整数，我们映射 int->int (0..N)。
                # 这里我们假设我们仅记录唯一值计数。
                pass 

        # 保存
        if save:
            self.save_config()

    def transform_continuous(self, data, feature_names_list):
        """
        基于配置转换连续特征。
        Args:
            data: (N, D) 数组
            feature_names_list: 对应列 [0..D-1] 的名称列表
        """
        if not self.config: self.load_config()
        self.validate_config(raise_on_error=False)
        
        result = data.copy()
        
        for i, name in enumerate(feature_names_list):
            if name not in self.config.get('features', {}):
                continue
                
            conf = self.config['features'][name]
            method = conf.get('method', 'minmax')
            stats = conf.get('stats', {})
            
            if method == 'minmax':
                d_min = stats.get('min')
                d_max = stats.get('max')
                if d_min is not None and d_max is not None and (d_max - d_min) > 1e-9:
                    result[:, i] = (result[:, i] - d_min) / (d_max - d_min)
            elif method == 'maxabs':
                d_abs = stats.get('abs_max')
                if d_abs is not None and d_abs > 1e-9:
                    result[:, i] = result[:, i] / d_abs
            elif method == 'fixed_scale':
                scale = conf.get('params', {}).get('scale')
                if scale:
                    result[:, i] = result[:, i] / scale
                    
        return result

    def transform_discrete(self, data, feature_names_list):
        """
        基于配置转换离散特征。
        Args:
            data: (N, D) 数组
            feature_names_list: 对应列 [0..D-1] 的名称列表
        """
        if not self.config: self.load_config()
        result = data.copy()

        for i, name in enumerate(feature_names_list):
            conf = self.config.get('discrete', {}).get(name)
            if not conf:
                continue

            mapping = conf.get('mapping', {})
            value_to_index = mapping.get('value_to_index', {})
            unknown_policy = mapping.get('unknown_policy', 'error')
            unknown_index = mapping.get('unknown_index', -1)

            def _map_value(v):
                key = str(v)
                if key in value_to_index:
                    return value_to_index[key]
                if unknown_policy == 'keep':
                    return v
                if unknown_policy == 'unk':
                    return unknown_index
                raise ValueError(f"离散特征 {name} 出现未见值: {v}")

            result[:, i] = np.vectorize(_map_value)(result[:, i])
        return result

    def transform(self, dataset_dict, feature_names=None):
        """统一转换入口，返回新的数据字典。"""
        if not self.config:
            self.load_config()
        self.validate_config(raise_on_error=True)
        output = dict(dataset_dict)

        # 以 CrashDataset 的输入为主：x_att_raw -> (x_att_continuous, x_att_discrete)
        if 'x_att_raw' in dataset_dict:
            x_att_raw = np.asarray(dataset_dict['x_att_raw'])
            if x_att_raw.ndim != 2 or x_att_raw.shape[1] != 13:
                raise ValueError("x_att_raw 必须是形状 (N,13) 的数组")

            cont_raw = x_att_raw[:, CONTINUOUS_INDICES_11].astype(float)
            disc_raw = x_att_raw[:, DISCRETE_INDICES_2].astype(int)

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

    def transform_waveform(self, waveforms):
        """使用全局缩放因子转换波形。"""
        if not self.config: self.load_config()
        scale_factor = self.config.get('waveform', {}).get('scale_factor', 1.0)
        return waveforms / scale_factor

    def get_discrete_info(self):
        """返回嵌入层所需的离散信息等。"""
        if not self.config: self.load_config()
        discrete = self.config.get('discrete', {})
        return {
            "is_driver_side": {
                "num_classes": discrete.get('is_driver_side', {}).get('stats', {}).get('num_classes', 2),
                "unknown_policy": discrete.get('is_driver_side', {}).get('mapping', {}).get('unknown_policy', 'error'),
                "unknown_index": discrete.get('is_driver_side', {}).get('mapping', {}).get('unknown_index', -1)
            },
            "OT": {
                "num_classes": discrete.get('OT', {}).get('stats', {}).get('num_classes', 3),
                "unknown_policy": discrete.get('OT', {}).get('mapping', {}).get('unknown_policy', 'error'),
                "unknown_index": discrete.get('OT', {}).get('mapping', {}).get('unknown_index', -1)
            }
        }

    def transform_continuous_crashdataset(self, cont_raw_11: np.ndarray) -> np.ndarray:
        """CrashDataset 连续特征转换：11维输入 -> 11维输出。"""
        cfg = self.config.get('continuous', {})
        out = cont_raw_11.astype(np.float32).copy()

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

    def transform_discrete_crashdataset(self, disc_raw_2: np.ndarray) -> np.ndarray:
        """CrashDataset 离散特征转换：2维输入 -> 2维输出（LabelEncoder等价编码）。"""
        disc_cfg = self.config.get('discrete', {})

        # is_driver_side
        m_driver = disc_cfg.get('is_driver_side', {}).get('mapping', {}).get('value_to_index', DISCRETE_VALUE_TO_INDEX['is_driver_side'])
        m_ot = disc_cfg.get('OT', {}).get('mapping', {}).get('value_to_index', DISCRETE_VALUE_TO_INDEX['OT'])

        def _map(col: np.ndarray, mapping: Dict[str, int], name: str) -> np.ndarray:
            out = np.empty_like(col, dtype=np.int64)
            for i, v in enumerate(col.tolist()):
                key = str(int(v))
                if key not in mapping:
                    raise ValueError(f"离散特征 {name} 出现未见值: {v}")
                out[i] = int(mapping[key])
            return out

        driver_enc = _map(disc_raw_2[:, 0], m_driver, 'is_driver_side')
        ot_enc = _map(disc_raw_2[:, 1], m_ot, 'OT')
        return np.stack([driver_enc, ot_enc], axis=1).astype(np.int64)
