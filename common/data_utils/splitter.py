import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Sequence
from common.utils.seeding import GLOBAL_SEED
from sklearn.model_selection import train_test_split


def build_case_id_index_map(case_ids: Sequence[int]) -> Dict[int, int]:
    """构建 case_id -> 数组位置 的映射（用于从case_id列表恢复indices）。"""
    case_ids_np = np.asarray(case_ids)
    if case_ids_np.ndim != 1:
        raise ValueError("case_ids 必须为一维")
    mapping: Dict[int, int] = {}
    for idx, cid in enumerate(case_ids_np.tolist()):
        cid_int = int(cid)
        if cid_int in mapping:
            raise ValueError(f"检测到重复case_id: {cid_int}")
        mapping[cid_int] = int(idx)
    return mapping

def case_ids_to_indices(case_ids_all: Sequence[int], subset_case_ids: Sequence[int]) -> np.ndarray:
    """将case_id列表转换为在case_ids_all中的indices。"""
    mapping = build_case_id_index_map(case_ids_all)
    out = []
    for cid in subset_case_ids:
        cid_int = int(cid)
        if cid_int not in mapping:
            raise ValueError(f"case_id 不存在于全量数据中: {cid_int}")
        out.append(mapping[cid_int])
    return np.asarray(out, dtype=int)

def stratified_split(
    case_ids: np.ndarray,
    stratify_labels: np.ndarray,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    special_case_assignments: Optional[Dict[str, List[str]]] = None,
    seed: int = GLOBAL_SEED
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    对数据集索引执行分层划分。
    返回的是索引数组，而不是 case_id 列表。
    
    Args:
        case_ids: case ID 数组 (N,)。比如仅is_injury_ok==True的injury_case_ids。
        stratify_labels: 用于分层的标签数组 (N,) (例如 MAIS)。
        train_ratio: 训练集比例。
        val_ratio: 验证集比例。
        test_ratio: 测试集比例。
        special_case_assignments: 包含 'train', 'valid', 'test', 'exclude' 键和 case_id 列表的字典。
        seed: 随机种子。
        
    Returns:
        (train_indices, val_indices, test_indices, split_summary):
            train_indices: 训练集索引数组。(不是case_id，而是对应于输入case_ids的索引)
            val_indices: 验证集索引数组。
            test_indices: 测试集索引数组。
            split_summary: 包含划分统计信息的字典。
    """
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例总和必须为 1"
    
    # 将 case_id 映射到索引
    case_id_map = build_case_id_index_map(case_ids) # Dict: case_id -> index(从0开始连续的整数索引)
    
    # 强制分配的集合
    forced_train_set = set()
    forced_val_set = set()
    forced_test_set = set()
    exclude_set = set()
    
    counts = {'train': 0, 'valid': 0, 'test': 0, 'exclude': 0}

    # 处理特殊分配
    if special_case_assignments:
        # 先处理 exclude，保证其优先级最高
        for cid in special_case_assignments.get('exclude', []):
            if cid in case_id_map:
                exclude_set.add(case_id_map[cid])
                counts['exclude'] += 1

        for cid in special_case_assignments.get('train', []):
            if cid in case_id_map and case_id_map[cid] not in exclude_set:
                forced_train_set.add(case_id_map[cid])
                counts['train'] += 1
        for cid in special_case_assignments.get('valid', []):
            if cid in case_id_map and case_id_map[cid] not in exclude_set:
                forced_val_set.add(case_id_map[cid])
                counts['valid'] += 1
        for cid in special_case_assignments.get('test', []):
            if cid in case_id_map and case_id_map[cid] not in exclude_set:
                forced_test_set.add(case_id_map[cid])
                counts['test'] += 1

        # 强制分配冲突检测（exclude 已优先处理）
        overlap_tv = forced_train_set & forced_val_set
        overlap_tt = forced_train_set & forced_test_set
        overlap_vt = forced_val_set & forced_test_set
        if overlap_tv or overlap_tt or overlap_vt:
            raise ValueError("强制分配中检测到同一 case_id 同时属于多个集合。")
    
    forced_indices = forced_train_set | forced_val_set | forced_test_set
    
    all_indices = np.arange(len(case_ids))
    remaining_indices = np.array(list(set(all_indices) - forced_indices - exclude_set))
    remaining_labels = stratify_labels[remaining_indices]
    
    # 准备划分
    train_indices_list = []
    
    # 处理稀有类别 (出现少于2次)，强制放入训练集
    label_counts = pd.Series(remaining_labels).value_counts()
    rare_labels = label_counts[label_counts < 2].index.tolist()
    
    processed_remaining_indices = remaining_indices
    processed_remaining_labels = remaining_labels
    
    if rare_labels:
        mask = np.isin(remaining_labels, rare_labels)
        train_indices_list.extend(remaining_indices[mask])
        
        # 更新剩余数据
        processed_remaining_indices = remaining_indices[~mask]
        processed_remaining_labels = remaining_labels[~mask]
        
    # 标准划分逻辑
    temp_size = val_ratio + test_ratio
    
    # 第一次划分: 训练集 vs (验证集+测试集)
    if len(processed_remaining_indices) < 2:
        train_auto = processed_remaining_indices
        temp_indices = np.array([], dtype=int)
    elif len(np.unique(processed_remaining_labels)) < 2:
        # 如果只剩下一个类别，无法分层
        train_auto, temp_indices = train_test_split(
            processed_remaining_indices, test_size=temp_size, random_state=seed
        )
    else:
        train_auto, temp_indices = train_test_split(
            processed_remaining_indices, 
            test_size=temp_size, 
            random_state=seed, 
            stratify=processed_remaining_labels
        )
        
    train_indices_list.extend(train_auto)
    
    # 第二次划分: 验证集 vs 测试集
    if len(temp_indices) > 0 and test_ratio > 0:
        relative_test = test_ratio / temp_size
        if len(temp_indices) < 2:
            val_auto = temp_indices
            test_auto = np.array([], dtype=int)
        else:
            # 数据量太小通常无法再次分层，仅使用随机划分
            val_auto, test_auto = train_test_split(
                temp_indices, test_size=relative_test, random_state=seed
            )
    else:
        val_auto = temp_indices
        test_auto = np.array([], dtype=int)
        
    # 合并强制分配和自动分配的索引
    final_train = np.unique(np.concatenate([list(forced_train_set), train_indices_list])).astype(int)
    final_val = np.unique(np.concatenate([list(forced_val_set), val_auto])).astype(int)
    final_test = np.unique(np.concatenate([list(forced_test_set), test_auto])).astype(int)
    
    # 验证
    intersection = len(set(final_train) & set(final_val)) + \
                   len(set(final_train) & set(final_test)) + \
                   len(set(final_val) & set(final_test))
                   
    if intersection > 0:
        raise ValueError("划分中检测到重叠索引!")
        
    summary = {
        "total_final": len(case_ids) - len(exclude_set),
        "train_final": len(final_train),
        "val_final": len(final_val),
        "test_final": len(final_test),
        "forced": counts,
        "total_ori": len(case_ids)
    }
    
    return final_train, final_val, final_test, summary

def stratified_split_case_ids(
    case_ids: np.ndarray,
    stratify_labels: np.ndarray,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    special_case_assignments: Optional[Dict[str, List[int]]] = None,
    seed: int = GLOBAL_SEED
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    基于case_id输出划分结果。
    返回的是 case_id 列表，而不是数组位置索引。
    """
    train_idx, val_idx, test_idx, summary = stratified_split(
        case_ids=np.asarray(case_ids),
        stratify_labels=np.asarray(stratify_labels),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        special_case_assignments=special_case_assignments,
        seed=seed
    )
    case_ids_np = np.asarray(case_ids)
    return case_ids_np[train_idx], case_ids_np[val_idx], case_ids_np[test_idx], summary
