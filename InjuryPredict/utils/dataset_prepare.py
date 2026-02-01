'''
描述：数据集处理准备的模块函数
功能包括：
1. 定义数据集类 CrashDataset，用于加载和存储原始及处理后的碰撞数据。
2. 定义数据处理类 DataProcessor，封装数据预处理逻辑，包括拟合(fit)、转换(transform)和结果展示。
3. 定义数据集划分函数 split_data，用于将数据集划分为训练集、验证集和测试集，支持特殊案例的强制分配。
作用：输出预处理后的数据集实例（.pt文件）和数据处理器实例（.joblib文件），便于后续模型训练和评估使用。
'''
import warnings
warnings.filterwarnings("ignore")
import torch
from torch.utils.data import Dataset, Subset, DataLoader
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import joblib
import time

try:
    from utils.set_random_seed import GLOBAL_SEED, set_random_seed  # 作为包导入时使用
except ImportError:
    from set_random_seed import GLOBAL_SEED, set_random_seed   # 直接运行时使用

set_random_seed()

class CrashDataset(Dataset):
    """
    数据集类，负责加载和存储原始及处理后的碰撞数据。
    """
    def __init__(self, input_file='./data/data_input.npz', label_file='./data/data_labels.npz'):
        """
        Args:
            input_file (str): 包含碰撞波形和特征数据的 .npz 文件路径。
            label_file (str): 包含标签数据的 .npz 文件路径。
        """
        with np.load(input_file) as inputs, np.load(label_file) as labels:
            # --- 对齐校验 ---
            inp_ids = inputs['case_ids']
            lab_ids = labels['case_ids']
            assert np.array_equal(inp_ids, lab_ids), (
                f"Case ID 不匹配：input_file 中 {inp_ids[:5]}… vs label_file 中 {lab_ids[:5]}…"
            )

            self.case_ids = inp_ids

            # --- 加载原始数据 ---
            self.x_acc_raw = inputs['waveforms'].astype(float) # 形状 (N, 2, 150) x/y direction acceleration waveforms
            self.x_att_raw = inputs['params'] # 形状 (N, 13)  attributes

            # 特征数据 (x_att_raw) 说明：形状 (N, 13)
            # 连续特征 (0-10): impact_velocity, impact_angle, overlap, LL1, LL2, BTF, LLATTF, AFT, SP, SH, RA
            # 离散特征 (11-12): is_driver_side, OT
            self.OT_raw = inputs['params'][:, 12].astype(int)  # OT 特征，形状 (N,)

            # --- 加载所有目标变量 ---
            self.y_HIC = labels['HIC'].astype(float)
            self.y_Dmax = labels['Dmax'].astype(float)
            self.y_Nij = labels['Nij'].astype(float)
            self.ais_head = labels['AIS_head'].astype(int)
            self.ais_chest = labels['AIS_chest'].astype(int)
            self.ais_neck = labels['AIS_neck'].astype(int)
            self.mais = labels['MAIS'].astype(int)
        
        self.x_acc = None
        self.x_att_continuous = None
        self.x_att_discrete = None

        # 连续特征索引: 0~10
        self.continuous_indices = list(range(11))
        # 离散特征索引: 11, 12 (is_driver_side, OT)
        self.discrete_indices = [11, 12]
        self.num_classes_of_discrete = None

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx):
        if self.x_acc is None or self.x_att_continuous is None or self.x_att_discrete is None:
            raise RuntimeError("数据集尚未预处理。请先运行数据处理流程。")

        return (
            torch.tensor(self.x_acc[idx], dtype=torch.float32),
            torch.tensor(self.x_att_continuous[idx], dtype=torch.float32),
            torch.tensor(self.x_att_discrete[idx], dtype=torch.int),
            torch.tensor(self.y_HIC[idx], dtype=torch.float32),
            torch.tensor(self.y_Dmax[idx], dtype=torch.float32),
            torch.tensor(self.y_Nij[idx], dtype=torch.float32),
            torch.tensor(self.ais_head[idx], dtype=torch.int),
            torch.tensor(self.ais_chest[idx], dtype=torch.int),
            torch.tensor(self.ais_neck[idx], dtype=torch.int),
            torch.tensor(self.mais[idx], dtype=torch.int),
            torch.tensor(self.OT_raw[idx], dtype=torch.int) # 额外返回 OT 特征
        )

class DataProcessor:
    """
    一个封装了数据预处理逻辑的类，包括拟合(fit)、转换(transform)和结果展示。
    """
    def __init__(self, top_k_waveform=50):
        self.waveform_norm_factor = None
        self.top_k_waveform = top_k_waveform
        self.scaler_minmax = None
        self.scaler_maxabs = None
        self.encoders_discrete = None

        # 定义连续与离散特征在原始13维向量中的索引
        self.continuous_indices = list(range(11)) # `0-10` 连续特征
        self.discrete_indices = [11, 12]
        
        # --- 定义预处理策略 ---
        # 连续特征子集中，应用 MaxAbsScaler 的索引 (归一化至 [-1, 1])
        # Idx 1: impact_angle, Idx 2: overlap
        self.maxabs_indices_in_continuous = [1, 2]
        
        # 连续特征子集中，应用 MinMaxScaler 的索引 (归一化至 [0, 1])
        # Idx 0: velocity, 3: LL1, 4: LL2, 5: BTF, 6: LLATTF, 7: AFT, 8: SP, 9: SH, 10: RA
        self.minmax_indices_in_continuous = [0, 3, 4, 5, 6, 7, 8, 9, 10]
        
        # 更新特征名称映射
        self.feature_names = {
            0: "impact_velocity", 1: "impact_angle", 2: "overlap", 
            3: "LL1", 4: "LL2", 5: "BTF", 6: "LLATTF", 7: "AFT", 8: "SP", 9: "SH", 10: "RA",
            11: "is_driver_side", 12: "OT"
        }

    def fit(self, train_indices, dataset):
        """
        仅使用训练集数据来拟合所有的scalers和encoders。
        """
        # --- 拟合波形数据的全局归一化因子 ---
        train_x_acc_raw = dataset.x_acc_raw[train_indices]
        # 展平所有波形数据并取绝对值
        flat_abs_waveforms = np.abs(train_x_acc_raw).flatten()
        # 排序并取top k
        top_k_values = np.sort(flat_abs_waveforms)[-self.top_k_waveform:]
        # 计算平均值作为归一化因子
        self.waveform_norm_factor = np.mean(top_k_values)
        if self.waveform_norm_factor < 1e-6: self.waveform_norm_factor = 1.0

        # --- 拟合标量特征的Scaler和Encoder ---
        train_x_att_continuous_raw = dataset.x_att_raw[train_indices][:, self.continuous_indices].astype(float)
        train_x_att_discrete_raw = dataset.x_att_raw[train_indices][:, self.discrete_indices].astype(int)

        self.scaler_minmax = MinMaxScaler(feature_range=(0, 1))
        self.scaler_maxabs = MaxAbsScaler()
        self.scaler_minmax.fit(train_x_att_continuous_raw[:, self.minmax_indices_in_continuous])
        self.scaler_maxabs.fit(train_x_att_continuous_raw[:, self.maxabs_indices_in_continuous])
        
        self.encoders_discrete = [LabelEncoder() for _ in range(train_x_att_discrete_raw.shape[1])]
        for i in range(train_x_att_discrete_raw.shape[1]):
            self.encoders_discrete[i].fit(train_x_att_discrete_raw[:, i])

    def transform(self, dataset):
        """
        使用已拟合的处理器转换整个数据集，并填充回dataset对象。
        """
        if self.waveform_norm_factor is None or self.scaler_minmax is None or self.encoders_discrete is None:
            raise RuntimeError("处理器尚未拟合。请先调用fit方法。")

        # --- 转换波形数据 ---
        dataset.x_acc = dataset.x_acc_raw / self.waveform_norm_factor
        
        # --- 转换标量数据 ---
        x_att_continuous_raw = dataset.x_att_raw[:, self.continuous_indices]
        x_att_discrete_raw = dataset.x_att_raw[:, self.discrete_indices]

        x_att_continuous_processed = np.zeros_like(x_att_continuous_raw, dtype=np.float32)
        x_att_continuous_processed[:, self.minmax_indices_in_continuous] = self.scaler_minmax.transform(x_att_continuous_raw[:, self.minmax_indices_in_continuous])
        x_att_continuous_processed[:, self.maxabs_indices_in_continuous] = self.scaler_maxabs.transform(x_att_continuous_raw[:, self.maxabs_indices_in_continuous])
        dataset.x_att_continuous = x_att_continuous_processed

        x_att_discrete_processed = np.zeros_like(x_att_discrete_raw, dtype=np.int64)
        num_classes = []
        for i in range(x_att_discrete_raw.shape[1]):
            x_att_discrete_processed[:, i] = self.encoders_discrete[i].transform(x_att_discrete_raw[:, i])
            num_classes.append(len(self.encoders_discrete[i].classes_))
        dataset.x_att_discrete = x_att_discrete_processed
        dataset.num_classes_of_discrete = num_classes
        
        return dataset
    
    def print_fit_summary(self):
        """
        打印已拟合的scalers和encoders的统计信息。
        """
        if self.waveform_norm_factor is None or self.scaler_minmax is None:
            print("处理器尚未拟合。")
            return
        
        print("\n--- 数据处理器拟合结果摘要 ---")
        
        print(f"\n[碰撞波形 (x_acc) 全局归一化因子]")
        print(f"  - 基于训练集Top {self.top_k_waveform} 最大绝对值点的平均值: {self.waveform_norm_factor:.4f}")

        print("\n[连续标量特征 (x_att_continuous) Scaler 统计量]")
        print("  - MinMaxScaler (归一化至 [0, 1]):")
        for i, idx_in_cont in enumerate(self.minmax_indices_in_continuous):
            orig_idx = self.continuous_indices[idx_in_cont]
            name = self.feature_names.get(orig_idx, f"特征 {orig_idx}")
            print(f"    - {name} (Idx {orig_idx}): Min={self.scaler_minmax.data_min_[i]:.4f}, Max={self.scaler_minmax.data_max_[i]:.4f}")
        
        print("  - MaxAbsScaler (归一化至 [-1, 1]):")
        for i, idx_in_cont in enumerate(self.maxabs_indices_in_continuous):
            orig_idx = self.continuous_indices[idx_in_cont]
            name = self.feature_names.get(orig_idx, f"特征 {orig_idx}")
            print(f"    - {name} (Idx {orig_idx}): MaxAbs={self.scaler_maxabs.max_abs_[i]:.4f}")

        print("\n[离散标量特征 (x_att_discrete) LabelEncoder 映射]")
        for i, encoder in enumerate(self.encoders_discrete):
            orig_idx = self.discrete_indices[i]
            name = self.feature_names.get(orig_idx, f"特征 {orig_idx}")
            mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
            print(f"  - {name} (Idx {orig_idx}):")
            print(f"    - 映射关系: {mapping}")
        print("---------------------------------\n")

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)

def split_data(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, special_case_assignments=None):
    """
    划分数据集为训练集、验证集和测试集。
    Args:
    dataset (CrashDataset): 要划分的数据集实例。
    train_ratio (float): 训练集比例。
    val_ratio (float): 验证集比例。
    test_ratio (float): 测试集比例。
    special_case_assignments (dict): 一个字典，用于强制分配 case_id。
    
    返回:
    (final_train_indices, final_val_indices, final_test_indices, split_summary)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    # --- 1. 处理特殊分配 (保持不变) ---
    case_id_map = {case_id: idx for idx, case_id in enumerate(dataset.case_ids)}
    
    forced_train_indices_set = set()
    forced_val_indices_set = set()
    forced_test_indices_set = set()
    exclude_indices_set = set()
    
    forced_counts = {'train': 0, 'valid': 0, 'test': 0, 'exclude': 0}

    if special_case_assignments:
        for case_id in special_case_assignments.get('train', []):
            if case_id in case_id_map:
                forced_train_indices_set.add(case_id_map[case_id])
                forced_counts['train'] += 1
        for case_id in special_case_assignments.get('valid', []):
            if case_id in case_id_map:
                forced_val_indices_set.add(case_id_map[case_id])
                forced_counts['valid'] += 1
        for case_id in special_case_assignments.get('test', []):
            if case_id in case_id_map:
                forced_test_indices_set.add(case_id_map[case_id])
                forced_counts['test'] += 1
        for case_id in special_case_assignments.get('exclude', []):
            if case_id in case_id_map:
                exclude_indices_set.add(case_id_map[case_id])
                forced_counts['exclude'] += 1
    
    forced_indices = forced_train_indices_set | forced_val_indices_set | forced_test_indices_set
    
    all_indices = np.arange(len(dataset))
    remaining_indices_for_split = np.array(list(set(all_indices) - forced_indices - exclude_indices_set))
    
    # --- 2. 严格按照原逻辑划分【剩余的】索引 ---
    
    labels = dataset.mais[remaining_indices_for_split]
    indices = remaining_indices_for_split

    label_counts = pd.Series(labels).value_counts()
    insufficient_samples_labels = label_counts[label_counts < 2].index.tolist()
    
    train_indices_final = []
    remaining_indices = indices
    if insufficient_samples_labels:
        singleton_mask = np.isin(labels, insufficient_samples_labels)
        train_indices_final.extend(indices[singleton_mask])
        remaining_indices_mask = ~singleton_mask
        remaining_indices = indices[remaining_indices_mask]
        labels = labels[remaining_indices_mask] # labels 必须同步更新
    
    remaining_labels = labels

    temp_size = val_ratio + test_ratio
    if len(remaining_indices) < 2:
        train_main_indices = remaining_indices
        temp_indices = np.array([], dtype=int)
    elif len(np.unique(remaining_labels)) < 2:
         train_main_indices, temp_indices = train_test_split(
            remaining_indices, test_size=temp_size, random_state=GLOBAL_SEED
         )
    else:
        # 第一次划分，使用 stratify
        train_main_indices, temp_indices, _, _ = train_test_split(
            remaining_indices, remaining_labels, test_size=temp_size, random_state=GLOBAL_SEED, stratify=remaining_labels
        )
    
    train_indices_final.extend(train_main_indices)
    auto_train_indices = np.array(train_indices_final) # 自动划分的训练集

    if len(temp_indices) > 0 and test_ratio > 0:
        relative_test_ratio = test_ratio / temp_size
        if len(temp_indices) < 2:
             auto_val_indices = temp_indices
             auto_test_indices = np.array([], dtype=int)
        else:
            # *** 不使用stratify ***
            auto_val_indices, auto_test_indices = train_test_split(
                temp_indices, 
                test_size=relative_test_ratio, 
                random_state=GLOBAL_SEED
            )
    else:
        auto_val_indices = temp_indices
        auto_test_indices = np.array([], dtype=int)

    # --- 3. 合并并确保 DTYPE ---
    forced_train_indices = np.array(list(forced_train_indices_set), dtype=int)
    forced_val_indices = np.array(list(forced_val_indices_set), dtype=int)
    forced_test_indices = np.array(list(forced_test_indices_set), dtype=int)

    auto_train_indices = auto_train_indices.astype(int)
    auto_val_indices = auto_val_indices.astype(int)
    auto_test_indices = auto_test_indices.astype(int)
    
    final_train_indices = np.unique(np.concatenate([forced_train_indices, auto_train_indices]))
    final_val_indices = np.unique(np.concatenate([forced_val_indices, auto_val_indices]))
    final_test_indices = np.unique(np.concatenate([forced_test_indices, auto_test_indices]))
    
    # --- 4. 完整性检查 ---
    assert len(set(final_train_indices) & set(final_val_indices)) == 0, "严重错误: 训练集和验证集存在交集!"
    assert len(set(final_train_indices) & set(final_test_indices)) == 0, "严重错误: 训练集和测试集存在交集!"
    assert len(set(final_val_indices) & set(final_test_indices)) == 0, "严重错误: 验证集和测试集存在交集!"
        
    # --- 5. 准备摘要字典 ---
    split_summary = {
        "total_cases_in_dataset": len(dataset),
        "forced_train_found": forced_counts['train'],
        "forced_valid_found": forced_counts['valid'],
        "forced_test_found": forced_counts['test'],
        "forced_exclude_found": forced_counts['exclude'],
        "remaining_for_auto_split": len(remaining_indices_for_split),
        "final_train_count": len(final_train_indices),
        "final_valid_count": len(final_val_indices),
        "final_test_count": len(final_test_indices)
    }
        
    return final_train_indices, final_val_indices, final_test_indices, split_summary

if __name__ == '__main__':
    start_time = time.time()
    
    dataset = CrashDataset(input_file='./data/data_input.npz', label_file='./data/data_labels.npz')
    # print(f"\n原始数据加载完成, 耗时: {time.time() - start_time:.2f}s")

    #####
    # exclude_case_ids = pd.read_excel(r'E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\仿真数据库相关\distribution\Injury_labels_1023.xlsx')['case_id'].tolist()
    #####

    special_assignments = {
        'train': [],   # 强制放入训练集的 case_id 列表
        'valid': [],   # 强制放入验证集的 case_id 列表
        'test': [],    # 强制放入测试集的 case_id 列表
        'exclude': []  # 强制排除的 case_id 列表
    }
    # print(f"应用特殊分配规则: {special_assignments}")

    # ---  捕获 split_summary 返回值 ---
    train_indices, val_indices, test_indices, split_summary = split_data(
        dataset, 
        train_ratio=0.8, 
        val_ratio=0.1, 
        test_ratio=0.1,
        special_case_assignments=special_assignments
    )
    
    processor = DataProcessor(top_k_waveform=50)
    
    # *** 检查训练集是否为空 ***
    if len(train_indices) == 0:
        raise ValueError("错误：根据划分规则，训练集为空。无法拟合 preprocessor。")
        
    processor.fit(train_indices, dataset)
    
    processor.print_fit_summary()

    dataset = processor.transform(dataset)
    print("整个数据集已使用训练集统计量完成转换。")

    processor.save('./data/preprocessors.joblib')
    print("处理器已保存至 './data/preprocessors.joblib'")

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    torch.save(train_dataset, './data/train_dataset.pt')
    torch.save(val_dataset, './data/val_dataset.pt')
    torch.save(test_dataset, './data/test_dataset.pt')
    print("\n处理后的训练、验证和测试数据集已保存。")

    # --- 打印详细的划分摘要 ---
    print(f"\n--- 数据集划分摘要 ---")
    print(f"  - 数据集总案例数: {split_summary['total_cases_in_dataset']}")
    print(f"  - 强制排除 (Exclude): {split_summary['forced_exclude_found']} (已找到并排除)")
    print(f"  - 强制分配 (Train): {split_summary['forced_train_found']} (已找到)")
    print(f"  - 强制分配 (Valid): {split_summary['forced_valid_found']} (已找到)")
    print(f"  - 强制分配 (Test): {split_summary['forced_test_found']} (已找到)")
    print(f"  - 剩余自动划分: {split_summary['remaining_for_auto_split']}")
    print(f"  ---------------------")
    print(f"  - 最终训练集大小: {split_summary['final_train_count']} (强制 + 自动)")
    print(f"  - 最终验证集大小: {split_summary['final_valid_count']} (强制 + 自动)")
    print(f"  - 最终测试集大小: {split_summary['final_test_count']} (强制 + 自动)")
    total_assigned = split_summary['final_train_count'] + split_summary['final_valid_count'] + split_summary['final_test_count']
    print(f"  - (总计已分配: {total_assigned})")
    # 校验总数
    total_check = total_assigned + split_summary['forced_exclude_found']
    print(f"  - (总计 = 已分配 + 排除: {total_check})")
    if total_check != split_summary['total_cases_in_dataset']:
         print(f"  *** 警告: 总数 {total_check} 与数据集案例数 {split_summary['total_cases_in_dataset']} 不匹配! ***")
    print(f"-----------------------\n")

    def get_label_distribution(subset):
        # 检查 subset.indices 是否为空
        if len(subset.indices) == 0: return "空"
        labels = [subset.dataset.mais[i] for i in subset.indices]
        unique, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique, counts))

    print("\n各子集的MAIS标签分布:")
    print(f"  - 训练集: {get_label_distribution(train_dataset)}")
    print(f"  - 验证集: {get_label_distribution(val_dataset)}")
    print(f"  - 测试集: {get_label_distribution(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
    print("\nTesting DataLoader...")
    
    # 检查训练集是否为空
    if len(train_dataset) > 0:
        try:
            batch_start_time = time.time()
            for i, batch in enumerate(train_loader):
                (x_acc, x_att_continuous, x_att_discrete, 
                 y_HIC, y_Dmax, y_Nij, 
                 ais_head, ais_chest, ais_neck, mais, ot) = batch
                
                print("x_acc shape:", x_acc.shape)
                print("y_HIC shape:", y_HIC.shape)
                print("MAIS shape:", mais.shape)
                print("OT shape:", ot.shape)
                # ot的值范围
                print("OT values in batch:", ot.unique().tolist())
                break
            print(f"batch loading time: {time.time() - batch_start_time:.4f}s")
        except Exception as e:
            print(f"DataLoader 测试失败: {e}")
    else:
        print("训练集为空，跳过 DataLoader 测试。")