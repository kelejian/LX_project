'''
描述：将输入工况参数和波形数据打包在一起的模块函数
功能包括：
1. 读取指定目录下的原始波形CSV文件。
2. 根据给定的 case_id 列表，匹配对应的工况参数。
3. 对波形数据进行降采样处理。
4. 将输入参数、输出波形和 case_id 作为一个整体保存到结构化的 .npz 文件中。
作用：输出打包后的数据 .npz 文件，便于后续模型训练和评估使用。
'''
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
try:
    from utils.AIS_cal import AIS_cal_head, AIS_cal_chest, AIS_cal_neck  # 作为包导入时使用
except ImportError:
    from AIS_cal import AIS_cal_head, AIS_cal_chest, AIS_cal_neck   # 直接运行时使用

# ============================================================================
CASE_ID_OFFSET = 50000  # 副驾侧case_id在主驾侧基础上加50000
# ============================================================================

def package_input_data(pulse_dir, params_path, case_id_list, output_path):
    """
    处理、降采样并将指定案例的输入工况参数和波形数据打包在一起。

    该函数会读取工况参数文件，并根据给定的 case_id 列表，匹配对应的
    原始波形CSV文件。然后将输入参数、输出波形和 case_id 作为一个整体
    保存到一个结构化的 .npz 文件中。

    :param pulse_dir: 存放原始波形CSV文件的目录。
    :param params_path: 包含所有工况参数的 distribution 文件路径 (包含 'case_id' 列)。
    :param case_id_list: 需要处理的案例ID列表。
    :param output_path: 打包后的 .npz 文件保存路径。
    """

    # --- 1. 加载并索引工况参数 ---
    # 读取distribution文件
    if params_path.endswith('.npz'):
        distribution_npz = np.load(params_path, allow_pickle=True)
        params_df = pd.DataFrame({
                key: distribution_npz[key]
                for key in distribution_npz.files
            }).set_index('case_id', drop=False)
    elif params_path.endswith('.csv'):
        params_df = pd.read_csv(params_path)
        params_df.set_index('case_id', inplace=True, drop=False)
    else:
        raise ValueError("Unsupported distribution file format. Use .csv or .npz")

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 用于存储最终数据的列表
    processed_case_ids = []
    processed_params = []
    processed_waveforms = []

    print(f"开始处理 {len(case_id_list)} 个案例，将输入和输出打包在一起...")
    for case_id in tqdm(case_id_list, desc="Packaging pulse Data"):
        try:
            # --- 2. 确认参数存在 ---
            if case_id not in params_df.index:
                print(f"警告：在参数文件中未找到案例 {case_id}，已跳过。")
                continue

            # --- 3. 读取并处理波形 ---
            params_row = params_df.loc[case_id]
            if params_row['is_driver_side'] == 1:
                x_path = os.path.join(pulse_dir, f'x{case_id}.csv')
                y_path = os.path.join(pulse_dir, f'y{case_id}.csv')
                # z_path = os.path.join(pulse_dir, f'z{case_id}.csv')
            else:
                x_path = os.path.join(pulse_dir, f'x{case_id - CASE_ID_OFFSET}.csv')
                y_path = os.path.join(pulse_dir, f'y{case_id - CASE_ID_OFFSET}.csv')

            

            if not all(os.path.exists(p) for p in [x_path, y_path]):  # Removed z_path check
                print(f"警告：案例 {case_id} 的波形文件不完整，已跳过。")
                continue

            time = pd.read_csv(x_path, sep='\t', header=None, usecols=[0]).values.flatten()
            total_length = len(time)
            dt = np.mean(np.diff(time))
            # ************************************************************************
            if np.isclose(dt, 1e-5, atol=1e-7):
                downsample_indices = np.arange(100, total_length, 100)
            elif np.isclose(dt,  5e-6, atol=5e-8):
                downsample_indices = np.arange(200, total_length, 200)
            else:
                raise ValueError(f"案例 {case_id} 的时间步长 {dt} 不符合预期。")
            # ************************************************************************
            # 读取完整波形数据
            ax_full = pd.read_csv(x_path, sep='\t', header=None, usecols=[1]).values.flatten()
            ay_full = pd.read_csv(y_path, sep='\t', header=None, usecols=[1]).values.flatten()
            # az_full = pd.read_csv(z_path, sep='\t', header=None, usecols=[1]).values.flatten()

            ax_sampled = ax_full[downsample_indices]
            ay_sampled = ay_full[downsample_indices]
            # az_sampled = az_full[downsample_indices]
            # ************************************************************************
            # 只取前150个点
            ax_sampled = ax_sampled[:150]
            ay_sampled = ay_sampled[:150]
            # az_sampled = az_sampled[:150]
            # ************************************************************************
            
            waveforms_np = np.stack([ax_sampled, ay_sampled]).squeeze() # 形状 (2, 150), 通道维度在前，分别是 x, y，对应索引 0, 1

            # --- 4. 提取匹配的参数 ---          
            # 输入的特征顺序必须严格按照此处 params_np 数组构建时的顺序
            # 打包后的 .npz 文件中，params 数组仅存储了纯数值矩阵（Numpy Array），不包含任何特征列名信息。因此推理时必须人工确保构建的输入数组顺序与打包代码中的逻辑完全一致。
            params_np = np.array([
                # --- 连续特征 (Continuous): 共 11 个---
                params_row['impact_velocity'], # 碰撞速度 kph
                params_row['impact_angle'], # 碰撞角度 °，分正负方向
                params_row['overlap'], # 重叠率，分正负方向
                params_row['LL1'], # 安全带一级限力值 KN
                params_row['LL2'],  # 安全带二级限力值 KN
                params_row['BTF'], # 预紧器点火时刻 ms
                params_row['LLATTF'],  # 安全带二级限力切换时刻 ms
                params_row['AFT'], # 气囊点火时刻 ms
                params_row['SP'], # 座椅前后位置 mm
                params_row['SH'], # 座椅高度 mm
                params_row['RA'], # 座椅靠背角 °；虽离散化但作为连续特征处理
                
                # --- 离散特征 (Discrete): 共 2 个---
                params_row['is_driver_side'], # 主驾侧标识 (0/1)
                params_row['OT'] # 乘员体征 (1/2/3)
            ], dtype=np.float32) # 形状 (13,)

            # --- 5. 添加到结果列表 ---
            processed_case_ids.append(case_id)
            processed_params.append(params_np)
            processed_waveforms.append(waveforms_np)

        except Exception as e:
            print(f"警告：处理案例 {case_id} 时发生错误 '{e}'，已跳过。")
            continue
            
    if not processed_case_ids:
        print("错误：没有成功处理任何数据，未生成输出文件。")
        return

    # --- 6. 将数据列表转换为Numpy数组并保存 ---
    final_case_ids = np.array(processed_case_ids, dtype=int) # 形状 (N,)
    final_params = np.stack(processed_params, axis=0) # 形状 (N, 13)
    final_waveforms = np.stack(processed_waveforms, axis=0) # 形状 (N, 2, 150)
    # 断言检查：输出 case_ids 顺序必须与传入的 case_id_list 一致
    assert np.array_equal(final_case_ids, np.array(case_id_list, dtype=int)), (
        f"case_ids 序列不匹配: 输出{final_case_ids[:5]} vs 输入{case_id_list[:5]}"
    )
    np.savez(
        output_path,
        case_ids=final_case_ids,
        params=final_params,
        waveforms=final_waveforms
    )
    print(f"数据打包完成，已保存至 {output_path}")
    print(f"成功处理并打包的数据数目：{len(final_case_ids)}")
    print(f"打包后文件内容: case_ids shape={final_case_ids.shape}, params shape={final_params.shape}, waveforms shape={final_waveforms.shape}")

if __name__ == '__main__':
    pulse_dir = r'G:\VCS_acc_data\acc_data_before1111_6134'
    params_path = r'E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\仿真数据库相关\distribution\distribution_0123.csv'
    output_dir = r'E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\LX_model_injurypredict\data'
    # 读取distribution文件
    if params_path.endswith('.npz'):
        distribution_npz = np.load(params_path, allow_pickle=True)
        distribution_df = pd.DataFrame({
                key: distribution_npz[key]
                for key in distribution_npz.files
            }).set_index('case_id', drop=False)
    elif params_path.endswith('.csv'):
        distribution_df = pd.read_csv(params_path)
        distribution_df.set_index('case_id', inplace=True, drop=False)
    else:
        raise ValueError("Unsupported distribution file format. Use .csv or .npz")
    
    # 筛选is_pulse_ok和is_injury_ok均为True的行, 并提取对应的case编号和标签
    filtered_df = distribution_df[(distribution_df['is_pulse_ok'] == True) & (distribution_df['is_injury_ok'] == True)]

    case_ids_need = filtered_df['case_id'].astype(int).tolist()
    hic15_labels = filtered_df['HIC15'].astype(float).values
    dmax_labels = filtered_df['Dmax'].astype(float).values
    ot = filtered_df['OT'].astype(int).values
    nij_labels = filtered_df['Nij'].astype(float).values

    # 计算对应的AIS标签
    ais_head_labels = AIS_cal_head(hic15_labels)
    ais_chest_labels = AIS_cal_chest(dmax_labels, ot)
    ais_neck_labels = AIS_cal_neck(nij_labels)

    # 计算MAIS
    mais_labels = np.maximum.reduce([ais_head_labels, ais_chest_labels, ais_neck_labels])

    ############################################################################################
    
    # 统计头部AIS标签分布
    unique, counts = np.unique(ais_head_labels, return_counts=True)
    label_distribution = {int(k): int(v) for k, v in zip(unique, counts)}
    print(f"AIS头部标签分布: {label_distribution}")

    # 按OT分别统计胸部AIS标签分布
    ot_names = {1: '5th Female', 2: '50th Male', 3: '95th Male'}
    for ot_val in [1, 2, 3]:
        ot_mask = (ot == ot_val)
        ais_chest_ot = ais_chest_labels[ot_mask]
        unique, counts = np.unique(ais_chest_ot, return_counts=True)
        label_distribution = {int(k): int(v) for k, v in zip(unique, counts)}
        print(f"AIS胸部标签分布 (OT={ot_val}, {ot_names[ot_val]}): {label_distribution}")

    # 统计颈部AIS标签分布
    unique, counts = np.unique(ais_neck_labels, return_counts=True)
    label_distribution = {int(k): int(v) for k, v in zip(unique, counts)}
    print(f"AIS颈部标签分布: {label_distribution}")

    # 按OT分别统计MAIS分布
    for ot_val in [1, 2, 3]:
        ot_mask = (ot == ot_val)
        mais_ot = mais_labels[ot_mask]
        unique, counts = np.unique(mais_ot, return_counts=True)
        label_distribution = {int(k): int(v) for k, v in zip(unique, counts)}
        print(f"MAIS标签分布 (OT={ot_val}, {ot_names[ot_val]}): {label_distribution}")
    # 合并所有OT的MAIS分布
    unique, counts = np.unique(mais_labels, return_counts=True)
    label_distribution = {int(k): int(v) for k, v in zip(unique, counts)}
    print(f"MAIS标签分布 (所有OT合并): {label_distribution}")

    ############################################################################################
    # 绘制碰撞速度和HIC/Dmax/Nij的散点图
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    # 确保数据顺序一一对应 - 按case_ids_need的顺序提取碰撞速度
    impact_velocities = distribution_df.loc[case_ids_need, 'impact_velocity'].values
    colors = ['blue', 'green', 'yellow', 'orange', 'red', 'darkred']

    # 头部散点图
    plt.figure(figsize=(8,6))
    ais_colors = [colors[min(ais, 5)] for ais in ais_head_labels]
    plt.scatter(impact_velocities, hic15_labels, c=ais_colors, alpha=0.6, s=50)
    legend_elements = [Patch(facecolor=colors[i], label=f'AIS {i}') for i in range(6) if i in ais_head_labels]
    plt.legend(handles=legend_elements, title='AIS LEVEL', loc='upper left')
    plt.title('impact velocity vs HIC15')
    plt.xlabel('impact velocity (km/h)')
    plt.ylabel('HIC15')
    plt.grid(True, alpha=0.3)

    # 胸部散点图 - 按OT分别绘制三张
    for ot_val in [1, 2, 3]:
        ot_mask = (ot == ot_val)
        plt.figure(figsize=(8,6))
        
        dmax_ot = dmax_labels[ot_mask]
        ais_chest_ot = ais_chest_labels[ot_mask]
        velocity_ot = impact_velocities[ot_mask]
        
        ais_colors = [colors[min(ais, 5)] for ais in ais_chest_ot]
        plt.scatter(velocity_ot, dmax_ot, c=ais_colors, alpha=0.6, s=50)
        
        legend_elements = [Patch(facecolor=colors[i], label=f'AIS {i}') for i in range(6) if i in ais_chest_ot]
        plt.legend(handles=legend_elements, title='AIS LEVEL', loc='upper left')
        plt.title(f'impact velocity vs Dmax (OT={ot_val}, {ot_names[ot_val]})')
        plt.xlabel('impact velocity (km/h)')
        plt.ylabel('Dmax (mm)')
        plt.grid(True, alpha=0.3)

    # 颈部散点图
    plt.figure(figsize=(8,6))
    ais_colors = [colors[min(ais, 5)] for ais in ais_neck_labels]
    plt.scatter(impact_velocities, nij_labels, c=ais_colors, alpha=0.6, s=50)
    legend_elements = [Patch(facecolor=colors[i], label=f'AIS {i}') for i in range(6) if i in ais_neck_labels]
    plt.legend(handles=legend_elements, title='AIS LEVEL', loc='upper left')
    plt.title('impact velocity vs Nij')
    plt.xlabel('impact velocity (km/h)')
    plt.ylabel('Nij')
    plt.grid(True, alpha=0.3)

    plt.show()
    ############################################################################################


    print(f"筛选出的case数量: {len(case_ids_need)}")

    print("\n打包输入数据...")
    package_input_data(
        pulse_dir=pulse_dir,
        params_path=params_path,
        case_id_list=case_ids_need,
        output_path=os.path.join(output_dir, 'data_input.npz')
    )

    print("\n打包标签数据...")
    labels_output_path = os.path.join(output_dir, 'data_labels.npz')
    np.savez(
        labels_output_path,
        case_ids=case_ids_need,
        HIC=hic15_labels,
        Dmax=dmax_labels,
        Nij=nij_labels,
        AIS_head=ais_head_labels,
        AIS_chest=ais_chest_labels,
        AIS_neck=ais_neck_labels,
        MAIS=mais_labels
    )
    print(f"对应的标签已保存至: {labels_output_path}")