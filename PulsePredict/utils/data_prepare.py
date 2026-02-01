import os
import numpy as np
import pandas as pd
from tqdm import tqdm

def package_pulse_data(pulse_dir, params_path, case_id_list, output_path):
    """
    处理、降采样并将指定案例的输入参数和输出波形数据打包在一起。
    注意输入的路径用 pathlib.Path 处理。

    该函数会读取工况参数文件，并根据给定的 case_id 列表，匹配对应的
    原始波形CSV文件。然后将输入参数、输出波形和 case_id 作为一个整体
    保存到一个结构化的 .npz 文件中。

    :param pulse_dir: 存放原始波形CSV文件的目录。
    :param params_path: 包含所有工况参数的文件路径 (包含 'case_id' 列)。.npz 或 .csv 格式均可。
    :param case_id_list: 需要处理的案例ID列表。
    :param output_path: 打包后的 .npz 文件保存路径。
    """
    # --- 1. 加载并索引工况参数 ---
    try:
        if params_path.suffix == '.csv':
            all_params_data = pd.read_csv(params_path)
            params_df = pd.DataFrame({
                'case_id': all_params_data['case_id'].to_numpy(),
                'impact_velocity': all_params_data['impact_velocity'].to_numpy(),
                'impact_angle': all_params_data['impact_angle'].to_numpy(),
                'overlap': all_params_data['overlap'].to_numpy()
            }).set_index('case_id')
        elif params_path.suffix == '.npz':
            all_params_data = np.load(params_path)
            params_df = pd.DataFrame({
                'case_id': all_params_data['case_id'],
                'impact_velocity': all_params_data['impact_velocity'],
                'impact_angle': all_params_data['impact_angle'],
                'overlap': all_params_data['overlap']
            }).set_index('case_id')
        else:
            raise ValueError("参数文件必须是 .csv 或 .npz 格式。")
    except Exception as e:
        print(f"错误：加载或处理工况参数文件 '{params_path}' 时出错: {e}")
        return

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
            x_path = os.path.join(pulse_dir, f'x{case_id}.csv')
            y_path = os.path.join(pulse_dir, f'y{case_id}.csv')
            z_path = os.path.join(pulse_dir, f'z{case_id}.csv')

            if not all(os.path.exists(p) for p in [x_path, y_path, z_path]):
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
            az_full = pd.read_csv(z_path, sep='\t', header=None, usecols=[1]).values.flatten()

            ax_sampled = ax_full[downsample_indices]
            ay_sampled = ay_full[downsample_indices]
            az_sampled = az_full[downsample_indices]

            # ************************************************************************
            # 只取前150个点
            ax_sampled = ax_sampled[:150]
            ay_sampled = ay_sampled[:150]
            az_sampled = az_sampled[:150]
            # ************************************************************************
                 
            waveforms_np = np.stack([ax_sampled, ay_sampled, az_sampled]).squeeze() # 形状 (3, 150)

            # --- 4. 提取匹配的参数 ---
            params_row = params_df.loc[case_id]
            params_np = np.array([
                params_row['impact_velocity'],
                params_row['impact_angle'],
                params_row['overlap']
            ], dtype=np.float32) # 形状 (3,)

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
    final_params = np.stack(processed_params, axis=0) # 形状 (N, 3)
    final_waveforms = np.stack(processed_waveforms, axis=0) # 形状 (N, 3, 150)

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
    from pathlib import Path
    pulse_dir = Path(r'G:\VCS_acc_data\acc_data_before1111_6134')
    params_path = Path(r'E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\仿真数据库相关\distribution\distribution_1112.csv')
    output_dir = Path(r'E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\仿真数据库相关\VCS波形数据集打包\acc_data_before1111_6134')

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    # 分割训练集和测试集
    if params_path.suffix == '.csv':  # 
        # all_case_ids = pd.read_csv(params_path)['case_id']
        is_pulse_ok = pd.read_csv(params_path)['is_pulse_ok'] # 只选择波形数据OK的工况
        pulse_ok_case_ids = pd.read_csv(params_path).loc[is_pulse_ok == True, 'case_id']
        # 只选<50000的case_id
        pulse_ok_case_ids = pulse_ok_case_ids[pulse_ok_case_ids < 50000]
    elif params_path.suffix == '.npz':
        # all_case_ids = np.load(params_path)['case_id']
        is_pulse_ok = np.load(params_path)['is_pulse_ok']
        pulse_ok_case_ids = np.load(params_path)['case_id'][is_pulse_ok == True]
        # 只选<50000的case_id
        pulse_ok_case_ids = pulse_ok_case_ids[pulse_ok_case_ids < 50000]
    else:
        raise ValueError("参数文件必须是 .csv 或 .npz 格式。")
    
    all_case_ids = pulse_ok_case_ids.tolist()
    print(f"*** 总共有 {len(all_case_ids)} 个工况波形数据 OK ***")
    np.random.shuffle(all_case_ids)
    num_train = int(len(all_case_ids) * 0.86)
    train_case_ids = all_case_ids[:num_train]
    test_case_ids = all_case_ids[num_train:]


    print("\n打包训练集数据...")
    package_pulse_data(
        pulse_dir=pulse_dir,
        params_path=params_path,
        case_id_list=train_case_ids,
        output_path=os.path.join(output_dir, 'packaged_data_train.npz')
    )

    print("\n打包测试集数据...")
    package_pulse_data(
        pulse_dir=pulse_dir,
        params_path=params_path,
        case_id_list=test_case_ids,
        output_path=os.path.join(output_dir, 'packaged_data_test.npz')
    )