# export_pulse_model.py

import warnings
warnings.filterwarnings('ignore')
import os
import torch
import numpy as np
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import joblib

# 导入您项目中的模型定义和工具函数
import model.model as module_arch
from utils.util import InputScaler, inverse_transform

def create_sample_raw_input(raw_input_dict=None):
    """
    创建一个符合物理范围的、未经处理的原始输入样本。
    这模拟了在实际推理场景中,模型接口接收到的原始数据。

    Args:
        raw_input_dict (dict, optional): 包含 'impact_velocity', 'impact_angle', 'overlap' 键的字典。
                                         若提供,则使用其中的值;否则随机生成。

    Returns:
        np.ndarray:
            - 描述: 原始的、未经归一化的标量碰撞参数。
            - 数据类型: numpy.float32
            - 数据形状: (3,)
            - 数据格式: [impact_velocity, impact_angle, overlap]
    """
    if raw_input_dict is None:
        raw_input_dict = {}
        print("  ⚠ 未提供原始输入,使用随机生成的样本进行演示和验证。")

    impact_velocity = raw_input_dict.get('impact_velocity', np.random.uniform(25, 65))
    impact_angle = raw_input_dict.get('impact_angle', np.random.uniform(-45, 45))
    if 'overlap' in raw_input_dict:
        overlap = raw_input_dict['overlap']
    else:
        overlap = np.random.uniform(0.25, 1.0) if np.random.rand() > 0.5 else np.random.uniform(-1.0, -0.25)

    print("  ✓ 生成的原始输入 (impact_velocity, impact_angle, overlap):", f"({impact_velocity:.2f}, {impact_angle:.2f}, {overlap:.2f})")

    return np.array([impact_velocity, impact_angle, overlap], dtype=np.float32)

def preprocess_input(raw_params, scaler):
    """
    使用与训练时相同的InputScaler对原始输入数据进行预处理,使其符合模型输入要求。

    Args:
        raw_params (np.ndarray): 形状为 (3,) 的原始标量特征。

    Returns:
        torch.Tensor:
            - 描述: 经过归一化处理、可直接输入模型的张量。
            - 数据类型: torch.float32
            - 数据形状: (1, 3)
            - 数据格式: [[norm_velocity, norm_angle, norm_overlap]]
    """
    if scaler is None:
        raise ValueError("preprocess_input 函数必须提供有效的 InputScaler 实例。")
    velocity, angle, overlap = raw_params
    norm_velocity, norm_angle, norm_overlap = scaler.transform(velocity, angle, overlap)
    processed_params = np.array([norm_velocity, norm_angle, norm_overlap], dtype=np.float32)
    
    return torch.tensor(processed_params).unsqueeze(0)

def export_model(model, sample_input, output_path, opset_version=11):
    """
    将原始的PulseCNN模型直接导出为ONNX格式,保留其所有原生输出。
    """
    model.eval()

    # --- ONNX模型接口定义 ---
    # 验证模型是否支持GauNll_use,以确定输出数量
    if not getattr(model, 'GauNll_use', False):
        raise TypeError("此脚本当前只支持 GauNll_use=True 的 PulseCNN 模型导出。")

    # 输入节点 (Input Node)
    # - 名称 (Name): 'input' (与图中保持一致)
    # - 数据类型 (Data Type): float32
    # - 形状 (Shape): [batch_size, 3]
    input_names = ["input"]

    # 输出节点 (Output Nodes) - 包含所有中间尺度和方差
    # - 名称 (Name): 'out_mean0', 'out_mean1', 'out_mean2', 'out_var0', 'out_var1', 'out_var2'
    # - 数据类型 (Data Type): float32
    # - 形状 (Shape): 分别为 [batch_size, 3, 37], [batch_size, 3, 75], [batch_size, 3, 150] ...
    output_names = [
    "out_mean_s1", "out_var_s1",
    "out_mean_s2", "out_var_s2",
    "out_mean_s3", "out_var_s3"
    ]
    dynamic_axes = {"input": {0: "batch_size"}}
    for name in output_names:
        dynamic_axes[name] = {0: "batch_size"}

    torch.onnx.export(
        model,
        sample_input,
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        opset_version=opset_version,
    )
    print(f"✔ 模型已导出至: {output_path}")

    try:
        import onnx
        from onnxsim import simplify
        print("  正在简化ONNX模型...")
        onnx_model = onnx.load(output_path)
        model_simp, check = simplify(onnx_model)
        if check:
            onnx.save(model_simp, output_path)
            print("  ✔ ONNX模型简化完成。")
        else:
            print("  ✘ ONNX模型简化验证失败,保留原模型。")
    except ImportError:
        print("  ⚠ 未安装onnx-simplifier,跳过简化步骤。")

def plot_verification_comparison(pt_wave_orig, onnx_wave_orig, output_path, raw_params):
    """
    绘制PyTorch和ONNX模型【真实物理尺度】最终预测的波形对比图。
    """
    pt_wave_orig = pt_wave_orig.reshape(3, 150)
    onnx_wave_orig = onnx_wave_orig.reshape(3, 150)

    time = np.arange(1, 151)
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    
    title = (f'ONNX vs. PyTorch Verification (Real Physical Scale)\n'
             f'Raw Input: vel={raw_params[0]:.1f}, ang={raw_params[1]:.1f}, ov={raw_params[2]:.2f}')
    fig.suptitle(title, fontsize=15, fontweight='bold')

    labels = ['X-direction Accel (m/s^2)', 'Y-direction Accel (m/s^2)', 'Z-direction Rot. Accel (rad/s^2)']
    np.set_printoptions(suppress=True, precision=4)
    for i in range(3):
        axes[i].plot(time, pt_wave_orig[i, :], 'b-', linewidth=2.5, label='PyTorch Output', alpha=0.8)
        axes[i].plot(time, onnx_wave_orig[i, :], 'r--', linewidth=2, label='ONNX Output')
        axes[i].set_title(labels[i])
        axes[i].grid(True, linestyle='--', alpha=0.6)
        axes[i].legend()
        print(f"\n--- {labels[i]} (Peak Values) ---")
        print(f"  PyTorch Peak: {np.min(pt_wave_orig[i, :]):.4f}")
        print(f"  ONNX    Peak: {np.min(onnx_wave_orig[i, :]):.4f}")

    axes[2].set_xlabel('Time (ms)')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path)
    plt.close()
    print(f"\n  ✓ 对比图已保存至: {output_path}")

def verify_onnx_model(onnx_path, pytorch_model, sample_input, raw_params, scaler=None):
    """
    验证多输出的ONNX模型,并提取最终预测进行对比。
    此版本调用项目中已有的 inverse_transform 函数。
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("  ⚠ 未安装onnxruntime,跳过验证步骤。")
        return
    
    print("\n========== 验证 ONNX 模型 ==========")
    
    # --- PyTorch 推理 ---
    pytorch_model.eval()
    with torch.no_grad():
        pt_output_raw = pytorch_model(sample_input)
        pt_pred_normalized = pt_output_raw[0][-1]

    # --- ONNX Runtime 推理 ---
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    onnx_inputs = {"input": sample_input.cpu().numpy()}
    onnx_outputs_list = sess.run(None, onnx_inputs)
    onnx_pred_normalized = onnx_outputs_list[2]
    
    # --- 逆归一化,得到真实物理尺度输出 ---
    print("对模型最终预测(out_mean2)进行逆归一化...")
    if scaler:
        # 使用项目中已有的 inverse_transform 函数
        # 注意: 该函数返回tensor,且需要两个输入,我们只关心第一个输出
        pt_pred_orig_tensor, _ = inverse_transform(pt_pred_normalized, pt_pred_normalized, scaler)
        pt_pred_orig = pt_pred_orig_tensor.cpu().numpy()

        # 为ONNX的numpy输出创建一个tensor以使用该函数
        onnx_pred_tensor = torch.from_numpy(onnx_pred_normalized).to(sample_input.device)
        onnx_pred_orig_tensor, _ = inverse_transform(onnx_pred_tensor, onnx_pred_tensor, scaler)
        onnx_pred_orig = onnx_pred_orig_tensor.cpu().numpy()
    else:
        # 如果没有scaler,直接使用numpy数组
        print("  - 信息: 未提供scaler,跳过逆归一化,输出为归一化尺度。")
        pt_pred_orig = pt_pred_normalized.cpu().numpy()
        onnx_pred_orig = onnx_pred_normalized
    
    # --- 绘制对比图并打印关键信息 ---
    comparison_plot_path = Path(onnx_path).parent / f"{Path(onnx_path).stem}_comparison.png"
    plot_verification_comparison(pt_pred_orig, onnx_pred_orig, comparison_plot_path, raw_params)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="导出PulseCNN模型为ONNX格式(多输出)")
    parser.add_argument('-r', '--resume', type=str, default="E:\\WPS Office\\1628575652\\WPS企业云盘\\清华大学\\我的企业文档\\课题组相关\\理想项目\\LX_model_PulsePredict\\saved\\models\\HybridPulseCNN\\1213_095952\\model_best.pth", help="已训练模型的权重文件路径")
    parser.add_argument("--output_dir", type=str, default="./onnx_models", help="ONNX模型输出目录")
    parser.add_argument("--opset_version", type=int, default=17, help="ONNX opset版本")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cpu')

    print(f"从检查点加载配置: {args.resume}")
    checkpoint_path = Path(args.resume)
    config_path = checkpoint_path.parent / 'config.json'
    if not config_path.exists():
        config_path_try = checkpoint_path.parent.parent / 'config.json'
        if not config_path_try.exists():
             raise FileNotFoundError(f"错误: 在 {checkpoint_path.parent} 或其父目录中找不到 'config.json'。")
        config_path = config_path_try

    with open(config_path) as f:
        config = json.load(f)

    if config['arch']['type'] != 'PulseCNN':
        raise TypeError(f"此脚本专为 PulseCNN 模型设计,但配置文件中的模型类型为 {config['arch']['type']}。")

    scaler = None
    if config['data_loader_train']['args'].get('pulse_norm_mode', 'none') != 'none':
        scaler_path_str = config['data_loader_train']['args'].get('scaler_path')
        if scaler_path_str:
            scaler_path = Path(scaler_path_str)
            if scaler_path.exists():
                print(f"  加载波形归一化Scaler: {scaler_path}")
                scaler = joblib.load(scaler_path)
            else:
                print(f"  ⚠ 警告: Scaler文件 '{scaler_path}' 不存在,无法进行逆归一化。")
        else:
            print("  ⚠ 警告: 配置中未指定 'scaler_path',无法进行逆归一化。")

    print("\n创建并预处理样本输入...")

    dl_args = config['data_loader_train']['args']
    if 'physics_bounds' in dl_args:
        bounds = dl_args['physics_bounds']
    else:
        raise ValueError("config文件中缺少 'physics_bounds' 参数，请更新配置文件。")

    input_scaler = InputScaler(**bounds)

    raw_inputs = {
        'impact_velocity': 43.97580257,  # kph
        'impact_angle': 1.257534638,     # degrees
        'overlap': 0.468948606            #  overlap
    }
    raw_params = create_sample_raw_input(raw_inputs)
    sample_input = preprocess_input(raw_params, input_scaler).to(device)

    print("\n" + "="*50)
    print("加载PyTorch模型")
    print("="*50)
    
    model = module_arch.PulseCNN(**config['arch']['args']).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print("✔ 模型 'PulseCNN' 加载成功。")

    model_name = Path(config['name']).name
    onnx_path = os.path.join(args.output_dir, f"{model_name}.onnx")
    export_model(model, sample_input, onnx_path, args.opset_version)
    
    verify_onnx_model(onnx_path, model, sample_input, raw_params, scaler)

    print("\n" + "="*50)
    print("✔ 所有流程执行完毕！")
    print(f"ONNX 模型已保存在: {args.output_dir}")
    print("="*50)
