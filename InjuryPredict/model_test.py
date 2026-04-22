"""
通用化模型测试函数,用于查看模型结构、检测是否正常前向和反向传播：
1. 接受任意模型实例化对象 `model`。
2. 自定义输入 `inputs` 和标签 `labels`。
3. 支持前向传播、反向传播、损失计算。
4. 导出 ONNX 模型并验证。
5. 输出模型详细信息。
6. 计算 FLOPS 和推理速度（用于评估算力需求）。
"""
import warnings
warnings.filterwarnings('ignore')
import os
import json
import time

import torch
import torch.nn as nn
import torch.onnx
from torchinfo import summary
from torchviz import make_dot
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset

from InjuryPredict.Injurydata_prepare import load_processed_subset
from common.tools.seeding import set_random_seed
from InjuryPredict.utils import models
from common.settings import get_injury_processed_dataset_path
# ==========================================================================================
# inference timing utility (merged from test_inference_time.py)
def test_inference_time(model, loader):
    """
    测试模型推理时间
    参数:
        model: 模型实例。
        loader: 数据加载器。
    """
    model.eval()
    total_time = 0.0
    num_runs = 200  # 推理次数

    with torch.no_grad():
        for i in range(num_runs):
            for batch in loader:
                # 只取模型输入需要的部分，并移动到设备
                batch_x_acc_pred = batch[1].to(device)
                batch_x_att_continuous = batch[2].to(device)
                batch_x_att_discrete = batch[3].to(device)

                # 预热阶段 (仅在第一次迭代时执行)
                if i == 0:
                    for _ in range(50):
                        model(batch_x_acc_pred, batch_x_att_continuous, batch_x_att_discrete) # [B, 2, L], [B, C], [B, D] -> [B, 3]

                # 开始计时
                if device.type == 'cuda':
                    torch.cuda.synchronize() # 确保CUDA操作同步
                start_time = time.time()

                model(batch_x_acc_pred, batch_x_att_continuous, batch_x_att_discrete) # [B, 2, L], [B, C], [B, D] -> [B, 3]

                # 结束计时
                if device.type == 'cuda':
                    torch.cuda.synchronize() # 确保CUDA操作同步
                elapsed_time = time.time() - start_time
                total_time += elapsed_time

    # 计算平均推理时间
    avg_time = total_time / (num_runs * len(loader)) # 平均到每个批次
    print(f"Average inference time per batch: {avg_time:.6f} seconds")

# ==========================================================================================
# 自定义 FLOPs 计算 Hooks (用于 ptflops 不默认支持的操作)
# ==========================================================================================

def gru_flops_counter_hook(module, input, output):
    """
    GRU 的 FLOPs 计算 Hook
    
    GRU 每个时间步的计算量:
    - 3个门 (reset, update, new): 每个门需要 2 * (input_size + hidden_size) * hidden_size 次乘加
    - 总计: 6 * (input_size + hidden_size) * hidden_size 次乘加 (MACs)
    - 双向: 再乘以 2
    - 多层: 第一层 input_size 为原始输入，后续层为 hidden_size * num_directions
    """
    input_tensor = input[0]
    batch_size = input_tensor.size(0)
    seq_len = input_tensor.size(1)
    input_size = module.input_size
    hidden_size = module.hidden_size
    num_layers = module.num_layers
    bidirectional = module.bidirectional
    num_directions = 2 if bidirectional else 1
    
    total_macs = 0
    
    for layer in range(num_layers):
        layer_input_size = input_size if layer == 0 else hidden_size * num_directions
        macs_per_step = 3 * 2 * (layer_input_size + hidden_size) * hidden_size
        layer_macs = macs_per_step * seq_len * batch_size
        layer_macs *= num_directions
        total_macs += layer_macs
    
    module.__flops__ += int(total_macs)


def lstm_flops_counter_hook(module, input, output):
    """
    LSTM 的 FLOPs 计算 Hook
    
    LSTM 每个时间步的计算量:
    - 4个门 (input, forget, cell, output): 每个门需要 2 * (input_size + hidden_size) * hidden_size 次乘加
    - 总计: 8 * (input_size + hidden_size) * hidden_size 次乘加 (MACs)
    """
    input_tensor = input[0]
    batch_size = input_tensor.size(0)
    seq_len = input_tensor.size(1)
    input_size = module.input_size
    hidden_size = module.hidden_size
    num_layers = module.num_layers
    bidirectional = module.bidirectional
    num_directions = 2 if bidirectional else 1
    
    total_macs = 0
    
    for layer in range(num_layers):
        layer_input_size = input_size if layer == 0 else hidden_size * num_directions
        macs_per_step = 4 * 2 * (layer_input_size + hidden_size) * hidden_size
        layer_macs = macs_per_step * seq_len * batch_size
        layer_macs *= num_directions
        total_macs += layer_macs
    
    module.__flops__ += int(total_macs)


def embedding_flops_counter_hook(module, input, output):
    """
    Embedding 的 FLOPs 计算 Hook
    
    Embedding 仅进行查表操作，理论上没有浮点运算
    """
    module.__flops__ += 0


def calculate_flops_with_ptflops(model, input_data, device, print_per_layer=False):
    """
    使用 ptflops 计算模型的 FLOPs
    
    参数:
        model: PyTorch 模型
        input_data: 输入数据 (tuple 或 tensor)
        device: 计算设备
        print_per_layer: 是否打印每层的 FLOPs 详情
    
    返回:
        dict: 包含 FLOPs、MACs、参数量等信息的字典
    """
    try:
        from ptflops import get_model_complexity_info
    except ImportError:
        print("   ⚠️  ptflops 未安装，请运行: pip install ptflops")
        return {
            'macs': 0,
            'flops': 0,
            'params': sum(p.numel() for p in model.parameters()),
            'success': False,
            'error': 'ptflops not installed'
        }
    
    # 获取输入形状 (不包含 batch 维度)
    if isinstance(input_data, (tuple, list)):
        # 多输入情况，取第一个输入的形状
        input_shape = tuple(input_data[0].shape[1:])
    else:
        input_shape = tuple(input_data.shape[1:])
    
    # 创建自定义 hooks 字典
    custom_hooks = {
        nn.GRU: gru_flops_counter_hook,
        nn.LSTM: lstm_flops_counter_hook,
        nn.Embedding: embedding_flops_counter_hook,
    }
    
    # 定义输入构造函数（处理多输入情况）
    def input_constructor(input_res):
        if isinstance(input_data, (tuple, list)):
            # 返回与原始输入相同结构的数据
            return tuple(
                torch.randn(1, *inp.shape[1:]).to(device) if inp.dtype.is_floating_point 
                else torch.zeros(1, *inp.shape[1:], dtype=inp.dtype).to(device)
                for inp in input_data
            )
        else:
            return torch.randn(1, *input_res).to(device)
    
    model.eval()
    
    try:
        macs, params = get_model_complexity_info(
            model, 
            input_shape,
            input_constructor=input_constructor,
            as_strings=False,
            print_per_layer_stat=print_per_layer,
            verbose=print_per_layer,
            custom_modules_hooks=custom_hooks,
        )
        
        # MACs to FLOPs: 1 MAC ≈ 2 FLOPs
        flops = macs * 2
        
        return {
            'macs': macs,
            'flops': flops,
            'params': params,
            'success': True
        }
        
    except Exception as e:
        print(f"   ⚠️  ptflops 计算失败: {e}")
        return {
            'macs': 0,
            'flops': 0,
            'params': sum(p.numel() for p in model.parameters()),
            'success': False,
            'error': str(e)
        }


def print_flops_analysis(flops_info, batch_size=1):
    """
    打印 FLOPs 分析结果和指标解释
    """
    macs = flops_info['macs']
    flops = flops_info['flops']
    params = flops_info['params']
    
    print("\n   " + "─" * 60)
    print("   📊 核心指标")
    print("   " + "─" * 60)
    
    # 参数量
    print(f"   参数量 (Params)      : {params:>15,} ({params/1e6:.2f} M)")
    
    # MACs
    print(f"   乘加次数 (MACs)      : {macs:>15,} ({macs/1e9:.3f} G)")
    
    # FLOPs
    print(f"   浮点运算数 (FLOPs)   : {flops:>15,} ({flops/1e9:.3f} G)")
    
    # 换算为 TOPS (Tera Operations Per Second)
    print("\n   " + "─" * 60)
    print("   ⚡ 算力需求 (假设满负载推理)")
    print("   " + "─" * 60)
    
    fps_targets = [30, 60, 120, 1000]
    for fps in fps_targets:
        tops = (flops * fps) / 1e12
        gflops = (flops * fps) / 1e9
        label = f"@{fps}fps (batch={batch_size})"
        print(f"   {label:<25}: {gflops:>8.2f} GFLOPS = {tops:.4f} TOPS")
    
    # 添加典型硬件参考
    print("\n   " + "─" * 60)
    print("   🖥️  典型硬件算力参考 (FP32)")
    print("   " + "─" * 60)
    print("   RTX 3060              :      12.74 TFLOPS")
    print("   RTX 3080              :      29.77 TFLOPS")
    print("   RTX 4090              :      82.58 TFLOPS")
    print("   A100 (40GB)           :      19.49 TFLOPS")
    print("   嵌入式 Jetson Orin    :       5.32 TFLOPS")
    print("   车规级 TDA4VM         :       8.00 TOPS (INT8)")
    
    # 指标解释
    print("\n   " + "─" * 60)
    print("   📖 指标解释")
    print("   " + "─" * 60)
    print("""
   • 参数量 (Parameters)
     模型中可学习权重的总数。影响模型存储大小和内存占用。
     1M 参数 ≈ 4MB (FP32) / 2MB (FP16) / 1MB (INT8)

   • MACs (Multiply-Accumulate Operations)
     乘加运算次数。一次 MAC = 一次乘法 + 一次加法。
     这是深度学习中最核心的计算单元。

   • FLOPs (Floating Point Operations)
     浮点运算次数。通常 FLOPs ≈ 2 × MACs。
     注意: 不同文献可能混用 FLOPs 和 MACs，需注意区分。

   • GFLOPS / TFLOPS (Giga/Tera FLOPs Per Second)
     每秒十亿/万亿次浮点运算，衡量算力需求或硬件性能。
     模型需求: FLOPs × FPS; 硬件供给: 峰值 TFLOPS

   • TOPS (Tera Operations Per Second)
     每秒万亿次运算 (通常指 INT8/INT4 整数运算)。
     1 TOPS = 1e12 OPS。常用于衡量 NPU/VPU 性能。

   • 算力利用率
     实际推理时，由于内存带宽、并行效率等因素，
     通常只能达到硬件峰值算力的 30%-70%。
""")


def test_model(
    model,
    inputs,
    labels,
    criterion=None,
    optimizer=None,
    onnx_file_path="model_test.onnx",
    test_inference_speed=True,
    num_warmup=10,
    num_iterations=100,
    print_flops_per_layer=False
):
    """
    通用化模型测试函数：
    1. 接受任意模型实例化对象 `model`。
    2. 自定义输入 `inputs` 和标签 `labels`。
    3. 支持前向传播、反向传播、损失计算。
    4. 导出 ONNX 模型并验证。
    5. 输出模型详细信息。
    6. 计算 FLOPS 和推理速度（用于评估算力需求）。
    
    参数：
    - model: PyTorch 模型实例化对象
    - inputs: 模型的输入张量 (tensor / tuple / list)
    - labels: 模型的真实标签张量（用于损失计算）
    - criterion: 损失函数实例化对象，默认为 nn.MSELoss
    - optimizer: 优化器实例化对象，默认为 Adam
    - onnx_file_path: 导出的 ONNX 文件路径
    - test_inference_speed: 是否测试推理速度，默认 True
    - num_warmup: 预热次数，默认 10
    - num_iterations: 测试推理次数，默认 100
    - print_flops_per_layer: 是否打印每层的 FLOPs 详情，默认 False
    """
    # ==================== 初始化设置 ====================
    if criterion is None:
        criterion = nn.MSELoss()
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 统一处理输入数据格式
    _input_data = tuple(inputs) if isinstance(inputs, (tuple, list)) else inputs
    batch_size = inputs[0].shape[0] if isinstance(inputs, (tuple, list)) else inputs.shape[0]
    device = next(model.parameters()).device
    original_training = model.training
    
    # batch_size=1 时必须使用 eval 模式（避免 BatchNorm 错误）
    use_eval_mode = (batch_size == 1)
    
    print("\n" + "=" * 80)
    print("🚀 开始测试神经网络模型")
    print("=" * 80)
    
    if use_eval_mode:
        print(f"\n⚠️  batch_size=1，全程使用评估模式 (model.eval())")
        print("   提示：如需测试反向传播，请将 batch_size 设置为 > 1")
    else:
        print(f"\n✅ batch_size={batch_size}，支持训练模式测试")
    
    # ==================== 模型结构信息 ====================
    print("\n" + "-" * 40)
    print("📊 模型结构信息")
    print("-" * 40)
    
    try:
        model.eval()
        summary(
            model,
            input_data=_input_data,
            col_names=["input_size", "output_size", "num_params"],
            depth=3,
            device=str(device)
        )
    except Exception as e:
        print(f"⚠️  torchinfo.summary 执行失败: {e}")
    
    # ==================== 算力需求评估 (使用 ptflops) ====================
    print("\n" + "-" * 40)
    print("⚡ 算力需求评估 (使用 ptflops)")
    print("-" * 40)
    
    flops_info = calculate_flops_with_ptflops(
        model, 
        _input_data, 
        device,
        print_per_layer=print_flops_per_layer
    )
    
    if flops_info['success']:
        print_flops_analysis(flops_info, batch_size=batch_size)
    else:
        print(f"   ⚠️  FLOPs 统计失败，尝试使用 fvcore 备用方案...")
        try:
            from fvcore.nn import FlopCountAnalysis, parameter_count
            
            model.eval()
            flops_analysis = FlopCountAnalysis(model, _input_data)
            total_flops = flops_analysis.total()
            params = parameter_count(model)['']
            
            backup_info = {
                'macs': total_flops // 2,
                'flops': total_flops,
                'params': params,
                'success': True
            }
            print_flops_analysis(backup_info, batch_size=batch_size)
        except Exception as e:
            print(f"   ⚠️  备用方案也失败: {e}")
            print(f"   参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # ==================== 推理速度测试 ====================
    if test_inference_speed:
        print("\n" + "-" * 40)
        print("⏱️  推理速度测试")
        print("-" * 40)
        
        model.eval()
        with torch.no_grad():
            # 预热
            for _ in range(num_warmup):
                # 通用测试入口不固定模型输入形状；具体输入形状由 test_model 的 inputs 参数决定，并在后续统一打印。
                if isinstance(_input_data, tuple):
                    _ = model(*_input_data)
                else:
                    _ = model(_input_data)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            # 计时测试
            start_time = time.time()
            for _ in range(num_iterations):
                # 通用测试入口不固定模型输入形状；此处仅测量同一 inputs 下的重复前向耗时。
                if isinstance(_input_data, tuple):
                    _ = model(*_input_data)
                else:
                    _ = model(_input_data)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()
            
            avg_time = (end_time - start_time) / num_iterations * 1000
            fps = 1000 / avg_time
            
            print(f"   预热次数    : {num_warmup}")
            print(f"   测试次数    : {num_iterations}")
            print(f"   平均耗时    : {avg_time:.2f} ms")
            print(f"   推理速度    : {fps:.2f} FPS")
            print(f"   吞吐量      : {fps * batch_size:.2f} samples/s")
            
            # 计算实际算力消耗
            if flops_info['success']:
                actual_gflops = (flops_info['flops'] * fps) / 1e9
                actual_tops = actual_gflops / 1000
                print(f"   实际算力消耗: {actual_gflops:.2f} GFLOPS ({actual_tops:.4f} TOPS)")
            
            if device.type == 'cuda':
                memory = torch.cuda.max_memory_allocated(device) / 1024**2
                print(f"   GPU内存占用 : {memory:.2f} MB")
                torch.cuda.reset_peak_memory_stats(device)
    
    # ==================== 前向传播 ====================
    print("\n" + "-" * 40)
    print("🔄 前向传播测试")
    print("-" * 40)
    
    # 根据 batch_size 设置模式
    if use_eval_mode:
        model.eval()
    else:
        model.train()
    
    # 通用测试入口不固定模型输入形状；输出结构由后续分支根据实际返回值打印。
    if isinstance(_input_data, tuple):
        outputs = model(*_input_data)
    else:
        outputs = model(_input_data)
    
    # 打印输入信息
    if isinstance(inputs, (tuple, list)):
        input_shapes = [str(tuple(inp.shape)) for inp in inputs]
        print(f"   输入形状    : [{', '.join(input_shapes)}]")
    else:
        print(f"   输入形状    : {tuple(inputs.shape)}")
    
    # 打印输出信息
    if isinstance(outputs, (tuple, list)) and not isinstance(outputs, torch.Tensor):
        output_info = _format_output_structure(outputs)
        print(f"   输出结构    : [{', '.join(output_info)}]")
        loss, matched_output = _compute_loss_multi_output(outputs, labels, criterion)
    else:
        print(f"   输出形状    : {tuple(outputs.shape)}")
        loss, matched_output = _compute_loss_single_output(outputs, labels, criterion)
    
    if loss is not None:
        print(f"   损失值      : {loss.item():.6f}")
    
    # ==================== 反向传播 ====================
    print("\n" + "-" * 40)
    print("🔙 反向传播测试")
    print("-" * 40)
    
    if use_eval_mode:
        print("   ⏭️  跳过（batch_size=1 不支持训练模式）")
    elif loss is None:
        print("   ⏭️  跳过（无有效损失值）")
    else:
        try:
            model.train()
            optimizer.zero_grad()
            
            # 需要重新前向传播
            # 反向传播测试复用相同 inputs，确保前向、损失和梯度检查处于同一输入形状下。
            if isinstance(_input_data, tuple):
                outputs_train = model(*_input_data)
            else:
                outputs_train = model(_input_data)
            
            if isinstance(outputs_train, (tuple, list)) and not isinstance(outputs_train, torch.Tensor):
                loss_train, _ = _compute_loss_multi_output(outputs_train, labels, criterion)
            else:
                loss_train, _ = _compute_loss_single_output(outputs_train, labels, criterion)
            
            loss_train.backward()
            optimizer.step()
            print("   ✅ 反向传播正常")
        except Exception as e:
            print(f"   ❌ 反向传播失败: {e}")
    
    # ==================== 计算图可视化 ====================
    print("\n" + "-" * 40)
    print("📈 计算图可视化")
    print("-" * 40)
    
    try:
        if loss is not None:
            graph = make_dot(loss, params=dict(model.named_parameters()))
            graph.render("model_computation_graph", format="png", cleanup=True)
            print("   ✅ 已保存: model_computation_graph.png")
        else:
            print("   ⏭️  跳过（无有效损失值）")
    except Exception as e:
        print(f"   ⚠️  失败: {e}")
    
    # ==================== ONNX 导出 ====================
    print("\n" + "-" * 40)
    print("📦 ONNX 模型导出")
    print("-" * 40)
    
    try:
        model.eval()
        
        # 配置输入输出名称
        if isinstance(inputs, (tuple, list)):
            input_names = [f"input_{i}" for i in range(len(inputs))]
            dynamic_axes = {name: {0: "batch_size"} for name in input_names}
        else:
            input_names = ["input"]
            dynamic_axes = {"input": {0: "batch_size"}}
        
        output_names, output_dynamic = _get_output_names(outputs)
        dynamic_axes.update(output_dynamic)
        
        torch.onnx.export(
            model, _input_data, onnx_file_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=11,
        )
        print(f"   ✅ 已保存: {onnx_file_path}")
        print("   📎 可视化: https://netron.app/")
    except Exception as e:
        print(f"   ⚠️  导出失败: {e}")
    
    # 恢复原始状态
    model.train(original_training)
    
    print("\n" + "=" * 80)
    print("✅ 模型测试完成")
    print("=" * 80 + "\n")


def _format_output_structure(outputs):
    """格式化多输出结构信息"""
    info = []
    for output in outputs:
        if isinstance(output, torch.Tensor):
            info.append(f"Tensor{tuple(output.shape)}")
        elif isinstance(output, (tuple, list)):
            sub = [f"Tensor{tuple(o.shape)}" if isinstance(o, torch.Tensor) else str(type(o).__name__) for o in output]
            info.append(f"({', '.join(sub)})")
        else:
            info.append(type(output).__name__)
    return info


def _compute_loss_multi_output(outputs, labels, criterion):
    """从多输出中计算损失"""
    for output in outputs:
        current = output[0] if isinstance(output, (tuple, list)) else output
        if isinstance(current, torch.Tensor) and current.shape == labels.shape:
            return criterion(current, labels), current
    
    # 使用第一个输出
    first = outputs[0]
    first = first[0] if isinstance(first, (tuple, list)) else first
    if isinstance(first, torch.Tensor):
        try:
            return criterion(first, labels), first
        except:
            return None, None
    return None, None


def _compute_loss_single_output(outputs, labels, criterion):
    """从单输出计算损失"""
    if outputs.shape == labels.shape:
        return criterion(outputs, labels), outputs
    try:
        return criterion(outputs, labels), outputs
    except:
        return None, None


def _get_output_names(outputs):
    """获取输出名称和动态轴配置"""
    names, axes = [], {}
    idx = 0
    
    if isinstance(outputs, (tuple, list)) and not isinstance(outputs, torch.Tensor):
        for out in outputs:
            if isinstance(out, (tuple, list)):
                for sub in out:
                    if isinstance(sub, torch.Tensor):
                        names.append(f"output_{idx}")
                        axes[f"output_{idx}"] = {0: "batch_size"}
                        idx += 1
            elif isinstance(out, torch.Tensor):
                names.append(f"output_{idx}")
                axes[f"output_{idx}"] = {0: "batch_size"}
                idx += 1
    else:
        names = ["output"]
        axes = {"output": {0: "batch_size"}}
    
    return names, axes


def _run_demo_model_tests():
    """原先 __main__ 中执行的结构与 FLOPs 等演示逻辑"""
    import os
    import numpy as np

    from utils.models import InjuryPredictModel
    from config import model_params
    
    from common.settings import get_injury_processed_dataset_path
    train_pt = get_injury_processed_dataset_path("train")
    train_dataset = torch.load(train_pt.as_posix()) 

    # --- 从 config 加载超参数 ---
    Ksize_init = model_params['Ksize_init']
    Ksize_mid = model_params['Ksize_mid']
    num_blocks_of_tcn = model_params['num_blocks_of_tcn']
    tcn_channels_list = model_params['tcn_channels_list']
    tcn_output_dim = model_params['tcn_output_dim']
    num_layers_of_mlpE = model_params['num_layers_of_mlpE']
    num_layers_of_mlpD = model_params['num_layers_of_mlpD']
    mlpE_hidden = model_params['mlpE_hidden']
    mlpD_hidden = model_params['mlpD_hidden']
    mlp_encoder_output_dim = model_params['mlp_encoder_output_dim']
    mlp_decoder_output_dim = model_params['mlp_decoder_output_dim']
    dropout_TCN = model_params['dropout_TCN']
    dropout_MLP = model_params['dropout_MLP']
    use_channel_attention = model_params['use_channel_attention']
    fixed_channel_weight = model_params['fixed_channel_weight']

    model = InjuryPredictModel(
        Ksize_init=Ksize_init,
        Ksize_mid=Ksize_mid,
        num_classes_of_discrete=train_dataset.dataset.num_classes_of_discrete,
        num_blocks_of_tcn=num_blocks_of_tcn,
        tcn_channels_list=tcn_channels_list,
        tcn_output_dim=tcn_output_dim,
        num_layers_of_mlpE=num_layers_of_mlpE,
        num_layers_of_mlpD=num_layers_of_mlpD,
        mlpE_hidden=mlpE_hidden,
        mlpD_hidden=mlpD_hidden,
        mlp_encoder_output_dim=mlp_encoder_output_dim,
        mlp_decoder_output_dim=mlp_decoder_output_dim,
        dropout_MLP=dropout_MLP,
        dropout_TCN=dropout_TCN,
        use_channel_attention=use_channel_attention,
        fixed_channel_weight=fixed_channel_weight
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 示例输入数据（模拟数据集第1个batch）
    batch_size = 64

    x_acc_pred = torch.tensor(train_dataset.dataset.x_acc_pred[:batch_size], dtype=torch.float32).to(device)
    x_att_con = torch.tensor(train_dataset.dataset.x_att_continuous[:batch_size], dtype=torch.float32).to(device)
    x_att_dis = torch.tensor(train_dataset.dataset.x_att_discrete[:batch_size], dtype=torch.long).to(device)
    y_HIC = torch.tensor(train_dataset.dataset.y_HIC[:batch_size], dtype=torch.float32).to(device)
    y_Dmax = torch.tensor(train_dataset.dataset.y_Dmax[:batch_size], dtype=torch.float32).to(device)
    y_Nij = torch.tensor(train_dataset.dataset.y_Nij[:batch_size], dtype=torch.float32).to(device)
    y = torch.stack([y_HIC, y_Dmax, y_Nij], dim=1) # [B], [B], [B] -> [B, 3]

    print(f"\n{'='*80}")
    print(f"模型: {type(model).__name__} | 设备: {device}")
    print(f"输入: x_acc_pred{tuple(x_acc_pred.shape)}, x_att_con{tuple(x_att_con.shape)}, x_att_dis{tuple(x_att_dis.shape)}")
    print(f"标签: {tuple(y.shape)}")
    print(f"{'='*80}")

    criterion = nn.MSELoss()
    
    test_model(
        model, 
        inputs=(x_acc_pred, x_att_con, x_att_dis), 
        labels=y,
        criterion=criterion,
        print_flops_per_layer=False
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="模型测试脚本，包含结构/推理/ONNX/FLOPs 等功能")
    parser.add_argument("--mode", choices=["model", "inference"], default="model",
                        help="运行模式：model - 演示模型结构及FLOPs；inference - 测试推理时间")
    parser.add_argument("--run_dir", "-r", type=str, default=r'E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\LX_model_injurypredict\runs\InjuryPredictModel_03032051',
                        help="训练结果目录（仅在 inference 模式下需要）")
    parser.add_argument("--weight_file", "-w", type=str, default="best_val_loss.pth",
                        help="模型权重文件名（仅在 inference 模式下需要）")
    args = parser.parse_args()
    set_random_seed()

    if args.mode == "inference":
        # copy logic from original test_inference_time main
        if args.run_dir is None:
            parser.error("--run_dir required when mode is inference")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # load training_record json
        with open(os.path.join(args.run_dir, "TrainingRecord.json"), "r", encoding="utf-8") as f:
            training_record = json.load(f)
        model_params = training_record["hyperparameters"]["model"]

        # prepare dataset
        test_dataset1 = load_processed_subset(get_injury_processed_dataset_path("val"))
        test_dataset2 = load_processed_subset(get_injury_processed_dataset_path("test"))
        test_dataset = ConcatDataset([test_dataset1, test_dataset2])
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

        model = models.InjuryPredictModel(**model_params).to(device)
        model.load_state_dict(torch.load(os.path.join(args.run_dir, args.weight_file), map_location=device))

        print(f"Start testing inference time for model: {args.weight_file}")
        test_inference_time(model, test_loader)
    else:
        _run_demo_model_tests()
