import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = 'T'
import warnings
warnings.filterwarnings('ignore')
import os
import json
import torch
import time
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
from utils import models
from utils.dataset_prepare import CrashDataset

from utils.set_random_seed import set_random_seed
set_random_seed()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
                batch_x_acc = batch[0].to(device)
                batch_x_att_continuous = batch[1].to(device)
                batch_x_att_discrete = batch[2].to(device)

                # 预热阶段 (仅在第一次迭代时执行)
                if i == 0:
                    for _ in range(50):
                        model(batch_x_acc, batch_x_att_continuous, batch_x_att_discrete)

                # 开始计时
                torch.cuda.synchronize() # 确保CUDA操作同步
                start_time = time.time()

                model(batch_x_acc, batch_x_att_continuous, batch_x_att_discrete)

                # 结束计时
                torch.cuda.synchronize() # 确保CUDA操作同步
                elapsed_time = time.time() - start_time
                total_time += elapsed_time

    # 计算平均推理时间
    avg_time = total_time / (num_runs * len(loader)) # 平均到每个批次
    print(f"Average inference time per batch: {avg_time:.6f} seconds")

if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser(description="Test Model Inference Time")
    # parser.add_argument("--run_dir", '-r', type=str, required=True, help="Directory of the training run.")
    # parser.add_argument("--weight_file", '-w', type=str, default="best_mais_accu.pth", help="Name of the model weight file.")
    # args = parser.parse_args()
    from dataclasses import dataclass
    @dataclass
    class args:
        run_dir: str = r'E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\LX_model_injurypredict\runs\InjuryPredictModel_10261509'
        weight_file: str = 'final_model.pth'

    # 加载超参数和训练记录
    with open(os.path.join(args.run_dir, "TrainingRecord.json"), "r") as f:
        training_record = json.load(f)

    # --- 从JSON结构中提取模型超参数 ---
    model_params = training_record["hyperparameters"]["model"]
    
    # 加载数据集
    dataset = CrashDataset()
    test_dataset1 = torch.load("./data/val_dataset.pt", weights_only=False)
    test_dataset2 = torch.load("./data/test_dataset.pt", weights_only=False)
    test_dataset = ConcatDataset([test_dataset1, test_dataset2])
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

    # --- 根据模型类型加载模型 ---
    model = models.InjuryPredictModel(
            **model_params
        ).to(device)
    
    model.load_state_dict(torch.load(os.path.join(args.run_dir, args.weight_file)))

    print(f"Start testing inference time for model: {args.weight_file}")
    test_inference_time(model, test_loader)