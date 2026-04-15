import os

os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "T"

import warnings

warnings.filterwarnings("ignore")

import argparse
import collections
import torch

import PulsePredict.data_loader.data_loaders as module_data
import PulsePredict.model.loss as module_loss
import PulsePredict.model.metric as module_metric
import PulsePredict.model.model as module_arch
from PulsePredict.parse_config import ConfigParser
from PulsePredict.trainer import Trainer
from PulsePredict.utils import prepare_device, get_parameter_groups
from common.tools.seeding import set_random_seed


def main(config):
    logger = config.get_logger("train")

    data_loader = config.init_obj("data_loader_train", module_data) # 只包含训练集数据的加载器，但内部已经划分好了验证集的范围
    valid_data_loader = data_loader.split_validation() # 把验证集加载器作为一个独立的对象分离出来

    logger.info(f"训练集样本数: {len(data_loader.train_test_indices)}")
    logger.info(f"验证集样本数: {len(data_loader.val_indices) if valid_data_loader is not None else 0}")

    model = config.init_obj("arch", module_arch)
    logger.info(model)

    device, device_ids = prepare_device(config["n_gpu"])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    criterion = config.init_obj("loss", module_loss).to(device)
    metrics = [getattr(module_metric, metric_name) for metric_name in config["metrics"]]

    global_weight_decay = config["optimizer"]["args"].get("weight_decay", 1e-3)
    head_decay_ratio = config["optimizer"].get("head_decay_ratio", 0.1)
    param_groups = get_parameter_groups(model, global_weight_decay, head_decay_ratio)

    criterion_params = [param for param in criterion.parameters() if param.requires_grad]
    if not criterion_params:
        raise ValueError("当前损失函数未提供可训练参数，无法进行任务级自动加权。")
    param_groups.append({"params": criterion_params, "weight_decay": 0.0})

    optimizer = config.init_obj("optimizer", torch.optim, param_groups)
    lr_scheduler = config.init_obj("lr_scheduler", torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(
        model,
        criterion,
        metrics,
        optimizer,
        config=config,
        device=device,
        data_loader=data_loader,
        valid_data_loader=valid_data_loader,
        lr_scheduler=lr_scheduler,
    )
    trainer.train()


if __name__ == "__main__":
    set_random_seed()

    args = argparse.ArgumentParser(description="PulsePredict 训练入口")
    args.add_argument(
        "-c",
        "--config",
        default="PulsePredict/config.json",
        type=str,
        help='config file path (default: PulsePredict/config.json)',
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help='path to the checkpoint which to resume (default: None)',
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help='indices of GPUs to enable (default: all)',
    )

    # 自定义命令行参数，用于覆盖配置文件中的设置
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size"),
    ]

    config = ConfigParser.from_args(args, options)
    main(config)
