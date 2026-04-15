import torch
from abc import abstractmethod
from numpy import inf
import numpy as np

from PulsePredict.logger import TensorboardWriter


class BaseTrainer:
    """训练器基类，负责通用训练流程、检查点保存与恢复。"""

    def __init__(self, model, criterion, metric_ftns, optimizer, config, lr_scheduler=None):
        self.config = config
        self.logger = config.get_logger("trainer", config["trainer"]["verbosity"])

        self.model = model
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        trainer_cfg = config["trainer"]
        self.epochs = trainer_cfg["epochs"]
        self.save_period = trainer_cfg["save_period"]
        self.monitor = trainer_cfg.get("monitor", "off")

        if self.monitor == "off":
            self.mnt_mode = "off"
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            if self.mnt_mode not in ["min", "max"]:
                raise ValueError("`trainer.monitor` 的模式只能是 `min` 或 `max`。")

            self.mnt_best = inf if self.mnt_mode == "min" else -inf
            self.early_stop = trainer_cfg.get("early_stop", inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.start_epoch = 1
        self.checkpoint_dir = config.save_dir
        self.writer = TensorboardWriter(config.log_dir, self.logger, trainer_cfg["tensorboard"])

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        """执行完整训练流程。"""
        not_improved_count = 0

        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            log = {"epoch": epoch}
            log.update(result)

            for key, value in log.items():
                self.logger.info("    {:15s}: {}".format(str(key), value))

            best = False
            if self.mnt_mode != "off":
                if self.mnt_metric not in log:
                    raise KeyError(f"监控指标 `{self.mnt_metric}` 不存在，无法执行模型保存策略。")

                improved = (
                    self.mnt_mode == "min" and log[self.mnt_metric] <= self.mnt_best
                ) or (
                    self.mnt_mode == "max" and log[self.mnt_metric] >= self.mnt_best
                )

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info(f"验证集指标连续 {self.early_stop} 个 epoch 未提升，训练提前结束。")
                    break

            if epoch % self.save_period == 0 or best:
                self._save_checkpoint(epoch, save_best=best)

    def _save_checkpoint(self, epoch, save_best=False):
        """保存检查点。"""
        state = {
            "arch": type(self.model).__name__,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "criterion_state_dict": self.criterion.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
            "monitor_best": self.mnt_best,
            "config": self.config,
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all(),
            "numpy_rng_state": np.random.get_state(),
        }

        if epoch % self.save_period == 0:
            filename = str(self.checkpoint_dir / f"checkpoint-epoch{epoch}.pth")
            torch.save(state, filename)
            self.logger.info(f"已保存模型检查点：{filename}")

        if save_best:
            best_path = str(self.checkpoint_dir / "model_best.pth")
            torch.save(state, best_path)
            self.logger.info("已更新当前最佳模型：model_best.pth")

    def _resume_checkpoint(self, resume_path):
        """从检查点恢复训练。"""
        resume_path = str(resume_path)
        self.logger.info(f"正在加载检查点：{resume_path}")
        checkpoint = torch.load(resume_path)

        if checkpoint["config"]["arch"] != self.config["arch"]:
            raise ValueError("当前配置中的模型结构与检查点不一致，不能继续恢复训练。")
        if checkpoint["config"]["optimizer"]["type"] != self.config["optimizer"]["type"]:
            raise ValueError("当前配置中的优化器类型与检查点不一致，不能继续恢复训练。")
        if "criterion_state_dict" not in checkpoint:
            raise KeyError("检查点中缺少 `criterion_state_dict`，无法恢复任务级损失权重。")

        self.model.load_state_dict(checkpoint["state_dict"])
        self.criterion.load_state_dict(checkpoint["criterion_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        if self.lr_scheduler is not None:
            if "lr_scheduler" not in checkpoint or checkpoint["lr_scheduler"] is None:
                raise KeyError("当前训练配置启用了学习率调度器，但检查点中缺少 `lr_scheduler` 状态。")
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            self.logger.info("已恢复学习率调度器状态。")

        self.start_epoch = checkpoint["epoch"] + 1
        self.mnt_best = checkpoint["monitor_best"]

        if "rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["rng_state"])
        if "cuda_rng_state" in checkpoint and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        if "numpy_rng_state" in checkpoint:
            np.random.set_state(checkpoint["numpy_rng_state"])

        self.logger.info(f"检查点加载完成，将从 epoch {self.start_epoch} 继续训练。")
