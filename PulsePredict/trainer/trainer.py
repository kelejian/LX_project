import torch
from PulsePredict.base import BaseTrainer
from PulsePredict.utils import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    """PulsePredict 训练器。"""

    def __init__(
        self,
        model,
        criterion,
        metric_ftns,
        optimizer,
        config,
        device,
        data_loader,
        valid_data_loader=None,
        lr_scheduler=None,
        len_epoch=None,
    ):
        super().__init__(model, criterion, metric_ftns, optimizer, config, lr_scheduler=lr_scheduler)
        self.config = config
        self.device = device
        self.data_loader = data_loader

        if len_epoch is None:
            self.len_epoch = len(self.data_loader)
        else:
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.log_step = max(1, len(self.data_loader) // 5)

        loss_names = self.criterion.loss_names
        metric_names = [metric.__name__ for metric in self.metric_ftns]
        self.train_metrics = MetricTracker("loss", *loss_names, *metric_names, writer=self.writer)
        self.valid_metrics = MetricTracker("loss", *loss_names, *metric_names, writer=self.writer)

    def _train_epoch(self, epoch):
        self.model.train()
        self.train_metrics.reset()

        processor = getattr(self.data_loader, "processor", None)
        if processor is None:
            raise RuntimeError("数据集缺少 `processor`，无法执行反归一化评估。")

        for batch_idx, (data, target, case_ids) in enumerate(self.data_loader):
            data = data.to(self.device)
            target = target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss, loss_components = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update("loss", loss.item())
            # 这里写入 TensorBoard 的各子项 `Loss` 来自 `loss_components`，对应的是各任务的原始损失值；
            # 这些数值不包含 Kendall 自适应缩放系数，也不包含 `log_var` 正则项，目的在于保留可解释的物理监控曲线。
            for loss_name, loss_value in loss_components.items():
                self.train_metrics.update(loss_name, loss_value)

            metrics_output = self.model.get_metrics_output(output)
            with torch.no_grad():
                metrics_output_np = processor.process_waveform(metrics_output.detach().cpu().numpy(), inverse=True)
                target_np = processor.process_waveform(target.detach().cpu().numpy(), inverse=True)

            metrics_output_orig = torch.from_numpy(metrics_output_np).to(metrics_output.device).type_as(metrics_output)
            target_orig = torch.from_numpy(target_np).to(target.device).type_as(target)

            for metric in self.metric_ftns:
                self.train_metrics.update(metric.__name__, metric(metrics_output_orig, target_orig))

            if batch_idx % self.log_step == 0:
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(epoch, self._progress(batch_idx), loss.item())
                )

            if batch_idx + 1 >= self.len_epoch:
                break

        for name, param in self.model.named_parameters():
            self.writer.add_histogram(name, param, bins="auto")

        # 这里额外记录的是任务级权重状态，而不是原始子损失；
        # 因此 TensorBoard 中会同时存在两类曲线：一类是原始任务损失，另一类是 Kendall 权重相关统计量。
        for loss_name, stats in self.criterion.get_weight_state().items():
            for stat_name, stat_value in stats.items():
                self.writer.add_scalar(f"{stat_name}/{loss_name}", stat_value)

        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{f"val_{key}": value for key, value in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        self.model.eval()
        self.valid_metrics.reset()

        processor = getattr(self.data_loader, "processor", None)
        if processor is None:
            raise RuntimeError("数据集缺少 `processor`，无法执行反归一化评估。")

        with torch.no_grad():
            for batch_idx, (data, target, case_ids) in enumerate(self.valid_data_loader):
                data = data.to(self.device)
                target = target.to(self.device)

                output = self.model(data)
                loss, loss_components = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, "valid")
                self.valid_metrics.update("loss", loss.item())
                # 验证阶段记录的各子项 `Loss` 与训练阶段一致，均为未乘入自适应权重的原始任务损失。
                for loss_name, loss_value in loss_components.items():
                    self.valid_metrics.update(loss_name, loss_value)

                metrics_output = self.model.get_metrics_output(output)
                metrics_output_np = processor.process_waveform(metrics_output.detach().cpu().numpy(), inverse=True)
                target_np = processor.process_waveform(target.detach().cpu().numpy(), inverse=True)

                metrics_output_orig = torch.from_numpy(metrics_output_np).to(metrics_output.device).type_as(metrics_output)
                target_orig = torch.from_numpy(target_np).to(target.device).type_as(target)

                for metric in self.metric_ftns:
                    self.valid_metrics.update(metric.__name__, metric(metrics_output_orig, target_orig))

        for name, param in self.model.named_parameters():
            self.writer.add_histogram(name, param, bins="auto")

        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        template = "[{}/{} ({:.0f}%)]"
        if hasattr(self.data_loader, "n_samples"):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return template.format(current, total, 100.0 * current / total)
