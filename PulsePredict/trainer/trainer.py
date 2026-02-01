import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker, inverse_transform
import torch.nn.functional as F

class Trainer(BaseTrainer):
    """
    Generic Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
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
        # self.lr_scheduler = lr_scheduler
        
        len_epoch = len(self.data_loader)
        self.log_step = max(1, len_epoch // 5)
        
        # loss_names_to_track = [type(item['instance']).__name__ for item in self.criterion] # Old: 从配置中动态获取各项loss的名称用于监控
        # 从 AutoWeightedLoss 实例中直接获取 Loss 名称列表
        loss_names_to_track = self.criterion.loss_names

        self.train_metrics = MetricTracker('loss', *loss_names_to_track, *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *loss_names_to_track, *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        """
        self.model.train()
        self.train_metrics.reset()
        scaler = getattr(self.data_loader, 'target_scaler', None)
        
        for batch_idx, (data, target, case_ids) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            
            output = self.model(data)
            loss, loss_components = self.criterion(output, target)

            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            # 更新总损失和各项损失到MetricTracker
            self.train_metrics.update('loss', loss.item())
            for loss_name, loss_val in loss_components.items():
                self.train_metrics.update(loss_name, loss_val)

            # Get the primary model output for metrics calculation
            metrics_output = self.model.get_metrics_output(output)
            metrics_output_orig, target_orig = inverse_transform(metrics_output, target, scaler)

            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(metrics_output_orig, target_orig))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch, self._progress(batch_idx), loss.item()))

            if batch_idx == self.len_epoch:
                break

        # <--- 在每个训练 Epoch 结束时记录权重直方图 --->
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
            
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        """
        self.model.eval()
        self.valid_metrics.reset()
        scaler = getattr(self.data_loader, 'target_scaler', None)
        
        with torch.no_grad():
            for batch_idx, (data, target, case_ids) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss, loss_components = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                # 更新总损失和各项损失到MetricTracker
                self.valid_metrics.update('loss', loss.item())
                for loss_name, loss_val in loss_components.items():
                    self.valid_metrics.update(loss_name, loss_val)
                
                metrics_output = self.model.get_metrics_output(output)
                metrics_output_orig, target_orig = inverse_transform(metrics_output, target, scaler)

                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(metrics_output_orig, target_orig))

        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)