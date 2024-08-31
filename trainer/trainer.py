import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import (
    inf_loop,
    MetricTracker,
    association_discrepancy,
    association_discrepancy_t,
    make_plot_image_array,
)


class Trainer(BaseTrainer):
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
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        self.k = self.config["trainer"]["k"]
        self.win_size = self.config["data_loader"]["args"]["win_size"]
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker(
            "loss", *[m.__name__ for m in self.metric_ftns], writer=self.writer
        )
        self.valid_metrics = MetricTracker(
            "loss", *[m.__name__ for m in self.metric_ftns], writer=self.writer
        )

    def _train_epoch(self, epoch):
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output, series, priors, _ = self.model(data)

            # calculate Association discrepancy
            series_loss = 0.0
            priors_loss = 0.0
            for u in range(len(priors)):
                s_loss, p_loss = association_discrepancy(
                    series[u], priors[u], self.win_size
                )
                series_loss += s_loss
                priors_loss += p_loss
            series_loss = series_loss / len(priors)
            priors_loss = priors_loss / len(priors)

            reconstruction_loss = self.criterion(output, data)
            # total loss : minmax association learning
            loss = reconstruction_loss - self.k * series_loss

            loss1 = loss
            loss2 = reconstruction_loss + self.k * priors_loss

            loss1.backward(retain_graph=True)
            loss2.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update("loss", loss.item())
            met_values = [reconstruction_loss.item(), series_loss.item()]
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(met_values))

            if batch_idx % self.log_step == 0:
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), loss.item()
                    )
                )
                inputs = np.concatenate(data[:60].cpu().numpy(), axis=0)
                output = np.concatenate(output[:60].detach().cpu().numpy(), axis=0)
                images = make_plot_image_array(inputs, output)
                self.writer.add_image("input", images, dataformats="HWC")
                # self.writer.add_image("input", make_grid(data.cpu(), nrow=8, normalize=True))
            if batch_idx == self.len_epoch:
                break

        log = self.train_metrics.result()
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{"val_" + k: v for k, v in val_log.items()})
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)
                output, series, priors, _ = self.model(data)

                series_loss = 0.0
                for u in range(len(series)):
                    s_loss, _ = association_discrepancy(
                        series[u], priors[u], self.win_size
                    )
                    series_loss += s_loss
                series_loss = series_loss / len(series)

                reconstruction_loss = self.criterion(output, data)
                loss = reconstruction_loss - self.k * series_loss

                self.writer.set_step(
                    (epoch - 1) * len(self.valid_data_loader) + batch_idx, "valid"
                )
                self.valid_metrics.update("loss", loss.item())
                met_values = [reconstruction_loss.item(), series_loss.item()]
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(met_values))
                # self.writer.add_image("input", make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins="auto")
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.data_loader, "n_samples"):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
