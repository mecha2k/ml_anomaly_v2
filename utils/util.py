import torch
import pandas as pd
import json
import platform
from pathlib import Path
from itertools import repeat
from collections import OrderedDict


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    for loader in repeat(data_loader):
        yield from loader


def prepare_device(n_gpu_use):
    if platform.system() == "Darwin":
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        list_ids = list(range(n_gpu_use))
        print(f"{device} is available in torch")
        return device, list_ids

    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print(
            "Warning: There's no GPU available on this machine,"
            "training will be performed on CPU."
        )
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(
            f"Warning: The number of GPU's configured to use is {n_gpu_use}, but only {n_gpu} are "
            "available on this machine."
        )
        n_gpu_use = n_gpu
    device = torch.device("cuda:0" if n_gpu_use > 0 else "cpu")
    list_ids = list(range(n_gpu_use))
    return device, list_ids


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=["total", "counts", "average"])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        print("Updating learning rate to {}".format(lr))


def association_discrepancy(series, prior, win_size=100):
    prior_d = torch.unsqueeze(torch.sum(prior, dim=-1), dim=-1)
    prior_d = prior_d.repeat(1, 1, 1, win_size)
    series1 = my_kl_loss(series, (prior / prior_d).detach())
    series2 = my_kl_loss((prior / prior_d).detach(), series)
    priors1 = my_kl_loss((prior / prior_d), series.detach())
    priors2 = my_kl_loss(series.detach(), (prior / prior_d))
    return torch.mean(series1 + series2), torch.mean(priors1 + priors2)


def association_discrepancy_t(series, prior, win_size=100, temperature=50):
    prior_d = torch.unsqueeze(torch.sum(prior, dim=-1), dim=-1)
    prior_d = prior_d.repeat(1, 1, 1, win_size)
    series1 = my_kl_loss(series, (prior / prior_d).detach()) * temperature
    priors1 = my_kl_loss((prior / prior_d), series.detach()) * temperature
    return series1, priors1
