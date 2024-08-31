import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
        device = (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
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
        self._data.loc[key, "total"] += value * n
        self._data.loc[key, "counts"] += n
        self._data.loc[key, "average"] = self._data.total[key] / self._data.counts[key]

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


def make_plot_image_array(inputs, output):
    fig, ax = plt.subplots(2, 1, figsize=(12, 6))
    ax[0].set_ylim(0, 1.2)
    ax[1].set_ylim(0, 1.2)
    ax[0].plot(inputs)
    ax[1].plot(output)
    ax[0].legend(["input (true)"], loc="upper right")
    ax[1].legend(["output (reconstructed)"], loc="upper right")
    ax[0].grid()
    ax[1].grid()
    plt.close("all")

    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    image = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 3)
    return image


def anomaly_scores(data_loader, model, device, criterion, win_size, temperature=50):
    model.eval()
    preds, ass_scores, rec_scores = [], [], []
    for i, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        output, series, prior, _ = model(data)
        loss = torch.mean(criterion(data, output), dim=-1)
        series_loss = 0.0
        priors_loss = 0.0
        for u in range(len(prior)):
            s_loss, p_loss = association_discrepancy_t(
                series[u], prior[u], win_size, temperature=temperature
            )
            series_loss += s_loss
            priors_loss += p_loss
        metric = torch.softmax((-series_loss - priors_loss), dim=-1)

        output = output.detach().cpu().numpy()
        association = (metric * loss).detach().cpu().numpy()
        reconstruction = np.abs(output - data.detach().cpu().numpy())
        output = np.concatenate(output, axis=0)
        association = np.concatenate(association, axis=0)
        reconstruction = np.concatenate(reconstruction, axis=0)
        preds.append(output)
        ass_scores.append(association)
        rec_scores.append(reconstruction)
    return (
        np.concatenate(preds, axis=0),
        [np.concatenate(ass_scores, axis=0), np.concatenate(rec_scores, axis=0)],
    )

    # def inference(model, data_loader, device="cuda"):
    #     model.eval()
    #     timestamps = []
    #     distances = []
    #     with torch.no_grad():
    #         for batch in tqdm(data_loader, desc="Inference", unit="batch"):
    #             inputs = batch["input"].to(device)
    #             targets = batch["target"].to(device)
    #             predictions = model(inputs)
    #             timestamps.extend(batch["timestamps"])
    #             distances.extend(torch.abs(targets - predictions).cpu().tolist())
    #     return np.array(timestamps), np.array(distances)
    # preds = []
    # with torch.no_grad():
    #     for i, (data, target) in enumerate(tqdm(train_loader)):
    #         data, target = data.to(device), target.to(device)
    #         output, _, _, _ = model(data)
    #         data = output.cpu().numpy()
    #         data = np.concatenate(data, axis=0)
    #         preds.append(data)
    # preds = np.concatenate(preds, axis=0)
