import argparse

import numpy as np
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_model
from parse_config import ConfigParser
from utils import association_discrepancy, anomaly_scores


def main(config):
    logger = config.get_logger("test")

    # setup data_loader instances
    data_loader = getattr(module_data, config["data_loader"]["type"])(
        config["data_loader"]["args"]["data_dir"],
        batch_size=config["data_loader"]["args"]["batch_size"],
        training=False,
    )

    # build model architecture
    model = config.init_obj("model", module_model)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config["loss"])
    metric_fns = [getattr(module_metric, met) for met in config["metrics"]]

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps" if torch.backends.mps.is_available() else device)
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))
    win_size = config["data_loader"]["args"]["win_size"]

    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            output, series, priors, _ = model(data)
            # calculate Association discrepancy
            series_loss = 0.0
            for u in range(len(priors)):
                s_loss, p_loss = association_discrepancy(series[u], priors[u], win_size)
                series_loss += s_loss
            series_loss = series_loss / len(priors)
            reconstruction_loss = loss_fn(output, data)
            loss = reconstruction_loss - config["trainer"]["k"] * series_loss
            # computing loss, metrics on test set
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for j, metric in enumerate(metric_fns):
                total_metrics[j] += metric(output, data) * batch_size

    n_samples = len(data_loader.sampler)
    log = {"loss": total_loss / n_samples}
    log.update(
        {
            met.__name__: total_metrics[i].item() / n_samples
            for i, met in enumerate(metric_fns)
        }
    )
    logger.info(log)

    train_loader = getattr(module_data, config["data_loader"]["type"])(
        config["data_loader"]["args"]["data_dir"],
        batch_size=config["data_loader"]["args"]["batch_size"],
        training=True,
    )

    train_scores = anomaly_scores(
        train_loader, model, device, loss_fn, win_size, temperature=50
    )
    test_scores = anomaly_scores(
        data_loader, model, device, loss_fn, win_size, temperature=50
    )
    combined_scores = np.concatenate([train_scores, test_scores], axis=0)
    threshold = np.percentile(combined_scores, 100 - config["trainer"]["anomaly_ratio"])
    print("Threshold :", threshold)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c", "--config", default="config.json", type=str, help="config file path"
    )
    args.add_argument(
        "-r", "--resume", default=None, type=str, help="path to latest checkpoint"
    )
    args.add_argument(
        "-d", "--device", default=None, type=str, help="indices of GPUs to enable"
    )

    config = ConfigParser.from_args(args)
    main(config)
