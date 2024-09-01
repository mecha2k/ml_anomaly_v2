import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
from pathlib import Path

import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_model
from parse_config import ConfigParser
from utils import (
    association_discrepancy,
    anomaly_scores,
    check_graphs_v1,
    check_graphs_v2,
    check_graphs_v3,
)


def main(config):
    logger = config.get_logger("test")

    # setup data_loader instances
    test_loader = getattr(module_data, config["data_loader"]["type"])(
        config["data_loader"]["args"]["data_dir"],
        batch_size=config["data_loader"]["args"]["batch_size"],
        stride=config["data_loader"]["args"]["win_size"],
        training=False,
    )

    # build model architecture
    model = config.init_obj("model", module_model)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config["loss"])
    metric_fns = [getattr(module_metric, met) for met in config["metrics"]]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("mps" if torch.backends.mps.is_available() else device)
    print(f"Using device: {device}")

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(Path(config.resume), map_location=device)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))
    win_size = config["data_loader"]["args"]["win_size"]

    # with torch.no_grad():
    #     for i, (data, target) in enumerate(tqdm(test_loader)):
    #         data, target = data.to(device), target.to(device)
    #         output, series, priors, _ = model(data)
    #         # calculate Association discrepancy
    #         series_loss = 0.0
    #         for u in range(len(priors)):
    #             s_loss, p_loss = association_discrepancy(series[u], priors[u], win_size)
    #             series_loss += s_loss
    #         series_loss = series_loss / len(priors)
    #         reconstruction_loss = loss_fn(output, data)
    #         loss = reconstruction_loss - config["trainer"]["k"] * series_loss
    #         # computing loss, metrics on test set
    #         batch_size = data.shape[0]
    #         total_loss += loss.item() * batch_size
    #         met_values = [reconstruction_loss.item(), series_loss.item()]
    #         for j, metric in enumerate(metric_fns):
    #             total_metrics[j] += metric(met_values) * batch_size

    n_samples = len(test_loader.sampler)
    log = {"loss": total_loss / n_samples}
    log.update(
        {met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)}
    )
    logger.info(log)

    train_loader = getattr(module_data, config["data_loader"]["type"])(
        config["data_loader"]["args"]["data_dir"],
        batch_size=config["data_loader"]["args"]["batch_size"],
        stride=config["data_loader"]["args"]["win_size"],
        training=True,
    )

    # preds = []
    # with torch.no_grad():
    #     for i, (data, target) in enumerate(tqdm(train_loader)):
    #         data, target = data.to(device), target.to(device)
    #         output, _, _, _ = model(data)
    #         data = output.cpu().numpy()
    #         data = np.concatenate(data, axis=0)
    #         preds.append(data)
    # preds = np.concatenate(preds, axis=0)

    # check_graphs_v3(
    #     train_loader.train, preds, img_path=Path("saved/images"), mode="train"
    # )

    # fig = plt.figure(figsize=(12, 6))
    # plt.ylim(0, 1)
    # plt.plot(data[:3200])
    # fig.savefig("saved/images/test_loader_samples.png")
    #
    # print("-" * 100)
    # fig = plt.figure(figsize=(12, 6))
    # plt.ylim(0, 1)
    # plt.plot(train_loader.train[:3200])
    # fig.savefig("saved/images/test_data_samples.png")
    #

    train_preds, train_scores = anomaly_scores(
        train_loader, model, device, win_size, temperature=50
    )
    test_preds, test_scores = anomaly_scores(test_loader, model, device, win_size, temperature=50)
    # combined_assoc = np.concatenate([train_scores[0], test_scores[0]], axis=0)
    # combined_recon = np.concatenate([train_scores[1], test_scores[1]], axis=0)
    anomaly_ratio = config["trainer"]["anomaly_ratio"]
    threshold = np.percentile(test_scores[0], 100 - anomaly_ratio)
    print(f"train data : {len(train_scores[0])}, test data : {len(test_scores[0])}")
    print(f"mean association discrepancy : {np.mean(test_scores[0])}")
    print(f"mean reconstruction error : {np.mean(test_scores[1])}")
    print(f"Threshold with {100-anomaly_ratio}% percentile : {threshold:.4e}")

    data_path = Path(config["data_loader"]["args"]["data_dir"])
    img_path = Path("saved/images")

    with open(data_path / "test_anomaly.pkl", "wb") as f:
        data_dict = {
            "train_preds": train_preds,
            "train_score": train_scores,
            "test_preds": test_preds,
            "test_score": test_scores,
            "threshold": {"assoc": 0.02, "recon": 0.06},
        }
        pickle.dump(data_dict, f)

    threshold = data_dict["threshold"]
    anomalies = np.zeros_like(test_scores[0])
    anomalies[test_scores[0] > data_dict["threshold"]["assoc"]] = 1
    # anomalies[test_scores[1] > data_dict["threshold"]["recon"]] = 1

    fig = plt.figure(figsize=(12, 6))
    plt.ylim(0, 1)
    plt.plot(anomalies, linewidth=2)
    fig.savefig(img_path / "test_anomaly_prediction.png")

    # check_graphs_v3(
    #     train_loader.train,
    #     train_preds,
    #     train_scores,
    #     anomalies,
    #     threshold["assoc"],
    # img_path = img_path,
    # mode = "train",
    # )
    check_graphs_v3(
        test_loader.test,
        test_preds,
        test_scores,
        anomalies,
        threshold=threshold["assoc"],
        img_path=img_path,
        mode="test",
    )

    # fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    # # fig.subtitle("Anomaly Scores Histogram")
    # axes[0].set_ylabel("Association Discrepancy")
    # axes[0].set_ylim(0, 0.3)
    # axes[0].ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))
    # axes[0].plot(test_scores[0], color="b")
    # axes[1].set_ylabel("Reconstruction Error")
    # axes[1].set_ylim(0, 0.3)
    # axes[1].ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))
    # axes[1].plot(test_scores[1], color="g")
    # fig.savefig("saved/images/anomaly_scores_plot.png")

    # check_graphs_v1(
    #     test_scores, prediction, threshold, name=image_path / "test_anomaly"
    # )
    #
    # test_df = pd.read_pickle(data_path / "test.pkl")
    # check_graphs_v2(
    #     test_df.values, prediction, test_scores, img_path=image_path, mode="test"
    # )
    #
    sample_submission = pd.read_csv(data_path / "sample_submission.csv")
    sample_submission["anomaly"] = anomalies
    sample_submission.to_csv(data_path / "final_submission.csv", encoding="UTF-8-sig", index=False)
    print(sample_submission["anomaly"].value_counts())


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument("-c", "--config", default="config.json", type=str, help="config file path")
    args.add_argument("-r", "--resume", default=None, type=str, help="path to latest checkpoint")
    args.add_argument("-d", "--device", default=None, type=str, help="indices of GPUs to enable")

    config = ConfigParser.from_args(args)
    main(config)
