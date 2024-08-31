import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from pathlib import Path
from datetime import datetime

from sympy.abc import alpha


def check_graphs_v1(data, preds, threshold=None, name="default", piece=15):
    interval = len(data) // piece
    fig, axes = plt.subplots(piece, figsize=(20, 4 * piece))
    # data_mean = np.round(data.mean() * 1.5, 3)
    print(data.max(), data.min())
    for i in range(piece):
        start = i * interval
        end = min(start + interval, len(data))
        axes[i].set_ylim(0, 1.5)
        axes[i].plot(data[start:end], color="blue")
        axes[i].plot(preds[start:end], color="green")
        if threshold is not None:
            axes[i].axhline(y=threshold, color="red")
    plt.tight_layout()
    plt.savefig(name)
    plt.close("all")


def check_graphs_v2(data, preds, anomaly, interval=10000, img_path=None, mode="train"):
    pieces = int(len(data) // interval)
    for i in range(pieces):
        start = i * interval
        end = min(start + interval, len(data))
        xticks = list(range(start, end, 1000))
        values = data[start:end]
        plt.figure(figsize=(16, 8))
        plt.ylim(-0.5, 1.5)
        plt.xticks(xticks)
        plt.grid()
        plt.plot(values)
        plt.plot(preds[start:end], color="green", linewidth=8)
        plt.plot(anomaly[start:end], color="blue", linewidth=5)
        plt.savefig(img_path / f"{mode}_raw_data" / f"raw_{i+1:02d}_pages")
        plt.close("all")


def check_graphs_v3(
    data, preds, scores, threshold=None, interval=10000, img_path=None, mode="train"
):
    plt.rcParams["font.size"] = 16
    piece = len(data) // interval
    for i in range(piece):
        start = i * interval
        end = min(start + interval, len(data))
        xticks = range(start, end)
        fig, axes = plt.subplots(3, figsize=(16, 12))
        axes[0].ticklabel_format(style="plain", axis="both", scilimits=(0, 0))
        axes[0].set_xticks(np.arange(start, end, step=interval // 10))
        axes[0].set_ylim(-0.25, 1.25)
        axes[0].plot(xticks, data[start:end])
        axes[0].grid()
        axes[0].legend([f"{mode} data (true)"], loc="upper right")
        axes[1].ticklabel_format(style="plain", axis="both", scilimits=(0, 0))
        axes[1].set_xticks(np.arange(start, end, step=interval // 10))
        axes[1].set_ylim(-0.25, 1.25)
        axes[1].plot(xticks, preds[start:end], alpha=1.0)
        axes[1].grid()
        axes[1].legend([f"{mode} data (reconstruction)"], loc="upper right")
        axes[2].ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))
        axes[2].set_xticks(np.arange(start, end, step=interval // 10))
        axes[2].plot(xticks, scores[0][start:end], color="b", alpha=1)
        axes[2].grid()
        axes[2].legend([f"{mode} association"], loc="upper right")
        axes[2].axhline(y=threshold, color="r", linewidth=5)
        axes[2].set_ylabel("Association Scores")
        twins = axes[2].twinx()
        # twins.set_ylim(0, 1)
        twins.plot(xticks, scores[1][start:end], color="g", alpha=0.1)
        twins.set_ylabel("Reconstruction Scores")
        plt.tight_layout()
        fig.savefig(img_path / f"{mode}_preds_data" / f"pred_{i+1:02d}_pages")
        if i == 2:
            break


def fill_blank_data(timestamps, datasets, total_ts):
    # create dataframes with total_ts index and 0 values
    df_total = pd.DataFrame(0, index=total_ts, columns=["outputs"]).astype(float)
    df_total.index = pd.to_datetime(df_total.index)
    df_partial = pd.DataFrame(datasets, index=timestamps, columns=["outputs"])
    df_partial.index = pd.to_datetime(df_partial.index)
    df_total.update(df_partial)
    return df_total["outputs"].values


def anomaly_prediction(scores, piece=15):
    mean_std, percentile = [], []
    interval = len(scores) // piece
    for i in range(piece):
        start = i * interval
        end = min(start + interval, len(scores))
        mean_std.append(scores[start:end].mean() + 2 * scores[start:end].std())
        percentile.append(np.percentile(scores[start:end], 99))
    return mean_std, percentile


def inference(model, data_loader, device="cuda"):
    model.eval()
    timestamps = []
    distances = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Inference", unit="batch"):
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            predictions = model(inputs)
            timestamps.extend(batch["timestamps"])
            distances.extend(torch.abs(targets - predictions).cpu().tolist())
    return np.array(timestamps), np.array(distances)


def final_submission(model, data_loader, device, data_path):
    timestamps, distances = inference(model, data_loader, device=device)
    anomaly_score = np.mean(distances, axis=1)
    attacks = np.zeros_like(anomaly_score)
    timestamps_raw = data_loader.test_df_raw["Timestamp"]

    with open(data_path / "test_anomaly.pkl", "wb") as f:
        data_dict = {
            "timestamps": timestamps,
            "anomaly_score": anomaly_score,
            "attacks": attacks,
            "timestamps_raw": timestamps_raw,
        }
        pickle.dump(data_dict, f)

    image_path = Path("../saved/images")
    # threshold = np.percentile(anomaly_score, 99)
    threshold = np.mean(anomaly_score) + 2 * np.std(anomaly_score)
    print(f"mean-std based Threshold: {threshold}")
    anomaly_score = fill_blank_data(timestamps, anomaly_score, np.array(timestamps_raw))
    prediction = np.zeros_like(anomaly_score)
    prediction[anomaly_score > threshold] = 1
    check_graphs(
        anomaly_score, prediction, threshold=threshold, name=image_path / "test_anomaly"
    )

    sample_submission = pd.read_csv(data_path / "sample_submission.csv")
    sample_submission["anomaly"] = prediction
    sample_submission.to_csv(
        data_path / "final_submission.csv", encoding="UTF-8-sig", index=False
    )
    print(sample_submission["anomaly"].value_counts())


if __name__ == "__main__":
    data_path = Path("../datasets/open")
    image_path = Path("../saved/images")

    with open(data_path / "test_anomaly.pkl", "rb") as f:
        data_dict = pickle.load(f)
    test_scores = data_dict["test_score"]
    threshold = data_dict["threshold"]

    prediction = np.zeros_like(test_scores)
    prediction[test_scores > threshold] = 1
    preds_df = pd.DataFrame(prediction, columns=["anomaly"])
    print(preds_df["anomaly"].value_counts())
    print(
        test_scores.max(),
        test_scores.min(),
        np.round(test_scores.mean(), 3),
        test_scores.std(),
    )
    check_graphs_v1(
        test_scores, prediction, threshold, name=image_path / "test_anomaly"
    )

    # train_df = pd.read_pickle(data_path / "train.pkl")
    # train = train_df.values
    # check_graphs_v2(train, np.zeros_like(train), img_path=image_path, mode="train")
    test_df = pd.read_pickle(data_path / "test.pkl")
    check_graphs_v2(
        test_df.values, prediction, test_scores, img_path=image_path, mode="test"
    )

    sample_submission = pd.read_csv(data_path / "sample_submission.csv")
    sample_submission["anomaly"] = prediction
    sample_submission.to_csv(
        data_path / "final_submission.csv", encoding="UTF-8-sig", index=False
    )
    print(sample_submission["anomaly"].value_counts())
