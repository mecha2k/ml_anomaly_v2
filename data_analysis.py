import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from datetime import datetime
from utils import check_graphs_v3


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

    image_path = Path("saved/images")
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


def drop_columns(df):
    drop_columns = []
    for col in df.columns:
        if df[col].std() < 0.01:
            drop_columns.append(col)
    return df.drop(columns=drop_columns)


def plot_train_test_dist(df, pieces=10, img_path=None, mode="train"):
    if mode == "preds":
        anomalies = df["anomaly"].values
        df = df.drop(columns=["anomaly"])
    ncols = len(df.columns.values)
    num_plots = max(np.round(ncols / pieces, 0).astype(int), 1)
    xticks = np.arange(0, len(df), np.ceil(len(df) // 10))
    fig, ax = plt.subplots(pieces, 1, figsize=(12, 3 * pieces))
    for i in range(pieces):
        start = i * num_plots
        end = min(start + num_plots, ncols)
        ax[i].plot(df.iloc[:, start:end])
        if mode == "preds":
            ax[i].plot(anomalies * 1.5, color="cyan", alpha=0.3, linewidth=5)
        ax[i].set_xticks(xticks)
        ax[i].set_title(f"Columns {start} to {end}")
        ax[i].ticklabel_format(style="scientific", axis="x", scilimits=(0, 0))
        ax[i].legend(df.columns[start:end], loc="upper right")
        ax[i].grid()
        ax[i].set_ylim(-0.5, 1.5)
        ax[i].axhline(0, color="r", alpha=0.5)
        ax[i].axhline(1, color="r", alpha=0.5)
    fig.tight_layout()
    fig.savefig(img_path / f"{mode}_dist.png")


if __name__ == "__main__":
    data_path = Path("datasets/open")
    image_path = Path("saved/images")

    train_df = pd.read_pickle(data_path / "train.pkl")
    test_df = pd.read_pickle(data_path / "test.pkl")
    # train_df = train_df[:100000]
    # test_df = test_df[:100000]
    # test_df = test_df.iloc[:100000, :3]
    print(train_df.shape, test_df.shape)

    # plot_train_test_dist(train_df, pieces=17, img_path=image_path, mode="train")
    # plot_train_test_dist(test_df, pieces=17, img_path=image_path, mode="test")

    margins = 0.2
    anomaly_cols = []
    anomaly_dict = []
    predictions = np.zeros_like(test_df.values).transpose()
    for i, col in enumerate(test_df.columns):
        data = test_df[col].values
        loc_max = np.where(data > 1 + margins)[0]
        loc_min = np.where(data < 0 - margins)[0]
        if len(loc_max) or len(loc_min) > 0:
            anomaly_cols.append(col)
            predictions[i, loc_max] = 1
            predictions[i, loc_min] = 1
            loc = np.where(predictions[i] == 1)[0]
            anomay_ratio = len(loc) / len(data) * 100
            anomaly_dict.append(
                {"col": col, "anomalies": len(loc), "ratio": anomay_ratio}
            )
    print(f"Total {len(anomaly_cols)} columns have anomalies")
    for ano in anomaly_dict:
        print(
            f"{ano['col']}: {ano['anomalies']} anomalies detected with {ano['ratio']:.2f}% ratio"
        )
    print(f"Total {predictions.sum()} anomalies detected")

    preds_df = pd.DataFrame(predictions.transpose(), columns=test_df.columns)
    print(preds_df.shape)
    print(preds_df.sum(axis=0))  # sum of all anomalies detected in each column
    preds_df["anomaly"] = preds_df.sum(axis=1).astype(int)
    print(preds_df["anomaly"].value_counts())
    print(preds_df["anomaly"][3000:3500].values)

    test_df["anomaly"] = preds_df["anomaly"].apply(lambda x: 1 if x > 1 else 0)
    print(test_df["anomaly"].value_counts())
    print(test_df["anomaly"][3000:3500].values)
    print(f"Anomaly ratio: {test_df['anomaly'].sum() / len(test_df) * 100:.2f}%")
    plot_train_test_dist(test_df, pieces=17, img_path=image_path, mode="preds")

    # with open(data_path / "test_anomaly.pkl", "wb") as f:
    #     data_dict = {
    #         "train_preds": train_preds,
    #         "train_score": train_scores,
    #         "test_preds": test_preds,
    #         "test_score": test_scores,
    #         "threshold": {"assoc": 0.02, "recon": threshold},
    #     }
    #     pickle.dump(data_dict, f)

    # threshold = data_dict["threshold"]
    # anomalies = np.zeros_like(test_scores[0])
    # # anomalies[test_scores[0] > data_dict["threshold"]["assoc"]] = 1
    # anomalies[test_scores[1] > data_dict["threshold"]["recon"]] = 1
    #
    # fig = plt.figure(figsize=(12, 6))
    # plt.hist(test_scores[1], range=(0, 0.2), bins=100)
    # plt.grid()
    # fig.savefig(img_path / "test_recon_hist.png")
    #
    # check_graphs_v3(
    #     test_loader.test,
    #     test_preds,
    #     test_scores,
    #     anomalies,
    #     threshold=threshold["recon"],
    #     img_path=img_path,
    #     mode="test"
    # )

    sample_submission = pd.read_csv(data_path / "sample_submission.csv")
    sample_submission["anomaly"] = test_df["anomaly"]
    sample_submission.to_csv(
        data_path / "final_submission.csv", encoding="UTF-8-sig", index=False
    )
    print(sample_submission["anomaly"].value_counts())
