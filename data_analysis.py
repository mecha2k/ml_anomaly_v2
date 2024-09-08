import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from datetime import datetime

from networkx.algorithms.bipartite import color


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
            ax[i].plot(anomalies * 1.5, color="red", alpha=0.3, linewidth=5)
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


def minmax_anomaly_detection(data_path, image_path):
    train_df = pd.read_pickle(data_path / "train.pkl")
    test_df = pd.read_pickle(data_path / "test.pkl")
    print(train_df.shape, test_df.shape)

    plot_train_test_dist(train_df, pieces=17, img_path=image_path, mode="train")
    plot_train_test_dist(test_df, pieces=17, img_path=image_path, mode="test")

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


def check_graphs_v3(
    data,
    preds,
    scores,
    anomalies,
    threshold=None,
    interval=10000,
    img_path=None,
    mode="train",
):
    plt.rcParams["font.size"] = 16
    save_dir = img_path / f"{mode}_preds_data"
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    submit_path = Path("datasets/open") / "submissions"
    submit_files = submit_path.glob("*.csv")
    submit_files = sorted(sub for sub in submit_files)
    submit_df = pd.read_csv(submit_files[-1])
    submit_data = submit_df["anomaly"].values
    submit_name = submit_files[-1].name.split(".")[0]

    piece = len(data) // interval
    inputs = [data, preds, scores[0], scores[1]]
    titles = [
        "Input data (true)",
        "Outputs (Reconstruction)",
        "Association Scores",
        "Reconstruction Scores",
    ]
    for i in range(piece):
        start = i * interval
        end = min(start + interval, len(data))
        xticks = range(start, end)
        fig, axes = plt.subplots(4, figsize=(16, 16))

        for j, dt in enumerate(inputs):
            axes[j].ticklabel_format(style="plain", axis="both", scilimits=(0, 0))
            axes[j].set_xticks(np.arange(start, end, step=interval // 10))
            axes[j].set_ylim(-0.5, 2)
            axes[j].plot(xticks, dt[start:end])
            axes[j].grid()
            axes[j].legend([f"{titles[j]}"], loc="upper right")
            if mode == "test":
                if j == 2:
                    axes[j].set_ylim(0, 0.05)
                    axes[j].plot(
                        xticks,
                        submit_data[start:end] * 0.03,
                        color="green",
                        linewidth=5,
                    )
                    axes[j].axhline(y=0.02, color="r", linewidth=2, alpha=0.5)
                    axes[j].set_title(
                        f"Anomaly Detection with 0.{submit_name}", color="g"
                    )
                if j == 3:
                    axes[j].set_ylim(0, 0.3)
                    axes[j].plot(
                        xticks, anomalies[start:end] * 0.1, color="green", linewidth=5
                    )
                    axes[j].axhline(y=threshold, color="r", linewidth=2, alpha=0.5)

        plt.tight_layout()
        fig.savefig(save_dir / f"pred_{i+1:02d}_pages")
        plt.close("all")
        break


def transformer_anomaly_detection(data_path, img_path, anomaly_ratio=10):
    with open(data_path / "test_anomaly.pkl", "rb") as f:
        data_dict = pickle.load(f)

    # train_df = pd.read_pickle(data_path / "train.pkl")
    # train_preds = data_dict["train_preds"]
    # train_scores = data_dict["train_score"]

    # check_graphs_v3(
    #     train_df,
    #     train_preds,
    #     train_scores,
    #     np.zeros_like(train_scores[0]),
    #     threshold=threshold,
    #     img_path=img_path,
    #     mode="train",
    # )

    test_df = pd.read_pickle(data_path / "test.pkl")
    test_preds = data_dict["test_preds"]
    test_scores = data_dict["test_score"]

    # combined_assoc = np.concatenate([train_scores[0], test_scores[0]], axis=0)
    # combined_recon = np.concatenate([train_scores[1], test_scores[1]], axis=0)
    threshold = min(np.percentile(test_scores[0], 100 - anomaly_ratio), 0.03)
    print(f"Threshold with {100 - anomaly_ratio}% percentile : {threshold:.3e}")

    threshold = 0.04

    predictions = np.zeros_like(test_scores[1])
    predictions[test_scores[1] > threshold] = 1

    # Input(true), Outputs(Reconstruction), Association Scores, Reconstruction Scores
    check_graphs_v3(
        test_df,
        test_preds,
        test_scores,
        predictions,
        threshold=threshold,
        img_path=img_path,
        mode="test",
    )

    return predictions


if __name__ == "__main__":
    data_path = Path("datasets/open")
    image_path = Path("saved/images")

    # minmax_anomaly_detection(data_path, image_path)
    predictions = transformer_anomaly_detection(data_path, image_path, anomaly_ratio=10)

    sample_submission = pd.read_csv(data_path / "sample_submission.csv")
    sample_submission["anomaly"] = predictions
    sample_submission.to_csv(
        data_path / "final_submission.csv", encoding="UTF-8-sig", index=False
    )
    print(sample_submission["anomaly"].value_counts())
