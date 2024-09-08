import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from datetime import datetime
from data_analysis import plot_train_test_dist


def check_graphs_v4(
    data,
    preds,
    interval=10000,
    img_path=None,
):
    plt.rcParams["font.size"] = 12
    if not img_path.exists():
        img_path.mkdir(parents=True)

    columns = preds.columns
    preds_dict = []
    for col in columns:
        preds_dict.append({"name": col, "preds": preds[col].values})

    piece = len(data) // interval
    for i in range(piece):
        start = i * interval
        end = min(start + interval, len(data))
        xticks = range(start, end)
        fig, axes = plt.subplots(len(columns) + 1, figsize=(12, 3 * (len(columns) + 1)))
        axes[0].ticklabel_format(style="plain", axis="both", scilimits=(0, 0))
        axes[0].set_xticks(np.arange(start, end, step=interval // 10))
        axes[0].set_ylim(-0.5, 2)
        axes[0].plot(xticks, data[start:end])
        axes[0].grid()
        axes[0].legend(["test data (true)"], loc="upper right")

        for j, pred in enumerate(preds_dict):
            axes[j + 1].ticklabel_format(style="plain", axis="both", scilimits=(0, 0))
            axes[j + 1].set_xticks(np.arange(start, end, step=interval // 10))
            axes[j + 1].plot(xticks, pred["preds"][start:end], color="blue", alpha=0.6)
            axes[j + 1].grid()
            axes[j + 1].legend([f"0.{pred['name']}"], loc="upper right")

        plt.tight_layout()
        fig.savefig(img_path / f"preds_{i+1:02d}")
        plt.close("all")


if __name__ == "__main__":
    data_path = Path("datasets/open")
    submit_path = data_path / "submissions"
    img_path = Path("saved/images")

    test_df = pd.read_pickle(data_path / "test.pkl")
    with open(data_path / "test_anomaly.pkl", "rb") as f:
        data_dict = pickle.load(f)
    test_preds = data_dict["test_preds"]
    test_scores = data_dict["test_score"]

    # test_dist.png
    plot_train_test_dist(test_df, pieces=17, img_path=img_path, mode="test")

    submit_files = submit_path.glob("*.csv")
    data_dict = []
    for sub in submit_files:
        sub_df = pd.read_csv(sub).drop("Timestamp", axis=1).reset_index(drop=True)
        name = sub.name.split(".")[0]
        preds = sub_df["anomaly"].values
        data_dict.append({"name": name, "preds": preds})

    columns = [d["name"] for d in data_dict]
    values = np.array([d["preds"] for d in data_dict])
    preds_df = pd.DataFrame(values.T, columns=columns)
    print(preds_df.info())

    check_graphs_v4(test_df, preds_df, 10000, img_path / "submissions")
