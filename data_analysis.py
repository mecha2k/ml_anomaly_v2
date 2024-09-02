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
    check_graphs(anomaly_score, prediction, threshold=threshold, name=image_path / "test_anomaly")

    sample_submission = pd.read_csv(data_path / "sample_submission.csv")
    sample_submission["anomaly"] = prediction
    sample_submission.to_csv(data_path / "final_submission.csv", encoding="UTF-8-sig", index=False)
    print(sample_submission["anomaly"].value_counts())


if __name__ == "__main__":
    data_path = Path("datasets/open")
    image_path = Path("saved/images")

    #     train_std = train_df.std()
    #     for col in train_df.columns:
    #         if train_std[col] < 0.01:
    #             print(f"Column {col} has {train_std[col]} std values")

    with open(data_path / "test_anomaly.pkl", "wb") as f:
        data_dict = {
            "train_preds": train_preds,
            "train_score": train_scores,
            "test_preds": test_preds,
            "test_score": test_scores,
            "threshold": {"assoc": 0.02, "recon": threshold},
        }
        pickle.dump(data_dict, f)

    threshold = data_dict["threshold"]
    anomalies = np.zeros_like(test_scores[0])
    # anomalies[test_scores[0] > data_dict["threshold"]["assoc"]] = 1
    anomalies[test_scores[1] > data_dict["threshold"]["recon"]] = 1

    fig = plt.figure(figsize=(12, 6))
    plt.hist(test_scores[1], range=(0, 0.2), bins=100)
    plt.grid()
    fig.savefig(img_path / "test_recon_hist.png")

    check_graphs_v3(
        test_loader.test,
        test_preds,
        test_scores,
        anomalies,
        threshold=threshold["recon"],
        img_path=img_path,
        mode="test",
    )

    sample_submission = pd.read_csv(data_path / "sample_submission.csv")
    sample_submission["anomaly"] = anomalies
    sample_submission.to_csv(data_path / "final_submission.csv", encoding="UTF-8-sig", index=False)
    print(sample_submission["anomaly"].value_counts())
