import torch
import numpy as np
import pandas as pd
import pickle
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
from base import BaseDataLoader


class HmcDataset(Dataset):
    def __init__(self, data, win_size=100):
        self.data = data
        self.win_size = win_size

    def __len__(self):
        return (self.data.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        idx = index * self.win_size
        data = np.float32(self.data[idx : idx + self.win_size])
        target = np.zeros_like(data)
        return data, target


class HmcDataLoader(BaseDataLoader):
    def __init__(
        self,
        data_dir,
        batch_size,
        win_size=100,
        training=True,
    ):
        data_path = Path(data_dir)
        self.scaler = MinMaxScaler()

        train_df_raw = pd.read_csv(data_path / "train.csv")
        train_df_raw = train_df_raw[:1000]
        data_columns = train_df_raw.columns.drop(["Timestamp", "anomaly"])
        train_df = train_df_raw[data_columns].astype(float)
        scaler = self.scaler.fit(train_df)
        train = scaler.transform(train_df)
        train_df = pd.DataFrame(train, columns=train_df.columns, index=list(train_df.index.values))
        train_df = train_df.ewm(alpha=0.9).mean()
        train_df.to_pickle(data_path / "train.pkl")

        test_df_raw = pd.read_csv(data_path / "test.csv")
        test_df_raw = test_df_raw[:1000]
        test_df = test_df_raw[data_columns].astype(float)
        test = scaler.transform(test_df)
        test_df = pd.DataFrame(test, columns=test_df.columns, index=list(test_df.index.values))
        test_df = test_df.ewm(alpha=0.9).mean()
        test_df.to_pickle(data_path / "test.pkl")

        self.train_df = train_df
        self.test_df = test_df
        self.test_timestamps = test_df_raw["Timestamp"]
        self.train = np.array(train_df.values)
        self.test = np.array(test_df.values)

        if training:
            shuffle = True
            self.dataset = HmcDataset(self.train, win_size)
        else:
            shuffle = False
            self.dataset = HmcDataset(self.test, win_size)

        super().__init__(self.dataset, batch_size, shuffle, validation_split=0.0)
