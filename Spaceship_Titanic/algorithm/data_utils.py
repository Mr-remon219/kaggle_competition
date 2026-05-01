import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

filepath = {"train": "./data/train.csv", "test": "./data/test.csv"}
str_process = 5

def data_init(file_path):
    df = pd.read_csv(file_path)
    for i in range(1, len(df.columns) - 1):
        if i in [5, 7, 8, 9, 10, 11]:
            df[df.columns[i]] = df[df.columns[i]].fillna(df[df.columns[i]].median())
        else:
            df[df.columns[i]] = df[df.columns[i]].fillna(df[df.columns[i]].mode()[0])

    for i in range(str_process):
        value, keys = pd.factorize(df[df.columns[i]])
        df[df.columns[i]] = value
    col = df.columns[6]
    df[col] = df[col].replace({False: 0, True: 1})
    if file_path == "train":
        col = df.columns[-1]
        df[col] = df[col].replace({False: 0, True: 1})
    value, keys = pd.factorize(df[df.columns[12]])
    df[df.columns[12]] = value
    value, keys = pd.factorize(df[df.columns[11]])
    df[df.columns[11]] = value

    return df


class TrainDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.file = data_init(filepath["train"])
        self.feature = self.file.iloc[:, 1 : -1].to_numpy(dtype=np.float32)
        self.label = self.file.iloc[:,  -1].to_numpy(dtype=np.long)

    def __len__(self):
        return len(self.feature)

    def __getitem__(self, idx):
        x = torch.tensor(self.feature[idx], dtype=torch.float32)
        y = torch.tensor(self.label[idx], dtype=torch.long)

        return x, y

class TestDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.file = data_init(filepath["test"])
        self.feature = self.file.iloc[:, 1 : ].to_numpy(dtype=np.float32)

    def __len__(self):
        return len(self.feature)

    def __getitem__(self, idx):
        x = torch.tensor(self.feature[idx], dtype=torch.float32)

        return x

if __name__ == "__main__":
    train_dataset = TrainDataset()
    test_dataset = TestDataset()
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    for x in test_loader:
        print(x)
        break
