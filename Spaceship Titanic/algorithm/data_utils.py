import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

file_path = "data.cropdata_updated.csv"
str_process = 5

def data_init():
    df = pd.read_csv(file_path)
    for i in range(str_process):
        value, keys = pd.factorize(df[df.columns[i]])
        df[df.columns[i]] = value
    df.loc[df.columns[6] == False, df.columns[6]] = 0
    df.loc[df.columns[6] == True, df.columns[6]] = 1
    df.loc[df.columns[-1] == False, df.columns[-1]] = 0
    df.loc[df.columns[-1] == True, df.columns[-1]] = 1
    value, keys = pd.factorize(df[df.columns[-2]])
    df[df.columns[-2]] = value
    value, keys = pd.factorize(df[df.columns[-3]])
    df[df.columns[-3]] = value

    return df





class WheatDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.file = data_init()
        self.feature = self.file.iloc[:, : -1].to_numpy(dtype=np.float32)
        self.label = self.file.iloc[:,  -1].to_numpy(dtype=np.long)

    def __len__(self):
        return len(self.feature)

    def __getitem__(self, idx):
        x = torch.tensor(self.feature[idx], dtype=torch.float32)
        y = torch.tensor(self.label[idx], dtype=torch.long)

        return x, y

if __name__ == "__main__":
    dataset = WheatDataset()
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    for x, y in loader:
        print(x)
        print(y)
        break
