import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from data_utils import TestDataset
from resnet18_for_1d.resnet181d import ResNet181D
import torch
import pandas as pd

if __name__ == "__main__":
    dataset = TestDataset()
    test_loader = DataLoader(dataset, batch_size=256, shuffle=False)
    model = ResNet181D(1, 2)
    state_dict = torch.load("model.pth", map_location="cpu")
    model.load_state_dict(state_dict["model_state_dict"])

    model.eval()
    preds = []
    with torch.no_grad():
        for data in test_loader:
            data = data.unsqueeze(1)
            output = model(data)
            batch_pred = torch.argmax(output, dim=1)
            preds.extend(batch_pred.numpy())
    
    sub_df = pd.read_csv("./data/sample_submission.csv")
    sub_df["Transported"] = pd.Series(preds).astype(bool)
    sub_df.to_csv("./data/submission_1.csv", index=False)
    
            