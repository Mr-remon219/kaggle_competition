import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch

from .data_utils import TrainDataset
from .model.resnet181d import ResNet181D
from .config import MODEL_DIR


def train():
    dataset = TrainDataset()

    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    num_classes = 2
    model = ResNet181D(1, num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(100):
        model.train()

        total_loss = 0
        total_sample = 0

        for x, y in loader:
            batch_size = y.shape[0]
            total_sample += batch_size
            x = x.unsqueeze(1)
            pred = model(x)
            loss = criterion(pred, y)
            total_loss += loss.item() * batch_size
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("当前循环为第%d轮, 误差为：%.3f" % (epoch, total_loss / total_sample))

    torch.save(model.state_dict(), MODEL_DIR / "model.pth")


if __name__ == "__main___":
    train()
