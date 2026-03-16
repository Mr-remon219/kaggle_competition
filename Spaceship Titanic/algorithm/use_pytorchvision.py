import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from data_utils import WheatDataset
from resnet18_for_1d.resnet181d import ResNet181D


if __name__ == "__main__":
    dataset = WheatDataset()
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    num_classes = 2
    model = ResNet181D(1, 2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(100):
        model.train()

        total_loss = 0
        total_sample = 0

        for x, y in loader:
            total_sample += 1
            x = x.unsqueeze(1)
            pred = model(x)
            loss = criterion(pred, y)
            total_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("当前循环为第%d轮， 误差为：%.3f" % (epoch, total_loss / total_sample))
