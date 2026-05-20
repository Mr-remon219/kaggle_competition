import torch
from torch import nn

from kaggle_competition.common.layer.BasicBlock import BasicBlock


class ResNet181D(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.conv1 = nn.Conv1d(
            in_channels=in_channels, out_channels=64, kernel_size=7, stride=1, padding=3
        )
        self.bn1 = nn.BatchNorm1d(64)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = BasicBlock(in_channels=64, out_channels=64, process=False)
        self.layer2 = BasicBlock(in_channels=64, out_channels=128, process=True)
        self.layer3 = BasicBlock(in_channels=128, out_channels=256, process=True)
        self.layer4 = BasicBlock(in_channels=256, out_channels=512, process=True)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
