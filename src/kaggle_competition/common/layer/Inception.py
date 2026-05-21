import torch.nn as nn
import torch
from torch.nn import functional as F


class Inception(nn.Moudle):
    def __init__(
        self, in_channel: int, c1: int, c2: int, c3: int, c4: int, is_1D: bool = False
    ):
        super().__init__()
        if is_1D:
            self.p1 = nn.Conv1d(in_channel, c1, kernel_size=1)

            self.p2_1 = nn.Conv1d(in_channel, c2[0], kernel_size=1)
            self.p2_2 = nn.Conv1d(c2[0], c2[1], kernel_size=3, padding=1)

            self.p3_1 = nn.Conv1d(in_channel, c3[0], kernel_size=1)
            self.p3_2 = nn.Conv1d(c3[0], c3[1], kernel_size=5, padding=2)

            self.p4_1 = nn.MaxPool1d(in_channel, kernel_size=3, padding=1)
            self.p4_2 = nn.Conv1d(in_channel, c4, kernel_size=1)

        else:
            self.p1 = nn.Conv2d(in_channel, c1, kernel_size=1)

            self.p2_1 = nn.Conv2d(in_channel, c2[0], kernel_size=1)
            self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)

            self.p3_1 = nn.Conv2d(in_channel, c3[0], kernel_size=1)
            self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)

            self.p4_1 = nn.MaxPool2d(in_channel, kernel_size=3, padding=1)
            self.p4_2 = nn.Conv2d(in_channel, c4, kernel_size=1)

    def forword(self, x):
        x_1 = F.relu(self.p1(x))
        x_2 = F.relu(self.p2_2(self.p2_1(x)))
        x_3 = F.relu(self.p3_2(self.p3_2(x)))
        x_4 = F.relu(self.p4_2(self.p4_1(x)))

        return torch.cat((x_1, x_2, x_3, x_4), dim=1)
