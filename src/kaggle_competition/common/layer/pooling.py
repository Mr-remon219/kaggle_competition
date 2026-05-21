import torch.nn as nn


def maxpool_x3(
    stride: int = 1, padding: int = 1, is_1D: bool = False
) -> nn.MaxPool1d | nn.MaxPool2d:
    if is_1D:
        return nn.MaxPool1d(kernel_size=3, stride=stride, padding=padding)
    else:
        return nn.MaxPool2d(kernel_size=3, stride=stride, padding=padding)
