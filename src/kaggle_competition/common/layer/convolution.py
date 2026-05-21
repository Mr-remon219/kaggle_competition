import torch.nn as nn


def conv_x1(
    in_channels: int,
    channels: int,
    stride: int = 1,
    padding: int = 0,
    is_1D: bool = False,
) -> nn.Conv1d | nn.Conv2d:
    if is_1D:
        return nn.Conv1d(
            in_channels=in_channels,
            out_channels=channels,
            kernel_size=1,
            stride=stride,
            padding=padding,
            bias=False,
        )
    else:
        return nn.Conv2d(
            in_channels=in_channels,
            out_channels=channels,
            kernel_size=1,
            stride=stride,
            padding=padding,
            bias=False,
        )


def conv_x3(
    in_channels: int,
    channels: int,
    stride: int = 1,
    padding: int = 1,
    is_1D: bool = False,
) -> nn.Conv1d | nn.Conv2d:
    if is_1D:
        return nn.Conv1d(
            in_channels=in_channels,
            out_channels=channels,
            kernel_size=3,
            stride=stride,
            padding=padding,
            bias=False,
        )
    else:
        return nn.Conv2d(
            in_channels=in_channels,
            out_channels=channels,
            kernel_size=3,
            stride=stride,
            padding=padding,
            bias=False,
        )


def conv_x7(
    in_channels: int,
    channels: int,
    stride: int = 1,
    padding: int = 3,
    is_1D: bool = False,
) -> nn.Conv1d | nn.Conv2d:
    if is_1D:
        return nn.Conv1d(
            in_channels=in_channels,
            out_channels=channels,
            kernel_size=7,
            stride=stride,
            padding=padding,
            bias=False,
        )
    else:
        return nn.Conv2d(
            in_channels=in_channels,
            out_channels=channels,
            kernel_size=7,
            stride=stride,
            padding=padding,
            bias=False,
        )
