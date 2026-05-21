import torch.nn as nn
import torch
from torch import Tensor
from typing import Optional

from .convolution import conv_x1, conv_x3, conv_x7
from .pooling import maxpool_x3


class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: int,
        stride: int = 1,
        norm_layer: type[nn.Module] = None,
        downsample: Optional[nn.Module] = None,
        is_1D: bool = False,
    ) -> None:

        super().__init__()
        self.in_channels = in_channels
        self.channels = channels

        if is_1D:
            if norm_layer is None:
                norm_layer = nn.BatchNorm1d
            self.conv1 = conv_x3(in_channels, channels, stride, is_1D=True)
            self.bn1 = norm_layer(channels)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = conv_x3(channels, channels, is_1D=True)
            self.bn2 = norm_layer(channels)

        else:
            if norm_layer is None:
                norm_layer = nn.BatchNorm2d
            self.conv1 = conv_x3(in_channels, channels, stride)
            self.bn1 = norm_layer(channels)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = conv_x3(channels, channels)
            self.bn2 = norm_layer(channels)

        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        init_channels: int,
        num_classes: int,
        layers: list[int],
        strides: list[int],
        norm_layer: Optional[nn.Module] = None,
        is_1D: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = 64
        self.num_classes = num_classes
        self.is_1D = is_1D

        if norm_layer is None:
            if is_1D:
                self.norm_layer = nn.BatchNorm1d
                norm_layer = self.norm_layer
            else:
                self.norm_layer = nn.BatchNorm2d
                norm_layer = self.norm_layer

        self.conv1 = conv_x7(init_channels, self.in_channels, is_1D=is_1D)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = maxpool_x3(is_1D=is_1D)

        self.layer1 = self._make_layer(64, layers[0], strides[0])
        self.layer2 = self._make_layer(128, layers[1], strides[1])
        self.layer3 = self._make_layer(256, layers[2], strides[2])
        self.layer4 = self._make_layer(512, layers[3], strides[3])

        if is_1D:
            self.avgpool = nn.AdaptiveAvgPool1d(1)
        else:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, channels: int, blocks: int, stride: int = 1) -> nn.Module:

        norm_layer = self.norm_layer
        is_1D = self.is_1D

        downsample = None

        if self.in_channels != channels or stride != 1:
            downsample = nn.Sequential(
                conv_x1(self.in_channels, channels, is_1D=is_1D), norm_layer(channels)
            )

        layer = []

        layer.append(
            BasicBlock(
                self.in_channels, channels, stride, norm_layer, downsample, is_1D=is_1D
            )
        )

        self.in_channels = channels

        for i in range(1, blocks):
            layer.append(
                BasicBlock(channels, channels, norm_layer=norm_layer, is_1D=is_1D)
            )

        return nn.Sequential(*layer)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out


def _resnet(
    init_channels: int,
    num_classes: int,
    layers: list[int],
    strides: list[int],
    norm_layer: Optional[nn.Module] = None,
    is_1D: bool = False,
) -> nn.Module:

    return ResNet(init_channels, num_classes, layers, strides, norm_layer, is_1D=is_1D)
