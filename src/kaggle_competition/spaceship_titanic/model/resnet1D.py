from kaggle_competition.common.layer.resnet import _resnet


def ResNet1D(init_channels: int, num_classes: int, dropout: float = 0.3):
    return _resnet(init_channels, num_classes, [2, 2, 2, 2], [1, 1, 1, 1], is_1D=True, dropout=dropout)
