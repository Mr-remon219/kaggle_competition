from kaggle_competition.common.layer.resnet import _resnet


def ResNet181D(init_channels: int, num_classes: int):
    return _resnet(init_channels, num_classes, [3, 4, 6, 3], [1, 1, 1, 1], is_1D=True)
