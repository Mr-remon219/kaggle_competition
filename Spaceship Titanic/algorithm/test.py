import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from data_utils import TestDataset
from resnet18_for_1d.resnet181d import ResNet181D

if __name__ == "__main__":
    dataset = TestDataset()
