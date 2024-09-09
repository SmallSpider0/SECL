import torch.nn as nn
import torch
from torchvision import models

CLASS_SIZE = None
class SmallMLP(nn.Module):
    """
    Multi-Layer Perceptron
    """

    def __init__(self):
        super(SmallMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, CLASS_SIZE),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class MediumMLP(nn.Module):
    """
    Multi-Layer Perceptron
    """

    def __init__(self):
        super(MediumMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, CLASS_SIZE),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class LargeMLP(nn.Module):
    """
    Multi-Layer Perceptron
    """

    def __init__(self):
        super(LargeMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, CLASS_SIZE),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits



# 模型：多层感知机
def mlp(SIZE="Small", class_size=10):
    global CLASS_SIZE
    CLASS_SIZE = class_size
    if SIZE == "Small":
        return SmallMLP()
    elif SIZE == "Medium":
        return MediumMLP()
    elif SIZE == "Large":
        return LargeMLP()


# 模型：resnet18
def resnet18(num_classes=10):
    return models.resnet18(num_classes=num_classes)
