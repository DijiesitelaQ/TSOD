import torch
import torch.nn as nn
from models.backbone import ResNet18


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ResNet18()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        for param in self.parameters():
            param.requires_grad = False
        self.fc1 = nn.Linear(512, 1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv = nn.Conv2d(512, 64, kernel_size=3)
        self.relu = nn.LeakyReLU()
        self.fc2 = nn.Linear(64, 4)

    def forward(self, x):
        x = self.backbone(x)  # Nx512x7x7
        bb = self.maxpool(x)  # Nx512x3x3
        bb = self.conv(bb)  # Nx64x1x1
        bb = torch.flatten(bb, 1)  # Nx64
        bb = self.relu(bb)  # Nx64
        bb = self.fc2(bb)  # Nx4
        x = self.avgpool(x)  # Nx512x1x1
        x = torch.flatten(x, 1)  # Nx512
        x = self.fc1(x)  # Nx1
        x = torch.cat([x, bb], 1)
        return x
