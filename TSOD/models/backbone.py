import torch
import torch.nn as nn


class BasicBlock1(nn.Module):
    """一个不带下采样的残差块"""
    def __init__(self, in_channel):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.conv2 = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += identity
        x = self.relu(x)
        return x


class BasicBlock2(nn.Module):
    """一个带下采样的残差块"""
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=2, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.conv3(x)
        identity = self.bn3(identity)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += identity
        x = self.relu(x)
        return x


class ResNet18(nn.Module):

    def __init__(self):
        super().__init__()
        self.channel = 64
        self.conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(*[BasicBlock1(64), BasicBlock1(64)])
        self.layer2 = self.make_layer(128)
        self.layer3 = self.make_layer(256)
        self.layer4 = self.make_layer(512)

    def make_layer(self, channel):
        layers = [BasicBlock2(self.channel, channel), BasicBlock1(channel)]
        self.channel = channel
        return nn.Sequential(*layers)

    def forward(self, x):  # Nx3x224x224
        x = self.conv(x)  # Nx64x112x112
        x = self.bn(x)  # Nx64x112x112
        x = self.relu(x)  # Nx64x112x112
        x = self.maxpool(x)  # Nx64x56x56
        x = self.layer1(x)  # Nx64x56x56
        x = self.layer2(x)  # Nx128x28x28
        x = self.layer3(x)  # Nx256x14x14
        x = self.layer4(x)  # Nx512x7x7
        return x
