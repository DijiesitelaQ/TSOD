import logging
import math
import warnings
from copy import copy
from pathlib import Path

from typing import Optional
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from PIL import Image
from torch.cuda import amp
import torch.nn.functional as F
from utils.activations import FReLU


class ImplicitA(nn.Module):
    def __init__(self, channel):
        super(ImplicitA, self).__init__()
        self.channel = channel
        self.implicit = nn.Parameter(torch.zeros(1, channel, 1, 1))
        nn.init.normal_(self.implicit, std=.02)

    def forward(self):
        return self.implicit


class ControlChannel(nn.Module):  # weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, layers):
        super(ControlChannel, self).__init__()
        self.layers = layers  # layer indices

    def forward(self, x, outputs):
        a = outputs[self.layers[0]]
        return a.expand_as(x) * x


class ShiftChannel(nn.Module):  # weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, layers):
        super(ShiftChannel, self).__init__()
        self.layers = layers  # layer indices

    def forward(self, x, outputs):
        a = outputs[self.layers[0]]
        return a.expand_as(x) + x

class ResdiualBlock(nn.Module):
    """
    实现子module：Residual Block
    """
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(ResdiualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outchannel)
        )

        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)


class ResNet(nn.Module):
    """
    实现主Module：ResNet34
    ResNet34包含多个layer， 每个layer又包含多个residual block
    用子module实现residual block， 用_make_layer函数实现layer
    """

    def __init__(self, num_classes=1000):
        super(ResNet, self).__init__()
        # 图像转换
        self.pre = nn.Sequential(
            # in_channel, out_channel, kernel_size, stride, padding
            nn.Conv2d(3, 64, 7, 2, 3, bias=False, ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )

        # 重复的layer， 分别有3，4，6，3 个residual block
        self.layer1 = self._make_layer(64, 128, 3)
        self.layer2 = self._make_layer(128, 256, 4, stride=2)
        self.layer3 = self._make_layer(256, 512, 6, stride=2)
        self.layer4 = self._make_layer(512, 512, 3, stride=2)

        # 全连接分类
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, inchannel, outchannel, block_num, stride=1):
        """
        构建residual block
        """
        shortcut = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
            nn.BatchNorm2d(outchannel)
        )

        layers = []
        layers.append(ResdiualBlock(inchannel, outchannel, stride, shortcut))
        for i in range(1, block_num):
            layers.append(ResdiualBlock(outchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x, 7)
        x = x.view(x.size(0), -1)
        return self.fc(x)


if __name__ == '__main__':
    net = ResNet()
    print(net)
