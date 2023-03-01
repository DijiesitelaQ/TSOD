# YOLOv5 � by Ultralytics, GPL-3.0 license
"""
Common modules
"""

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

from utils.datasets import exif_transpose, letterbox
from utils.general import colorstr, increment_path, is_ascii, make_divisible, non_max_suppression, save_one_box, \
    scale_coords, xyxy2xywh
from utils.plots import Annotator, colors
from utils.torch_utils import time_sync

LOGGER = logging.getLogger(__name__)


def get_activation(act=True):
    act_name = None
    if isinstance(act, str):
        if act == "silu":
            m = nn.SiLU()
        elif act == "relu":
            m = nn.ReLU(inplace=True)
            act_name = 'relu'
        elif act == "lrelu":
            m = nn.LeakyReLU(0.1, inplace=True)
            act_name = 'leaky_relu'
        # elif act =="frelu":
        #     m = FReLU()
        #     act_name = "frelu"
        else:
            raise AttributeError("Unsupported act type: {}".format(act))
    else:
        m = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
    return m, act_name


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv1(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act, self.act_name = get_activation(act=act)
        self.init_weights()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))
    # def fuseforward(self, x):
    #     return self.act(self.conv(x))

    def init_weights(self):
        if self.act_name == "relu":
            # nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity="relu")
            pass

def _conv_bn(input_channel, output_channel, kernel_size=1, stride=1, padding=1, groups=1):
    res = nn.Sequential()
    res.add_module('conv',nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=kernel_size,padding=padding, padding_mode='zeros', stride=stride, groups=groups, bias=False))
    res.add_module('bn', nn.BatchNorm2d(output_channel))
    return res
#这个特意为C3准备的，因为目前的non-deep有跳转连接，不能有下采样
class Conv_C3(nn.Module):
    # Standard convolution
    # self.cv1 = Conv_C3(c1, c1, k=1,s=1,g=g)
    def __init__(self, c1, c2, k=3, s=1,deploy=False, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        # self.conv = nn.Conv2d(c1, c2, k, s, padding=k//2, groups=g, bias=False)
        # self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.deploy = deploy
        self.input_channel = c1

        self.k =  3
        self.groups = g
        self.padding = k//2
        # class Conv_non_deep(nn.Module):
        if (not self.deploy):#not  deploy=detect
            self.brb_3x3 = _conv_bn(c1, c2, kernel_size=3, padding=self.padding,groups=g)
            self.brb_1x1 = _conv_bn(c1, c2, kernel_size=1, padding=0, groups=g)
            self.brb_identity = nn.BatchNorm2d(c1) if c1 == c2 else None   #这个在接下来的C3处会造成self.cv3无法重参数化
        else:  #detect
            self.brb_rep = nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=k,padding=self.padding, padding_mode='zeros', stride=s, bias=True)
    # 将identity，1x1,3x3的卷积融合到一起，变成一个3x3卷积的参数

    def _get_equivalent_kernel_bias(self):
        brb_3x3_weight, brb_3x3_bias = self._fuse_conv_bn(self.brb_3x3)
        brb_1x1_weight, brb_1x1_bias = self._fuse_conv_bn(self.brb_1x1)
        brb_id_weight, brb_id_bias = self._fuse_conv_bn(self.brb_identity)
        return brb_3x3_weight + self._pad_1x1_kernel(
            brb_1x1_weight) + brb_id_weight, brb_3x3_bias + brb_1x1_bias + brb_id_bias

    def _switch_to_deploy(self):
        self.deploy = True
        kernel, bias = self._get_equivalent_kernel_bias()
        self.brb_rep = nn.Conv2d(in_channels=self.brb_3x3.conv.in_channels, out_channels=self.brb_3x3.conv.out_channels,
                                 kernel_size=3, padding=1,
                                 padding_mode=self.brb_3x3.conv.padding_mode, stride=self.brb_3x3.conv.stride,
                                 groups=1, bias=True)
        # print(self.brb_3x3.conv.padding,'11111111111')
        self.brb_rep.weight.data = kernel
        self.brb_rep.bias.data = bias
        # 消除梯度更新
        for para in self.parameters():
            para.detach_()
        # 删除没用的分支
        self.__delattr__('brb_3x3')
        self.__delattr__('brb_1x1')
        self.__delattr__('brb_identity')

        #wogaide
    def _fuse_conv_bn(self, branch):
        if (branch is None):
            return 0, 0
        elif (isinstance(branch, nn.Sequential)):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                # self.brb_identity.num_features
                # print(self.input_channel)
                #wogaide
                self.groups = 1

                input_dim = self.input_channel // self.groups
                kernel_value = np.zeros((self.input_channel, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.input_channel):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps

        std = (running_var + eps).sqrt()
        t = gamma / std
        t = t.view(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    # 将1x1的卷积变成3x3的卷积参数
    def _pad_1x1_kernel(self, kernel):
        if (kernel is None):
            return 0
        else:
            return nn.functional.pad(kernel, [1] * 4)

    def forward(self, inputs):

        if (self.deploy):#Conv._switch_to_deploy 会改变deploy
            return self.act(self.brb_rep(inputs))

        if (self.brb_identity == None):
            identity_out = 0
        else:
            identity_out = self.brb_identity(inputs)
        return self.act(self.brb_1x1(inputs) + self.brb_3x3(inputs) + identity_out)

    def forward_fuse(self, x):
        if (self.brb_identity == None):
            identity_out = 0
        else:
            identity_out = self.brb_identity(x)
        return self.act(self.brb_1x1(x) + self.brb_3x3(x) + identity_out)
        # return self.act(self.conv(x))


class DWConv(Conv):
    # Depth-wise convolution class
    def __init__(self, c1, c2, k=1, s=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers, act=True):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2, act=act)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*[TransformerLayer(c2, num_heads) for _ in range(num_layers)])
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).unsqueeze(0).transpose(0, 3).squeeze(3)
        return self.tr(p + self.linear(p)).unsqueeze(3).transpose(0, 3).reshape(b, self.c2, w, h)


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5, act=True):  # ch_in, ch_out, shortcut, groups, expansion, act
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, act=act)
        self.cv2 = Conv(c_, c2, 3, 1, g=g, act=act)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, act=True):  # ch_in, ch_out, number, shortcut, groups, expansion, act
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, act=act)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1, act=act)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0, act=True) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, act=True):  # ch_in, ch_out, number, shortcut, groups, expansion, act
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        if act == 'relu_silu':
            act = 'relu'
            last_act = 'silu'
        elif act == 'relu_lrelu':
            act = 'relu'
            last_act = 'lrelu'
        else:
            last_act = act
        self.cv1 = Conv(c1, c_, 1, 1, act=act)
        self.cv2 = Conv(c1, c_, 1, 1, act=act)
        self.cv3 = Conv(2 * c_, c2, 1, act=last_act)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0, act=act) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut, act=act) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class C3TR(C3):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, act=True):
        super().__init__(c1, c2, n, shortcut, g, e, act)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n, act=act)


class C3SPP(C3):
    # C3 module with SPP()
    def __init__(self, c1, c2, k=(5, 9, 13), n=1, shortcut=True, g=1, e=0.5, act=True):
        super().__init__(c1, c2, n, shortcut, g, e, act)
        c_ = int(c2 * e)
        self.m = SPP(c_, c_, k, act=act)


class C3Ghost(C3):
    # C3 module with GhostBottleneck()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, act=True):
        super().__init__(c1, c2, n, shortcut, g, e, act)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*[GhostBottleneck(c_, c_, act=act) for _ in range(n)])


class SPP(nn.Module):
    # Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729
    def __init__(self, c1, c2, k=(5, 9, 13), act=True):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        if act == 'relu_silu':
            act = 'relu'
            last_act = 'silu'
        elif act == 'relu_lrelu':
            act = 'relu'
            last_act = 'lrelu'
        else:
            last_act = act
        self.cv1 = Conv(c1, c_, 1, 1, act=act)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1, act=last_act)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5, act=True):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        if act == 'relu_silu':
            act = 'relu'
            last_act = 'silu'
        elif act == 'relu_lrelu':
            act = 'relu'
            last_act = 'lrelu'
        else:
            last_act = act
        self.cv1 = Conv(c1, c_, 1, 1, act=act)
        self.cv2 = Conv(c_ * 4, c2, 1, 1, act=last_act)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        c_ = c2 // 2  # hidden channels
        if act == 'relu_silu':
            act = 'relu'
            last_act = 'silu'
        elif act == 'relu_lrelu':
            act = 'relu'
            last_act = 'lrelu'
        else:
            last_act = act
        self.cv1 = Conv(c1, c_, k, s, None, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, last_act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat([y, self.cv2(y)], 1)


class GhostBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=3, s=1, act=True):  # ch_in, ch_out, kernel, stride, act
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(GhostConv(c1, c_, 1, 1, act=act),  # pw
                                  DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
                                  GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False),
                                      Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert (h / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(b, c, h // s, s, w // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(b, c * s * s, h // s, w // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(b, s, s, c // s ** 2, h, w)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(b, c // s ** 2, h * s, w * s)  # x(1,16,160,160)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)

class Concat_bifpn(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, c1, c2,act=True):
        super(Concat_bifpn, self).__init__()
        self.w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
       # self.w3 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        self.conv = Conv(c1, c2, 1 ,1 ,0 , act=act)
        #self.act= nn.ReLU()
        self.act = nn.SiLU()

    def forward(self, x): # mutil-layer 1-3 layers
        # print("bifpn:",x.shape)
        if len(x) == 2:
            w = self.w1
            weight = w / (torch.sum(w, dim=0) + self.epsilon)
            x = self.conv(self.act(weight[0] * x[0] + weight[1] * x[1]))
        elif len(x) == 3: 
            w = self.w2
            weight = w / (torch.sum(w, dim=0) + self.epsilon)
            x = self.conv(self.act (weight[0] * x[0] + weight[1] * x[1] + weight[2] * x[2]))
        # elif len(x) == 4:    
        #     w = self.w3
        #     weight = w / (torch.sum(w, dim=0) + self.epsilon)
        #     x = self.conv(self.act(weight[0] * x[0] + weight[1] * x[1] + weight[2] *x[2] + weight[3]*x[3] ))
        return x 


class AutoShape(nn.Module):
    # YOLOv5 input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    classes = None  # (optional list) filter by class
    max_det = 1000  # maximum number of detections per image

    def __init__(self, model):
        super().__init__()
        self.model = model.eval()

    def autoshape(self):
        LOGGER.info('AutoShape already enabled, skipping... ')  # model already converted to model.autoshape()
        return self

    @torch.no_grad()
    def forward(self, imgs, size=640, augment=False, profile=False):
        # Inference from various sources. For height=640, width=1280, RGB images example inputs are:
        #   file:       imgs = 'data/images/zidane.jpg'  # str or PosixPath
        #   URI:             = 'https://ultralytics.com/images/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(640,1280,3)
        #   PIL:             = Image.open('image.jpg') or ImageGrab.grab()  # HWC x(640,1280,3)
        #   numpy:           = np.zeros((640,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,320,640)  # BCHW (scaled to size=640, 0-1 values)
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        t = [time_sync()]
        p = next(self.model.parameters())  # for device and type
        if isinstance(imgs, torch.Tensor):  # torch
            with amp.autocast(enabled=p.device.type != 'cpu'):
                return self.model(imgs.to(p.device).type_as(p), augment, profile)  # inference

        # Pre-process
        n, imgs = (len(imgs), imgs) if isinstance(imgs, list) else (1, [imgs])  # number of images, list of images
        shape0, shape1, files = [], [], []  # image and inference shapes, filenames
        for i, im in enumerate(imgs):
            f = f'image{i}'  # filename
            if isinstance(im, (str, Path)):  # filename or uri
                im, f = Image.open(requests.get(im, stream=True).raw if str(im).startswith('http') else im), im
                im = np.asarray(exif_transpose(im))
            elif isinstance(im, Image.Image):  # PIL Image
                im, f = np.asarray(exif_transpose(im)), getattr(im, 'filename', f) or f
            files.append(Path(f).with_suffix('.jpg').name)
            if im.shape[0] < 5:  # image in CHW
                im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
            im = im[..., :3] if im.ndim == 3 else np.tile(im[..., None], 3)  # enforce 3ch input
            s = im.shape[:2]  # HWC
            shape0.append(s)  # image shape
            g = (size / max(s))  # gain
            shape1.append([y * g for y in s])
            imgs[i] = im if im.data.contiguous else np.ascontiguousarray(im)  # update
        shape1 = [make_divisible(x, int(self.stride.max())) for x in np.stack(shape1, 0).max(0)]  # inference shape
        x = [letterbox(im, new_shape=shape1, auto=False)[0] for im in imgs]  # pad
        x = np.stack(x, 0) if n > 1 else x[0][None]  # stack
        x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))  # BHWC to BCHW
        x = torch.from_numpy(x).to(p.device).type_as(p) / 255.  # uint8 to fp16/32
        t.append(time_sync())

        with amp.autocast(enabled=p.device.type != 'cpu'):
            # Inference
            y = self.model(x, augment, profile)[0]  # forward
            t.append(time_sync())

            # Post-process
            y = non_max_suppression(y, self.conf, iou_thres=self.iou, classes=self.classes, max_det=self.max_det)  # NMS
            for i in range(n):
                scale_coords(shape1, y[i][:, :4], shape0[i])

            t.append(time_sync())
            return Detections(imgs, y, files, t, self.names, x.shape)


class Detections:
    # YOLOv5 detections class for inference results
    def __init__(self, imgs, pred, files, times=None, names=None, shape=None):
        super().__init__()
        d = pred[0].device  # device
        gn = [torch.tensor([*[im.shape[i] for i in [1, 0, 1, 0]], 1., 1.], device=d) for im in imgs]  # normalizations
        self.imgs = imgs  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.ascii = is_ascii(names)  # names are ascii (use PIL for UTF-8)
        self.files = files  # image filenames
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)  # number of images (batch size)
        self.t = tuple((times[i + 1] - times[i]) * 1000 / self.n for i in range(3))  # timestamps (ms)
        self.s = shape  # inference BCHW shape

    def display(self, pprint=False, show=False, save=False, crop=False, render=False, save_dir=Path('')):
        for i, (im, pred) in enumerate(zip(self.imgs, self.pred)):
            str = f'image {i + 1}/{len(self.pred)}: {im.shape[0]}x{im.shape[1]} '
            if pred.shape[0]:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    str += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                if show or save or render or crop:
                    annotator = Annotator(im, pil=not self.ascii)
                    for *box, conf, cls in reversed(pred):  # xyxy, confidence, class
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        if crop:
                            save_one_box(box, im, file=save_dir / 'crops' / self.names[int(cls)] / self.files[i])
                        else:  # all others
                            annotator.box_label(box, label, color=colors(cls))
                    im = annotator.im
            else:
                str += '(no detections)'

            im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im  # from np
            if pprint:
                LOGGER.info(str.rstrip(', '))
            if show:
                im.show(self.files[i])  # show
            if save:
                f = self.files[i]
                im.save(save_dir / f)  # save
                if i == self.n - 1:
                    LOGGER.info(f"Saved {self.n} image{'s' * (self.n > 1)} to {colorstr('bold', save_dir)}")
            if render:
                self.imgs[i] = np.asarray(im)

    def print(self):
        self.display(pprint=True)  # print results
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {tuple(self.s)}' %
                    self.t)

    def show(self):
        self.display(show=True)  # show results

    def save(self, save_dir='runs/detect/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/detect/exp', mkdir=True)  # increment save_dir
        self.display(save=True, save_dir=save_dir)  # save results

    def crop(self, save_dir='runs/detect/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/detect/exp', mkdir=True)  # increment save_dir
        self.display(crop=True, save_dir=save_dir)  # crop results
        LOGGER.info(f'Saved results to {save_dir}\n')

    def render(self):
        self.display(render=True)  # render results
        return self.imgs

    def pandas(self):
        # return detections as pandas DataFrames, i.e. print(results.pandas().xyxy[0])
        new = copy(self)  # return copy
        ca = 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'  # xyxy columns
        cb = 'xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name'  # xywh columns
        for k, c in zip(['xyxy', 'xyxyn', 'xywh', 'xywhn'], [ca, ca, cb, cb]):
            a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]  # update
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
        return new

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        x = [Detections([self.imgs[i]], [self.pred[i]], self.names, self.s) for i in range(self.n)]
        for d in x:
            for k in ['imgs', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
                setattr(d, k, getattr(d, k)[0])  # pop out of list
        return x

    def __len__(self):
        return self.n


class Classify(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g)  # to x(b,c2,1,1)
        self.flat = nn.Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
        return self.flat(self.conv(z))  # flatten to x(b,c2)


# build repvgg block
# -------------------------------------------------------------------------
def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                                        bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))

    return result


class SEBlock(nn.Module):

    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1,
                              bias=True)
        self.up = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1,
                            bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x = F.avg_pool2d(inputs, kernel_size=inputs.size(3))
        x = self.down(x)
        x = F.relu(x)
        x = self.up(x)
        x = torch.sigmoid(x)
        x = x.view(-1, self.input_channels, 1, 1)
        return inputs * x
        
class RepVGGBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.SiLU()

        # self.nonlinearity = nn.ReLU()

        if use_se:
            self.se = SEBlock(out_channels, internal_neurons=out_channels // 16)
        else:
            self.se = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=True,
                                         padding_mode=padding_mode)

        else:
            self.rbr_identity = nn.BatchNorm2d(
                num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                   padding=padding_11, groups=groups)
            # print('RepVGG Block, identity = ', self.rbr_identity)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))

        # RepVGGBlock(in_channels=self.in_planes, out_channels=planes, kernel_size=3,
        #                           stride=stride, padding=1, groups=1, deploy=self.deploy, use_se=self.use_se))

    def fusevggforward(self, x):
        return self.nonlinearity(self.rbr_dense(x))


# repvgg block end
# -----------------------------

################# DropBlock2D
class DropBlock2D(nn.Module):
    r"""Randomly zeroes 2D spatial blocks of the input tensor.
    As described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.
    Args:
        drop_prob (float): probability of an element to be dropped.
        block_size (int): size of the block to drop
    Shape:
        - Input: `(N, C, H, W)`
        - Output: `(N, C, H, W)`
    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890
    """

    def __init__(self, drop_prob, block_size):
        super(DropBlock2D, self).__init__()

        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        # shape: (bsize, channels, height, width)
        assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"
        if not self.training or self.drop_prob == 0.:
            return x
        else:
            # get gamma value
            gamma = self._compute_gamma(x)
            # sample mask
            mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()
            # place mask on input device
            mask = mask.to(x.device)
            # compute block mask
            block_mask = self._compute_block_mask(mask)
            # apply block mask
            out = x * block_mask[:, None, :, :]
            # scale output
            out = out * block_mask.numel() / block_mask.sum()

            return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(input=mask[:, None, :, :],
                                  kernel_size=(self.block_size, self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)
        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]
        block_mask = 1 - block_mask.squeeze(1)
        
        return block_mask

    def _compute_gamma(self, x):
        return self.drop_prob / (self.block_size ** 2)
##############

# CBAM
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool= nn.AdaptiveAvgPool2d(1)
        self.max_pool= nn.AdaptiveMaxPool2d(1)

        self.fc1= nn.Conv2d(in_planes, in_planes//ratio, 1, bias=False)
        self.relu1= nn.ReLU()
        self.fc2= nn.Conv2d(in_planes//ratio, in_planes, 1, bias=False)
        self.sigmoid= nn.Sigmoid()

    def forward(self,x):
        avg_out= self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out= self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out= avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention,self).__init__()
        self.conv1= nn.Conv2d(2, 1, kernel_size, padding=3, bias=False)  # kernel size = 7 Padding is 3: (n - 7 + 1) + 2P = n 
        self.sigmoid= nn.Sigmoid()

    def forward(self,x):
        avg_out = torch.mean(x, dim=1, keepdim=True) 
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, channelIn, channelOut):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channelIn)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        out = self.channel_attention(x) * x
        # print('outchannels:{}'.format(out.shape))
        out = self.spatial_attention(out) * out
        return out

# -------------------------------------------------------------------------

# coordattion
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)
        
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        

    def forward(self, x):
        identity = x
        
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out 

################
#
# build Lcnet
# -----------------------------
class CBH(nn.Module):
    def __init__(self, num_channels, num_filters, filter_size, stride, num_groups=1):
        super().__init__()
        self.conv = nn.Conv2d(
            num_channels,
            num_filters,
            filter_size,
            stride,
            padding=(filter_size - 1) // 2,
            groups=num_groups,
            bias=False)
        self.bn = nn.BatchNorm2d(num_filters)
        self.hardswish = nn.Hardswish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.hardswish(x)
        return x

    def fuseforward(self, x):
        return self.hardswish(self.conv(x))

# class Hardsigmoid(nn.Module):
#     def forward(self, x):
#         out = F.relu6(x + 3, inplace=True) / 6
#         return out


class LC_SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(
            in_channels=channel,
            out_channels=channel // reduction,
            kernel_size=1,
            stride=1,
            padding=0)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=channel // reduction,
            out_channels=channel,
            kernel_size=1,
            stride=1,
            padding=0)
        self.SiLU = nn.SiLU()
        # self.hardsigmoid = nn.Hardsigmoid()

    def forward(self, x):
        identity = x
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        # x = self.hardsigmoid(x)
        x = self.SiLU(x)
        out = identity * x
        return out


class LC_Block(nn.Module):
    def __init__(self, num_channels, num_filters, stride, dw_size, use_se=False):
        super().__init__()
        self.use_se = use_se
        self.dw_conv = CBH(
            num_channels=num_channels,
            num_filters=num_channels,
            filter_size=dw_size,
            stride=stride,
            num_groups=num_channels)
        if use_se:
            self.se = LC_SEModule(num_channels)
        self.pw_conv = CBH(
            num_channels=num_channels,
            filter_size=1,
            num_filters=num_filters,
            stride=1)

    def forward(self, x):
        x = self.dw_conv(x)
        if self.use_se:
            x = self.se(x)
        x = self.pw_conv(x)
        return x


class Dense(nn.Module):
    def __init__(self, num_channels, num_filters, filter_size, dropout_prob):
        super().__init__()
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dense_conv = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=1,
            padding=0,
            bias=False)
        self.hardswish = nn.Hardswish()
        self.dropout = nn.Dropout(p=dropout_prob)
        # self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        # self.fc = nn.Linear(num_filters, num_filters)

    def forward(self, x):
        # x = self.avg_pool(x)
        # b, _, w, h = x.shape
        x = self.dense_conv(x)
        # b, _, w, h = x.shape
        x = self.hardswish(x)
        x = self.dropout(x)
        # x = self.flatten(x)
        # x = self.fc(x)
        # x = x.reshape(b, self.c2, w, h)
        return x
# end

# build shuffle block
# -------------------------------------------------------------------------

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class conv_bn_relu_maxpool(nn.Module):
    def __init__(self, c1, c2):  # ch_in, ch_out
        super(conv_bn_relu_maxpool, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

    def forward(self, x):
        return self.maxpool(self.conv(x))


class Shuffle_Block(nn.Module):
    def __init__(self, inp, oup, stride):
        super(Shuffle_Block, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out


# shuffle block end
# -------------------------------------------------------------------------

# build DWConvblock
# -------------------------------------------------------------------------
class DWConvblock(nn.Module):
    "Depthwise conv + Pointwise conv"

    def __init__(self, in_channels, out_channels, k, s):
        super(DWConvblock, self).__init__()
        self.p = k // 2
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=k, stride=s, padding=self.p, groups=in_channels,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        return x

# DWConvblock end
# -------------------------------------------------------------------------

# build GDWConvblock
# -------------------------------------------------------------------------
class GSC(nn.Module):
    "Depthwise conv + Pointwise conv"

    def __init__(self, in_channels, out_channels, k, s):
        super(GSC, self).__init__()
        _c = out_channels // 2
        self.conv1 = Conv(in_channels, _c, k=k, s=s, p=self.p)
        self.Shuffle_Block = Shuffle_Block(out_channels,out_channels,s=1)
        self.dwconv = DWConvblock(_c, _c, k=1, s=1)

    def forward(self, x):
        x = self.conv1(x)
        x1 = self.dwconv(x)
        x = torch.cat(x,x1,dim=1)
        x = self.Shuffle_Block(x)
        return x
# build GDWConvblock
# -------------------------------------------------------------------------

class ADD(nn.Module):
    # Stortcut a list of tensors along dimension
    def __init__(self, alpha=0.5):
        super(ADD, self).__init__()
        self.a = alpha

    def forward(self, x):
        x1, x2 = x[0], x[1]
        return torch.add(x1, x2, alpha=self.a)

class CSPT(nn.Module):
    # C3 module with SwinTransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5,k=(5, 9, 13), act=True):
        super().__init__()
        c_ = int(c2 * e)
        if act == 'relu_silu':
            act = 'relu'
            last_act = 'silu'
        elif act == 'relu_lrelu':
            act = 'relu'
            last_act = 'lrelu'
        else:
            last_act = act
        self.cv1 = Conv(c1, c_, 1, 1, act=act)
        self.cv2 = Conv(c1, c_, 1, 1, act=act)
        self.cv3 = Conv(2 * c_, c2, 1, act=last_act)  # act=FReLU(c2)
        self.spp = SPP(c_, c_, k, act=act)
        self.st = SwinTransformerBlock(c_, c_, c_//32, n)
    def forward(self, x):
        out = self.cv1(x)
        return self.cv3(torch.cat((self.st(out), self.spp(out)),dim=1))

class CSPT2(nn.Module):
    # C3 module with SwinTransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5,k=(5, 9, 13), act=True):
        super().__init__()
        c_ = int(c2 * e)
        if act == 'relu_silu':
            act = 'relu'
            last_act = 'silu'
        elif act == 'relu_lrelu':
            act = 'relu'
            last_act = 'lrelu'
        else:
            last_act = act
        self.cv1 = Conv(c1, c_, 1, 1, act=act)
        self.cv2 = Conv(c1, c_, 1, 1, act=act)
        self.cv3 = Conv(2 * c_, c2, 1, act=last_act)  # act=FReLU(c2)
        self.cv4 = Conv(c_, c_, 1, 1, act=act)
        self.spp = SPP(c_, c_, k, act=act)
        self.st = SwinTransformerBlock(c_, c_, c_//32, n)
    def forward(self, x):
        out1 = self.cv1(x)
        out2 = self.cv4(torch.add(self.st(out1), self.spp(out1)))
        return self.cv3(torch.cat((out2,self.cv2(x)),dim=1))

class CSPT3(nn.Module):
    # C3 module with SwinTransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5,k=(5, 9, 13), act=True):
        super().__init__()
        c_ = int(c2 * e)
        if act == 'relu_silu':
            act = 'relu'
            last_act = 'silu'
        elif act == 'relu_lrelu':
            act = 'relu'
            last_act = 'lrelu'
        else:
            last_act = act
        self.cv1 = Conv(c1, c_, 1, 1, act=act)
        self.cv2 = Conv(c1, c_, 1, 1, act=act)
        self.cv3 = Conv(2 * c_, c2, 1, act=last_act)  # act=FReLU(c2)
        self.cv4 = Conv(c_, c_, 1, 1, act=act)
        self.spp = SPP(c_, c_, k, act=act)
        self.st = SwinTransformerBlock(c_, c_, c_//32, n)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0, act=act) for _ in range(n)])
    def forward(self, x):
        out1 = self.cv1(x)
        out2 = self.cv4(torch.add(self.st(out1), self.spp(out1)))
        out3 = self.m(out1)
        out4 = out2+out3
        # return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
        return self.cv3(torch.cat((out4,self.cv2(x)),dim=1))

class C3STR(C3):
    # C3 module with SwinTransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = SwinTransformerBlock(c_, c_, c_//32, n)

class SwinTransformerBlock(nn.Module):
    def __init__(self, c1, c2, num_heads, num_layers, window_size=8):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)

        self.window_size = window_size
        self.shift_size = window_size // 2
        self.tr = nn.Sequential(*(SwinTransformerLayer(c2, num_heads=num_heads, window_size=window_size,  shift_size=0 if (i % 2 == 0) else self.shift_size ) for i in range(num_layers)))

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        x = self.tr(x)
        return x

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class LePEAttention(nn.Module):
    def __init__(self, dim, resolution, idx, split_size=7, dim_out=None, num_heads=8, attn_drop=0., proj_drop=0.,
                 qk_scale=None):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.resolution = resolution
        self.split_size = split_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        if idx == -1:
            H_sp, W_sp = self.resolution, self.resolution
        elif idx == 0:
            H_sp, W_sp = self.resolution, self.split_size
        elif idx == 1:
            W_sp, H_sp = self.resolution, self.split_size
        else:
            print("ERROR MODE", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp
        stride = 1
        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.attn_drop = nn.Dropout(attn_drop)

    def im2cswin(self, x):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.H_sp * self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def get_lepe(self, x, func):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)

        H_sp, W_sp = self.H_sp, self.W_sp
        x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp)  ### B', C, H', W'

        lepe = func(x)  ### B', C, H', W'
        lepe = lepe.reshape(-1, self.num_heads, C // self.num_heads, H_sp * W_sp).permute(0, 1, 3, 2).contiguous()

        x = x.reshape(-1, self.num_heads, C // self.num_heads, self.H_sp * self.W_sp).permute(0, 1, 3, 2).contiguous()
        return x, lepe

    def forward(self, qkv):
        """
        x: B L C
        """
        q, k, v = qkv[0], qkv[1], qkv[2]

        ### Img2Window
        H = W = self.resolution
        B, L, C = q.shape
        assert L == H * W, "flatten img_tokens has wrong size"

        q = self.im2cswin(q)
        k = self.im2cswin(k)
        v, lepe = self.get_lepe(v, self.get_v)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N
        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = (attn @ v) + lepe
        x = x.transpose(1, 2).reshape(-1, self.H_sp * self.W_sp, C)  # B head N N @ B head N C

        ### Window2Img
        x = windows2img(x, self.H_sp, self.W_sp, H, W).view(B, -1, C)  # B H' W' C

        return x


class CSWinBlock(nn.Module):

    def __init__(self, dim, reso, num_heads,
                 split_size=7, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 last_stage=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.patches_resolution = reso
        self.split_size = split_size
        self.mlp_ratio = mlp_ratio
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm1 = norm_layer(dim)

        if self.patches_resolution == split_size:
            last_stage = True
        if last_stage:
            self.branch_num = 1
        else:
            self.branch_num = 2
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        if last_stage:
            self.attns = nn.ModuleList([
                LePEAttention(
                    dim, resolution=self.patches_resolution, idx=-1,
                    split_size=split_size, num_heads=num_heads, dim_out=dim,
                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
                for i in range(self.branch_num)])
        else:
            self.attns = nn.ModuleList([
                LePEAttention(
                    dim // 2, resolution=self.patches_resolution, idx=i,
                    split_size=split_size, num_heads=num_heads // 2, dim_out=dim // 2,
                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
                for i in range(self.branch_num)])

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
                       drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):
        """
        x: B, H*W, C
        b, c, w, h = x.shape
        """

        H = W = self.patches_resolution
        B,  C,w,h = x.shape
        L = w*h
        assert L == H * W, "flatten img_tokens has wrong size"
        img = self.norm1(x)
        qkv = self.qkv(img).reshape(B, -1, 3, C).permute(2, 0, 1, 3)

        if self.branch_num == 2:
            x1 = self.attns[0](qkv[:, :, :, :C // 2])
            x2 = self.attns[1](qkv[:, :, :, C // 2:])
            attened_x = torch.cat([x1, x2], dim=2)
        else:
            attened_x = self.attns[0](qkv)
        attened_x = self.proj(attened_x)
        x = x + self.drop_path(attened_x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class C3CSTR(C3):
    # C3 module with SwinTransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = CSWinBlock(c_, c_, c_//32, n)

def img2windows(img, H_sp, W_sp):
    """
    img: B C H W
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp * W_sp, C)
    return img_perm


def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    img_splits_hw: B' H W C
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img


class Merge_Block(nn.Module):
    def __init__(self, dim, dim_out, norm_layer=nn.LayerNorm):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim_out, 3, 2, 1)
        self.norm = norm_layer(dim_out)

    def forward(self, x):
        B, new_HW, C = x.shape
        H = W = int(np.sqrt(new_HW))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = self.conv(x)
        B, C = x.shape[:2]
        x = x.view(B, C, -1).transpose(-2, -1).contiguous()
        x = self.norm(x)

        return x


class SwinTransformerLayer(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4, qkv_bias=False, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        if num_heads > 10:
            drop_path = 0.1
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(c)
        self.attn = WindowAttention(
            c, window_size=(self.window_size, self.window_size), num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(c)
        mlp_hidden_dim = int(c * mlp_ratio)
        self.mlp = Mlp(in_features=c, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def create_mask(self, x, H, W):
        # calculate attention mask for SW-MSA
        # 保证Hp和Wp是window_size的整数倍
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        # 拥有和feature map一样的通道排列顺序，方便后续window_partition
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # [1, Hp, Wp, 1]
        h_slices = ((0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # [nW, Mh, Mw, 1]
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # [nW, Mh*Mw]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]
        # [nW, Mh*Mw, Mh*Mw]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, torch.tensor(-100.0)).masked_fill(attn_mask == 0,
                                                                                            torch.tensor(0.0))
        return attn_mask

    def forward(self, x):
        b, c, w, h = x.shape
        x = x.permute(0, 3, 2, 1).contiguous()  # [b,h,w,c]

        attn_mask = self.create_mask(x, h, w)  # [nW, Mh*Mw, Mh*Mw]
        shortcut = x
        x = self.norm1(x)

        pad_l = pad_t = 0
        pad_r = (self.window_size - w % self.window_size) % self.window_size
        pad_b = (self.window_size - h % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, hp, wp, _ = x.shape

        if self.shift_size > 0:
            # print(f"shift size: {self.shift_size}")
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition(shifted_x, self.window_size)  # [nW*B, Mh, Mw, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c)  # [nW*B, Mh*Mw, C]

        attn_windows = self.attn(x_windows, mask=attn_mask)  # [nW*B, Mh*Mw, C]

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)  # [nW*B, Mh, Mw, C]
        shifted_x = window_reverse(attn_windows, self.window_size, hp, wp)  # [B, H', W', C]

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            # 把前面pad的数据移除掉
            x = x[:, :h, :w, :].contiguous()

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        x = x.permute(0, 3, 2, 1).contiguous()
        return x  # (b, self.c2, w, h)

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # [Mh, Mw]
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # [2*Mh-1 * 2*Mw-1, nH]

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # [2, Mh, Mw]
        coords_flatten = torch.flatten(coords, 1)  # [2, Mh*Mw]
        # [2, Mh*Mw, 1] - [2, 1, Mh*Mw]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, Mh*Mw, Mh*Mw]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [Mh*Mw, Mh*Mw, 2]
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # [Mh*Mw, Mh*Mw]
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: input features with shape of (num_windows*B, Mh*Mw, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # [batch_size*num_windows, Mh*Mw, total_embed_dim]
        B_, N, C = x.shape
        # qkv(): -> [batch_size*num_windows, Mh*Mw, 3 * total_embed_dim]
        # reshape: -> [batch_size*num_windows, Mh*Mw, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size*num_windows, num_heads, embed_dim_per_head, Mh*Mw]
        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, Mh*Mw]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # relative_position_bias_table.view: [Mh*Mw*Mh*Mw,nH] -> [Mh*Mw,Mh*Mw,nH]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [nH, Mh*Mw, Mh*Mw]
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # mask: [nW, Mh*Mw, Mh*Mw]
            nW = mask.shape[0]  # num_windows
            # attn.view: [batch_size, num_windows, num_heads, Mh*Mw, Mh*Mw]
            # mask.unsqueeze: [1, nW, 1, Mh*Mw, Mh*Mw]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        # transpose: -> [batch_size*num_windows, Mh*Mw, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size*num_windows, Mh*Mw, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

def window_reverse(windows, window_size: int, H: int, W: int):
    """
    将一个个window还原成一个feature map
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size(M)
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # view: [B*num_windows, Mh, Mw, C] -> [B, H//Mh, W//Mw, Mh, Mw, C]
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    # permute: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B, H//Mh, Mh, W//Mw, Mw, C]
    # view: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H, W, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

def window_partition(x, window_size: int):
    """
    将feature map按照window_size划分成一个个没有重叠的window
    Args:
        x: (B, H, W, C)
        window_size (int): window size(M)

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # permute: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C]
    # view: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B*num_windows, Mh, Mw, C]
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)

def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


# """
# dbb block
#
# """
def transI_fusebn(kernel, bn):
    gamma = bn.weight
    std = (bn.running_var + bn.eps).sqrt()
    return kernel * ((gamma / std).reshape(-1, 1, 1, 1)), bn.bias - bn.running_mean * gamma / std

def transII_addbranch(kernels, biases):
    return sum(kernels), sum(biases)

def transIII_1x1_kxk(k1, b1, k2, b2, groups):
    if groups == 1:
        k = F.conv2d(k2, k1.permute(1, 0, 2, 3))      #
        b_hat = (k2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3))
    else:
        k_slices = []
        b_slices = []
        k1_T = k1.permute(1, 0, 2, 3)
        k1_group_width = k1.size(0) // groups
        k2_group_width = k2.size(0) // groups
        for g in range(groups):
            k1_T_slice = k1_T[:, g*k1_group_width:(g+1)*k1_group_width, :, :]
            k2_slice = k2[g*k2_group_width:(g+1)*k2_group_width, :, :, :]
            k_slices.append(F.conv2d(k2_slice, k1_T_slice))
            b_slices.append((k2_slice * b1[g*k1_group_width:(g+1)*k1_group_width].reshape(1, -1, 1, 1)).sum((1, 2, 3)))
        k, b_hat = transIV_depthconcat(k_slices, b_slices)
    return k, b_hat + b2

def transIV_depthconcat(kernels, biases):
    return torch.cat(kernels, dim=0), torch.cat(biases)

def transV_avg(channels, kernel_size, groups):
    input_dim = channels // groups
    k = torch.zeros((channels, input_dim, kernel_size, kernel_size))
    k[np.arange(channels), np.tile(np.arange(input_dim), groups), :, :] = 1.0 / kernel_size ** 2
    return k

#   This has not been tested with non-square kernels (kernel.size(2) != kernel.size(3)) nor even-size kernels
def transVI_multiscale(kernel, target_kernel_size):
    H_pixels_to_pad = (target_kernel_size - kernel.size(2)) // 2
    W_pixels_to_pad = (target_kernel_size - kernel.size(3)) // 2
    return F.pad(kernel, [H_pixels_to_pad, H_pixels_to_pad, W_pixels_to_pad, W_pixels_to_pad])

def conv_bn(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                   padding_mode='zeros'):
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                           stride=stride, padding=padding, dilation=dilation, groups=groups,
                           bias=False, padding_mode=padding_mode)
    bn_layer = nn.BatchNorm2d(num_features=out_channels, affine=True)
    se = nn.Sequential()
    se.add_module('conv', conv_layer)
    se.add_module('bn', bn_layer)
    return se


class IdentityBasedConv1x1(nn.Conv2d):

    def __init__(self, channels, groups=1):
        super(IdentityBasedConv1x1, self).__init__(in_channels=channels, out_channels=channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=False)

        assert channels % groups == 0
        input_dim = channels // groups
        id_value = np.zeros((channels, input_dim, 1, 1))
        for i in range(channels):
            id_value[i, i % input_dim, 0, 0] = 1
        self.id_tensor = torch.from_numpy(id_value).type_as(self.weight)
        nn.init.zeros_(self.weight)

    def forward(self, input):
        kernel = self.weight + self.id_tensor.to(self.weight.device)
        result = F.conv2d(input, kernel, None, stride=1, padding=0, dilation=self.dilation, groups=self.groups)
        return result

    def get_actual_kernel(self):
        return self.weight + self.id_tensor.to(self.weight.device)


class BNAndPadLayer(nn.Module):
    def __init__(self,
                 pad_pixels,
                 num_features,
                 eps=1e-5,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True):
        super(BNAndPadLayer, self).__init__()
        self.bn = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
        self.pad_pixels = pad_pixels

    def forward(self, input):
        output = self.bn(input)
        if self.pad_pixels > 0:
            if self.bn.affine:
                pad_values = self.bn.bias.detach() - self.bn.running_mean * self.bn.weight.detach() / torch.sqrt(self.bn.running_var + self.bn.eps)
            else:
                pad_values = - self.bn.running_mean / torch.sqrt(self.bn.running_var + self.bn.eps)
            output = F.pad(output, [self.pad_pixels] * 4)
            pad_values = pad_values.view(1, -1, 1, 1)
            output[:, :, 0:self.pad_pixels, :] = pad_values
            output[:, :, -self.pad_pixels:, :] = pad_values
            output[:, :, :, 0:self.pad_pixels] = pad_values
            output[:, :, :, -self.pad_pixels:] = pad_values
        return output

    @property
    def weight(self):
        return self.bn.weight

    @property
    def bias(self):
        return self.bn.bias

    @property
    def running_mean(self):
        return self.bn.running_mean

    @property
    def running_var(self):
        return self.bn.running_var

    @property
    def eps(self):
        return self.bn.eps


class DiverseBranchBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 internal_channels_1x1_3x3=None,
                 deploy=False, nonlinear=None, single_init=False):
        super(DiverseBranchBlock, self).__init__()
        self.deploy = deploy

        if nonlinear is None:
            self.nonlinear = nn.Identity()
        else:
            self.nonlinear = nonlinear

        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.groups = groups
        assert padding == kernel_size // 2

        if deploy:
            self.dbb_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True)

        else:

            self.dbb_origin = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups)

            self.dbb_avg = nn.Sequential()
            if groups < out_channels:
                self.dbb_avg.add_module('conv',
                                        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                                  stride=1, padding=0, groups=groups, bias=False))
                self.dbb_avg.add_module('bn', BNAndPadLayer(pad_pixels=padding, num_features=out_channels))
                self.dbb_avg.add_module('avg', nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0))
                self.dbb_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                       padding=0, groups=groups)
            else:
                self.dbb_avg.add_module('avg', nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding))

            self.dbb_avg.add_module('avgbn', nn.BatchNorm2d(out_channels))


            if internal_channels_1x1_3x3 is None:
                internal_channels_1x1_3x3 = in_channels if groups < out_channels else 2 * in_channels   # For mobilenet, it is better to have 2X internal channels

            self.dbb_1x1_kxk = nn.Sequential()
            if internal_channels_1x1_3x3 == in_channels:
                self.dbb_1x1_kxk.add_module('idconv1', IdentityBasedConv1x1(channels=in_channels, groups=groups))
            else:
                self.dbb_1x1_kxk.add_module('conv1', nn.Conv2d(in_channels=in_channels, out_channels=internal_channels_1x1_3x3,
                                                            kernel_size=1, stride=1, padding=0, groups=groups, bias=False))
            self.dbb_1x1_kxk.add_module('bn1', BNAndPadLayer(pad_pixels=padding, num_features=internal_channels_1x1_3x3, affine=True))
            self.dbb_1x1_kxk.add_module('conv2', nn.Conv2d(in_channels=internal_channels_1x1_3x3, out_channels=out_channels,
                                                            kernel_size=kernel_size, stride=stride, padding=0, groups=groups, bias=False))
            self.dbb_1x1_kxk.add_module('bn2', nn.BatchNorm2d(out_channels))

        #   The experiments reported in the paper used the default initialization of bn.weight (all as 1). But changing the initialization may be useful in some cases.
        if single_init:
            #   Initialize the bn.weight of dbb_origin as 1 and others as 0. This is not the default setting.
            self.single_init()

    def get_equivalent_kernel_bias(self):
        k_origin, b_origin = transI_fusebn(self.dbb_origin.conv.weight, self.dbb_origin.bn)

        if hasattr(self, 'dbb_1x1'):
            k_1x1, b_1x1 = transI_fusebn(self.dbb_1x1.conv.weight, self.dbb_1x1.bn)
            k_1x1 = transVI_multiscale(k_1x1, self.kernel_size)
        else:
            k_1x1, b_1x1 = 0, 0

        if hasattr(self.dbb_1x1_kxk, 'idconv1'):
            k_1x1_kxk_first = self.dbb_1x1_kxk.idconv1.get_actual_kernel()
        else:
            k_1x1_kxk_first = self.dbb_1x1_kxk.conv1.weight
        k_1x1_kxk_first, b_1x1_kxk_first = transI_fusebn(k_1x1_kxk_first, self.dbb_1x1_kxk.bn1)
        k_1x1_kxk_second, b_1x1_kxk_second = transI_fusebn(self.dbb_1x1_kxk.conv2.weight, self.dbb_1x1_kxk.bn2)
        k_1x1_kxk_merged, b_1x1_kxk_merged = transIII_1x1_kxk(k_1x1_kxk_first, b_1x1_kxk_first, k_1x1_kxk_second, b_1x1_kxk_second, groups=self.groups)

        k_avg = transV_avg(self.out_channels, self.kernel_size, self.groups)
        k_1x1_avg_second, b_1x1_avg_second = transI_fusebn(k_avg.to(self.dbb_avg.avgbn.weight.device), self.dbb_avg.avgbn)
        if hasattr(self.dbb_avg, 'conv'):
            k_1x1_avg_first, b_1x1_avg_first = transI_fusebn(self.dbb_avg.conv.weight, self.dbb_avg.bn)
            k_1x1_avg_merged, b_1x1_avg_merged = transIII_1x1_kxk(k_1x1_avg_first, b_1x1_avg_first, k_1x1_avg_second, b_1x1_avg_second, groups=self.groups)
        else:
            k_1x1_avg_merged, b_1x1_avg_merged = k_1x1_avg_second, b_1x1_avg_second

        return transII_addbranch((k_origin, k_1x1, k_1x1_kxk_merged, k_1x1_avg_merged), (b_origin, b_1x1, b_1x1_kxk_merged, b_1x1_avg_merged))

    def switch_to_deploy(self):
        if hasattr(self, 'dbb_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.dbb_reparam = nn.Conv2d(in_channels=self.dbb_origin.conv.in_channels, out_channels=self.dbb_origin.conv.out_channels,
                                     kernel_size=self.dbb_origin.conv.kernel_size, stride=self.dbb_origin.conv.stride,
                                     padding=self.dbb_origin.conv.padding, dilation=self.dbb_origin.conv.dilation, groups=self.dbb_origin.conv.groups, bias=True)
        self.dbb_reparam.weight.data = kernel
        self.dbb_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('dbb_origin')
        self.__delattr__('dbb_avg')
        if hasattr(self, 'dbb_1x1'):
            self.__delattr__('dbb_1x1')
        self.__delattr__('dbb_1x1_kxk')

    def forward(self, inputs):

        if hasattr(self, 'dbb_reparam'):
            return self.nonlinear(self.dbb_reparam(inputs))

        out = self.dbb_origin(inputs)
        if hasattr(self, 'dbb_1x1'):
            out += self.dbb_1x1(inputs)
        out += self.dbb_avg(inputs)
        out += self.dbb_1x1_kxk(inputs)
        return self.nonlinear(out)

    def init_gamma(self, gamma_value):
        if hasattr(self, "dbb_origin"):
            torch.nn.init.constant_(self.dbb_origin.bn.weight, gamma_value)
        if hasattr(self, "dbb_1x1"):
            torch.nn.init.constant_(self.dbb_1x1.bn.weight, gamma_value)
        if hasattr(self, "dbb_avg"):
            torch.nn.init.constant_(self.dbb_avg.avgbn.weight, gamma_value)
        if hasattr(self, "dbb_1x1_kxk"):
            torch.nn.init.constant_(self.dbb_1x1_kxk.bn2.weight, gamma_value)

    def single_init(self):
        self.init_gamma(0.0)
        if hasattr(self, "dbb_origin"):
            torch.nn.init.constant_(self.dbb_origin.bn.weight, 1.0)

class BottleneckDBB(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5, act=None):  # ch_in, ch_out, shortcut, groups, expansion, act
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = DiverseBranchBlock(c1, c_, 1, 1, nonlinear=act)
        self.cv2 = DiverseBranchBlock(c_, c2, 3, 1,padding=1, groups=g, nonlinear=act)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3DBB(nn.Module):
    # DBB Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, act=True):  # ch_in, ch_out, number, shortcut, groups, expansion, act
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        if act == 'relu_silu':
            act = 'relu'
            last_act = 'silu'
        elif act == 'relu_lrelu':
            act = 'relu'
            last_act = 'lrelu'
        else:
            last_act = act
        self.cv1 = Conv(c1, c_, 1, 1, act=act)
        self.cv2 = Conv(c1, c_, 1, 1, act=act)
        self.cv3 = Conv(2 * c_, c2, 1, act=last_act)  # act=FReLU(c2)
        self.m = nn.Sequential(*[BottleneckDBB(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut, act=act) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

class BottleneckResVGG(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv_C3(c1, c1, k=3,s=1,g=g)
        self.cv2 = Conv_C3(c1, c2, k=3,s= 1, g=g)
        self.add = shortcut and c1 == c2

    ###### wogaide
    def _switch_to_deploy(self):
        self.cv1._switch_to_deploy()
        self.cv2._switch_to_deploy()

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3RVGG(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, act=True):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv1(c1, c_, 1, 1)
        self.cv2 = Conv1(c1, c_, 1, 1)
        self.cv3 = Conv1(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*(BottleneckResVGG(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])
    def _switch_to_deploy(self):

        # self.cv1 = self.cv1._switch_to_deploy()
        # self.cv2 = self.cv2._switch_to_deploy()
        # self.cv3 = self.cv3._switch_to_deploy()  # 为啥后面两个都可以，这个不行？
        # self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
        for i in self.m:
            i._switch_to_deploy()
        # self.m = nn.Sequential(*(i._switch_to_deploy() for i in self.m))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

# class C3RVGG(nn.Module):
#     # DBB Bottleneck with 3 convolutions
#     def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, act=True):  # ch_in, ch_out, number, shortcut, groups, expansion, act
#         super().__init__()
#         c_ = int(c2 * e)  # hidden channels
#         if act == 'relu_silu':
#             act = 'relu'
#             last_act = 'silu'
#         elif act == 'relu_lrelu':
#             act = 'relu'
#             last_act = 'lrelu'
#         else:
#             last_act = act
#         self.cv1 = Conv(c1, c_, 1, 1, act=act)
#         self.cv2 = Conv(c1, c_, 1, 1, act=act)
#         self.cv3 = Conv(2 * c_, c2, 1, act=last_act)  # act=FReLU(c2)
#         self.m = nn.Sequential(*[BottleneckResVGG(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
#         # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut, act=act) for _ in range(n)])
#
#     def forward(self, x):
#         return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
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
        a = outputs[self.layers]
        return a.expand_as(x) * x

class ShiftChannel(nn.Module):  # weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, layers):
        super(ShiftChannel, self).__init__()
        self.layers = layers  # layer indices

    def forward(self, x, outputs):
        a = outputs[self.layers[0]]
        return a.expand_as(x) + x