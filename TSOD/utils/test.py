import glob
import hashlib
import json
import logging
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import ThreadPool, Pool
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective, \
    box_candidates
from utils.general import check_requirements, check_file, check_dataset, xywh2xyxy, xywhn2xyxy, xyxy2xywhn, \
    xyn2xy, xywhs2xyxy, segments2boxes, clean_str
from utils.torch_utils import torch_distributed_zero_first




def collate_fn4(batch):
    img, label, path, shapes = zip(*batch)  # transposed
    n = len(shapes) // 4
    img4, label4, path4, shapes4 = [], [], path[:n], shapes[:n]

    ho = torch.tensor([[0., 0, 0, 1, 0, 0]])
    wo = torch.tensor([[0., 0, 1, 0, 0, 0]])
    s = torch.tensor([[1, 1, .5, .5, .5, .5]])  # scale
    for i in range(n):  # zidane torch.zeros(16,3,720,1280)  # BCHW
        i *= 4
        if random.random() < 0.5:
            im = F.interpolate(img[i].unsqueeze(0).float(), scale_factor=2., mode='bilinear', align_corners=False)[
                0].type(img[i].type())
            l = label[i]
        else:
            im = torch.cat((torch.cat((img[i], img[i + 1]), 1), torch.cat((img[i + 2], img[i + 3]), 1)), 2)
            l = torch.cat((label[i], label[i + 1] + ho, label[i + 2] + wo, label[i + 3] + ho + wo), 0) * s
        img4.append(im)
        label4.append(l)

    for i, l in enumerate(label4):
        l[:, 0] = i  # add target image index for build_targets()

    return torch.stack(img4, 0), torch.cat(label4, 0), path4, shapes4

