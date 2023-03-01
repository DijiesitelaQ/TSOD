import torch
import os
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, utils
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.dataset import Dataset
from torchvision.transforms import transforms
from pathlib import Path
import cv2
from PIL import Image
import torch.nn.functional as F
from tqdm import tqdm
from models.simplenet import SimpleNet
from utils.datasets import create_dataloader
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    check_requirements, print_mutation, set_logging, one_cycle, colorstr, methods

transform = transforms.Compose([ToTensor(),
                                transforms.Normalize(
                                    mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5]
                                ),
                                transforms.Resize((224, 224))
                                ])

train_path = "/home/zeta/LuoQiang/data/datasets/data/train.txt"
imgsz = 1280
batch_size = 32
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
gs = 32 # grid size (max stride)
single_cls = True
cache = False
train_loader, dataset = create_dataloader(train_path, imgsz, batch_size // WORLD_SIZE, gs, single_cls,
                                              hyp=None, augment=False, cache=cache, rect=False, rank=-1,
                                              workers=8, image_weights=False, quad=False,
                                              prefix=colorstr('train: '))

# 保持数据集和测试机能完整划分
# train_data = DataLoader(dataset=training_data, batch_size=batch_size, shuffle=True, drop_last=True)
# test_data = DataLoader(dataset=testing_data, batch_size=batch_size, shuffle=True, drop_last=True)
pbar = enumerate(train_loader)
# images, labels = next(iter(train_loader))
# print(images.shape)
# img = utils.make_grid(images)
# img = img.numpy().transpose(1, 2, 0)
# mean = [0.5, 0.5, 0.5]
# std = [0.5, 0.5, 0.5]
# img = img * std + mean
# print([labels[i] for i in range(64)])
# plt.imshow(img)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleNet()
cost = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
nb = len(train_loader)
print(nb)
# print(len(test_data))
epochs = 10
# total_nb = nb * epochs
# pbar = tqdm(pbar, total=nb)
for epoch in range(epochs):
    running_loss = 0.0
    running_correct = 0.0
    model.train()
    print("Epoch {}/{}".format(epoch + 1, epochs))
    print("-" * 10)
    # for X_train, y_train in train_loader:
    #     # X_train,y_train = torch.autograd.Variable(X_train),torch.autograd.Variable(y_train)
    #     X_train, y_train = X_train.to(device), y_train.to(device)
    #     outputs = model(X_train)
    #     _, pred = torch.max(outputs.data, 1)
    #     optimizer.zero_grad()
    #     loss = cost(outputs, y_train)
    #
    #     loss.backward()
    #     optimizer.step()
    #     running_loss += loss.item()
    #     running_correct += torch.sum(pred == y_train.data)
    for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
        ni = i + nb * epoch  # number integrated batches (since train start)
        imgs = imgs.to(device, non_blocking=True).float() / 255.0  #
        print(imgs.shape)
        outputs = model(imgs)
        _, pred = torch.max(outputs.data, 1)
        optimizer.zero_grad()
        loss = cost(outputs, targets.to(device))

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        # running_correct += torch.sum(pred == targets.data)

    testing_correct = 0
    test_loss = 0
    model.eval()
    # for X_test, y_test in test_data:
    #     # X_test,y_test = torch.autograd.Variable(X_test),torch.autograd.Variable(y_test)
    #     X_test, y_test = X_test.to(device), y_test.to(device)
    #     outputs = model(X_test)
    #     loss = cost(outputs, y_test)
    #     _, pred = torch.max(outputs.data, 1)
    #     testing_correct += torch.sum(pred == y_test.data)
    #     test_loss += loss.item()
    # print("Train Loss is:{:.4f}, Train Accuracy is:{:.4f}%, Test Loss is::{:.4f} Test Accuracy is:{:.4f}%".format(
    #     running_loss / len(training_data), 100 * running_correct / len(training_data),
    #     test_loss / len(testing_data),
    #     100 * testing_correct / len(testing_data)
    # ))

