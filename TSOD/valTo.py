# YOLOv5 � by Ultralytics, GPL-3.0 license
"""
Validate a trained YOLOv5 model accuracy on a custom dataset

Usage:
    $ python path/to/val.py --data coco128.yaml --weights yolov5s.pt --img 640
"""

import argparse
import json
import os
import sys
from pathlib import Path
from threading import Thread
import cv2
import numpy as np
import torch
from tqdm import tqdm
from torchvision import transforms
import torch.nn as nn

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target, plot_study_txt
from utils.torch_utils import select_device, time_sync
from utils.callbacks import Callbacks
import itertools
from terminaltables import AsciiTable

def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')


def save_one_json(predn, jdict, path, class_map):
    # Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append({'image_id': image_id,
                      'category_id': class_map[int(p[5])],
                      'bbox': [round(x, 3) for x in b],
                      'score': round(p[4], 5)})


def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU above threshold and classes match
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[0],im.shape[1]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    # if auto:  # minimum rectangle
    #     dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    # elif scaleFill:  # stretch
    #     dw, dh = 0.0, 0.0
    #     new_unpad = (new_shape[1], new_shape[0])
    #     ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    # top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    # left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    # w1,h1 = im.shape[0],im.shape[1]
    # if w1 < 640:
    #     right = int(640-w1)
    # if h1 < 640:
    #     bottom = int(640-w1)
    # top = left = 0
    top, bottom = 0, int(round(dh + 0.2))+int(round(dh - 0.1))
    left, right = 0, int(round(dw + 0.2))+int(round(dw - 0.1))  # 将填充的黑边添加到右边和下边
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def letterbox2(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32,imgsz = 3.2,target=None):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[0],im.shape[1]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    target[:,2:] *= imgsz
    target[:, 2:] *= r
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    # if auto:  # minimum rectangle
    #     dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    # elif scaleFill:  # stretch
    #     dw, dh = 0.0, 0.0
    #     new_unpad = (new_shape[1], new_shape[0])
    #     ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    # top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    # left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    # w1,h1 = im.shape[0],im.shape[1]
    # if w1 < 640:
    #     right = int(640-w1)
    # if h1 < 640:
    #     bottom = int(640-w1)
    # top = left = 0
    top, bottom = 0, int(round(dh + 0.2))+int(round(dh - 0.1))
    left, right = 0, int(round(dw + 0.2))+int(round(dw - 0.1))  # 将填充的黑边添加到右边和下边
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    # for i in target:
    #     cls = int(i[1].item())
    #     x1 = float(i[2].item())*640
    #     y1 = float(i[3].item())*640
    #     w2 = float(i[4].item())*640
    #     h2 = float(i[5].item())*640
    #     bb = [x1, y1, w2, h2]
    #     bb = xywh2xyxy(bb)
    return im, target

@torch.no_grad()
def run(data,
        weights=None,  # model.pt path(s)
        batch_size=8,  # batch size
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.6,  # NMS IoU threshold
        task='val',  # train, val, test, speed or study
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        single_cls=False,  # treat as single-class dataset
        augment=False,  # augmented inference
        verbose=False,  # verbose output
        save_txt=False,  # save results to *.txt
        save_hybrid=False,  # save label+prediction hybrid results to *.txt
        save_conf=True,  # save confidences in --save-txt labels
        save_json=False,  # save a COCO-JSON results file
        project='runs/val',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=False,  # use FP16 half-precision inference
        classwise=False,  # Whether to evaluating the AP for each class
        model=None,
        dataloader=None,
        save_dir=Path(''),
        plots=True,
        callbacks=Callbacks(),
        compute_loss=None,
        ):
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        device = select_device(device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        imgsz = check_img_size(imgsz, s=gs)  # check image size

        # Multi-GPU disabled, incompatible with .half() https://github.com/ultralytics/yolov5/issues/99
        # if device.type != 'cpu' and torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)

        # Data
        data = check_dataset(data)  # check

    # Half
    half &= device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    model.eval()
    is_coco = type(data['val']) is str and data['val'].endswith('coco/val2017.txt')  # COCO dataset
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    if not training:
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        dataloader = create_dataloader(data[task], imgsz, batch_size, gs, single_cls, pad=0, rect=True,
                                       prefix=colorstr(f'{task}: '))[0]

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1, t2 = 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    count = 0
    erro = []
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        t_ = time_sync()
        img1 = img.to(device, non_blocking=True)    # load_image(self, i) -> loads 1 image from dataset index 'i', returns im, original hw, resized hw
        img = img1.half() if half else img1.float()  # uint8 to fp16/32
        img = img/255.0  # 0 - 255 to 0.0 - 1.0
        print("原始imgshape:",img.shape)
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width
        t = time_sync()
        t0 += t - t_
        # img = nn.functional.interpolate(img, size=[640,640], mode='bilinear', align_corners=False)
        # Run model
        outputs = model(img, augment=augment)  # inference and training outputs
        t1 += time_sync() - t

        if len(outputs) >= 2:
            out, train_out = outputs[:2]
            # Compute loss
            if compute_loss:
                loss += compute_loss([x.float() for x in train_out], targets)[1]  # box, obj, cls
        else:
            out = outputs[0]    # [bs, ,class+5] ->[32,5292,205]

        # Run NMS
        nb, _, height1, width1 = img.shape
        mytarget = targets.clone().to(device, non_blocking=True)
        mytarget[:, 1] = 0
        mytarget[:, 2:] *= torch.Tensor([width1, height1, width1, height1]).to(device)  # to pixels
        lb = [mytarget[mytarget[:, 0] == i, 1:] for i in range(nb)] if True else []  # for autolabelling
        out = non_max_suppression(out, 0.25, 0.6, labels=lb, multi_label=True, agnostic=True)
        newimgs = torch.zeros(batch_size, 3, 640, 640).to(device)  # 创建一个shape为[d1,d2,d3]的tensor
        targets = targets.to(device)
        mylist = [int(targets[i, 0].item()) for i in range(targets.shape[0])]
        # Statistics per image  out->[xyxy, conf, cls]=list[bach-size->32张图片]
        for si, pred in enumerate(out):
            # print(pred.shape)
            # pred = torch.topk(pred,10,4)
            # print(pred)
            name  = paths[si].split('/')[-1]
            txtname = name.split('.')[0]+'.txt'
            sublist = [k for k in range(len(mylist)) if mylist[k] == si]
            if pred.shape[0] == 0 :
                print("该图片没有交通标志--->",paths[si])
                imga = img[si]
                array1 = imga * 255  # normalize，将图像数据扩展到[0,255] --69906这张图片可以测试一下
                mat = np.uint8(array1.cpu())
                mat = mat.transpose(1, 2, 0)  # mat_shape: (982, 814，3)
                mat = letterbox(mat, new_shape=[640, 640], stride=32)[0]
                mat = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
                print('------------------->', name)
                # cv2.imwrite('/home/zeta/LuoQiang/data/datasets/data/newdata/images/' + name, mat)
                continue
            # print("--->", paths[si],pred.shape)
            x_min, y_min, x_max, y_max = min(pred[:, 0]), min(pred[:, 1]), max(pred[:, 2]), max(pred[:, 3])
            bbox_x1,bbox_y1,bbox_x2,bbox_y2 = int(x_min * 0.95),int(y_min * 0.95),int(x_max * 1.05),int(y_max * 1.05)
            print("起初值：",[bbox_x1,bbox_y1,bbox_x2,bbox_y2])
            if bbox_x2 > 2047:
                bbox_x2 = 2047
            if bbox_y2 > 2047:
                bbox_y2 = 2047
            if bbox_x1 < 0:
                bbox_x1 = 0
            if bbox_y1 < 0:
                bbox_y1 = 0
            w1 = bbox_x2-bbox_x1
            h1 = bbox_y2-bbox_y1
            l1 = 2047 - bbox_x2
            l2 = 2047 - bbox_y2
            print(w1,h1)

            if w1 < 640 and h1 < 640:
                if l1 > bbox_x1:
                    bbox_x1 = bbox_x2-640
                else:
                    bbox_x2 = bbox_x1+640
                if l2 > bbox_y1:
                    bbox_y1 = bbox_y2-640
                else:
                    bbox_y2 = bbox_y1+640
            else:
                if h1 > w1:
                    if l1 > bbox_x1:
                        bbox_x2 = bbox_x2 + (h1-w1)
                    else:
                        bbox_x1 = bbox_x1 - (h1-w1)
                else:
                    if l2 > bbox_y1:
                        bbox_y2 = bbox_y2 + (w1-h1)
                    else:
                        bbox_y1 = bbox_y1 - (w1-h1)

            if bbox_x2 > 2047:
                bbox_x1 = bbox_x1 - (bbox_x2 - 2047)
                bbox_x2 = 2047
            if bbox_y2 > 2047:
                bbox_y1 = bbox_y1 - (bbox_y2 - 2047)
                bbox_y2 = 2047
            if bbox_x1 < 0:
                bbox_x2 = bbox_x2 - bbox_x1
                bbox_x1 = 0
            if bbox_y1 < 0:
                bbox_y2 = bbox_y2 - bbox_y1
                bbox_y1 = 0
            print(bbox_x1,bbox_y1,bbox_x2,bbox_y2)
            bbox = [bbox_x1 / 2048.0, bbox_y1 / 2048.0]
            # 以来代码是用来输出子网络提取效果的
            imga = img[si]
            array1 = imga * 255  # normalize，将图像数据扩展到[0,255] --69906这张图片可以测试一下
            mat = np.uint8(array1.cpu())
            mat = mat.transpose(1, 2, 0)  # mat_shape: (982, 814，3)
            mat = mat[bbox_y1: bbox_y2, bbox_x1: bbox_x2]
            targets[sublist, 2:3] -= bbox[0]
            targets[sublist, 3:4] -= bbox[1]
            mat,targets[sublist] = letterbox2(mat, new_shape=[640,640], stride=32,target=targets[sublist])
            mat = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
            # for i in sublist:
            #     cls = int(targets[i, 1].item())
            #     x1 = float(targets[i, 2].item())
            #     y1 = float(targets[i, 3].item())
            #     w2 = float(targets[i, 4].item())
            #     h2 = float(targets[i, 5].item())
            #     bb = [x1, y1, w2, h2]
            #     c1 = bb[0] - bb[2] / 2  # top left x
            #     c2 = bb[1] - bb[3] / 2  # top left y
            #     c3 = bb[0] + bb[2] / 2  # bottom right x
            #     c4 = bb[1] + bb[3] / 2  # bottom right y
            #     c1 = int(c1 * 640)
            #     c2 = int(c2 * 640)
            #     c3 = int(c3 * 640)
            #     c4 = int(c4 * 640)
            #     cc = [c1,c2,c3,c4]
            #     try:
            #         # cv2.imshow(str(cls), mat[c2:c4, c1:c3])
            #         cv2.imwrite("/home/zeta/LuoQiang/data/datasets/data/3class/marks/"+name+str(i)+'_'+str(cls)+'.jpg',mat[c2:c4, c1:c3])
            #     except:
            #         print(cc)
            #         count += 1
            with open('/home/zeta/LuoQiang/data/datasets/data/3class/labels/'+txtname,'w+') as f:
                for j in sublist:
                    cls = int(targets[j, 1].item())
                    x1 = float(targets[j, 2].item())
                    y1 = float(targets[j, 3].item())
                    w2 = float(targets[j, 4].item())
                    h2 = float(targets[j, 5].item())
                    bb = [x1, y1, w2, h2]
                    c1 = bb[0] - bb[2] / 2  # top left x
                    c2 = bb[1] - bb[3] / 2  # top left y
                    c3 = bb[0] + bb[2] / 2  # bottom right x
                    c4 = bb[1] + bb[3] / 2  # bottom right y
                    c1 = int(c1 * 640)
                    c2 = int(c2 * 640)
                    c3 = int(c3 * 640)
                    c4 = int(c4 * 640)
                    cc = [c1, c2, c3, c4]
                    try:
                        # cv2.imshow(str(cls), mat[c2:c4, c1:c3])
                        cv2.imwrite("/home/zeta/LuoQiang/data/datasets/data/3class/marks/" + name + str(j) + '_' + str(
                            cls) + '.jpg', mat[c2:c4, c1:c3])
                        cv2.imwrite('/home/zeta/LuoQiang/data/datasets/data/3class/images/' + name, mat)
                    except:
                        erro.append(paths[si])
                        count += 1
                    f.write(str(cls) + " " + " ".join([str(a) for a in bb]) + '\n')
            # array1 = imga * 255  # normalize，将图像数据扩展到[0,255] --69906这张图片可以测试一下
            # mat = np.uint8(array1.cpu())  # float32-->uint8
            # print('左上角点的坐标为:', mat.shape)  # mat_shape: (3, 982, 814)
            # print("----------------------",w1,h1)
            # print("----------------------",paths[si])

            print('------------------->处理完毕',name)
            # cv2.imwrite('/home/zeta/LuoQiang/data/datasets/data/newdata/images/'+name,mat)
            # cv2.imshow("img", mat)
            # cv2.waitKey()
            # bboxes.append(bbox)
        print('出错图片数目：', count,erro)



def parse_opt():
    parser = argparse.ArgumentParser(prog='val.py')
    parser.add_argument('--data', type=str, default='data/test.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default='runs/train/exp108/weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=2048, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment',default=False, action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt',default=False, action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', default=True,action='store_true', help='save a COCO-JSON results file')
    parser.add_argument('--project', default='runs/val', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument("--classwise", action="store_true", help="Whether to evaluating the AP for each class")
    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.save_txt |= opt.save_hybrid
    opt.data = check_file(opt.data)  # check file
    return opt


def main(opt):
    set_logging()
    print(colorstr('val: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    check_requirements(requirements=FILE.parent / 'requirements.txt', exclude=('tensorboard', 'thop'))

    if opt.task in ('train', 'val', 'test'):  # run normally
        run(**vars(opt))

    elif opt.task == 'speed':  # speed benchmarks
        for w in opt.weights if isinstance(opt.weights, list) else [opt.weights]:
            run(opt.data, weights=w, batch_size=opt.batch_size, imgsz=opt.imgsz, conf_thres=.25, iou_thres=.45,
                save_json=False, plots=False)

    elif opt.task == 'study':  # run over a range of settings and save/plot
        # python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5s.pt yolov5m.pt yolov5l.pt yolov5x.pt
        x = list(range(256, 1536 + 128, 128))  # x axis (image sizes)
        for w in opt.weights if isinstance(opt.weights, list) else [opt.weights]:
            f = f'study_{Path(opt.data).stem}_{Path(w).stem}.txt'  # filename to save to
            y = []  # y axis
            for i in x:  # img-size
                print(f'\nRunning {f} point {i}...')
                r, _, t = run(opt.data, weights=w, batch_size=opt.batch_size, imgsz=i, conf_thres=opt.conf_thres,
                              iou_thres=opt.iou_thres, save_json=opt.save_json, plots=False)
                y.append(r + t)  # results and times
            np.savetxt(f, y, fmt='%10.4g')  # save
        os.system('zip -r study.zip study_*.txt')
        plot_study_txt(x=x)  # plot


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
