# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
import os
from os import getcwd
import cv2

classes = ["1"]
abs_path = os.getcwd()
print(abs_path)

def convert(size, box):
    # dw = 1. / (size[0])
    # dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    # x = x * dw
    # w = w * dw
    # y = y * dh
    # h = h * dh
    return x, y, w, h

def convert_annotation(image_id,res):
    in_file = open('/home/zeta/LuoQiang/data/datasets/data/XML/%s.xml' % (image_id), encoding='UTF-8')
    # out_file = open('/home/zeta/LuoQiang/data/datasets/data/labels/%s.txt' % (image_id), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    s = ['i1', 'i2', 'i3', 'i4', 'i5', 'i6', 'i7', 'i8', 'i9', 'i10', 'i11', 'i12', 'i13', 'i14', 'i15', 'il50', 'il60', 'il70', 'il80', 'il90',
        'il100', 'il110', 'ilx', 'ip', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11', 'p12', 'p13', 'p14', 'p15', 'p16', 'p17',
        'p18', 'p19', 'p20', 'p21', 'p22', 'p23', 'p24', 'p25', 'p26', 'p27', 'p28', 'p29', 'pa8', 'pa10', 'pa12', 'pa13', 'pa14', 'pax', 'pb', 'pc',
        'pd', 'pe', 'pg', 'ph1.5', 'ph2.1', 'ph2.2', 'ph2.4', 'ph2.5', 'ph2.6', 'ph2.8', 'ph2.9', 'ph3.2', 'ph3.3', 'ph3.5', 'ph3.8', 'ph4.2', 'ph4.3',
        'ph4.4', 'ph4.5', 'ph4.8', 'ph5.3', 'ph5.5', 'ph2', 'ph3', 'ph4', 'ph5', 'phx', 'pl0', 'pl3', 'pl4', 'pl5', 'pl10', 'pl15', 'pl20', 'pl25', 'pl30',
        'pl35', 'pl40', 'pl50', 'pl60', 'pl65', 'pl70', 'pl80', 'pl90', 'pl100', 'pl110', 'pl120', 'plx', 'pm1.5', 'pm2.5', 'pm2', 'pm5', 'pm8', 'pm10',
        'pm13', 'pm15', 'pm20', 'pm25', 'pm30', 'pm35', 'pm40', 'pm46', 'pm50', 'pm55', 'pmx', 'pn', 'pn40', 'pne', 'pnl', 'pr10', 'pr20', 'pr30', 'pr40',
        'pr45', 'pr50', 'pr60', 'pr70', 'pr80', 'pr100', 'prx', 'ps', 'pw2.5', 'pw3.2', 'pw3.5', 'pw4.2', 'pw4.5', 'pw2', 'pw3', 'pw4', 'pwx', 'w1', 'w2',
        'w3', 'w4', 'w5', 'w6', 'w7', 'w8', 'w9', 'w10', 'w11', 'w12', 'w13', 'w14', 'w15', 'w16', 'w17', 'w18', 'w19', 'w20', 'w21', 'w22', 'w23', 'w24',
        'w25', 'w26', 'w27', 'w28', 'w29', 'w30', 'w31', 'w32', 'w33', 'w34', 'w35', 'w36', 'w37', 'w38', 'w39', 'w40', 'w41', 'w42', 'w43', 'w44', 'w45',
        'w46', 'w47', 'w48', 'w49', 'w50', 'w51', 'w52', 'w53', 'w54', 'w55', 'w56', 'w57', 'w58', 'w59', 'w60', 'w61', 'w62', 'w63', 'w64', 'w65', 'w66', 'w67']
    for obj in root.iter('object'):
        # difficult = obj.find('difficult').text
        # difficult = obj.find('Difficult').text
        name = obj.find('name')
        strname = name.text
        # print(strname)
        if strname in ['io','po','wo']:
            continue
        ids = s.index(strname)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        b1, b2, b3, b4 = b
        # 标注越界修正
        if b2 > w:
            b2 = w
        if b4 > h:
            b4 = h
        b = (b1, b2, b3, b4)
        bb = convert((w, h), b)
        if bb[2]*bb[3]<32*32:
            res['s'].append(image_id)
        elif 96*96>bb[2]*bb[3]>32*32:
            res['m'].append(image_id)
        else:
            res['l'].append(image_id)
        # out_file.write(str(ids) + " " + " ".join([str(a) for a in bb]) + '\n')
        # out_file.write(str(0) + " " + " ".join([str(a) for a in bb]) + '\n')
    # print(image_id+"========> 生成！")
    in_file.close()
    return res


wd = getcwd()

image_set = ["train","val","test"]
listImg = os.listdir("/home/zeta/LuoQiang/data/datasets/data/images/")

# print(test_img, len(test_img))
# for j in listImg:
#     imgid = j.split(".")[0]
#     convert_annotation(imgid)

# /home/zeta/LuoQiang/data/datasets/data/ImageSets/Main
# print(len(listImg))  以下算法是用来处理train.txt的生成
# abs_path = "/home/zeta/LuoQiang/data/datasets/data/images/"
# image_ids = open('/home/zeta/LuoQiang/model/Target_Detection/CVPR/tph-yolov5/paper2_data/Main/%s.txt' % (image_set[1])).read().strip().split()
# test_list_file = open('/home/zeta/LuoQiang/data/datasets/data/%s.txt' % (image_set[2]), 'w+')
# train_list_file = open('/home/zeta/LuoQiang/data/datasets/data/%s.txt' % (image_set[0]), 'w+')
def getlist():
    res = {'s': [], 'm': [], 'l': []}
    with open("/home/zeta/LuoQiang/data/datasets/data/val.txt",'r') as f:
        for lin in f.readlines():
            lin = lin.strip('\n')
            imgid = lin.split('/')[-1].split('.')[0]
            res = convert_annotation(imgid,res)
    return res
# print(len(getlist()['s']))

def find_erro_img(image_id,res,erro):
    in_file = open('/home/zeta/LuoQiang/data/datasets/data/XML/%s.xml' % (image_id), encoding='UTF-8')
    # out_file = open('/home/zeta/LuoQiang/data/datasets/data/labels/%s.txt' % (image_id), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    s = ['i1', 'i2', 'i3', 'i4', 'i5', 'i6', 'i7', 'i8', 'i9', 'i10', 'i11', 'i12', 'i13', 'i14', 'i15', 'il50', 'il60', 'il70', 'il80', 'il90',
        'il100', 'il110', 'ilx', 'ip', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11', 'p12', 'p13', 'p14', 'p15', 'p16', 'p17',
        'p18', 'p19', 'p20', 'p21', 'p22', 'p23', 'p24', 'p25', 'p26', 'p27', 'p28', 'p29', 'pa8', 'pa10', 'pa12', 'pa13', 'pa14', 'pax', 'pb', 'pc',
        'pd', 'pe', 'pg', 'ph1.5', 'ph2.1', 'ph2.2', 'ph2.4', 'ph2.5', 'ph2.6', 'ph2.8', 'ph2.9', 'ph3.2', 'ph3.3', 'ph3.5', 'ph3.8', 'ph4.2', 'ph4.3',
        'ph4.4', 'ph4.5', 'ph4.8', 'ph5.3', 'ph5.5', 'ph2', 'ph3', 'ph4', 'ph5', 'phx', 'pl0', 'pl3', 'pl4', 'pl5', 'pl10', 'pl15', 'pl20', 'pl25', 'pl30',
        'pl35', 'pl40', 'pl50', 'pl60', 'pl65', 'pl70', 'pl80', 'pl90', 'pl100', 'pl110', 'pl120', 'plx', 'pm1.5', 'pm2.5', 'pm2', 'pm5', 'pm8', 'pm10',
        'pm13', 'pm15', 'pm20', 'pm25', 'pm30', 'pm35', 'pm40', 'pm46', 'pm50', 'pm55', 'pmx', 'pn', 'pn40', 'pne', 'pnl', 'pr10', 'pr20', 'pr30', 'pr40',
        'pr45', 'pr50', 'pr60', 'pr70', 'pr80', 'pr100', 'prx', 'ps', 'pw2.5', 'pw3.2', 'pw3.5', 'pw4.2', 'pw4.5', 'pw2', 'pw3', 'pw4', 'pwx', 'w1', 'w2',
        'w3', 'w4', 'w5', 'w6', 'w7', 'w8', 'w9', 'w10', 'w11', 'w12', 'w13', 'w14', 'w15', 'w16', 'w17', 'w18', 'w19', 'w20', 'w21', 'w22', 'w23', 'w24',
        'w25', 'w26', 'w27', 'w28', 'w29', 'w30', 'w31', 'w32', 'w33', 'w34', 'w35', 'w36', 'w37', 'w38', 'w39', 'w40', 'w41', 'w42', 'w43', 'w44', 'w45',
        'w46', 'w47', 'w48', 'w49', 'w50', 'w51', 'w52', 'w53', 'w54', 'w55', 'w56', 'w57', 'w58', 'w59', 'w60', 'w61', 'w62', 'w63', 'w64', 'w65', 'w66', 'w67']
    for obj in root.iter('object'):
        # difficult = obj.find('difficult').text
        # difficult = obj.find('Difficult').text
        name = obj.find('name')
        strname = name.text
        # print(strname)
        if strname in ['io','po','wo']:
            continue
        ids = s.index(strname)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        b1, b2, b3, b4 = b
        # 标注越界修正
        if b2 > w:
            b2 = w
        if b4 > h:
            b4 = h
        b = (b1, b2, b3, b4)
        if strname == erro:
            res.update({image_id:b})
            # res.append(image_id)
        # out_file.write(str(ids) + " " + " ".join([str(a) for a in bb]) + '\n')
        # out_file.write(str(0) + " " + " ".join([str(a) for a in bb]) + '\n')
    # print(image_id+"========> 生成！")
    in_file.close()
    return res

def showPicture(err):
    res = dict()
    with open("/home/zeta/LuoQiang/data/datasets/data/newdata/val.txt",'r') as f:
        for lin in f.readlines():
            lin = lin.strip('\n')
            imgid = lin.split('/')[-1].split('.')[0]
            r = find_erro_img(imgid,res,err)
            res.update(r)
    return res

def show():
    err = 'ph2.9'
    res = showPicture(err)
    print(res)
    for i in res:
        pathroot = '/home/zeta/LuoQiang/data/datasets/data/newdata/images/'
        imgpath = pathroot+i+'.jpg'
        box = res[i]
        img = cv2.imread(imgpath)
        cv2.rectangle(img, (int(box[0]/3.2), int(box[1]/3.2)), (int(box[2]/3.2), int(box[3]/3.2)), (0, 0, 255), 2)
        cv2.imshow(err,img)
        cv2.waitKey()