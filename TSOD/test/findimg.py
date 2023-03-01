import os
import shutil

oldimg_path ="/home/zeta/LuoQiang/model/Target_Detection/CVPR/YOLOX-Lite/inference/oldimg/"
newimg_path = "/home/zeta/LuoQiang/model/Target_Detection/CVPR/YOLOX-Lite/inference/newimg/"
imgpath = "/home/zeta/LuoQiang/data/datasets/data/images/"

newimg_list = os.listdir(newimg_path)

for i in newimg_list:
    oldpath = imgpath+i
    shutil.copyfile(oldpath,oldimg_path+i)