import cv2
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image

def featuremap_2_heatmap(feature_map):
    assert isinstance(feature_map, torch.Tensor)
    feature_map = feature_map.detach()
    # print(feature_map.shape)
    heatmap = feature_map[:,0,:,:]*0 # 对第二个维度外的参数
    heatmaps = []
    for c in range(feature_map.shape[1]):
        heatmap+=feature_map[:,c,:,:]
    heatmap = heatmap.cpu().numpy()
    heatmap = np.mean(heatmap, axis=0)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmaps.append(heatmap)

    return heatmaps

def draw_feature_map(features,img_path,type = None,cout=0):
    # img = cv2.imread("/home/zeta/LuoQiang/model/Target_Detection/CVPR/yolor/inference/newimg/2.jpg")
    save_path = "/home/zeta/LuoQiang/model/Target_Detection/CVPR/YOLOX-Lite/demo/demo5/"
    # 图像加载&预处理
    if img_path == None:
        return
    if isinstance(features,torch.Tensor):
        print(features.shape)
        imgname = img_path.strip().split('/')[-1]
        for heat_maps in features:
            heat_maps=heat_maps.unsqueeze(0)
            heatmaps = featuremap_2_heatmap(heat_maps)
            # 这里的h,w指的是你想要把特征图resize成多大的尺寸
            # heatmap = cv2.resize(heatmap, (h, w))
            for heatmap in heatmaps:
                cout += 1
                # heatmap = np.uint8(255 * heatmap)
                # # 下面这行将热力图转换为RGB格式 ，如果注释掉就是灰度图
                # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                # superimposed_img = heatmap
                # # cv2.imshow(superimposed_img)
                # # cv2.waitKey(0)
                # # cv2.destroyAllWindows()
                # plt.imshow(superimposed_img)
                # plt.show()
                # plt.savefig(save_dir + 'feature_map_{}.png'
                #             .format(str(cout)), dpi=300)

                img = cv2.imread(img_path)  # 用cv2加载原始图像
                # print(heatmap.shape,img.shape)
                heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
                heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
                superimposed_img = heatmap * 0.7 + img  # 这里的0.4是热力图强度因子
                # plt.savefig(save_dir + 'feature_map_{}.png'
                #             .format(str(cout)), dpi=300)
                cv2.imwrite(save_path+"_"+str(cout)+type+imgname, superimposed_img)  # 将图像保存到硬盘
    else:
        for featuremap in features:
            #img = np.float32(img)
            heatmaps = featuremap_2_heatmap(featuremap)
            #heatmaps = cv2.resize(heatmaps, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
            for heatmap in heatmaps:
                heatmap = np.uint8(255 * heatmap)
                #img=np.uint8(255 * heatmap)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))# 将热力图转换为RGB格式
                superimposed_img = heatmap * 0.5+ img*0.3
                #superimposed_img = heatmap
                plt.imshow(superimposed_img)
                plt.show()
                #下面这些是对特征图进行保存，使用时取消注释
                #cv2.imshow(superimposed_img)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()
                cv2.imwrite(os.path.join("/home/zeta/LuoQiang/model/Target_Detection/CVPR/YOLOX-Lite/demo","feavis" +str(i)+'.jpg'), superimposed_img)
                #i=i+1



# path = "F:/gts/gtsdate/"
# b = os.path.join(path,"/abc")