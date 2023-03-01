import os
import cv2
txt_file = "/home/zeta/LuoQiang/data/datasets/STS/annotations2.txt"
lable_file = "/home/zeta/LuoQiang/data/datasets/STS/test/"
img_file = "/home/zeta/LuoQiang/data/datasets/STS/images/"
imglist = os.listdir(lable_file)

# with open("/home/zeta/LuoQiang/data/datasets/STS/test.txt","w") as f:
#     for i in imglist:
#         f.write(img_file+i+"\n")
i = 0
olist = []
cout = 0
# while cout < 750:
# #         # f.write(img_file+imglist[i]+"\n")
#     olist.append(i)
#     i = i + 2
#     cout += 1

# print(olist,cout)

with open("/home/zeta/LuoQiang/data/datasets/STS/test.txt","w") as f:
    while cout < 750:
#         #
#         olist.append(i)
        i = i + 2
        cout += 1
        olist.append(imglist[i])
        print(img_file+imglist[i]+"\n")
        f.write(img_file+imglist[i]+"\n")

total = os.listdir(img_file)
with open("/home/zeta/LuoQiang/data/datasets/STS/train.txt","w") as f:
    for i in total:
        if i not in olist:

            f.write(img_file+i+"\n")