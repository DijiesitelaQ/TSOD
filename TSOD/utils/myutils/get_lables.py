import os
import cv2
txt_file = "/home/zeta/LuoQiang/data/datasets/STS/annotations2.txt"
lable_file = "/home/zeta/LuoQiang/data/datasets/STS/labels/"

imgsize = [1280,960]
# a = set()
b = ['MANDATORY', 'UNREADABLE', 'INFORMATION', 'OTHER', 'WARNING', 'PROHIBITORY']

def convert(size, box):
    print(box)
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

with open(txt_file,"r") as f:
    for lin in f.readlines():
        lin = lin.strip('\n')
        isemp =  lin.split(":")
        imgname = isemp[0].split(".")[0]
        if isemp[1] == '':
            with open(lable_file+imgname+".txt","w+") as f1:
                f1.write(" ")
            print(imgname+"========="+"为空")
        else:
            lables = isemp[1].split(";")
            print(imgname)
            with open(lable_file+imgname+".txt","w+") as f2:
                for lable in lables:
                    if lable=='':
                        break
                    if lable == "MISC_SIGNS":
                        continue
                    lable = lable.replace(' ', '')
                    lable = lable.split(",")
                    print(lable)
                    if lable == '':
                        continue
                    taget_name = lable[-2]
                    x2,y2,x1,y1 = lable[-6],lable[-5],lable[-4],lable[-3]
                    box = [float(x1),float(x2),float(y1),float(y2)]
                    if imgname=="1277381680Image000009":
                        print("================",box)
                    bb = convert(imgsize,box)
                    cls_id = b.index(taget_name)
                    f2.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
                print(imgname+"------------------->完成")

# image = cv2.imread("/home/zeta/LuoQiang/data/datasets/STS/images/1277381655Image000022.jpg")
# # jietu_image = image[915: 426, 840: 349]
# # jietu_image = image[349: 426, 840: 915]
# # jietu_image = image[ 840: 915,349: 426]
# cv2.imshow("jietu_image",jietu_image)

# cv2.waitKey(0)

# print(a)