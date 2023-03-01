import cv2
import os

path = "/home/zeta/LuoQiang/data/datasets/DFG/images/0004085.jpg"

img = cv2.imread(path)
print(img.shape) # (1080, 1920, 3)
img2 = img[120:600,1400:1680]
cv2.imshow("img",img2)

cv2.waitKey(0)
