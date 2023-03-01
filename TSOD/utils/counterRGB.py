import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt

img = "/home/zeta/LuoQiang/data/datasets/data/images/2.jpg"

img = cv.imread(img)
print(img.shape)
print(img[0][0])