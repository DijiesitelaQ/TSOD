from skimage.metrics  import structural_similarity as  compare_ssim, peak_signal_noise_ratio as compare_psnr
import cv2


def getSSIM(img1, img2):
    print("111")
    return compare_ssim(img1, img2, multichannel=True)  # 对于多通道图像(RGB、HSV等)关键词multichannel要设置为True


def getPSNR(img1, img2):
    return compare_psnr(img1, img2)


def getMSE(img1, img2):
    return compare_mse(img1, img2)



markpath = "/home/zeta/LuoQiang/data/datasets/data/marks/pad-all/w22.png"
labelpath = "../mydata/w22/3304_3_.jpg"
img1 = cv2.imread(markpath)
img2 = cv2.imread(labelpath)
print(img2.shape)
img1 = cv2.resize(img1,(img2.shape[1],img2.shape[0]))
cv2.imshow("a",img1)
key = cv2.waitKey(0)
print(img1.shape)

res = getSSIM(img1, img2)
print(res)
pscore = getPSNR(img1, img2)
print(pscore)
# getMSE(img1, img2)
