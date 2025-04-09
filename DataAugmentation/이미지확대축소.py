import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread("image02.jpeg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 이미지 크기 설정
h, w = image.shape[:2]

# 확대/축소할 배율 설정
zoom_scale = 4

# 이미지 확대
enlarged_img = cv2.resize(image, (w*zoom_scale, h*zoom_scale), interpolation=cv2.INTER_CUBIC)
            # resize(원본 이미지, (최종 너비, 최종 높이), 이미지 보간 방법 (ex: cv2.INTER_CUBIC))
plt.imshow(enlarged_img)
plt.show()

# 이미지 축소
center = [enlarged_img.shape[0] // 2, enlarged_img.shape[1] // 2]
cut_half = 300
zoomed_img = enlarged_img[center[0]-cut_half:center[0]+cut_half, center[1]-cut_half:center[1]+cut_half]
plt.imshow(zoomed_img)
plt.show()