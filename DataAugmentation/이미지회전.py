import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread("image02.jpeg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# rotation
angle = 30

# 이미지 중심점 기준 회전 행렬 생성
h, w = image.shape[:2]
center = (w//2, h//2)
M = cv2.getRotationMatrix2D(center, angle, 1.0)
# getRotationMatrix2D(중심 좌표, 회전 각도, 크기 변환 비율)

# 회전 적용
rotated_img = cv2.warpAffine(image, M, (w, h))
                # warpAffine(원본 이미지, 회전 행렬, 이미지 크기)

plt.imshow(image)
plt.show()

plt.imshow(rotated_img)
plt.show()



