import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread("image02.jpeg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

hsv_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

# hsv[h, w, c]
hsv_img[:, :, 0] += 50 # Hue -> 50도 증가
hsv_img[:, :, 1] = np.uint8(hsv_img[:, :, 1] * 0.5)  # 채도
hsv_img[:, :, 2] = np.uint8(hsv_img[:, :, 2] * 1.5)  # 밝기

# imshow <- BGR / RGB 로 강제로 디코딩
rgb_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
plt.imshow(rgb_img)
plt.show()