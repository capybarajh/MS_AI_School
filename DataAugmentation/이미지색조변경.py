import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread("image02.jpeg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

hsv_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
hue_shift = 30
hsv_img[:, :, 0] = (hsv_img[:, :, 0] + hue_shift) % 180
rgb_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)

plt.imshow(image)
plt.show()
plt.imshow(rgb_img)
plt.show()