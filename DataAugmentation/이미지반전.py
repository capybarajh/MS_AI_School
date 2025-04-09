import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread("image02.jpeg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

flipped_img_updown = cv2.flip(image, 0) # 상하반전
flipped_img_leftright = cv2.flip(image, 1) # 좌우반전
flipped_img_lr_other = cv2.flip(image, -1) # 상하 & 좌우반전

plt.imshow(image)
plt.show()
plt.imshow(flipped_img_updown)
plt.show()
plt.imshow(flipped_img_leftright)
plt.show()
plt.imshow(flipped_img_lr_other)
plt.show()