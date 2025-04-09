import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread("image02.jpeg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


x, y, w, h = 300, 300, 200, 200  # (100, 100) 좌표에서 (200 * 200) 크기로 자를 것임
crop_img_wide = image[y-h:y+h, x-w:x+w] # (x, y) 를 중심으로 2w, 2h 크기로 자름
crop_img_lt = image[y:y+h, x:x+w] # (x, y) 를 기점으로 (w, h) 만큼 오른쪽 아래로 간 크기로 자름

plt.imshow(image)
plt.show()
plt.imshow(crop_img_wide)
plt.show()
plt.imshow(crop_img_lt)
plt.show()
