import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread("image02.jpeg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

### 채도 조정 ####
img = cv2.imread('image02.jpeg')
org_img = img.copy()

img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
saturation_factor = 1.5
img_hsv[:, :, 1] = img_hsv[:, :, 1] * saturation_factor

saturated_img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

cv2.imshow("", org_img)
cv2.waitKey()
cv2.imshow("", saturated_img)
cv2.waitKey()


#### 밝기 조정 ####
img_temp = cv2.imread('image02.jpeg')

bright_diff = 50
img_brighten = cv2.convertScaleAbs(img_temp, alpha=1, beta=bright_diff)

cv2.imshow("brighten", img_brighten)
cv2.waitKey()
