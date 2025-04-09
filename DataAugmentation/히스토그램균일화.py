import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread("image02.jpeg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

img_gray = cv2.imread("image02.jpeg", cv2.IMREAD_GRAYSCALE)
img_equalized = cv2.equalizeHist(img_gray)

cv2.imshow("org", img_gray)
cv2.imshow("hist_equal", img_equalized)
cv2.waitKey()
