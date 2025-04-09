import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread("image02.jpeg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

gray_img = cv2.imread('image02.jpeg', cv2.IMREAD_GRAYSCALE)
h, w = gray_img.shape

mean = 0
var = 100
sigma = var ** 0.5

gaussian = np.random.normal(mean, sigma, (h, w))
noisy_image = gray_img + gaussian.astype(np.uint8)
# uint8 -> 0 ~ 255
cv2.imshow("", noisy_image)
cv2.waitKey()
