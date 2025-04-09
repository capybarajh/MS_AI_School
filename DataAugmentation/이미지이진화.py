import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread("image02.jpeg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
# ADAPTIVE_THRESH_MEAN_C: 적응형 임계값 처리, 임계값 기준을 평균치를 사용함
# 인자 11: 블록 크기, 11x11 블록으로 이미지를 나눈 후 해당 영역
plt.imshow(img_gray, 'gray')
plt.show()
plt.imshow(thresh, 'gray')
plt.show()
