import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread("image02.jpeg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

org_img = image.copy()
balance = [0.8, 0.7, 0.8]

for i, value in enumerate(balance):
    if value != 1.0:
        org_img[:, :, i] = cv2.addWeighted(org_img[:,:,i], value, 0, 0, 0)
                            # addWeighted: src에 대해 value만큼의 가중치로 색온도 조절

plt.imshow(org_img)
plt.show()
