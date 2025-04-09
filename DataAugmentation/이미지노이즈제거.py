import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread("image02.jpeg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

img_filtered = cv2.medianBlur(image, 5)
plt.imshow(image)
plt.show()
plt.imshow(img_filtered)
plt.show()
