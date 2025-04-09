import cv2
import matplotlib.pyplot as plt
import numpy as np


# ex1
image = cv2.imread("image02.jpeg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

mean = 0
var = 100
sigma = var ** 0.5

gauss = np.random.normal(mean, sigma, image.shape)
gauss = gauss.astype('uint8')

noisy_img = cv2.add(image, gauss)

plt.imshow(noisy_img)
plt.show()

# ex2
img = cv2.imread("image02.jpeg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

variance = 0.1
sigma2 = variance ** 2
gaussian_noise = np.random.normal(0, sigma2, img.shape)

img_noisy = img + gaussian_noise

plt.imshow(np.uint8(np.clip(img_noisy, 0, 255)))
plt.show()

