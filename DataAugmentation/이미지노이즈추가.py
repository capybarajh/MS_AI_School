import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread("image02.jpeg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

noise = np.zeros(image.shape, np.uint8) # uint8 = unsigned int 8-bit (부호 없는 1바이트 정수)
cv2.randu(noise, 0, 255)
black = noise < 30 # [True, True, False, False, False, ...] 형태의 Mask 생성
white = noise > 225
noise[black] = 0
noise[white] = 255

noise_b = noise[:, :, 0] # image.shape (h, w, c) -> h*w*c -> color channel : B, G, R
noise_g = noise[:, :, 1]
noise_r = noise[:, :, 2]
noisy_img = cv2.merge([
    cv2.add(image[:, :, 0], noise_b),
    cv2.add(image[:, :, 1], noise_g),
    cv2.add(image[:, :, 2], noise_r)
])

plt.imshow(image)
plt.show()
plt.imshow(noisy_img)
plt.show()